from pathlib import Path
import re
from typing import List

import hydra
from omegaconf import DictConfig
from pyspark.sql import SparkSession, functions as F, types as T, Window

# ---- Helpers to read wide train_date ----
def _read_header(p: str):
    with open(p, "r", encoding="utf-8") as f:
        return f.readline().strip().split(",")

def _schema(header):
    fs = []
    for c in header:
        if c == "Id":
            fs.append(T.StructField(c, T.LongType(), False))
        else:
            fs.append(T.StructField(c, T.LongType(), True))  # ticks as long
    return T.StructType(fs)

def _read_date_csv(spark, path: str):
    header = _read_header(path)
    return (
        spark.read.option("header", True)
        .option("mode", "PERMISSIVE").option("nullValue", "")
        .schema(_schema(header)).csv(path)
    )

# ---- Parse L{line}_S{station}_D{d} once ----
_pat = re.compile(r"^L(\d+)_S(\d+)_D\d+$")
def _parse_col(name: str):
    m = _pat.match(name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))

# ---- Melt a batch of date columns to long ----
def _melt_batch(df_date, cols: List[str], start_ord: int):
    dfs = []
    ord_idx = start_ord
    for c in cols:
        parsed = _parse_col(c)
        if not parsed:
            continue
        line, station = parsed
        dfs.append(
            df_date.select(
                F.col("Id").alias("Id"),
                F.lit(line).cast("int").alias("line"),
                F.lit(station).cast("int").alias("station"),
                F.col(c).cast("long").alias("tick"),
                F.lit(ord_idx).cast("int").alias("col_ordinal"),
            ).where(F.col(c).isNotNull())
        )
        ord_idx += 1
    if not dfs:
        return None, start_ord
    out = dfs[0]
    for d in dfs[1:]:
        out = out.unionByName(d)
    return out, ord_idx

# ---- Sum of events (validation) ----
def _n_events_sum_batched(df_date, cols_all: List[str], batch_size: int):
    total = 0
    for i in range(0, len(cols_all), batch_size):
        cols = cols_all[i:i+batch_size]
        aggs = [
            F.sum(F.when(F.col(c).isNotNull(), 1).otherwise(0)).cast("long").alias(c)
            for c in cols
        ]
        row = df_date.agg(*aggs).collect()[0].asDict()
        total += sum(int(v) for v in row.values())
    return total

@hydra.main(config_path="../../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    builder = (SparkSession.builder
               .appName("CIRP2026-Day2-EventsLong")
               .master(cfg.spark.master))
    for k, v in cfg.spark.extra_conf.items():
        builder = builder.config(k, v)
    spark = builder.getOrCreate()
    spark.conf.set("spark.sql.shuffle.partitions", cfg.spark.shuffle_partitions)

    raw = Path(cfg.data.raw_dir)
    path_date = str(raw/"train_date.csv")
    print("raw_date:", path_date)

    df_date = _read_date_csv(spark, path_date)
    dcols_all = [c for c in df_date.columns if c != "Id" and _parse_col(c) is not None]
    print(f"[events_long] date columns detected: {len(dcols_all)}")

    # --- batched melt -> direct write
    batch = int(cfg.seq.batch_date_cols)
    out_tmp = Path(cfg.seq.out_parquet) / "events_long_tmp"
    wrote_any = False
    for i in range(0, len(dcols_all), batch):
        cols = dcols_all[i:i+batch]
        part, _ = _melt_batch(df_date, cols, start_ord=i)
        if part is None:
            continue
        mode = "overwrite" if not wrote_any else "append"
        (part.repartition(256, F.col("Id"))
             .write.mode(mode).format("parquet")
             .save(str(out_tmp)))
        wrote_any = True
        print(f"[events_long] wrote batch {i//batch + 1}/{(len(dcols_all)+batch-1)//batch}")

    if not wrote_any:
        raise RuntimeError("No valid date columns found to melt.")

    # --- load written events, rank, persist final
    events_raw = spark.read.parquet(str(out_tmp))
    w = Window.partitionBy("Id").orderBy(
        F.col("tick").asc(), F.col("line").asc(), F.col("station").asc(), F.col("col_ordinal").asc()
    )
    events_long = events_raw.withColumn("rank", F.row_number().over(w)).drop("col_ordinal")

    dst = Path(cfg.data.lake_dir) / "parquet" / "events_long"
    events_long.repartition(256, "Id").write.mode("overwrite").parquet(str(dst))

    # ---- Checks ----
    n_events_sum = _n_events_sum_batched(df_date, dcols_all, batch)
    ev_count = events_long.count()
    print(f"[check] sum(n_events over wide) = {n_events_sum}")
    print(f"[check] events_long.count()     = {ev_count}")

    w2 = Window.partitionBy("Id").orderBy("rank")
    neg = (events_long.withColumn("prev_tick", F.lag("tick").over(w2))
                       .withColumn("dt", F.col("tick") - F.col("prev_tick"))
                       .where((F.col("rank")>1) & (F.col("dt") < 0)).limit(1).count())
    print(f"[check] any negative Δt? {'YES' if neg>0 else 'NO'}")

    print("[events_long] DONE")
    spark.stop()

if __name__ == "__main__":
    main()
