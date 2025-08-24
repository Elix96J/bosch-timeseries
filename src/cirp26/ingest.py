from pathlib import Path
from typing import List
import os
import hydra
from omegaconf import DictConfig
from pyspark.sql import SparkSession, functions as F, types as T
import mlflow
from functools import reduce
import operator

# ---------- CSV helpers ----------
def _read_header(p: str):
    with open(p, "r", encoding="utf-8") as f:
        return f.readline().strip().split(",")

def _schema(header, dom: str):
    m = {"numeric": T.DoubleType(), "categorical": T.StringType(), "date": T.LongType()}
    fields = []
    for c in header:
        if c == "Id":
            fields.append(T.StructField(c, T.LongType(), False))
        elif c == "Response":
            fields.append(T.StructField(c, T.IntegerType(), True))
        else:
            fields.append(T.StructField(c, m[dom], True))
    return T.StructType(fields)

def _read_csv(spark, path: str, dom: str):
    return (
        spark.read.option("header", True)
        .option("mode", "PERMISSIVE")
        .option("nullValue", "")
        .schema(_schema(_read_header(path), dom))
        .csv(path)
    )

# ---------- Keys from date (batched & safe) ----------
def _derive_keys_from_date(df_date):
    id_col = "Id"
    date_cols = [c for c in df_date.columns if c != id_col]
    line_vals, station_vals, event_vals = [], [], []
    for c in date_cols:
        parts = c.split("_")
        if len(parts) < 3 or not parts[2].startswith("D"):
            continue
        try:
            line = int(parts[0][1:])
            station = int(parts[1][1:])
        except ValueError:
            continue
        col = F.col(c)
        line_vals.append(F.when(col.isNotNull(), F.lit(line)))
        station_vals.append(F.when(col.isNotNull(), F.lit(station)))
        event_vals.append(F.when(col.isNotNull(), F.lit(1)).otherwise(F.lit(0)))

    line_key      = (F.array_min(F.array(*line_vals)) if line_vals else F.lit(None).cast("int")).alias("line_key")
    station_first = (F.array_min(F.array(*station_vals)) if station_vals else F.lit(None).cast("int")).alias("station_first")
    station_last  = (F.array_max(F.array(*station_vals)) if station_vals else F.lit(None).cast("int")).alias("station_last")
    n_events      = (reduce(operator.add, event_vals, F.lit(0)) if event_vals else F.lit(0)).alias("n_events")
    return df_date.select(F.col(id_col), line_key, station_first, station_last, n_events)

def _min_nullable(a, b):
    return F.when(a.isNull(), b).when(b.isNull(), a).otherwise(F.least(a, b))

def _max_nullable(a, b):
    return F.when(a.isNull(), b).when(b.isNull(), a).otherwise(F.greatest(a, b))

def _derive_keys_batched(df_date, batch_size: int = 200):
    date_cols_all = [c for c in df_date.columns if c != "Id"]
    batches: List[List[str]] = [date_cols_all[i:i+batch_size] for i in range(0, len(date_cols_all), batch_size)]
    agg = None
    for i, cols in enumerate(batches, 1):
        part = _derive_keys_from_date(df_date.select("Id", *cols))
        if agg is None:
            agg = part
        else:
            a, b = agg.alias("a"), part.alias("b")
            agg = (
                a.join(b, "Id", "outer")
                 .select(
                    F.col("Id"),
                    _min_nullable(F.col("a.line_key"),      F.col("b.line_key")).alias("line_key"),
                    _min_nullable(F.col("a.station_first"), F.col("b.station_first")).alias("station_first"),
                    _max_nullable(F.col("a.station_last"),  F.col("b.station_last")).alias("station_last"),
                    (F.coalesce(F.col("a.n_events"), F.lit(0)) + F.coalesce(F.col("b.n_events"), F.lit(0))).alias("n_events"),
                 )
            )
        print(f"[keys] merged batch {i}/{len(batches)} with {len(cols)} cols")
    return agg

# ---------- Null map ----------
def _null_stats(df, name, batch_size: int = 200):
    total = df.count()
    dtypes = dict(df.dtypes)
    cols = list(dtypes.keys())
    rows = []
    for i in range(0, len(cols), batch_size):
        chunk = cols[i:i+batch_size]
        aggs = []
        for c in chunk:
            t = dtypes[c]
            if t in ("double", "float"):
                aggs.append(F.sum(F.when(F.isnan(F.col(c)) | F.col(c).isNull(), 1).otherwise(0)).alias(c))
            else:
                aggs.append(F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c))
        d = df.agg(*aggs).collect()[0].asDict()
        for col, nulls in d.items():
            rows.append((name, col, int(nulls), (float(nulls)/total if total else 0.0), total))

    schema = T.StructType([
        T.StructField("dataset", T.StringType(), False),
        T.StructField("column", T.StringType(), False),
        T.StructField("null_count", T.LongType(), False),
        T.StructField("null_frac", T.DoubleType(), False),
        T.StructField("row_count", T.LongType(), False),
    ])
    return df.sparkSession.createDataFrame(rows, schema)

# ---------- Main ----------
@hydra.main(config_path="../../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    # Spark builder mit extra_conf vor dem Start
    builder = (SparkSession.builder
               .appName(cfg.spark.app_name)
               .master(cfg.spark.master))
    for k, v in cfg.spark.extra_conf.items():
        builder = builder.config(k, v)
    spark = builder.getOrCreate()
    spark.conf.set("spark.sql.shuffle.partitions", cfg.spark.shuffle_partitions)
    spark.conf.set("spark.local.dir", os.environ.get("SPARK_LOCAL_DIRS", "/tmp/spark_tmp"))

    raw = Path(cfg.data.raw_dir)
    print("raw_dir:", raw)
    missing = [f for f in ["train_numeric.csv","train_categorical.csv","train_date.csv"] if not (raw/f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing {missing} in {raw}. Set RAW_DIR or link data/raw.")

    # Einlesen
    df_num  = _read_csv(spark, str(raw/"train_numeric.csv"), "numeric")
    df_cat  = _read_csv(spark, str(raw/"train_categorical.csv"), "categorical").fillna(cfg.na_policy.categorical_fill)
    df_date = _read_csv(spark, str(raw/"train_date.csv"), "date")

    print(f"numeric: {len(df_num.columns)} Spalten")
    print(f"categorical: {len(df_cat.columns)} Spalten")
    print(f"date: {len(df_date.columns)} Spalten")

    # Keys (batched)
    keys = _derive_keys_batched(df_date, batch_size=200)
    cov = (keys.select((F.col("n_events") > 0).cast("int").alias("has"))
               .agg(F.sum("has").alias("with_events"), F.count("*").alias("total"))
               .collect()[0])
    print(f"coverage: {cov['with_events']} / {cov['total']} Ids mit Events")

    # Joins
    df_num_k  = df_num.join(keys, "Id", "left")
    df_cat_k  = df_cat.join(keys, "Id", "left")
    df_date_k = df_date.join(keys, "Id", "left")

    # MLflow
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.experiment)
    reports_dir = Path("reports"); reports_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=cfg.run_name):
        mlflow.log_params({
            "format": cfg.data.format,
            "compression": cfg.data.compression,
            "shuffle_partitions": cfg.spark.shuffle_partitions,
            "categorical_fill": cfg.na_policy.categorical_fill,
        })

        # Row counts
        rn, rc, rd = df_num_k.count(), df_cat_k.count(), df_date_k.count()
        mlflow.log_metrics({"rows_numeric": rn, "rows_categorical": rc, "rows_date": rd})
        print(f"rows: num={rn}, cat={rc}, date={rd}")

        # Null map
        null_all = (
            _null_stats(df_num_k,  "train_numeric", batch_size=200)
            .unionByName(_null_stats(df_cat_k,  "train_categorical", batch_size=200))
            .unionByName(_null_stats(df_date_k, "train_date", batch_size=200))
        )
        out_dir = reports_dir / "null_map.csv"
        (null_all.orderBy(F.col("dataset"), F.col("null_frac").desc())
                 .coalesce(1).write.mode("overwrite").option("header", True).csv(str(out_dir)))
        # log single part if present
        try:
            part = next((out_dir).glob("part-*.csv"))
            mlflow.log_artifact(str(part), artifact_path="reports")
        except StopIteration:
            pass

        # Persist Lake
        base = Path(cfg.data.lake_dir) / cfg.data.format
        def _save(df, sub):
            (df.write.mode("overwrite")
               .format(cfg.data.format)
               .option("compression", cfg.data.compression)
               .partitionBy("line_key","station_last")
               .save(str(base / sub)))
        _save(df_num_k,  "train_numeric")
        _save(df_cat_k,  "train_categorical")
        _save(df_date_k, "train_date")

    spark.stop()
    print("INGEST DONE")

if __name__ == "__main__":
    main()
