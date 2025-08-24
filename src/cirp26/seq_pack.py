from pathlib import Path
import hydra
from omegaconf import DictConfig
from pyspark.sql import SparkSession, functions as F

@hydra.main(config_path="../../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    base = Path(cfg.data.lake_dir)
    seq_base = Path(cfg.seq.out_parquet)
    src  = seq_base / "events_long_enriched"
    T    = int(cfg.seq.T)  # e.g., 128
    dst  = seq_base / f"sequences_T{T}"

    spark = (SparkSession.builder
             .appName("CIRP2026-Day2-Pack")
             .master(cfg.spark.master)
             .getOrCreate())
    for k, v in cfg.spark.extra_conf.items():
        spark.conf.set(k, v)

    ev = spark.read.parquet(str(src))
    ev = ev.withColumn("station_token", (F.col("line")*F.lit(1000) + F.col("station")).cast("int"))

    arr_tokens = F.transform(F.array_sort(F.collect_list(F.struct("rank","station_token"))), lambda x: x["station_token"])
    arr_dt     = F.transform(F.array_sort(F.collect_list(F.struct("rank","dt_min"))),      lambda x: x["dt_min"])
    arr_hour   = F.transform(F.array_sort(F.collect_list(F.struct("rank","hour"))),        lambda x: x["hour"])

    grp = ev.groupBy("Id").agg(
        F.count("*").alias("length_orig"),
        arr_tokens.alias("tokens_sorted"),
        arr_dt.alias("dt_sorted"),
        arr_hour.alias("hour_sorted")
    )

    def clip_pad(arr_col, pad_val):
        cut = F.slice(arr_col, 1, T)
        pad_len = (F.lit(T) - F.size(cut)).cast("int")
        return F.concat(cut, F.array_repeat(F.lit(pad_val), pad_len))

    tokens_T = clip_pad(F.col("tokens_sorted"), 0)
    dt_T     = clip_pad(F.col("dt_sorted"),     0.0)
    hour_T   = clip_pad(F.col("hour_sorted"),   0)

    length_eff = F.least(F.col("length_orig").cast("int"), F.lit(T))
    mask_T = F.concat(
        F.array_repeat(F.lit(1), length_eff),
        F.array_repeat(F.lit(0), (F.lit(T) - length_eff).cast("int"))
    )

    tokens_str = F.transform(F.slice(F.col("tokens_sorted"), 1, T), lambda x: x.cast("string"))
    path_id = F.sha2(F.concat_ws("-", tokens_str), 256)

    seq = (grp.select(
               "Id",
               "length_orig",
               length_eff.alias("length"),
               (F.col("length_orig") > F.lit(T)).cast("int").alias("truncated"),
               tokens_T.alias("tokens"),
               dt_T.alias("dt"),
               hour_T.alias("hour"),
               mask_T.alias("mask"),
               path_id.alias("path_id"))
    )

    labels = (spark.read.parquet(str(base / "parquet" / "train_numeric"))
              .select("Id","Response").dropDuplicates(["Id"]))
    seq = seq.join(labels, "Id", "left")

    seq.repartition(256, "Id").write.mode("overwrite").parquet(str(dst))

    dst_df = spark.read.parquet(str(dst))
    tot = dst_df.count()
    n_trunc = dst_df.where(F.col("truncated")==1).count()
    avg_len = dst_df.agg(F.avg("length_orig")).first()[0]
    bad = (dst_df.where(F.size("tokens") != T).limit(1).count() +
           dst_df.where(F.size("dt")     != T).limit(1).count() +
           dst_df.where(F.size("hour")   != T).limit(1).count() +
           dst_df.where(F.size("mask")   != T).limit(1).count())
    print(f"seq count: {tot}  truncated: {n_trunc} ({(n_trunc*100.0/max(tot,1)):.2f}%)  avg length_orig: {round(avg_len or 0,2)}")
    print("any wrong array length?:", "YES" if bad>0 else "NO")

    spark.stop()
    print(f"[sequences_T{T}] DONE")

if __name__ == "__main__":
    main()
