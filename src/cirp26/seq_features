from pathlib import Path
import hydra
from omegaconf import DictConfig
from pyspark.sql import SparkSession, functions as F, Window

@hydra.main(config_path="../../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    base = Path(cfg.data.lake_dir) / "parquet"
    src  = base / "events_long"
    dst  = Path(cfg.seq.out_parquet) / "events_long_enriched"
    mpt  = int(cfg.seq.minutes_per_tick)  # e.g., 6

    spark = (SparkSession.builder
             .appName("CIRP2026-Day2-EventsEnrich")
             .master(cfg.spark.master)
             .getOrCreate())
    for k, v in cfg.spark.extra_conf.items():
        spark.conf.set(k, v)

    ev = spark.read.parquet(str(src))
    w = Window.partitionBy("Id").orderBy("rank")

    ev2 = (ev
      .withColumn("prev_tick", F.lag("tick").over(w))
      .withColumn("dt_min", (F.when(F.col("prev_tick").isNull(), 0)
                               .otherwise(F.col("tick")-F.col("prev_tick")) * F.lit(mpt)).cast("double"))
      .drop("prev_tick")
      .withColumn("hour", ((F.col("tick")*F.lit(mpt)) / 60).cast("int") % 24)
      .withColumn("dow",  ((F.col("tick")*F.lit(mpt)) / (60*24)).cast("int") % 7)
      .withColumn("shift",
        F.when((F.col("hour") >= 22) | (F.col("hour") < 6),  F.lit("night"))
         .when((F.col("hour") >= 6)  & (F.col("hour") < 14), F.lit("early"))
         .otherwise(F.lit("late"))
      )
      .withColumn("is_jump", (F.col("station") != F.lag("station").over(w)).cast("int"))
    )

    ev2.repartition(256, "Id").write.mode("overwrite").parquet(str(dst))

    # Checks
    neg = ev2.where((F.col("rank")>1) & (F.col("dt_min")<0)).limit(1).count()
    print("counts src/dst:", ev.count(), ev2.count())
    print("any negative dt_min?:", "YES" if neg>0 else "NO")

    spark.stop()
    print("[events_long_enriched] DONE")

if __name__ == "__main__":
    main()
