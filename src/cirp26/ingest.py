import os
from pathlib import Path
import mlflow
from omegaconf import DictConfig
import hydra

from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.storagelevel import StorageLevel

# ---------- Utilities ----------
def _read_header(csv_path: str):
    # Read the first line safely to build the schema
    with open(csv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    return header

def _schema_for_domain(header, domain: str):
    """
    Enforce Bosch schemas:
    - Id: int
    - Response: int (only in train_numeric, if present)
    - numeric: DoubleType
    - categorical: StringType
    - date: IntegerType (date ticks)
    """
    fields = []
    for col in header:
        if col == "Id":
            fields.append(T.StructField(col, T.LongType(), False))
        elif col == "Response":
            fields.append(T.StructField(col, T.IntegerType(), True))
        else:
            if domain == "numeric":
                fields.append(T.StructField(col, T.DoubleType(), True))
            elif domain == "categorical":
                fields.append(T.StructField(col, T.StringType(), True))
            elif domain == "date":
                # ticks are integers in practice; use LongType for safety
                fields.append(T.StructField(col, T.LongType(), True))
            else:
                raise ValueError(f"Unknown domain: {domain}")
    return T.StructType(fields)

def _build_spark(app_name: str, master: str, extra_conf: dict, use_delta: bool) -> SparkSession:
    builder = SparkSession.builder.appName(app_name).master(master)
    if use_delta:
        builder = (
            builder
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        )
    for k, v in (extra_conf or {}).items():
        builder = builder.config(k, v)
    spark = builder.getOrCreate()
    return spark

def _read_csv_with_schema(spark, path: str, domain: str):
    header = _read_header(path)
    schema = _schema_for_domain(header, domain)
    # Read CSV with enforced schema; treat empty as null
    df = (
        spark.read
             .option("header", True)
             .option("mode", "PERMISSIVE")
             .option("nullValue", "")
             .schema(schema)
             .csv(path)
    )
    return df

def _derive_keys_from_date(df_date, sample=False):
    """
    Derive partition keys (line/station) from the date table:
    - Find which stations were visited (non-null tick)
    - Extract line and station from column name: L{line}_S{station}_F{feature}
    - Compute per-Id aggregates: min line, first station, last station, event count
    Implementation uses `stack` for pivot-to-long in one pass.
    """
    id_col = "Id"
    date_cols = [c for c in df_date.columns if c != id_col]

    # Optional: sample during development
    if sample:
        df_date = df_date.limit(100000)

    # Build stack expression: stack(N, `c1`,'c1', `c2`,'c2', ...)
    pairs = []
    for c in date_cols:
        pairs.append(f"`{c}`")
        pairs.append(f"'{c}'")
    stack_expr = f"stack({len(date_cols)}, {', '.join(pairs)}) as (tick, name)"

    exploded = (
        df_date.select(F.col(id_col), F.expr(stack_expr))
               .where(F.col("tick").isNotNull())
               .withColumn("line", F.regexp_extract("name", r"L(\\d+)_S(\\d+)_F\\d+", 1).cast("int"))
               .withColumn("station", F.regexp_extract("name", r"L(\\d+)_S(\\d+)_F\\d+", 2).cast("int"))
    )

    keys = (exploded.groupBy(id_col)
            .agg(
                F.min("line").alias("line_key"),
                F.min("station").alias("station_first"),
                F.max("station").alias("station_last"),
                F.count("*").alias("n_events")
            ))
    return keys

def _apply_na_policies(df_cat, fill_value: str):
    return df_cat.fillna(fill_value) if fill_value is not None else df_cat

def _null_stats(df, dataset_name: str):
    # Single-pass aggregated null counts for all columns
    total = df.count()
    # Build per-column expressions
    aggs = []
    for name, dtype in df.dtypes:
        if dtype in ("double", "float"):
            aggs.append(F.sum(F.when(F.isnan(F.col(name)) | F.col(name).isNull(), 1).otherwise(0)).alias(name))
        else:
            aggs.append(F.sum(F.when(F.col(name).isNull(), 1).otherwise(0)).alias(name))
    row = df.agg(*aggs).collect()[0].asDict()
    # Convert to long form
    rows = []
    for col, nulls in row.items():
        rows.append((dataset_name, col, int(nulls), float(nulls)/total if total else 0.0, total))
    schema = T.StructType([
        T.StructField("dataset", T.StringType(), False),
        T.StructField("column", T.StringType(), False),
        T.StructField("null_count", T.LongType(), False),
        T.StructField("null_frac", T.DoubleType(), False),
        T.StructField("row_count", T.LongType(), False),
    ])
    return df.sparkSession.createDataFrame(rows, schema)

def _write_table(df, out_dir: str, fmt: str, partitions: list, mode: str = "overwrite"):
    writer = df.write.mode(mode)
    if fmt == "delta":
        writer = writer.format("delta")
    else:
        writer = writer.format("parquet")
    if partitions:
        writer = writer.partitionBy(*partitions)
    writer.save(out_dir)

# ---------- Main pipeline ----------
@hydra.main(config_path="../../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    use_delta = (cfg.data.format.lower() == "delta")
    spark = _build_spark(
        app_name=cfg.spark.app_name,
        master=cfg.spark.master,
        extra_conf=cfg.spark.extra_conf,
        use_delta=use_delta,
    )
    spark.conf.set("spark.sql.shuffle.partitions", cfg.spark.shuffle_partitions)

    raw_dir = Path(cfg.data.raw_dir)
    lake_dir = Path(cfg.data.lake_dir)
    lake_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # MLflow
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.experiment)
    with mlflow.start_run(run_name=cfg.run_name):
        mlflow.log_params({
            "format": cfg.data.format,
            "compression": cfg.data.compression,
            "shuffle_partitions": cfg.spark.shuffle_partitions,
            "categorical_fill": cfg.na_policy.categorical_fill,
        })

        # Read CSVs with enforced schema
        path_num = str(raw_dir / "train_numeric.csv")
        path_cat = str(raw_dir / "train_categorical.csv")
        path_date = str(raw_dir / "train_date.csv")

        df_num = _read_csv_with_schema(spark, path_num, "numeric").persist(StorageLevel.DISK_ONLY)
        df_cat = _read_csv_with_schema(spark, path_cat, "categorical").persist(StorageLevel.DISK_ONLY)
        df_date = _read_csv_with_schema(spark, path_date, "date").persist(StorageLevel.DISK_ONLY)

        # NA fill policy (categorical only)
        df_cat_f = _apply_na_policies(df_cat, cfg.na_policy.categorical_fill)

        # Derive partition keys from date table, then join onto num & cat
        keys = _derive_keys_from_date(df_date).persist(StorageLevel.DISK_ONLY)

        df_num_k = (df_num.join(keys, on="Id", how="left"))
        df_cat_k = (df_cat_f.join(keys, on="Id", how="left"))
        df_date_k = (df_date.join(keys, on="Id", how="left"))

        # Basic EDA
        cnt_num = df_num_k.count()
        cnt_cat = df_cat_k.count()
        cnt_date = df_date_k.count()

        mlflow.log_metrics({
            "rows_numeric": cnt_num,
            "rows_categorical": cnt_cat,
            "rows_date": cnt_date
        })

        # Null maps
        null_num = _null_stats(df_num_k, "train_numeric")
        null_cat = _null_stats(df_cat_k, "train_categorical")
        null_date = _null_stats(df_date_k, "train_date")
        null_all = null_num.unionByName(null_cat).unionByName(null_date)

        # Write null map CSV for the report
        null_out = str(reports_dir / "null_map.csv")
        (null_all
            .coalesce(1)
            .orderBy(F.col("dataset"), F.col("null_frac").desc())
            .write
            .mode("overwrite")
            .option("header", True)
            .csv(null_out))

        # Write Parquet/Delta lake (ZSTD for Parquet is set via Spark conf)
        # Partition by line/station keys (line_key, station_last) — robust cardinality.
        base = lake_dir / cfg.data.format.lower()
        _write_table(df_num_k, str(base / "train_numeric"), cfg.data.format, partitions=["line_key", "station_last"])
        _write_table(df_cat_k, str(base / "train_categorical"), cfg.data.format, partitions=["line_key", "station_last"])
        _write_table(df_date_k, str(base / "train_date"), cfg.data.format, partitions=["line_key", "station_last"])

        # Log artifacts to MLflow
        # Grab the single CSV part
        # (We wrote a folder; find the file starting with part-)
        csv_dir = Path(null_out)
        # On Databricks/cluster you’d use DBFS utils. Locally:
        part_files = list(csv_dir.glob("*.csv")) or list(csv_dir.glob("part-*.csv"))
        for p in part_files:
            mlflow.log_artifact(str(p), artifact_path="reports")

        mlflow.set_tags(dict(cfg.mlflow.tags))

        print(f"Done. Wrote data lake to: {base}")
        print("Done. Logged run & report to MLflow")

    spark.stop()

if __name__ == "__main__":
    main()
