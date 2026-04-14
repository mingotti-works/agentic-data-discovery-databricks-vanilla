"""
agent_discovery.py
------------------
Discovery agent — answers "what data exists and where?"

Crawls:
  1. Databricks Volumes for Parquet and CSV files
  2. JDBC sources via information_schema (Postgres / MySQL / SQL Server)

Outputs:
  - list[DatasetMeta]  (in-memory, passed to subsequent agents)
  - Delta table        (config.output_tables["discovery"])
  - MLflow run params  (logged by orchestrator)
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from config import DatasetMeta, PipelineConfig, get_spark


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(config: PipelineConfig) -> list[DatasetMeta]:
    """
    Discover all datasets from configured Volumes and JDBC sources.
    Returns a list of DatasetMeta — one entry per discovered table / file.
    """
    spark = get_spark()
    results: list[DatasetMeta] = []

    print("=" * 60)
    print("DISCOVERY AGENT — starting")
    print("=" * 60)

    # ── 1. Volumes (Parquet / CSV) ─────────────────────────────────────
    for volume_path in config.volume_paths:
        discovered = _discover_volume(spark, volume_path, config)
        results.extend(discovered)
        print(f"  [volume] {volume_path} → {len(discovered)} dataset(s) found")

    # ── 2. JDBC sources ────────────────────────────────────────────────
    for jdbc_cfg in config.jdbc_sources:
        discovered = _discover_jdbc(spark, jdbc_cfg, config)
        results.extend(discovered)
        print(f"  [jdbc]   {jdbc_cfg['name']} → {len(discovered)} table(s) found")

    print(f"\nDISCOVERY AGENT — complete. Total datasets: {len(results)}")

    # ── Persist to Delta ───────────────────────────────────────────────
    _write_delta(spark, results, config)

    return results


# ---------------------------------------------------------------------------
# Volume discovery
# ---------------------------------------------------------------------------

def _discover_volume(
    spark,
    volume_path: str,
    config: PipelineConfig,
) -> list[DatasetMeta]:
    """
    Walk the volume path recursively and load every matching file.
    Returns one DatasetMeta per file.
    """
    import subprocess
    results: list[DatasetMeta] = []

    # List files using dbutils (available on Databricks) or os.walk fallback
    file_paths = _list_files(volume_path, config.volume_file_extensions)

    for file_path in file_paths:
        try:
            ext = os.path.splitext(file_path)[1].lower()

            if ext == ".parquet":
                df = spark.read.parquet(file_path)
            elif ext == ".csv":
                df = spark.read.option("header", "true").option("inferSchema", "true").csv(file_path)
            else:
                continue

            schema_map = {field.name: str(field.dataType) for field in df.schema.fields}
            row_count  = df.count()

            meta = DatasetMeta(
                source_type = "volume",
                source_name = volume_path,
                table_name  = os.path.basename(file_path),
                columns     = list(schema_map.keys()),
                row_count   = row_count,
                raw_schema  = schema_map,
                extra       = {"file_path": file_path, "format": ext.lstrip(".")},
            )
            results.append(meta)

        except Exception as exc:
            print(f"    [WARN] Could not read {file_path}: {exc}")

    return results


def _list_files(root_path: str, extensions: tuple[str, ...]) -> list[str]:
    """
    List files under root_path matching the given extensions.

    Uses os.walk with the native /Volumes/... path for all Unity Catalog
    Volume paths. This avoids the DBFS_DISABLED error entirely — dbutils.fs.ls
    returns dbfs:/Volumes/... paths which Spark tries to resolve via the legacy
    DBFS mount (/dbfs/), which is disabled on modern Databricks workspaces.

    os.walk works directly on /Volumes/... without any path translation,
    and the same paths are valid for spark.read.csv / spark.read.parquet.

    For legacy dbfs:/ paths (non-Volume), we fall back to dbutils.fs.ls.
    """
    matched: list[str] = []

    # Normalise: strip trailing slash, convert dbfs:/Volumes -> /Volumes
    normalised = root_path.rstrip("/")
    if normalised.startswith("dbfs:/Volumes/"):
        normalised = normalised[len("dbfs:"):]   # /Volumes/...

    if normalised.startswith("/Volumes/"):
        # Unity Catalog Volume — use os.walk directly on the native path.
        # No dbutils, no /dbfs/ translation needed.
        for dirpath, _, filenames in os.walk(normalised):
            for fname in filenames:
                if any(fname.lower().endswith(ext) for ext in extensions):
                    matched.append(os.path.join(dirpath, fname))
    else:
        # Legacy DBFS path — use dbutils.fs.ls
        try:
            from pyspark.dbutils import DBUtils
            from pyspark.sql import SparkSession
            dbutils = DBUtils(SparkSession.builder.getOrCreate())

            def _walk(path: str):
                try:
                    entries = dbutils.fs.ls(path)
                except Exception:
                    return
                for entry in entries:
                    if entry.isDir():
                        _walk(entry.path)
                    elif any(entry.path.endswith(ext) for ext in extensions):
                        matched.append(entry.path)

            _walk(normalised)

        except ImportError:
            for dirpath, _, filenames in os.walk(normalised):
                for fname in filenames:
                    if any(fname.lower().endswith(ext) for ext in extensions):
                        matched.append(os.path.join(dirpath, fname))

    return matched


# ---------------------------------------------------------------------------
# JDBC discovery
# ---------------------------------------------------------------------------

def _discover_jdbc(
    spark,
    jdbc_cfg: dict[str, str],
    config: PipelineConfig,
) -> list[DatasetMeta]:
    """
    Discover tables from a JDBC source via information_schema.
    Reads column metadata without loading full table data (schema only).
    """
    results: list[DatasetMeta] = []
    source_name = jdbc_cfg["name"]
    url         = jdbc_cfg["url"]
    driver      = jdbc_cfg["driver"]
    user        = jdbc_cfg["user"]
    password    = jdbc_cfg["password"]

    jdbc_props = {
        "user":     user,
        "password": password,
        "driver":   driver,
    }

    # ── Discover table list ────────────────────────────────────────────
    tables = _get_explicit_tables(jdbc_cfg) or _infer_tables_from_schema(
        spark, url, jdbc_props
    )

    for table_name in tables:
        try:
            # Read a zero-row sample just to get the schema
            df = spark.read.jdbc(
                url        = url,
                table      = f"(SELECT * FROM {table_name} LIMIT 0) AS t",
                properties = jdbc_props,
            )

            # Row count via a lightweight COUNT query
            count_df = spark.read.jdbc(
                url        = url,
                table      = f"(SELECT COUNT(*) AS cnt FROM {table_name}) AS t",
                properties = jdbc_props,
            )
            row_count = count_df.collect()[0]["cnt"]

            schema_map = {field.name: str(field.dataType) for field in df.schema.fields}

            meta = DatasetMeta(
                source_type = "jdbc",
                source_name = source_name,
                table_name  = table_name,
                columns     = list(schema_map.keys()),
                row_count   = row_count,
                raw_schema  = schema_map,
                extra       = {"jdbc_url": url, "driver": driver},
            )
            results.append(meta)

        except Exception as exc:
            print(f"    [WARN] Could not read JDBC table {table_name}: {exc}")

    return results


def _get_explicit_tables(jdbc_cfg: dict[str, str]) -> list[str]:
    """Return explicitly configured table list (if provided)."""
    raw = jdbc_cfg.get("tables", "")
    if not raw:
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]


def _infer_tables_from_schema(spark, url: str, props: dict) -> list[str]:
    """
    Fall back to querying information_schema.tables to discover tables.
    Works for Postgres, MySQL, SQL Server.
    """
    try:
        df = spark.read.jdbc(
            url   = url,
            table = (
                "(SELECT table_schema, table_name "
                "FROM information_schema.tables "
                "WHERE table_type = 'BASE TABLE') AS t"
            ),
            properties = props,
        )
        rows = df.collect()
        return [f"{r['table_schema']}.{r['table_name']}" for r in rows]
    except Exception as exc:
        print(f"    [WARN] information_schema query failed: {exc}")
        return []


# ---------------------------------------------------------------------------
# Persist results to Delta
# ---------------------------------------------------------------------------

def _write_delta(spark, results: list[DatasetMeta], config: PipelineConfig):
    """Write discovery results to a Delta table for downstream use."""
    if not results:
        return

    from pyspark.sql import Row
    import json

    rows = [
        Row(
            run_ts      = datetime.utcnow().isoformat(),
            source_type = m.source_type,
            source_name = m.source_name,
            table_name  = m.table_name,
            columns     = json.dumps(m.columns),
            row_count   = m.row_count,
            raw_schema  = json.dumps(m.raw_schema),
            extra       = json.dumps(m.extra),
        )
        for m in results
    ]

    df = spark.createDataFrame(rows)
    target = (
        f"{config.output_catalog}.{config.output_schema}"
        f".{config.output_tables['discovery']}"
    )
    (
        df.write
          .format("delta")
          .mode("overwrite")
          .option("mergeSchema", "true")
          .saveAsTable(target)
    )
    print(f"  [delta] Discovery results written → {target}")
