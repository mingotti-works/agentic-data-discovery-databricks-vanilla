"""
agent_profiling.py
------------------
Profiling agent — answers "what does the data look like?"

For every DatasetMeta from the discovery agent, this agent:
  - Samples up to config.profiling_sample_rows rows
  - Computes per-column statistics (null rate, cardinality, min/max,
    top-N values, inferred semantic type)
  - Flags columns that look like PKs or candidate join keys
  - Persists results to Delta and returns a structured dict

Output schema (one row per column per dataset):
  run_ts, source_type, source_name, table_name, column_name,
  dtype, null_count, null_rate, distinct_count, is_unique,
  min_value, max_value, mean_value, sample_values,
  inferred_type, pk_candidate
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from config import DatasetMeta, PipelineConfig, get_spark


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(
    config: PipelineConfig,
    datasets: list[DatasetMeta],
) -> list[dict[str, Any]]:
    """
    Profile every discovered dataset.
    Returns a flat list of column-level profile dicts.
    """
    spark  = get_spark()
    all_profiles: list[dict[str, Any]] = []

    print("=" * 60)
    print("PROFILING AGENT — starting")
    print("=" * 60)

    for meta in datasets:
        print(f"\n  Profiling: [{meta.source_type}] {meta.source_name} / {meta.table_name}")
        try:
            df = _load_dataframe(spark, meta, config)
            col_profiles = _profile_dataframe(df, meta, config)
            all_profiles.extend(col_profiles)
            print(f"    → {len(col_profiles)} column(s) profiled")
        except Exception as exc:
            print(f"    [WARN] Could not profile {meta.table_name}: {exc}")

    print(f"\nPROFILING AGENT — complete. Total column profiles: {len(all_profiles)}")

    _write_delta(spark, all_profiles, config)

    return all_profiles


# ---------------------------------------------------------------------------
# DataFrame loader (re-uses discovery metadata to avoid re-scanning)
# ---------------------------------------------------------------------------

def _load_dataframe(spark, meta: DatasetMeta, config: PipelineConfig):
    """Load a sampled DataFrame from a Volume file or JDBC table."""

    if meta.source_type == "volume":
        file_path = meta.extra.get("file_path", "")
        fmt       = meta.extra.get("format", "parquet")

        if fmt == "parquet":
            df = spark.read.parquet(file_path)
        else:
            df = (
                spark.read
                     .option("header", "true")
                     .option("inferSchema", "true")
                     .csv(file_path)
            )

    elif meta.source_type == "jdbc":
        # Find matching JDBC config
        jdbc_cfg  = _find_jdbc_config(meta.source_name, config)
        jdbc_props = {
            "user":     jdbc_cfg["user"],
            "password": jdbc_cfg["password"],
            "driver":   jdbc_cfg["driver"],
        }
        df = spark.read.jdbc(
            url        = jdbc_cfg["url"],
            table      = meta.table_name,
            properties = jdbc_props,
        )
    else:
        raise ValueError(f"Unknown source_type: {meta.source_type!r}")

    # Sample if large — deterministic seed for reproducibility
    if meta.row_count > config.profiling_sample_rows:
        fraction = config.profiling_sample_rows / meta.row_count
        df = df.sample(withReplacement=False, fraction=fraction, seed=42)

    return df


def _find_jdbc_config(source_name: str, config: PipelineConfig) -> dict:
    for jdbc_cfg in config.jdbc_sources:
        if jdbc_cfg["name"] == source_name:
            return jdbc_cfg
    raise KeyError(f"No JDBC config found for source_name={source_name!r}")


# ---------------------------------------------------------------------------
# Column-level profiling
# ---------------------------------------------------------------------------

def _profile_dataframe(
    df,
    meta: DatasetMeta,
    config: PipelineConfig,
) -> list[dict[str, Any]]:
    """Compute per-column stats for a sampled DataFrame."""
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        NumericType, StringType, BooleanType,
        DateType, TimestampType,
    )

    total_rows = df.count()
    profiles: list[dict[str, Any]] = []

    for col_field in df.schema.fields:
        col_name = col_field.name
        dtype    = str(col_field.dataType)
        col      = F.col(col_name)

        # ── Null stats ────────────────────────────────────────────────
        null_count = df.filter(col.isNull()).count()
        null_rate  = round(null_count / total_rows, 4) if total_rows > 0 else 0.0

        # ── Cardinality ───────────────────────────────────────────────
        distinct_count = df.select(col_name).distinct().count()
        is_unique      = (distinct_count == total_rows) and total_rows > 0

        # ── Min / max / mean ──────────────────────────────────────────
        min_val = max_val = mean_val = None
        try:
            agg_row = df.agg(
                F.min(col).alias("mn"),
                F.max(col).alias("mx"),
            ).collect()[0]
            min_val = str(agg_row["mn"]) if agg_row["mn"] is not None else None
            max_val = str(agg_row["mx"]) if agg_row["mx"] is not None else None
        except Exception:
            pass

        if isinstance(col_field.dataType, NumericType):
            try:
                mean_row = df.agg(F.mean(col).alias("mean")).collect()[0]
                mean_val = round(float(mean_row["mean"]), 4) if mean_row["mean"] else None
            except Exception:
                pass

        # ── Top-N sample values ───────────────────────────────────────
        sample_values: list[str] = []
        try:
            top_rows = (
                df.groupBy(col_name)
                  .count()
                  .orderBy(F.desc("count"))
                  .limit(config.profiling_sample_values)
                  .collect()
            )
            sample_values = [str(r[col_name]) for r in top_rows if r[col_name] is not None]
        except Exception:
            pass

        # ── Inferred semantic type ────────────────────────────────────
        inferred_type = _infer_semantic_type(
            col_name, dtype, sample_values, is_unique, null_rate
        )

        # ── PK candidate flag ─────────────────────────────────────────
        pk_candidate = is_unique and null_rate == 0.0 and total_rows > 0

        profiles.append({
            "run_ts":         datetime.utcnow().isoformat(),
            "source_type":    meta.source_type,
            "source_name":    meta.source_name,
            "table_name":     meta.table_name,
            "column_name":    col_name,
            "dtype":          dtype,
            "total_rows":     total_rows,
            "null_count":     null_count,
            "null_rate":      null_rate,
            "distinct_count": distinct_count,
            "is_unique":      is_unique,
            "min_value":      min_val,
            "max_value":      max_val,
            "mean_value":     mean_val,
            "sample_values":  json.dumps(sample_values),
            "inferred_type":  inferred_type,
            "pk_candidate":   pk_candidate,
        })

    return profiles


# ---------------------------------------------------------------------------
# Semantic type inference (rule-based, no LLM needed here)
# ---------------------------------------------------------------------------

_PATTERN_ID   = re.compile(r"\b(id|key|code|pk|uuid|guid)\b", re.IGNORECASE)
_PATTERN_DATE = re.compile(r"\b(date|time|ts|created|updated|modified|at)\b", re.IGNORECASE)
_PATTERN_AMT  = re.compile(r"\b(amount|price|cost|revenue|salary|fee|total|balance)\b", re.IGNORECASE)
_PATTERN_NAME = re.compile(r"\b(name|label|title|desc|description)\b", re.IGNORECASE)
_PATTERN_FLAG = re.compile(r"\b(flag|is_|has_|active|enabled|deleted)\b", re.IGNORECASE)
_PATTERN_GEO  = re.compile(r"\b(country|city|state|region|zip|postal|lat|lon|geo)\b", re.IGNORECASE)
_PATTERN_EMAIL = re.compile(r"^[^@]+@[^@]+\.[^@]+$")

def _infer_semantic_type(
    col_name: str,
    dtype: str,
    sample_values: list[str],
    is_unique: bool,
    null_rate: float,
) -> str:
    """
    Rule-based semantic type inference from column name, dtype, and samples.
    Returns a short label; downstream semantic agent refines with LLM.
    """
    name_lower = col_name.lower()

    # Check sample values for email pattern
    if sample_values and all(_PATTERN_EMAIL.match(v) for v in sample_values[:3]):
        return "email"

    if _PATTERN_ID.search(name_lower):
        return "identifier"
    if _PATTERN_DATE.search(name_lower) or "Timestamp" in dtype or "Date" in dtype:
        return "datetime"
    if _PATTERN_AMT.search(name_lower):
        return "monetary_amount"
    if _PATTERN_NAME.search(name_lower):
        return "label_or_name"
    if _PATTERN_FLAG.search(name_lower) or "Boolean" in dtype:
        return "boolean_flag"
    if _PATTERN_GEO.search(name_lower):
        return "geographic"
    if "String" in dtype and is_unique:
        return "unique_text"
    if "String" in dtype:
        return "categorical_or_text"
    if any(t in dtype for t in ("Int", "Long", "Double", "Float", "Decimal")):
        return "numeric"

    return "unknown"


# ---------------------------------------------------------------------------
# Persist results to Delta
# ---------------------------------------------------------------------------

def _write_delta(
    spark,
    profiles: list[dict[str, Any]],
    config: PipelineConfig,
):
    if not profiles:
        return

    from pyspark.sql import Row

    rows = [Row(**p) for p in profiles]
    df   = spark.createDataFrame(rows)

    target = (
        f"{config.output_catalog}.{config.output_schema}"
        f".{config.output_tables['profiling']}"
    )
    (
        df.write
          .format("delta")
          .mode("overwrite")
          .option("mergeSchema", "true")
          .saveAsTable(target)
    )
    print(f"  [delta] Profiling results written → {target}")
