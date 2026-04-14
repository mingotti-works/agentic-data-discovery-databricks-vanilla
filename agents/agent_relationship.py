"""
agent_relationship.py
---------------------
Relationship discovery agent — answers "how do these datasets connect?"

Strategy (three complementary signals, combined into one confidence score):

  1. Value overlap      — what fraction of distinct values in col A
                          appear in col B? High overlap → FK candidate.
  2. Column name cosine — are the column names lexically similar?
                          (edit-distance + token overlap heuristic)
  3. Cardinality fit    — does col B look like a PK (unique, not null)
                          that col A could reference?

Each candidate is emitted with:
  - source_table / source_col  (the FK side)
  - target_table / target_col  (the PK side)
  - overlap_score, name_score, cardinality_score, confidence
  - evidence (human-readable explanation string)

Confidence tiers (from config):
  ≥ auto_accept_confidence  → status = "auto_accepted"
  ≥ human_review_confidence → status = "needs_review"
  <  human_review_confidence → status = "discarded"
"""

from __future__ import annotations

import itertools
import json
from datetime import datetime
from typing import Any

from config import DatasetMeta, PipelineConfig, get_spark


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(
    config: PipelineConfig,
    datasets: list[DatasetMeta],
    profiles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Discover relationship candidates across all datasets.
    Returns a list of candidate dicts, sorted by confidence descending.
    """
    spark = get_spark()

    print("=" * 60)
    print("RELATIONSHIP AGENT — starting")
    print("=" * 60)

    # Build a lookup: (source_name, table_name, col_name) → profile dict
    profile_index = {
        (p["source_name"], p["table_name"], p["column_name"]): p
        for p in profiles
    }

    # Identify PK candidates from profiling results
    pk_candidates = {
        (p["source_name"], p["table_name"], p["column_name"])
        for p in profiles
        if p.get("pk_candidate") and p.get("null_rate", 1.0) == 0.0
    }

    print(f"  PK candidates found: {len(pk_candidates)}")

    # Load each dataset into Spark (caching for repeated value lookups)
    df_cache: dict[tuple[str, str], Any] = {}
    for meta in datasets:
        key = (meta.source_name, meta.table_name)
        df_cache[key] = _load_dataframe(spark, meta, config)

    candidates: list[dict[str, Any]] = []

    # Compare every (col_A, col_B) pair where col_B is a PK candidate
    # and col_A is not already a PK
    all_cols = [
        (p["source_name"], p["table_name"], p["column_name"])
        for p in profiles
    ]

    pairs_evaluated = 0
    for fk_key in all_cols:
        if fk_key in pk_candidates:
            continue  # skip — already a PK

        fk_profile = profile_index.get(fk_key)
        if not fk_profile:
            continue

        # Only compare columns with compatible inferred types
        fk_type = fk_profile.get("inferred_type", "unknown")
        if fk_type in ("monetary_amount", "boolean_flag", "datetime", "numeric"):
            continue  # These are unlikely FKs

        # Cap columns per table to avoid O(n²) explosion
        fk_src, fk_tbl, fk_col = fk_key
        cols_in_table = [
            k for k in all_cols
            if k[0] == fk_src and k[1] == fk_tbl
        ]
        if cols_in_table.index(fk_key) >= config.relationship_max_cols_per_table:
            continue

        for pk_key in pk_candidates:
            pk_src, pk_tbl, pk_col = pk_key

            # Skip self-comparison
            if (fk_src, fk_tbl) == (pk_src, pk_tbl):
                continue

            pairs_evaluated += 1

            try:
                candidate = _evaluate_pair(
                    spark     = spark,
                    df_cache  = df_cache,
                    fk_key    = fk_key,
                    pk_key    = pk_key,
                    fk_profile= fk_profile,
                    pk_profile= profile_index.get(pk_key, {}),
                    config    = config,
                )
                if candidate:
                    candidates.append(candidate)
            except Exception as exc:
                print(f"    [WARN] pair {fk_key} ↔ {pk_key}: {exc}")

    # Sort by confidence descending, apply status tiers
    candidates.sort(key=lambda c: c["confidence"], reverse=True)

    print(f"\n  Pairs evaluated : {pairs_evaluated}")
    print(f"  Candidates found: {len(candidates)}")


    print("\nRELATIONSHIP AGENT — complete")
    _write_delta(spark, candidates, config)

    return candidates


# ---------------------------------------------------------------------------
# Pair evaluation
# ---------------------------------------------------------------------------

def _evaluate_pair(
    spark,
    df_cache: dict,
    fk_key: tuple[str, str, str],
    pk_key: tuple[str, str, str],
    fk_profile: dict,
    pk_profile: dict,
    config: PipelineConfig,
) -> dict[str, Any] | None:
    """
    Evaluate one (FK column, PK column) pair.
    Returns a candidate dict or None if below discard threshold.
    """
    fk_src, fk_tbl, fk_col = fk_key
    pk_src, pk_tbl, pk_col = pk_key

    # ── Signal 1: value overlap ───────────────────────────────────────
    overlap_score = _compute_value_overlap(
        spark, df_cache, fk_key, pk_key
    )

    # Early exit — if overlap is essentially zero, skip
    if overlap_score < 0.05:
        return None

    # ── Signal 2: column name similarity ─────────────────────────────
    name_score = _name_similarity(fk_col, pk_col)

    # ── Signal 3: cardinality fit ─────────────────────────────────────
    cardinality_score = _cardinality_fit(fk_profile, pk_profile)

    # ── Combined confidence (weighted average) ────────────────────────
    confidence = round(
        0.60 * overlap_score +
        0.20 * name_score    +
        0.20 * cardinality_score,
        4,
    )

    # Apply status tier
    if confidence >= config.auto_accept_confidence:
        status = "auto_accepted"
    elif confidence >= config.human_review_confidence:
        status = "needs_review"
    else:
        status = "discarded"

    evidence = (
        f"overlap={overlap_score:.2f}, "
        f"name_similarity={name_score:.2f}, "
        f"cardinality_fit={cardinality_score:.2f}"
    )

    return {
        "run_ts":             datetime.utcnow().isoformat(),
        "source_table":       f"{fk_src}.{fk_tbl}",
        "source_col":         fk_col,
        "target_table":       f"{pk_src}.{pk_tbl}",
        "target_col":         pk_col,
        "overlap_score":      overlap_score,
        "name_score":         name_score,
        "cardinality_score":  cardinality_score,
        "confidence":         confidence,
        "status":             status,
        "evidence":           evidence,
        "relationship_type":  "FK → PK (candidate)",
    }


# ---------------------------------------------------------------------------
# Signal implementations
# ---------------------------------------------------------------------------

def _compute_value_overlap(
    spark,
    df_cache: dict,
    fk_key: tuple[str, str, str],
    pk_key: tuple[str, str, str],
) -> float:
    """
    What fraction of distinct FK values appear in the PK column?
    Uses a Spark broadcast join for efficiency on small PK sets.
    """
    fk_src, fk_tbl, fk_col = fk_key
    pk_src, pk_tbl, pk_col = pk_key

    from pyspark.sql import functions as F

    fk_df = df_cache.get((fk_src, fk_tbl))
    pk_df = df_cache.get((pk_src, pk_tbl))

    if fk_df is None or pk_df is None:
        return 0.0

    # Distinct non-null FK values
    fk_vals = (
        fk_df.select(F.col(fk_col).cast("string").alias("v"))
             .filter(F.col("v").isNotNull())
             .distinct()
    )
    fk_count = fk_vals.count()
    if fk_count == 0:
        return 0.0

    # Distinct PK values (as lookup set)
    pk_vals = (
        pk_df.select(F.col(pk_col).cast("string").alias("v"))
             .filter(F.col("v").isNotNull())
             .distinct()
             .withColumnRenamed("v", "pk_v")
    )

    # Inner join to find how many FK values exist in PK
    matched = fk_vals.join(
        F.broadcast(pk_vals),
        fk_vals["v"] == pk_vals["pk_v"],
        "inner",
    ).count()

    return round(matched / fk_count, 4)


def _name_similarity(col_a: str, col_b: str) -> float:
    """
    Lexical similarity between two column names.
    Combines token overlap and edit-distance ratio.
    """
    import difflib

    def tokenise(name: str) -> set[str]:
        # Split on underscores, camelCase boundaries, and digits
        import re
        tokens = re.sub(r"([A-Z])", r"_\1", name).lower()
        return set(t for t in re.split(r"[_\s\-]+", tokens) if t)

    tokens_a = tokenise(col_a)
    tokens_b = tokenise(col_b)

    # Token Jaccard similarity
    intersection = tokens_a & tokens_b
    union        = tokens_a | tokens_b
    jaccard      = len(intersection) / len(union) if union else 0.0

    # Edit distance ratio (works well for short column names)
    edit_ratio = difflib.SequenceMatcher(
        None, col_a.lower(), col_b.lower()
    ).ratio()

    return round(0.5 * jaccard + 0.5 * edit_ratio, 4)


def _cardinality_fit(fk_profile: dict, pk_profile: dict) -> float:
    """
    Score how well the PK column's cardinality fits a PK role
    and how well the FK column's cardinality fits a FK role.

    Perfect fit:
      - PK column: is_unique=True, null_rate=0
      - FK column: not unique (many rows referencing fewer PK values)
    """
    score = 0.0

    # PK side: should be unique and non-null
    if pk_profile.get("is_unique"):
        score += 0.5
    if pk_profile.get("null_rate", 1.0) == 0.0:
        score += 0.25

    # FK side: should NOT be unique (many-to-one relationship)
    if not fk_profile.get("is_unique"):
        score += 0.25

    return round(score, 4)


# ---------------------------------------------------------------------------
# DataFrame loader (mirrors profiling agent)
# ---------------------------------------------------------------------------

def _load_dataframe(spark, meta: DatasetMeta, config: PipelineConfig):
    if meta.source_type == "volume":
        file_path = meta.extra.get("file_path", "")
        fmt       = meta.extra.get("format", "parquet")
        if fmt == "parquet":
            return spark.read.parquet(file_path)
        return (
            spark.read
                 .option("header", "true")
                 .option("inferSchema", "true")
                 .csv(file_path)
        )
    elif meta.source_type == "jdbc":
        jdbc_cfg = next(
            (j for j in config.jdbc_sources if j["name"] == meta.source_name),
            None,
        )
        if not jdbc_cfg:
            raise KeyError(f"JDBC config not found: {meta.source_name}")
        props = {
            "user": jdbc_cfg["user"],
            "password": jdbc_cfg["password"],
            "driver": jdbc_cfg["driver"],
        }
        return spark.read.jdbc(url=jdbc_cfg["url"], table=meta.table_name, properties=props)
    raise ValueError(f"Unknown source_type: {meta.source_type!r}")


# ---------------------------------------------------------------------------
# Serverless-compatible Delta writer
# saveAsTable is not supported on serverless compute. Instead we:
#   1. Derive a deterministic storage path under the schema location
#   2. Write the data with spark.write.save(path)
#   3. Register (or refresh) the table with CREATE TABLE IF NOT EXISTS ... LOCATION
# ---------------------------------------------------------------------------

def _write_to_table(spark, df, full_table_name: str, mode: str = "overwrite"):
    """
    Write df to a Unity Catalog Delta table without using saveAsTable().
    Compatible with Databricks serverless compute.
    """
    from pyspark.sql import functions as F

    # Derive storage path: use the schema default location + table name
    parts       = full_table_name.split(".")          # [catalog, schema, table]
    catalog     = parts[0]
    schema      = parts[1]
    table       = parts[2]

    # Fetch schema location from Unity Catalog
    schema_info = spark.sql(f"DESCRIBE SCHEMA {catalog}.{schema}").collect()
    schema_loc  = next(
        (r["data_type"] for r in schema_info if r["database_description_item"] == "Location"),
        f"/Volumes/{catalog}/{schema}/_delta_tables",
    )
    path = f"{schema_loc}/{table}"

    # Write data
    (
        df.write
          .format("delta")
          .mode(mode)
          .option("mergeSchema", "true")
          .save(path)
    )

    # Register / refresh the table pointing at that path
    spark.sql(
        f"CREATE TABLE IF NOT EXISTS {full_table_name} "
        f"USING DELTA LOCATION '{path}'"
    )
    # If table already existed from a previous run, REFRESH so metadata is current
    spark.sql(f"REFRESH TABLE {full_table_name}")


# ---------------------------------------------------------------------------
# Persist results to Delta
# ---------------------------------------------------------------------------

def _write_delta(
    spark,
    candidates: list[dict[str, Any]],
    config: PipelineConfig,
):
    if not candidates:
        print("  [delta] No relationship candidates to write.")
        return

    from pyspark.sql import Row

    rows = [Row(**c) for c in candidates]
    df   = spark.createDataFrame(rows)

    target = (
        f"{config.output_catalog}.{config.output_schema}"
        f".{config.output_tables['relationship']}"
    )
    _write_to_table(spark, df, target, mode="overwrite")
    print(f"  [delta] Relationship candidates written → {target}")
