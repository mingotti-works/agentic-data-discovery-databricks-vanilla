"""
orchestrator.py
---------------
Orchestrator — runs the full agentic data discovery pipeline.
No MLflow dependency. Audit trail is written to:
  - Delta table  : <catalog>.<schema>.pipeline_runs   (one row per run)
  - Volume JSON  : <artefacts_path>/<run_id>/          (one file per artefact)

Execution order:
  1. Discovery    — finds all datasets
  2. Profiling    — statistical profiling of every column
  3. Relationship — FK/PK candidate detection
  4. Semantic     — LLM-based column meaning inference

Usage (Databricks notebook):
  %run ./run_pipeline

Or import directly:
  from orchestrator import run_pipeline
  run_pipeline()                          # uses pipeline_config.json
  run_pipeline("/path/to/other_cfg.json") # explicit config
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import agent_discovery
import agent_profiling
import agent_relationship
import agent_semantic
from config import PipelineConfig, get_spark


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_pipeline(config_path: str | None = None) -> dict[str, Any]:
    """
    Execute the full pipeline. Returns a summary dict.

    Args:
        config_path: optional explicit path to pipeline_config.json.
                     Falls back to PIPELINE_CONFIG_PATH env var,
                     then to pipeline_config.json beside this module.
    """
    config = PipelineConfig.from_json(config_path)
    spark  = get_spark()

    run_id  = _new_run_id()
    run_ts  = datetime.now(timezone.utc).isoformat()
    summary: dict[str, Any] = {}

    print(f"[ORCHESTRATOR] Config  : {config.config_path}")
    print(f"[ORCHESTRATOR] Run ID  : {run_id}")
    print(f"[ORCHESTRATOR] Run TS  : {run_ts}")

    # ── Ensure output schema exists ────────────────────────────────────
    spark.sql(f"CREATE CATALOG  IF NOT EXISTS {config.output_catalog}")
    spark.sql(f"CREATE SCHEMA   IF NOT EXISTS {config.output_catalog}.{config.output_schema}")

    # ── Ensure artefacts directory exists ──────────────────────────────
    run_artefacts_dir = _ensure_artefacts_dir(config, run_id)

    # ── Write config snapshot so every run is fully reproducible ───────
    _write_artefact(run_artefacts_dir, "config_snapshot.json", config.to_dict())

    try:
        # ── PHASE 1: Discovery ─────────────────────────────────────────
        t0       = time.time()
        datasets = agent_discovery.run(config)
        t_disc   = round(time.time() - t0, 1)

        if not datasets:
            print("\n[ORCHESTRATOR] No datasets discovered — stopping pipeline.")
            summary = {"status": "stopped_no_data", "run_id": run_id, "run_ts": run_ts}
            _write_run_record(spark, config, run_id, run_ts, summary)
            return summary

        # ── PHASE 2: Profiling ─────────────────────────────────────────
        t0       = time.time()
        profiles = agent_profiling.run(config, datasets)
        t_prof   = round(time.time() - t0, 1)

        pk_candidates = [p for p in profiles if p.get("pk_candidate")]

        # ── PHASE 3: Relationship discovery ───────────────────────────
        t0            = time.time()
        relationships = agent_relationship.run(config, datasets, profiles)
        t_rel         = round(time.time() - t0, 1)

        auto_accepted = [r for r in relationships if r.get("status") == "auto_accepted"]
        needs_review  = [r for r in relationships if r.get("status") == "needs_review"]
        discarded     = [r for r in relationships if r.get("status") == "discarded"]

        # ── PHASE 4: Semantic inference ───────────────────────────────
        t0        = time.time()
        semantics = agent_semantic.run(config, profiles)
        t_sem     = round(time.time() - t0, 1)

        sem_auto   = [s for s in semantics if s.get("status") == "auto_accepted"]
        sem_review = [s for s in semantics if s.get("status") == "needs_review"]
        sem_pii    = [s for s in semantics if s.get("sensitivity") == "pii"]

        # ── Write artefacts to Volume ─────────────────────────────────
        if relationships:
            _write_artefact(
                run_artefacts_dir, "relationship_candidates.json",
                {"top_candidates": relationships[:20]},
            )
        if sem_pii:
            _write_artefact(
                run_artefacts_dir, "pii_summary.json",
                {"pii_columns": [
                    {k: s[k] for k in ("source_name", "table_name", "column_name",
                                       "business_label", "confidence")}
                    for s in sem_pii
                ]},
            )

        # ── Build summary ─────────────────────────────────────────────
        total_duration = round(t_disc + t_prof + t_rel + t_sem, 1)

        summary = {
            "status":             "complete",
            "run_id":             run_id,
            "run_ts":             run_ts,
            "total_duration_sec": total_duration,
            "phase_durations": {
                "discovery_sec":    t_disc,
                "profiling_sec":    t_prof,
                "relationship_sec": t_rel,
                "semantic_sec":     t_sem,
            },
            "datasets_found":   len(datasets),
            "columns_profiled": len(profiles),
            "pk_candidates":    len(pk_candidates),
            "relationship_candidates": {
                "auto_accepted": len(auto_accepted),
                "needs_review":  len(needs_review),
                "discarded":     len(discarded),
            },
            "semantic_annotations": {
                "auto_accepted": len(sem_auto),
                "needs_review":  len(sem_review),
                "pii_columns":   len(sem_pii),
            },
            "artefacts_dir": str(run_artefacts_dir),
        }

        _write_artefact(run_artefacts_dir, "pipeline_summary.json", summary)

    except Exception as exc:
        summary = {
            "status":   "failed",
            "run_id":   run_id,
            "run_ts":   run_ts,
            "error":    str(exc),
        }
        _write_artefact(run_artefacts_dir, "pipeline_summary.json", summary)
        raise

    finally:
        # Always write the run record to Delta — even on failure
        _write_run_record(spark, config, run_id, run_ts, summary)

    _print_summary(summary, needs_review, sem_pii)
    return summary


# ---------------------------------------------------------------------------
# Human-review helper
# ---------------------------------------------------------------------------

def get_review_queue(config_path: str | None = None) -> dict[str, Any]:
    """
    Return Delta DataFrames for everything flagged 'needs_review'.
    Use in a review notebook to inspect and action human checkpoints.
    """
    config = PipelineConfig.from_json(config_path)
    spark  = get_spark()

    def read_needs_review(table_key: str):
        full = f"{config.output_catalog}.{config.output_schema}.{config.output_tables[table_key]}"
        try:
            return spark.read.table(full).filter("status = 'needs_review'")
        except Exception:
            from pyspark.sql.types import StructType
            return spark.createDataFrame([], StructType([]))

    rel_df = read_needs_review("relationship")
    sem_df = read_needs_review("semantic")

    return {
        "relationships_pending_review": rel_df.count(),
        "semantics_pending_review":     sem_df.count(),
        "relationship_df":              rel_df,
        "semantic_df":                  sem_df,
    }


# ---------------------------------------------------------------------------
# Audit trail — Delta run record
# ---------------------------------------------------------------------------

def _write_run_record(
    spark,
    config: PipelineConfig,
    run_id: str,
    run_ts: str,
    summary: dict[str, Any],
):
    """
    Append one row to the pipeline_runs Delta table.
    Creates the table on first run (mergeSchema=true handles evolution).
    """
    from pyspark.sql import Row

    target = f"{config.output_catalog}.{config.output_schema}.pipeline_runs"

    row = Row(
        run_id         = run_id,
        run_ts         = run_ts,
        config_path    = config.config_path,
        status         = summary.get("status", "unknown"),
        summary_json   = json.dumps(summary),
        artefacts_dir  = summary.get("artefacts_dir", ""),
    )

    df = spark.createDataFrame([row])
    (
        df.write
          .format("delta")
          .mode("append")
          .option("mergeSchema", "true")
          .saveAsTable(target)
    )
    print(f"  [delta] Run record appended → {target}  (run_id={run_id})")


# ---------------------------------------------------------------------------
# Artefact helpers — write JSON files to a Volume
# ---------------------------------------------------------------------------

def _ensure_artefacts_dir(config: PipelineConfig, run_id: str) -> Path:
    """
    Create a per-run subdirectory under config.artefacts_path.
    Falls back to a temp path if artefacts_path is not configured.
    """
    base = config.artefacts_path or f"/tmp/pipeline_artefacts"
    run_dir = Path(base) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_artefact(directory: Path, filename: str, data: Any):
    """Write a JSON artefact file to the run's artefact directory."""
    target = directory / filename
    with open(target, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    print(f"  [artefact] {target}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"run_{ts}_{uuid.uuid4().hex[:6]}"


def _print_summary(
    summary: dict[str, Any],
    needs_review: list,
    pii_cols: list,
):
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE — SUMMARY")
    print("=" * 60)
    print(f"  Run ID            : {summary['run_id']}")
    print(f"  Duration          : {summary['total_duration_sec']}s")
    print(f"  Datasets found    : {summary['datasets_found']}")
    print(f"  Columns profiled  : {summary['columns_profiled']}")
    print(f"  PK candidates     : {summary['pk_candidates']}")
    print()
    print("  Relationship candidates:")
    rel = summary["relationship_candidates"]
    print(f"    Auto-accepted   : {rel['auto_accepted']}")
    print(f"    Needs review    : {rel['needs_review']}")
    print(f"    Discarded       : {rel['discarded']}")
    print()
    print("  Semantic annotations:")
    sem = summary["semantic_annotations"]
    print(f"    Auto-accepted   : {sem['auto_accepted']}")
    print(f"    Needs review    : {sem['needs_review']}")
    print(f"    PII columns     : {sem['pii_columns']}")
    print()

    if needs_review:
        print("  TOP RELATIONSHIP CANDIDATES FOR REVIEW:")
        for r in needs_review[:5]:
            print(
                f"    {r['source_table']}.{r['source_col']}"
                f" → {r['target_table']}.{r['target_col']}"
                f"  conf={r['confidence']:.2f}"
            )

    if pii_cols:
        print("\n  PII COLUMNS DETECTED:")
        for s in pii_cols[:5]:
            print(
                f"    {s['source_name']}.{s['table_name']}.{s['column_name']}"
                f"  ({s['business_label']}, conf={s['confidence']:.2f})"
            )

    print(f"\n  Artefacts         : {summary.get('artefacts_dir', 'n/a')}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Run when executed directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_pipeline()
