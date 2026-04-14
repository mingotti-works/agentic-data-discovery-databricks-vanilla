"""
agent_semantic.py
-----------------
Semantic agent — answers "what does this data actually mean?"

For every column in every profiled dataset, this agent builds a
context prompt from:
  - column name
  - data type
  - rule-based inferred type (from profiling)
  - sample distinct values
  - table name (often a strong signal even when undocumented)

It then calls the LLM (or stub) and parses back a structured response:
  - business_label    : human-friendly name
  - description       : 1–2 sentence business meaning
  - sensitivity       : "low" | "medium" | "high" | "pii"
  - confidence        : 0.0–1.0 (self-assessed by LLM, normalised)

Results are batched (one LLM call per table) to keep token usage low.

Output schema (one row per column per dataset):
  run_ts, source_name, table_name, column_name,
  business_label, description, sensitivity, confidence, status, raw_response
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from config import LLMClient, PipelineConfig, get_spark


# ---------------------------------------------------------------------------
# System prompt (shared across all table-level calls)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are a senior data analyst specialising in undocumented datasets.
Given a table name, column names, data types, and sample values,
infer the business meaning of each column.

Respond ONLY with a valid JSON array — one object per column — in this format:
[
  {
    "column_name":    "<exact column name as given>",
    "business_label": "<short human-friendly name, 2-5 words>",
    "description":    "<1-2 sentences explaining business meaning>",
    "sensitivity":    "<one of: low | medium | high | pii>",
    "confidence":     <float between 0.0 and 1.0>
  },
  ...
]

Rules:
- Output ONLY the JSON array. No markdown, no explanation, no preamble.
- sensitivity = "pii" for names, emails, phone numbers, IDs, addresses.
- sensitivity = "high" for financial amounts, account numbers.
- sensitivity = "medium" for operational codes, internal identifiers.
- sensitivity = "low" for timestamps, flags, generic counts.
- confidence = 1.0 means you are certain; 0.5 means you are guessing.
- If you cannot infer meaning, set description = "Unknown — insufficient context"
  and confidence = 0.3.
""".strip()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(
    config: PipelineConfig,
    profiles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Run semantic inference over all column profiles.
    Batches by table — one LLM call per table.
    Returns a flat list of semantic result dicts.
    """
    spark  = get_spark()
    llm    = LLMClient(config)
    results: list[dict[str, Any]] = []

    print("=" * 60)
    print("SEMANTIC AGENT — starting")
    print("=" * 60)

    # Group profiles by (source_name, table_name)
    tables: dict[tuple[str, str], list[dict]] = {}
    for p in profiles:
        key = (p["source_name"], p["table_name"])
        tables.setdefault(key, []).append(p)

    for (source_name, table_name), col_profiles in tables.items():
        print(f"\n  Semantic: {source_name} / {table_name} ({len(col_profiles)} cols)")
        try:
            table_results = _infer_table(
                llm          = llm,
                source_name  = source_name,
                table_name   = table_name,
                col_profiles = col_profiles,
                config       = config,
            )
            results.extend(table_results)
        except Exception as exc:
            print(f"    [WARN] Semantic inference failed for {table_name}: {exc}")

    print(f"\nSEMANTIC AGENT — complete. Total columns annotated: {len(results)}")
    _write_delta(spark, results, config)

    return results


# ---------------------------------------------------------------------------
# Per-table inference
# ---------------------------------------------------------------------------

def _infer_table(
    llm: LLMClient,
    source_name: str,
    table_name: str,
    col_profiles: list[dict],
    config: PipelineConfig,
) -> list[dict[str, Any]]:
    """
    Build a single LLM prompt for all columns in a table.
    Parse the JSON array response into individual result dicts.
    """
    user_prompt = _build_user_prompt(table_name, col_profiles)
    raw_response = llm.call(SYSTEM_PROMPT, user_prompt)

    parsed = _parse_llm_response(raw_response, col_profiles)

    results: list[dict[str, Any]] = []
    for item in parsed:
        col_name   = item.get("column_name", "")
        confidence = float(item.get("confidence", 0.5))

        # Apply status tiers using same thresholds as relationship agent
        if confidence >= config.auto_accept_confidence:
            status = "auto_accepted"
        elif confidence >= config.human_review_confidence:
            status = "needs_review"
        else:
            status = "low_confidence"

        results.append({
            "run_ts":         datetime.utcnow().isoformat(),
            "source_name":    source_name,
            "table_name":     table_name,
            "column_name":    col_name,
            "business_label": item.get("business_label", ""),
            "description":    item.get("description", ""),
            "sensitivity":    item.get("sensitivity", "low"),
            "confidence":     confidence,
            "status":         status,
            "raw_response":   raw_response[:2000],  # truncate for storage
        })

    print(f"    → {len(results)} column(s) annotated")
    return results


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_user_prompt(table_name: str, col_profiles: list[dict]) -> str:
    """
    Build a compact user prompt with table context and column metadata.
    Keeps tokens low by summarising rather than dumping all profile data.
    """
    lines = [
        f"Table name: {table_name}",
        "",
        "Columns (name | type | inferred_type | sample_values):",
    ]

    for p in col_profiles:
        sample = json.loads(p.get("sample_values", "[]"))
        sample_str = ", ".join(str(v) for v in sample[:5])  # top 5 only
        lines.append(
            f"  - {p['column_name']} | {p['dtype']} "
            f"| {p.get('inferred_type', 'unknown')} "
            f"| [{sample_str}]"
        )

    lines += [
        "",
        "Additional context:",
        f"  - Total rows sampled: {col_profiles[0].get('total_rows', 'unknown')}",
        f"  - Null rates range: "
        f"{min(p.get('null_rate', 0) for p in col_profiles):.0%}"
        f" – {max(p.get('null_rate', 0) for p in col_profiles):.0%}",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def _parse_llm_response(
    raw: str,
    col_profiles: list[dict],
) -> list[dict]:
    """
    Extract and parse the JSON array from the LLM response.
    Falls back to a stub result per column if parsing fails.
    """
    # Strip markdown fences if the LLM ignored instructions
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # Try to extract first [...] block
    match = re.search(r"\[.*?\]", cleaned, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    # Fallback — return a placeholder per column so we never drop rows
    print("    [WARN] Could not parse LLM response — using fallback placeholders")
    return [
        {
            "column_name":    p["column_name"],
            "business_label": p["column_name"].replace("_", " ").title(),
            "description":    "Could not infer — LLM response parse failure.",
            "sensitivity":    "low",
            "confidence":     0.3,
        }
        for p in col_profiles
    ]


# ---------------------------------------------------------------------------
# Persist results to Delta
# ---------------------------------------------------------------------------

def _write_delta(
    spark,
    results: list[dict[str, Any]],
    config: PipelineConfig,
):
    if not results:
        return

    from pyspark.sql import Row

    rows = [Row(**r) for r in results]
    df   = spark.createDataFrame(rows)

    target = (
        f"{config.output_catalog}.{config.output_schema}"
        f".{config.output_tables['semantic']}"
    )
    (
        df.write
          .format("delta")
          .mode("overwrite")
          .option("mergeSchema", "true")
          .saveAsTable(target)
    )
    print(f"  [delta] Semantic results written → {target}")
