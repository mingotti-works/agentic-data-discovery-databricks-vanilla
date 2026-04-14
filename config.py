"""
config.py
---------
Shared configuration, Spark session factory, LLM client, and common types.
No MLflow dependency — audit trail is handled via Delta tables and Volume JSON files.

Config path resolution order (first match wins):
  1. Argument passed to PipelineConfig.from_json(path)
  2. Environment variable  PIPELINE_CONFIG_PATH
  3. pipeline_config.json beside this file  (default)

JDBC credentials are never stored in the JSON.
Each JDBC source specifies "user_env" / "pass_env" — the names of the
environment variables that hold the actual credentials.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_THIS_DIR = Path(__file__).parent


def _default_config_path() -> Path:
    env_path = os.getenv("PIPELINE_CONFIG_PATH")
    if env_path:
        return Path(env_path)
    return _THIS_DIR / "pipeline_config.json"


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    # ── Volumes ───────────────────────────────────────────────────────────
    volume_paths:           list[str]
    volume_file_extensions: tuple[str, ...]

    # ── JDBC sources (credentials resolved from env vars) ─────────────────
    jdbc_sources: list[dict[str, str]]

    # ── Delta output ──────────────────────────────────────────────────────
    output_catalog: str
    output_schema:  str
    output_tables:  dict[str, str]

    # ── Audit artefacts (JSON files written here after each run) ──────────
    # Must be a path inside a Databricks Volume, e.g.:
    #   /Volumes/my_catalog/my_schema/pipeline_artefacts
    artefacts_path: str

    # ── Profiling ─────────────────────────────────────────────────────────
    profiling_sample_rows:   int
    profiling_sample_values: int

    # ── Relationship discovery ────────────────────────────────────────────
    relationship_overlap_threshold:  float
    relationship_max_cols_per_table: int

    # ── Confidence thresholds ─────────────────────────────────────────────
    auto_accept_confidence:  float
    human_review_confidence: float

    # ── LLM ───────────────────────────────────────────────────────────────
    llm_provider: str

    # ── Resolved at load time ─────────────────────────────────────────────
    config_path: str = ""

    # -----------------------------------------------------------------------
    # Factory
    # -----------------------------------------------------------------------

    @classmethod
    def from_json(cls, path: str | Path | None = None) -> "PipelineConfig":
        resolved = Path(path) if path else _default_config_path()
        if not resolved.exists():
            raise FileNotFoundError(
                f"Pipeline config not found: {resolved}\n"
                "Set PIPELINE_CONFIG_PATH or pass the path explicitly."
            )
        with open(resolved, encoding="utf-8") as fh:
            raw = json.load(fh)
        raw.pop("_comment", None)

        return cls(
            volume_paths            = raw.get("volume_paths", []),
            volume_file_extensions  = tuple(raw.get("volume_file_extensions", [".parquet", ".csv"])),
            jdbc_sources            = _resolve_jdbc_credentials(raw.get("jdbc_sources", [])),
            output_catalog          = raw["output_catalog"],
            output_schema           = raw["output_schema"],
            output_tables           = raw["output_tables"],
            artefacts_path          = raw.get("artefacts_path", ""),
            profiling_sample_rows   = int(raw.get("profiling_sample_rows", 10_000)),
            profiling_sample_values = int(raw.get("profiling_sample_values", 10)),
            relationship_overlap_threshold  = float(raw.get("relationship_overlap_threshold", 0.80)),
            relationship_max_cols_per_table = int(raw.get("relationship_max_cols_per_table", 50)),
            auto_accept_confidence  = float(raw.get("auto_accept_confidence", 0.85)),
            human_review_confidence = float(raw.get("human_review_confidence", 0.50)),
            llm_provider            = raw.get("llm_provider", "stub"),
            config_path             = str(resolved),
        )

    def to_dict(self) -> dict[str, Any]:
        """Sanitised representation — safe to log (no credentials)."""
        return {
            "config_path":                    self.config_path,
            "volume_paths":                   self.volume_paths,
            "volume_file_extensions":         list(self.volume_file_extensions),
            "jdbc_source_names":              [j["name"] for j in self.jdbc_sources],
            "jdbc_source_count":              len(self.jdbc_sources),
            "output_catalog":                 self.output_catalog,
            "output_schema":                  self.output_schema,
            "output_tables":                  self.output_tables,
            "artefacts_path":                 self.artefacts_path,
            "profiling_sample_rows":          self.profiling_sample_rows,
            "profiling_sample_values":        self.profiling_sample_values,
            "relationship_overlap_threshold": self.relationship_overlap_threshold,
            "relationship_max_cols_per_table":self.relationship_max_cols_per_table,
            "auto_accept_confidence":         self.auto_accept_confidence,
            "human_review_confidence":        self.human_review_confidence,
            "llm_provider":                   self.llm_provider,
        }


# ---------------------------------------------------------------------------
# JDBC credential resolver
# ---------------------------------------------------------------------------

def _resolve_jdbc_credentials(sources: list[dict]) -> list[dict]:
    resolved = []
    for src in sources:
        entry    = dict(src)
        user_env = entry.pop("user_env", None)
        pass_env = entry.pop("pass_env", None)
        if user_env:
            entry["user"] = os.getenv(user_env, "")
            if not entry["user"]:
                print(f"  [WARN] JDBC '{entry.get('name')}': env var '{user_env}' not set.")
        if pass_env:
            entry["password"] = os.getenv(pass_env, "")
            if not entry["password"]:
                print(f"  [WARN] JDBC '{entry.get('name')}': env var '{pass_env}' not set.")
        resolved.append(entry)
    return resolved


# ---------------------------------------------------------------------------
# Spark session factory
# ---------------------------------------------------------------------------

def get_spark():
    try:
        from pyspark.sql import SparkSession
        return SparkSession.builder.getOrCreate()
    except Exception as exc:
        raise RuntimeError(
            "Could not obtain a SparkSession. "
            "Make sure this runs on a Databricks cluster."
        ) from exc


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Uniform interface for all LLM backends.
    Change llm_provider in pipeline_config.json to switch backends.
    The DBRX and Llama 3 providers call Databricks Model Serving directly
    via the requests library — no MLflow dependency.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    def call(self, system_prompt: str, user_prompt: str) -> str:
        p = self.config.llm_provider
        if p == "stub":   return self._stub(user_prompt)
        if p == "dbrx":   return self._databricks_serving("databricks-dbrx-instruct", system_prompt, user_prompt)
        if p == "llama3": return self._databricks_serving("databricks-meta-llama-3-1-70b-instruct", system_prompt, user_prompt)
        if p == "openai": return self._openai(system_prompt, user_prompt)
        raise ValueError(f"Unknown llm_provider: {p!r}")

    def _stub(self, user_prompt: str) -> str:
        return (
            "STUB RESPONSE | provider=stub | "
            f"prompt_chars={len(user_prompt)} | "
            "Set llm_provider in pipeline_config.json to activate a real model."
        )

    def _databricks_serving(self, endpoint: str, system_prompt: str, user_prompt: str) -> str:
        """
        Call Databricks Model Serving via the REST API directly.
        Requires DATABRICKS_HOST and DATABRICKS_TOKEN environment variables.
        """
        import requests
        host  = os.environ["DATABRICKS_HOST"].rstrip("/")
        token = os.environ["DATABRICKS_TOKEN"]
        url   = f"{host}/serving-endpoints/{endpoint}/invocations"
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "max_tokens": 512,
            "temperature": 0.0,
        }
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _openai(self, system_prompt: str, user_prompt: str) -> str:
        import openai
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=512,
            temperature=0.0,
        )
        return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# Shared result type
# ---------------------------------------------------------------------------

@dataclass
class DatasetMeta:
    source_type: str
    source_name: str
    table_name:  str
    columns:     list[str]
    row_count:   int
    raw_schema:  dict[str, str]
    extra:       dict[str, Any] = field(default_factory=dict)
