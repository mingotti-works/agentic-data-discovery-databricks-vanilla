# Databricks notebook source

# COMMAND ----------
# MAGIC %md
# MAGIC # Agentic Data Discovery — Pipeline Trigger
# MAGIC
# MAGIC All configuration lives in `pipeline_config.json`.
# MAGIC No MLflow setup required — audit trail is written to Delta and Volume JSON files.
# MAGIC
# MAGIC **Default run:** just run all cells.
# MAGIC **Override config:** set the `config_path` widget or the `PIPELINE_CONFIG_PATH`
# MAGIC cluster / Job environment variable.

# COMMAND ----------

dbutils.widgets.text(
    "config_path",
    defaultValue="",
    label="Config path (leave blank for default)",
)

# COMMAND ----------

import sys, os

AGENTS_DIR = "/Workspace/Users/leandro.mingoti@gmail.com/agentic-data-discovery-databricks-vanilla/agents" # adjust this
if AGENTS_DIR not in sys.path:
    sys.path.insert(0, AGENTS_DIR)

print(f"Agents directory is {AGENTS_DIR}")

# COMMAND ----------

from orchestrator import run_pipeline

config_path = dbutils.widgets.get("config_path").strip() or None
summary     = run_pipeline(config_path=config_path)

# COMMAND ----------

# MAGIC %md ## Pipeline summary

# COMMAND ----------

import json
print(json.dumps(summary, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Human review queue
# MAGIC Run the cells below to inspect what needs manual approval
# MAGIC before proceeding to data modelling.

# COMMAND ----------

from orchestrator import get_review_queue

queue = get_review_queue(config_path=config_path)
print(f"Relationships pending review : {queue['relationships_pending_review']}")
print(f"Semantic labels pending review: {queue['semantics_pending_review']}")

# COMMAND ----------

display(queue["relationship_df"].orderBy("confidence", ascending=False))

# COMMAND ----------

display(queue["semantic_df"].orderBy("confidence", ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Past runs
# MAGIC Query the pipeline_runs table to review the history of all pipeline executions.

# COMMAND ----------

from config import PipelineConfig
cfg = PipelineConfig.from_json(config_path)

display(
    spark.read
         .table(f"{cfg.output_catalog}.{cfg.output_schema}.pipeline_runs")
         .orderBy("run_ts", ascending=False)
)
