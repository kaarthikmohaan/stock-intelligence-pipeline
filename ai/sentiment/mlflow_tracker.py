"""
Phase 5 — MLflow Experiment Tracking
--------------------------------------
Tracks sentiment scoring runs with MLflow.
Records parameters, metrics, and artifacts
so every run is reproducible and comparable.

Usage:
    python ai/sentiment/mlflow_tracker.py
    python ai/sentiment/mlflow_tracker.py --sample 100
"""

import os
import sys
import logging
import argparse
import time
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import mlflow
import mlflow.sklearn
import pandas as pd
import duckdb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingestion.config import (
    GOLD_DIR, LOGS_DIR, LOG_LEVEL, LOG_FORMAT,
)

os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("mlflow_tracker")

MLFLOW_DIR       = os.path.join(GOLD_DIR, "mlflow")
SENTIMENT_PATH   = os.path.join(GOLD_DIR, "sentiment", "news_sentiment.parquet")
EXPERIMENT_NAME  = "stock_sentiment_scoring"


def run_tracked_experiment(batch_size: int = 16, sample: int = None) -> str:
    """
    Run FinBERT sentiment scoring as a tracked MLflow experiment.
    Returns the MLflow run ID.
    """
    # Set up MLflow — store locally in gold layer
    mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    log.info("=" * 55)
    log.info("MLflow Tracked Sentiment Experiment")
    log.info(f"Experiment : {EXPERIMENT_NAME}")
    log.info(f"Tracking   : {MLFLOW_DIR}")
    log.info("=" * 55)

    with mlflow.start_run(run_name=f"finbert_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        run_id = run.info.run_id
        log.info(f"MLflow run ID: {run_id}")

        # ── Log parameters ─────────────────────────────
        mlflow.log_params({
            "model_name"  : "ProsusAI/finbert",
            "batch_size"  : batch_size,
            "sample_size" : sample or "all",
            "device"      : "mps",
            "max_length"  : 512,
            "labels"      : "positive,negative,neutral",
        })

        start_time = time.time()

        # ── Run scoring ────────────────────────────────
        log.info("Running FinBERT scoring...")
        from ai.sentiment.finbert_scorer import run as score_run
        summary = score_run(batch_size=batch_size, sample=sample)

        elapsed = time.time() - start_time

        # ── Log metrics ────────────────────────────────
        dist      = summary.get("sentiment_dist", {})
        total     = summary.get("total_headlines", 0)
        headlines_per_sec = total / max(elapsed, 1)

        mlflow.log_metrics({
            "total_headlines"      : total,
            "headlines_per_second" : round(headlines_per_sec, 2),
            "elapsed_seconds"      : round(elapsed, 1),
            "positive_count"       : dist.get("positive", 0),
            "negative_count"       : dist.get("negative", 0),
            "neutral_count"        : dist.get("neutral", 0),
            "positive_pct"         : round(dist.get("positive", 0) / max(total, 1) * 100, 2),
            "negative_pct"         : round(dist.get("negative", 0) / max(total, 1) * 100, 2),
            "neutral_pct"          : round(dist.get("neutral", 0) / max(total, 1) * 100, 2),
        })

        # ── Log sentiment summary as artifact ──────────
        conn = duckdb.connect(":memory:")
        summary_df = conn.execute(f"""
            SELECT
                ticker,
                COUNT(*) AS total,
                SUM(CASE WHEN sentiment='positive' THEN 1 ELSE 0 END) AS positive,
                SUM(CASE WHEN sentiment='negative' THEN 1 ELSE 0 END) AS negative,
                SUM(CASE WHEN sentiment='neutral'  THEN 1 ELSE 0 END) AS neutral,
                ROUND(AVG(sentiment_score), 4) AS avg_confidence
            FROM read_parquet('{SENTIMENT_PATH}')
            GROUP BY ticker
            ORDER BY positive DESC
        """).df()
        conn.close()

        # Save summary CSV as MLflow artifact
        artifact_path = "/tmp/sentiment_by_ticker.csv"
        summary_df.to_csv(artifact_path, index=False)
        mlflow.log_artifact(artifact_path, "sentiment_summary")

        # ── Log model info as tags ──────────────────────
        mlflow.set_tags({
            "pipeline_phase"  : "phase_5_ai",
            "data_source"     : "newsapi_finbert",
            "model_type"      : "classification",
            "framework"       : "huggingface_transformers",
            "gpu"             : "apple_metal_mps",
        })

        log.info("\n" + "=" * 55)
        log.info("MLflow Run Complete")
        log.info(f"  Run ID          : {run_id}")
        log.info(f"  Total scored    : {total:,}")
        log.info(f"  Speed           : {headlines_per_sec:.1f} headlines/sec")
        log.info(f"  Positive        : {dist.get('positive', 0)} ({dist.get('positive', 0)/max(total,1)*100:.1f}%)")
        log.info(f"  Negative        : {dist.get('negative', 0)} ({dist.get('negative', 0)/max(total,1)*100:.1f}%)")
        log.info(f"  Neutral         : {dist.get('neutral', 0)} ({dist.get('neutral', 0)/max(total,1)*100:.1f}%)")
        log.info(f"  MLflow tracking : {MLFLOW_DIR}")
        log.info("=" * 55)

        return run_id


def list_runs() -> None:
    """Display all previous experiment runs."""
    mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")
    client = mlflow.MlflowClient()

    try:
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if not experiment:
            log.info("No experiments found — run the tracker first")
            return

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
        )

        print(f"\n{'='*65}")
        print(f" Experiment: {EXPERIMENT_NAME}")
        print(f" Total runs: {len(runs)}")
        print(f"{'='*65}")
        print(f"{'Run ID':15} {'Name':30} {'Headlines':>10} {'Speed':>8}")
        print(f"{'-'*65}")

        for r in runs:
            run_id   = r.info.run_id[:12]
            name     = r.info.run_name[:28]
            total    = int(r.data.metrics.get("total_headlines", 0))
            speed    = r.data.metrics.get("headlines_per_second", 0)
            print(f"{run_id:15} {name:30} {total:>10,} {speed:>7.1f}/s")

    except Exception as e:
        log.error(f"Error listing runs: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLflow tracked sentiment experiment")
    parser.add_argument("--sample", type=int, default=None,
                        help="Score N headlines (default: all)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--list-runs", action="store_true",
                        help="List all previous runs")
    args = parser.parse_args()

    if args.list_runs:
        list_runs()
    else:
        run_id = run_tracked_experiment(args.batch_size, args.sample)
        print(f"\nRun complete. Run ID: {run_id}")
        print(f"View runs: PYTHONPATH=. python ai/sentiment/mlflow_tracker.py --list-runs")
