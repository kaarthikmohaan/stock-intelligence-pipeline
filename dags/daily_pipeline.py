"""
Stock Intelligence Pipeline — Daily Airflow DAG
------------------------------------------------
Runs the full pipeline every weekday at 6pm.
Tasks run in this order:

  [ingest_yfinance] ─┐
  [ingest_news]     ─┼─► [clean_silver] ─► [spark_indicators] ─► [dbt_run] ─► [dbt_test]
  [ingest_sec]      ─┘
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

BASE = "/opt/pipeline"
PYTHON = f"{BASE}/venv/bin/python"
DBT = f"{BASE}/venv/bin/dbt"

default_args = {
    "owner"           : "karthik",
    "retries"         : 2,
    "retry_delay"     : timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="stock_intelligence_daily",
    description="Daily stock data pipeline: ingest → clean → indicators → dbt",
    schedule="0 18 * * 1-5",   # 6pm Monday–Friday
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["stock", "pipeline", "daily"],
) as dag:

    # ── INGESTION (parallel) ──────────────────────────
    ingest_prices = BashOperator(
        task_id="ingest_yfinance",
        bash_command=f"cd {BASE} && {PYTHON} ingestion/ingest_yfinance.py",
        execution_timeout=timedelta(minutes=15),
    )

    ingest_news = BashOperator(
        task_id="ingest_news",
        bash_command=f"cd {BASE} && {PYTHON} ingestion/ingest_news.py",
        execution_timeout=timedelta(minutes=10),
    )

    ingest_sec = BashOperator(
        task_id="ingest_sec_edgar",
        bash_command=f"cd {BASE} && {PYTHON} ingestion/ingest_sec_edgar.py",
        execution_timeout=timedelta(minutes=10),
    )

    # ── SILVER CLEANING ───────────────────────────────
    clean_silver = BashOperator(
        task_id="clean_silver_prices",
        bash_command=f"cd {BASE} && {PYTHON} transformations/clean_prices.py --source yfinance",
        execution_timeout=timedelta(minutes=10),
    )

    # ── SPARK INDICATORS ──────────────────────────────
    spark_indicators = BashOperator(
        task_id="compute_indicators",
        bash_command=f"cd {BASE} && {PYTHON} transformations/spark_jobs/compute_indicators.py --source yfinance",
        execution_timeout=timedelta(minutes=20),
    )

    # ── DBT TRANSFORMS ────────────────────────────────
    dbt_run = BashOperator(
        task_id="dbt_run",
        bash_command=f"cd {BASE}/dbt_project && PIPELINE_BASE_DIR={BASE} {DBT} run",
        execution_timeout=timedelta(minutes=10),
    )

    dbt_test = BashOperator(
        task_id="dbt_test",
        bash_command=f"cd {BASE}/dbt_project && PIPELINE_BASE_DIR={BASE} {DBT} test",
        execution_timeout=timedelta(minutes=5),
    )

    # ── TASK DEPENDENCIES ─────────────────────────────
    # Ingest in parallel, then clean, then indicators, then dbt
    [ingest_prices, ingest_news, ingest_sec] >> clean_silver
    clean_silver >> spark_indicators
    spark_indicators >> dbt_run
    dbt_run >> dbt_test
