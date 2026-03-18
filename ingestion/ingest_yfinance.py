"""
Ingestion Script 1 — yfinance Historical Prices
------------------------------------------------
Pulls daily OHLCV data for all 50 tickers (2019-2024)
and lands them as Parquet files in the bronze layer.

Usage:
    python ingestion/ingest_yfinance.py
    python ingestion/ingest_yfinance.py --tickers AAPL MSFT GOOGL
    python ingestion/ingest_yfinance.py --start 2023-01-01 --end 2023-12-31
"""

import os
import sys
import logging
import argparse
import time
from datetime import datetime

import pandas as pd
import yfinance as yf

# ── allow running from project root ──────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingestion.config import (
    TICKERS, HISTORICAL_START, HISTORICAL_END,
    PRICES_BRONZE, YFINANCE_INTERVAL,
    PARQUET_COMPRESSION, PARQUET_ENGINE,
    LOGS_DIR, LOG_LEVEL, LOG_FORMAT,
    MAX_RETRIES, RETRY_DELAY_SECONDS,
)

# ── logging setup ─────────────────────────────────
os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(LOGS_DIR, f"yfinance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        ),
    ],
)
log = logging.getLogger("ingest_yfinance")


def get_output_path(ticker: str, year: int) -> str:
    """
    Build the output file path using Hive-style partitioning.
    Example: data/bronze/prices/yfinance/year=2024/AAPL.parquet
    This partitioning lets DuckDB and PySpark skip irrelevant years
    when querying — critical for performance at scale.
    """
    partition_dir = os.path.join(PRICES_BRONZE, "yfinance", f"year={year}")
    os.makedirs(partition_dir, exist_ok=True)
    return os.path.join(partition_dir, f"{ticker}.parquet")


def fetch_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download OHLCV data for one ticker with retry logic.
    Returns a clean DataFrame or raises an exception if all retries fail.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.info(f"  Downloading {ticker} (attempt {attempt}/{MAX_RETRIES})")
            df = yf.download(
                ticker,
                start=start,
                end=end,
                interval=YFINANCE_INTERVAL,
                auto_adjust=True,   # adjusts for splits + dividends
                progress=False,     # suppress yfinance progress bar
            )

            if df.empty:
                raise ValueError(f"yfinance returned empty DataFrame for {ticker}")

            # ── flatten MultiIndex columns if present ──
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # ── standardise column names ───────────────
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]

            # ── reset index so Date becomes a column ───
            df = df.reset_index()
            df.rename(columns={"index": "date", "Date": "date"}, inplace=True)
            df["date"] = pd.to_datetime(df["date"])

            # ── add metadata columns ───────────────────
            df["ticker"]      = ticker
            df["source"]      = "yfinance"
            df["ingested_at"] = datetime.utcnow().isoformat()

            # ── enforce column order ───────────────────
            cols = ["date", "ticker", "open", "high", "low", "close",
                    "volume", "source", "ingested_at"]
            existing = [c for c in cols if c in df.columns]
            df = df[existing]

            log.info(f"  {ticker}: {len(df)} rows from {df['date'].min().date()} to {df['date'].max().date()}")
            return df

        except Exception as e:
            log.warning(f"  {ticker} attempt {attempt} failed: {e}")
            if attempt < MAX_RETRIES:
                log.info(f"  Waiting {RETRY_DELAY_SECONDS}s before retry...")
                time.sleep(RETRY_DELAY_SECONDS)

    raise RuntimeError(f"All {MAX_RETRIES} attempts failed for {ticker}")


def save_by_year(df: pd.DataFrame, ticker: str) -> list[str]:
    """
    Split the DataFrame by year and save each year as a separate Parquet file.
    Why split by year? Because when you query 'show me 2024 prices',
    DuckDB reads only the 2024 partition — not 5 years of data.
    This makes queries 5x faster at scale.
    """
    saved_files = []
    df["year"] = df["date"].dt.year

    for year, year_df in df.groupby("year"):
        year_df = year_df.drop(columns=["year"])
        path = get_output_path(ticker, year)
        year_df.to_parquet(
            path,
            engine=PARQUET_ENGINE,
            compression=PARQUET_COMPRESSION,
            index=False,
        )
        size_kb = os.path.getsize(path) / 1024
        log.info(f"  Saved: {path} ({size_kb:.1f} KB, {len(year_df)} rows)")
        saved_files.append(path)

    return saved_files


def run(tickers: list[str], start: str, end: str) -> dict:
    """
    Main ingestion loop. Returns a summary dict.
    """
    log.info("=" * 60)
    log.info("yfinance Historical Price Ingestion")
    log.info(f"Tickers : {len(tickers)}")
    log.info(f"Period  : {start} → {end}")
    log.info(f"Output  : {PRICES_BRONZE}/yfinance/")
    log.info("=" * 60)

    succeeded, failed = [], []
    total_rows = 0
    start_time = time.time()

    for i, ticker in enumerate(tickers, 1):
        log.info(f"\n[{i:02d}/{len(tickers)}] {ticker}")
        try:
            df = fetch_ticker(ticker, start, end)
            files = save_by_year(df, ticker)
            succeeded.append(ticker)
            total_rows += len(df)
        except Exception as e:
            log.error(f"  FAILED {ticker}: {e}")
            failed.append(ticker)

        # ── small delay to be polite to yfinance ──
        time.sleep(0.3)

    elapsed = time.time() - start_time
    summary = {
        "succeeded"  : succeeded,
        "failed"     : failed,
        "total_rows" : total_rows,
        "elapsed_sec": round(elapsed, 1),
    }

    log.info("\n" + "=" * 60)
    log.info("INGESTION COMPLETE")
    log.info(f"Succeeded : {len(succeeded)}/{len(tickers)} tickers")
    log.info(f"Failed    : {failed if failed else 'none'}")
    log.info(f"Total rows: {total_rows:,}")
    log.info(f"Time taken: {elapsed:.1f}s")
    log.info("=" * 60)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest yfinance historical prices")
    parser.add_argument("--tickers", nargs="+", default=TICKERS,
                        help="List of tickers (default: all 50)")
    parser.add_argument("--start", default=HISTORICAL_START,
                        help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=HISTORICAL_END,
                        help="End date YYYY-MM-DD")
    args = parser.parse_args()

    result = run(args.tickers, args.start, args.end)
    sys.exit(0 if not result["failed"] else 1)
