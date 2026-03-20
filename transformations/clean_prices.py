"""
Silver Layer — Price Data Cleaning
-----------------------------------
Reads raw bronze price Parquet files, validates with pandera,
cleans bad rows, enriches with calculated columns, and writes
to the silver layer.

Usage:
    python transformations/clean_prices.py --source yfinance
    python transformations/clean_prices.py --source kaggle
    python transformations/clean_prices.py --source all
    python transformations/clean_prices.py --source kaggle --sample 500000
"""

import os
import sys
import logging
import argparse
import time
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import pandas as pd
import pandera.pandas as pa
import duckdb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingestion.config import (
    PRICES_BRONZE, SILVER_DIR,
    PARQUET_COMPRESSION, PARQUET_ENGINE,
    LOGS_DIR, LOG_LEVEL, LOG_FORMAT,
)

os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(LOGS_DIR, f"silver_prices_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        ),
    ],
)
log = logging.getLogger("clean_prices")

SILVER_PRICES_DIR = os.path.join(SILVER_DIR, "prices")

# ── PANDERA SCHEMA ─────────────────────────────────────────
# Every row in the silver layer must pass these rules.
# Rows that fail are dropped and counted — pipeline never crashes.
PRICE_SCHEMA = pa.DataFrameSchema(
    columns={
        "date"  : pa.Column(pa.DateTime, nullable=False),
        "ticker": pa.Column(str, pa.Check.str_length(1, 15), nullable=False),
        "open"  : pa.Column(float, pa.Check.gt(0), nullable=True),
        "high"  : pa.Column(float, pa.Check.gt(0), nullable=True),
        "low"   : pa.Column(float, pa.Check.gt(0), nullable=True),
        "close" : pa.Column(float, pa.Check.gt(0), nullable=False),
        "volume": pa.Column(float, pa.Check.ge(0), nullable=True),
    },
    coerce=True,
    drop_invalid_rows=True,
)


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calculated columns needed by all downstream consumers.
    Computed once here — never recomputed in dbt, PySpark, or ML.
    """
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Daily return: % change from previous day's close
    # This is the single most important metric in quantitative finance
    df["daily_return"] = (
        df.groupby("ticker")["close"]
        .pct_change()
        .round(6)
    )

    # Price range: intraday movement (high - low)
    # Large range = high volatility day
    df["price_range"] = (df["high"] - df["low"]).round(4)

    # Typical price: (high + low + close) / 3
    # Standard input for VWAP and many technical indicators
    df["typical_price"] = (
        (df["high"] + df["low"] + df["close"]) / 3
    ).round(4)

    # Range as % of close — normalised volatility measure
    # Allows comparing volatility across different price levels
    df["range_pct"] = (
        df["price_range"] / df["close"] * 100
    ).round(4)

    return df


def clean_dataframe(df: pd.DataFrame, source: str) -> tuple:
    """
    Full cleaning pipeline for one batch.
    Returns (cleaned_df, stats_dict).
    """
    stats = {
        "source"           : source,
        "input_rows"       : len(df),
        "output_rows"      : 0,
        "dropped_nulls"    : 0,
        "dropped_zeros"    : 0,
        "dropped_dupes"    : 0,
        "validation_errors": 0,
    }

    # Check required columns exist
    required = ["date", "ticker", "close"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        log.warning(f"  Missing columns: {missing} — skipping batch")
        return pd.DataFrame(), stats

    # Cast types safely — coerce means bad values become NaN
    df["date"]   = pd.to_datetime(df["date"],   errors="coerce")
    df["close"]  = pd.to_numeric(df["close"],   errors="coerce")
    df["open"]   = pd.to_numeric(df.get("open",   pd.Series(dtype=float)), errors="coerce")
    df["high"]   = pd.to_numeric(df.get("high",   pd.Series(dtype=float)), errors="coerce")
    df["low"]    = pd.to_numeric(df.get("low",    pd.Series(dtype=float)), errors="coerce")
    df["volume"] = pd.to_numeric(df.get("volume", pd.Series(dtype=float)), errors="coerce")

    # Drop rows with null date, ticker, or close
    before = len(df)
    df = df.dropna(subset=["date", "close", "ticker"])
    stats["dropped_nulls"] = before - len(df)

    # Drop zero or negative close prices — physically impossible
    before = len(df)
    df = df[df["close"] > 0]
    stats["dropped_zeros"] = before - len(df)

    # Drop rows where high < low — corrupted OHLC data
    # Found in Kaggle source: decimal point errors cause high < low
    before = len(df)
    valid_ohlc = (
        df["high"].isna() | df["low"].isna() |
        (df["high"] >= df["low"])
    )
    df = df[valid_ohlc]
    stats["dropped_zeros"] += before - len(df)

    # Drop duplicate ticker + date combinations
    before = len(df)
    df = df.drop_duplicates(subset=["ticker", "date"], keep="last")
    stats["dropped_dupes"] = before - len(df)

    # pandera schema validation — drops invalid rows silently
    before = len(df)
    try:
        df = PRICE_SCHEMA.validate(df, lazy=True)
    except pa.errors.SchemaErrors as e:
        stats["validation_errors"] = before - len(df)
        log.warning(f"  pandera dropped {stats['validation_errors']} invalid rows")

    # Enrich with calculated columns
    if not df.empty:
        df = enrich(df)

    # Add silver metadata
    df["silver_processed_at"] = datetime.utcnow().isoformat()
    df["data_source"]         = source

    stats["output_rows"] = len(df)
    return df, stats


def process_source(source: str, sample: int = None) -> dict:
    """Read bronze → clean → write silver for one source."""
    bronze_glob = os.path.join(PRICES_BRONZE, source, "**", "*.parquet")
    silver_base = os.path.join(SILVER_PRICES_DIR, source)
    os.makedirs(silver_base, exist_ok=True)

    log.info(f"\nSource : {source}")
    log.info(f"Bronze : {bronze_glob}")
    log.info(f"Silver : {silver_base}")

    # DuckDB reads all Parquet files in one vectorised scan
    conn  = duckdb.connect(":memory:")
    query = f"SELECT * FROM read_parquet('{bronze_glob}')"
    if sample:
        query += f" USING SAMPLE {sample} ROWS"
        log.info(f"Sample : {sample:,} rows")

    log.info("Loading bronze data...")
    t0 = time.time()
    df = conn.execute(query).df()
    conn.close()
    log.info(f"Loaded {len(df):,} rows in {time.time()-t0:.1f}s")

    # Clean
    log.info("Cleaning and validating...")
    t1 = time.time()
    df_clean, stats = clean_dataframe(df, source)
    log.info(f"Cleaning done in {time.time()-t1:.1f}s")

    if df_clean.empty:
        log.error("  No valid rows after cleaning — check bronze data")
        return stats

    # Log cleaning summary
    dropped  = stats["input_rows"] - stats["output_rows"]
    drop_pct = dropped / max(stats["input_rows"], 1) * 100
    log.info(f"  Input rows   : {stats['input_rows']:,}")
    log.info(f"  Output rows  : {stats['output_rows']:,}")
    log.info(f"  Total dropped: {dropped:,} ({drop_pct:.2f}%)")
    log.info(f"    Nulls      : {stats['dropped_nulls']:,}")
    log.info(f"    Zeros      : {stats['dropped_zeros']:,}")
    log.info(f"    Dupes      : {stats['dropped_dupes']:,}")
    log.info(f"    Validation : {stats['validation_errors']:,}")

    # Write silver — one file per year (cross-ticker)
    # Unlike bronze (one file per ticker per year),
    # silver uses one file per year for efficient cross-ticker analytics
    log.info("Writing silver Parquet...")
    t2 = time.time()
    df_clean["year"] = df_clean["date"].dt.year

    for year, year_df in df_clean.groupby("year"):
        year_df  = year_df.drop(columns=["year"])
        out_dir  = os.path.join(silver_base, f"year={year}")
        out_path = os.path.join(out_dir, "prices.parquet")
        os.makedirs(out_dir, exist_ok=True)
        year_df.to_parquet(
            out_path,
            engine=PARQUET_ENGINE,
            compression=PARQUET_COMPRESSION,
            index=False,
        )
        size_kb = os.path.getsize(out_path) / 1024
        log.info(f"  year={year}: {len(year_df):,} rows → {size_kb:.0f} KB")

    log.info(f"Silver write done in {time.time()-t2:.1f}s")
    return stats


def run(sources: list, sample: int = None) -> dict:
    log.info("=" * 60)
    log.info("Silver Layer — Price Data Cleaning")
    log.info(f"Sources : {sources}")
    log.info(f"Output  : {SILVER_PRICES_DIR}/")
    log.info("=" * 60)

    start_time = time.time()
    all_stats  = {}

    for source in sources:
        all_stats[source] = process_source(source, sample)

    elapsed = time.time() - start_time
    log.info("\n" + "=" * 60)
    log.info("SILVER CLEANING COMPLETE")
    for src, s in all_stats.items():
        log.info(f"  {src}: {s.get('output_rows', 0):,} clean rows")
    log.info(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    log.info("=" * 60)

    return all_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean bronze prices → silver")
    parser.add_argument("--source", choices=["yfinance", "kaggle", "all"],
                        default="all", help="Which source to clean")
    parser.add_argument("--sample", type=int, default=None,
                        help="Sample N rows for testing")
    args = parser.parse_args()

    sources = ["yfinance", "kaggle"] if args.source == "all" else [args.source]
    run(sources, args.sample)
