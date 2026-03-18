"""
Ingestion Script 3 — Kaggle Bulk Stock Dataset
-----------------------------------------------
Downloads the "Huge Stock Market Dataset" from Kaggle
(8,000+ US stocks, 20+ years of daily OHLCV data)
and converts CSVs to partitioned Parquet in the bronze layer.

Usage:
    python ingestion/ingest_kaggle.py
    python ingestion/ingest_kaggle.py --dataset borismarjanovic/price-volume-data-for-all-us-stocks-etfs
"""

import os
import sys
import logging
import argparse
import time
import zipfile
import glob
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingestion.config import (
    PRICES_BRONZE, PARQUET_COMPRESSION, PARQUET_ENGINE,
    LOGS_DIR, LOG_LEVEL, LOG_FORMAT,
)

load_dotenv()

os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(LOGS_DIR, f"kaggle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        ),
    ],
)
log = logging.getLogger("ingest_kaggle")

# Default dataset — one of the most comprehensive free stock datasets
DEFAULT_DATASET = "borismarjanovic/price-volume-data-for-all-us-stocks-etfs"

# Where to extract the raw zip
EXTRACT_DIR = os.path.join(PRICES_BRONZE, "kaggle_raw")
# Where to write final Parquet files
KAGGLE_PARQUET_DIR = os.path.join(PRICES_BRONZE, "kaggle")


def download_dataset(dataset: str) -> str:
    """Download dataset from Kaggle and return path to the zip file."""
    import kaggle
    log.info(f"Downloading Kaggle dataset: {dataset}")
    log.info(f"This may take several minutes depending on your connection...")

    os.makedirs(EXTRACT_DIR, exist_ok=True)
    kaggle.api.dataset_download_files(
        dataset,
        path=EXTRACT_DIR,
        unzip=False,
        quiet=False,
    )

    # Find the downloaded zip
    zips = glob.glob(os.path.join(EXTRACT_DIR, "*.zip"))
    if not zips:
        raise FileNotFoundError(f"No zip file found in {EXTRACT_DIR}")

    log.info(f"Downloaded: {zips[0]}")
    return zips[0]


def extract_zip(zip_path: str) -> str:
    """Extract zip file and return the directory containing CSVs."""
    log.info(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(EXTRACT_DIR)
    log.info("Extraction complete")
    return EXTRACT_DIR


def convert_csv_to_parquet(csv_path: str, ticker: str) -> tuple[int, int]:
    """
    Convert one CSV file to Parquet.
    Returns (rows_written, bytes_saved).
    """
    try:
        df = pd.read_csv(csv_path, low_memory=False)

        if df.empty:
            return 0, 0

        # Standardise column names
        df.columns = [c.strip().lower() for c in df.columns]

        # Rename common variations
        rename_map = {
            "date": "date", "open": "open", "high": "high",
            "low": "low", "close": "close", "vol": "volume",
            "volume": "volume", "openint": "open_interest",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        # Ensure date column
        if "date" not in df.columns:
            return 0, 0

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

        if df.empty:
            return 0, 0

        df["ticker"] = ticker.upper()
        df["source"] = "kaggle_boris"
        df["ingested_at"] = datetime.utcnow().isoformat()

        # Add year for partitioning
        df["year"] = df["date"].dt.year

        # Save one Parquet per year per ticker
        rows_total = 0
        for year, year_df in df.groupby("year"):
            out_dir = os.path.join(KAGGLE_PARQUET_DIR, f"year={year}")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{ticker.upper()}.parquet")

            year_df.drop(columns=["year"]).to_parquet(
                out_path,
                engine=PARQUET_ENGINE,
                compression=PARQUET_COMPRESSION,
                index=False,
            )
            rows_total += len(year_df)

        csv_size  = os.path.getsize(csv_path)
        return rows_total, csv_size

    except Exception as e:
        log.warning(f"  Failed to convert {ticker}: {e}")
        return 0, 0


def run(dataset: str, limit: int = None) -> dict:
    """
    Main ingestion flow.
    limit: process only first N files (for testing)
    """
    log.info("=" * 60)
    log.info("Kaggle Bulk Stock Dataset Ingestion")
    log.info(f"Dataset : {dataset}")
    log.info(f"Output  : {KAGGLE_PARQUET_DIR}/")
    log.info("=" * 60)

    start_time = time.time()

    # Step 1: Download
    zip_path = download_dataset(dataset)
    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    log.info(f"Zip size: {zip_size_mb:.1f} MB")

    # Step 2: Extract
    extract_zip(zip_path)

    # Step 3: Find all CSV files
    csv_files = glob.glob(os.path.join(EXTRACT_DIR, "**", "*.csv"), recursive=True)
    csv_files += glob.glob(os.path.join(EXTRACT_DIR, "**", "*.txt"), recursive=True)

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {EXTRACT_DIR}")

    if limit:
        csv_files = csv_files[:limit]
        log.info(f"Limited to first {limit} files for testing")

    log.info(f"Found {len(csv_files):,} CSV files to convert")

    # Step 4: Convert each CSV to Parquet
    total_rows = 0
    total_csv_bytes = 0
    converted = 0
    skipped = 0

    for i, csv_path in enumerate(csv_files):
        # Extract ticker from filename
        filename = os.path.basename(csv_path)
        ticker = os.path.splitext(filename)[0].upper()

        rows, csv_bytes = convert_csv_to_parquet(csv_path, ticker)

        if rows > 0:
            total_rows += rows
            total_csv_bytes += csv_bytes
            converted += 1
        else:
            skipped += 1

        # Progress log every 500 files
        if (i + 1) % 500 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (len(csv_files) - i - 1) / rate
            log.info(f"  Progress: {i+1:,}/{len(csv_files):,} files "
                     f"| {total_rows:,} rows "
                     f"| ETA: {remaining/60:.1f} min")

    # Step 5: Calculate storage savings
    parquet_bytes = sum(
        os.path.getsize(f)
        for f in glob.glob(os.path.join(KAGGLE_PARQUET_DIR, "**", "*.parquet"), recursive=True)
    )
    savings_pct = (1 - parquet_bytes / max(total_csv_bytes, 1)) * 100

    elapsed = time.time() - start_time
    summary = {
        "converted"       : converted,
        "skipped"         : skipped,
        "total_rows"      : total_rows,
        "csv_size_mb"     : total_csv_bytes / (1024 * 1024),
        "parquet_size_mb" : parquet_bytes / (1024 * 1024),
        "savings_pct"     : savings_pct,
        "elapsed_sec"     : round(elapsed, 1),
    }

    log.info("\n" + "=" * 60)
    log.info("KAGGLE INGESTION COMPLETE")
    log.info(f"Converted  : {converted:,} tickers")
    log.info(f"Skipped    : {skipped:,} (empty/corrupt)")
    log.info(f"Total rows : {total_rows:,}")
    log.info(f"CSV size   : {total_csv_bytes/(1024*1024):.1f} MB")
    log.info(f"Parquet size: {parquet_bytes/(1024*1024):.1f} MB")
    log.info(f"Space saved: {savings_pct:.1f}%")
    log.info(f"Time taken : {elapsed:.1f}s ({elapsed/60:.1f} min)")
    log.info("=" * 60)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Kaggle bulk stock dataset")
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help="Kaggle dataset slug")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N files (for testing)")
    args = parser.parse_args()

    result = run(args.dataset, args.limit)
    sys.exit(0 if result["converted"] > 0 else 1)
