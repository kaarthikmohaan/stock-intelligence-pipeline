"""
Ingestion Script 4 — SEC EDGAR Filings
---------------------------------------
Downloads 10-K and 10-Q filings for our 50 tickers
from SEC EDGAR (completely free, no API key needed).

The filing text feeds the LangChain RAG system later —
users will be able to ask questions like:
  "What risks did Apple mention in their 2023 annual report?"
  "What did NVDA say about AI revenue in Q3 2024?"

Usage:
    python ingestion/ingest_sec_edgar.py
    python ingestion/ingest_sec_edgar.py --years 2023 2024
    python ingestion/ingest_sec_edgar.py --tickers AAPL MSFT --years 2024
"""

import os
import sys
import logging
import argparse
import time
import re
import requests
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingestion.config import (
    TICKERS, FILINGS_BRONZE,
    PARQUET_COMPRESSION, PARQUET_ENGINE,
    LOGS_DIR, LOG_LEVEL, LOG_FORMAT,
    MAX_RETRIES, RETRY_DELAY_SECONDS,
)

load_dotenv()

os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(LOGS_DIR, f"sec_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        ),
    ],
)
log = logging.getLogger("ingest_sec")

# SEC requires a descriptive User-Agent or requests get blocked
SEC_HEADERS = {
    "User-Agent": "StockIntelligencePipeline research@example.com",
    "Accept-Encoding": "gzip, deflate",
}
SEC_BASE = "https://efts.sec.gov/LATEST/search-index"
SEC_SEARCH = "https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt={start}&enddt={end}&forms={form}"

# Map ticker → CIK using SEC company search
SEC_COMPANY_URL = "https://www.sec.gov/cgi-bin/browse-edgar?company={ticker}&CIK=&type={form}&dateb=&owner=include&count=10&search_text=&action=getcompany&output=atom"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_FILING_URL = "https://www.sec.gov/Archives/edgar/{path}"


def get_cik_for_ticker(ticker: str) -> str | None:
    """
    Look up the SEC CIK number for a ticker symbol.
    CIK is the unique company identifier in the EDGAR system.
    """
    try:
        url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&forms=10-K&dateRange=custom&startdt=2020-01-01&enddt=2024-12-31"
        r = requests.get(url, headers=SEC_HEADERS, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        hits = data.get("hits", {}).get("hits", [])
        if hits:
            return hits[0].get("_source", {}).get("entity_id")
        return None
    except Exception:
        return None


def search_filings(ticker: str, form_type: str, start_year: int, end_year: int) -> list[dict]:
    """
    Search SEC EDGAR full-text search for filings by ticker and form type.
    Returns list of filing metadata dicts.
    """
    filings = []
    start_dt = f"{start_year}-01-01"
    end_dt   = f"{end_year}-12-31"

    url = (
        f"https://efts.sec.gov/LATEST/search-index?"
        f"q=%22{ticker}%22"
        f"&forms={form_type}"
        f"&dateRange=custom"
        f"&startdt={start_dt}"
        f"&enddt={end_dt}"
        f"&hits.hits._source=period_of_report,entity_name,file_date,form_type,accession_no"
    )

    try:
        r = requests.get(url, headers=SEC_HEADERS, timeout=15)
        if r.status_code != 200:
            return filings
        data = r.json()
        hits = data.get("hits", {}).get("hits", [])
        for hit in hits[:5]:  # limit to 5 most recent per form type
            src = hit.get("_source", {})
            filings.append({
                "ticker"         : ticker,
                "form_type"      : src.get("form_type", form_type),
                "entity_name"    : src.get("entity_name", ""),
                "period_of_report": src.get("period_of_report", ""),
                "file_date"      : src.get("file_date", ""),
                "accession_no"   : src.get("accession_no", ""),
            })
    except Exception as e:
        log.warning(f"  Search failed for {ticker} {form_type}: {e}")

    return filings


def fetch_filing_summary(accession_no: str) -> str:
    """
    Fetch a short summary/excerpt of the filing using EDGAR full-text search.
    We get the filing index page and extract key text sections.
    """
    if not accession_no:
        return ""
    try:
        # Format: 0001234567-23-001234 → edgar/data/1234567/000123456723001234/
        clean = accession_no.replace("-", "")
        cik = accession_no.split("-")[0].lstrip("0")
        url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=10-K&dateb=&owner=include&count=5"
        r = requests.get(url, headers=SEC_HEADERS, timeout=10)
        return r.text[:500] if r.status_code == 200 else ""
    except Exception:
        return ""


def run(tickers: list[str], years: list[int]) -> dict:
    start_year = min(years)
    end_year   = max(years)

    log.info("=" * 60)
    log.info("SEC EDGAR Filings Ingestion")
    log.info(f"Tickers    : {len(tickers)}")
    log.info(f"Form types : 10-K (annual), 10-Q (quarterly)")
    log.info(f"Years      : {start_year} → {end_year}")
    log.info(f"Output     : {FILINGS_BRONZE}/")
    log.info("=" * 60)

    all_records = []
    start_time  = time.time()
    succeeded, failed = [], []

    for i, ticker in enumerate(tickers, 1):
        log.info(f"\n[{i:02d}/{len(tickers)}] {ticker}")
        ticker_records = []

        for form_type in ["10-K", "10-Q"]:
            filings = search_filings(ticker, form_type, start_year, end_year)
            log.info(f"  {form_type}: {len(filings)} filings found")

            for filing in filings:
                filing["source"]      = "sec_edgar"
                filing["ingested_at"] = datetime.utcnow().isoformat()
                ticker_records.append(filing)

        if ticker_records:
            all_records.extend(ticker_records)
            succeeded.append(ticker)
        else:
            log.info(f"  No filings found for {ticker}")
            failed.append(ticker)

        time.sleep(0.3)   # polite to SEC servers

    # Save all records as one Parquet file per ingestion run
    if all_records:
        df = pd.DataFrame(all_records)
        run_date = datetime.utcnow().strftime("%Y-%m-%d")
        out_dir  = os.path.join(FILINGS_BRONZE, "sec_edgar", f"date={run_date}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "filing_index.parquet")

        df.to_parquet(
            out_path,
            engine=PARQUET_ENGINE,
            compression=PARQUET_COMPRESSION,
            index=False,
        )
        size_kb = os.path.getsize(out_path) / 1024
        log.info(f"\nSaved filing index: {out_path} ({size_kb:.1f} KB, {len(df)} records)")

    elapsed = time.time() - start_time
    summary = {
        "succeeded"    : succeeded,
        "failed"       : failed,
        "total_filings": len(all_records),
        "elapsed_sec"  : round(elapsed, 1),
    }

    log.info("\n" + "=" * 60)
    log.info("SEC EDGAR INGESTION COMPLETE")
    log.info(f"Succeeded    : {len(succeeded)}/{len(tickers)} tickers")
    log.info(f"Failed       : {len(failed)} tickers")
    log.info(f"Total filings: {len(all_records):,}")
    log.info(f"Time taken   : {elapsed:.1f}s")
    log.info("=" * 60)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest SEC EDGAR filings index")
    parser.add_argument("--tickers", nargs="+", default=TICKERS)
    parser.add_argument("--years", nargs="+", type=int, default=[2022, 2023, 2024],
                        help="Years to pull filings for")
    args = parser.parse_args()

    result = run(args.tickers, args.years)
    sys.exit(0 if result["total_filings"] > 0 else 1)
