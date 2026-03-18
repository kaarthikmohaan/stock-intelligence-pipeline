"""
Ingestion Script 2 — NewsAPI Financial Headlines (per-ticker)
-------------------------------------------------------------
Pulls news headlines for each ticker individually.
This ensures every article is tagged to a specific stock
so FinBERT can score sentiment per ticker later.

Usage:
    python ingestion/ingest_news.py
    python ingestion/ingest_news.py --tickers AAPL MSFT NVDA
"""

import os
import sys
import logging
import argparse
import time
from datetime import datetime, timedelta

import pandas as pd
from newsapi import NewsApiClient
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingestion.config import (
    TICKERS, NEWS_BRONZE, NEWS_PAGE_SIZE,
    PARQUET_COMPRESSION, PARQUET_ENGINE,
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
            os.path.join(LOGS_DIR, f"news_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        ),
    ],
)
log = logging.getLogger("ingest_news")


def fetch_ticker_news(api: NewsApiClient, ticker: str, from_date: str) -> list[dict]:
    """
    Search NewsAPI for articles mentioning this ticker.
    Returns a list of article dicts tagged with the ticker.
    """
    results = []
    try:
        response = api.get_everything(
            q=ticker,
            language="en",
            sort_by="publishedAt",
            page_size=NEWS_PAGE_SIZE,
            from_param=from_date,
        )
        if response["status"] == "ok":
            for article in response.get("articles", []):
                results.append({
                    "ticker"        : ticker,
                    "title"         : article.get("title", "") or "",
                    "description"   : article.get("description", "") or "",
                    "url"           : article.get("url", "") or "",
                    "source"        : article.get("source", {}).get("name", "") or "",
                    "published_at"  : article.get("publishedAt", "") or "",
                    "content"       : article.get("content", "") or "",
                })
    except Exception as e:
        log.warning(f"  API error for {ticker}: {e}")

    return results


def save_ticker_news(records: list[dict], ticker: str, run_date: str) -> str:
    """Save news for one ticker as a Parquet file partitioned by date."""
    partition_dir = os.path.join(NEWS_BRONZE, "newsapi_tickers", f"date={run_date}")
    os.makedirs(partition_dir, exist_ok=True)
    path = os.path.join(partition_dir, f"{ticker}.parquet")

    df = pd.DataFrame(records) if records else pd.DataFrame(columns=[
        "ticker", "title", "description", "url",
        "source", "published_at", "content"
    ])
    df["ingested_at"] = datetime.utcnow().isoformat()
    df["ingestion_date"] = run_date

    df.to_parquet(path, engine=PARQUET_ENGINE,
                  compression=PARQUET_COMPRESSION, index=False)
    size_kb = os.path.getsize(path) / 1024
    log.info(f"  Saved: {ticker}.parquet ({len(df)} articles, {size_kb:.1f} KB)")
    return path


def run(tickers: list[str]) -> dict:
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        raise ValueError("NEWS_API_KEY not set in .env file")

    api      = NewsApiClient(api_key=api_key)
    run_date = datetime.utcnow().strftime("%Y-%m-%d")
    from_date = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")

    log.info("=" * 60)
    log.info("NewsAPI Per-Ticker Ingestion")
    log.info(f"Tickers  : {len(tickers)}")
    log.info(f"Date range: {from_date} → {run_date}")
    log.info(f"Output   : {NEWS_BRONZE}/newsapi_tickers/")
    log.info("=" * 60)

    succeeded, failed = [], []
    total_articles = 0
    start_time = time.time()

    for i, ticker in enumerate(tickers, 1):
        log.info(f"\n[{i:02d}/{len(tickers)}] {ticker}")
        try:
            records = fetch_ticker_news(api, ticker, from_date)
            save_ticker_news(records, ticker, run_date)
            total_articles += len(records)
            succeeded.append(ticker)
            log.info(f"  {ticker}: {len(records)} articles")
        except Exception as e:
            log.error(f"  FAILED {ticker}: {e}")
            failed.append(ticker)

        time.sleep(0.8)   # ~75 requests/min — safely within free tier

    elapsed = time.time() - start_time
    summary = {
        "succeeded"     : succeeded,
        "failed"        : failed,
        "total_articles": total_articles,
        "elapsed_sec"   : round(elapsed, 1),
    }

    log.info("\n" + "=" * 60)
    log.info("NEWS INGESTION COMPLETE")
    log.info(f"Succeeded    : {len(succeeded)}/{len(tickers)}")
    log.info(f"Failed       : {failed if failed else 'none'}")
    log.info(f"Total articles: {total_articles:,}")
    log.info(f"Time taken   : {elapsed:.1f}s")
    log.info("=" * 60)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest NewsAPI headlines per ticker")
    parser.add_argument("--tickers", nargs="+", default=TICKERS,
                        help="Tickers to fetch (default: all 50)")
    args = parser.parse_args()

    result = run(args.tickers)
    sys.exit(0 if not result["failed"] else 1)
