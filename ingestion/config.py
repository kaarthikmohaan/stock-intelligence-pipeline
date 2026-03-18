"""
Pipeline configuration — single source of truth.
All ingestion scripts import from here.
"""

# ── TICKER UNIVERSE ───────────────────────────────
# 50 liquid US stocks across major sectors
TICKERS = [
    # Technology
    "AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN", "TSM", "AVGO", "ORCL", "AMD",
    # Finance
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "SCHW", "AXP", "V", "MA",
    # Healthcare
    "JNJ", "UNH", "LLY", "ABBV", "PFE", "MRK", "TMO", "ABT", "DHR", "BMY",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "KMI", "MPC", "PSX", "VLO", "OXY",
    # Consumer
    "WMT", "HD", "MCD", "NKE", "SBUX", "TGT", "COST", "LOW", "TJX", "DG",
]

# ── DATE RANGE ────────────────────────────────────
HISTORICAL_START = "2019-01-01"   # 5+ years of history
HISTORICAL_END   = "2024-12-31"   # end of last full year

# ── DATA PATHS ────────────────────────────────────
import os
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRONZE_DIR  = os.path.join(BASE_DIR, "data", "bronze")
SILVER_DIR  = os.path.join(BASE_DIR, "data", "silver")
GOLD_DIR    = os.path.join(BASE_DIR, "data", "gold")
LOGS_DIR    = os.path.join(BASE_DIR, "logs")

PRICES_BRONZE   = os.path.join(BRONZE_DIR, "prices")
NEWS_BRONZE     = os.path.join(BRONZE_DIR, "news")
FILINGS_BRONZE  = os.path.join(BRONZE_DIR, "filings")

# ── INGESTION SETTINGS ────────────────────────────
YFINANCE_INTERVAL   = "1d"     # daily bars
TWELVE_DATA_INTERVAL = "1day"  # daily bars from Twelve Data
NEWS_PAGE_SIZE      = 100      # max headlines per API call
MAX_RETRIES         = 3        # retry failed API calls
RETRY_DELAY_SECONDS = 5        # wait between retries

# ── PARQUET SETTINGS ──────────────────────────────
PARQUET_COMPRESSION = "snappy"  # best balance of speed + size
PARQUET_ENGINE      = "pyarrow"

# ── LOGGING ───────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
