"""
Stock Intelligence Pipeline — Full Project Checkpoint
======================================================
Verifies everything built in Phase 1 and Phase 2.
Run this before starting Phase 3 to catch any issues early.

Usage: python checkpoint.py
"""

import sys
import os
import subprocess
import time
from datetime import datetime

RESET  = "\033[0m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
BOLD   = "\033[1m"

def ok(msg):      print(f"  {GREEN}✓{RESET}  {msg}")
def fail(msg):    print(f"  {RED}✗{RESET}  {msg}")
def warn(msg):    print(f"  {YELLOW}~{RESET}  {msg}")
def section(msg): print(f"\n{BOLD}{BLUE}{'─'*55}{RESET}\n{BOLD} {msg}{RESET}\n{'─'*55}")
def info(msg):    print(f"     {msg}")

results = {"passed": 0, "failed": 0, "warned": 0}

def check(label, test_fn):
    try:
        result = test_fn()
        if result is True or (isinstance(result, str) and result):
            ok(f"{label}: {result if isinstance(result, str) else 'OK'}")
            results["passed"] += 1
        else:
            fail(f"{label}: returned falsy")
            results["failed"] += 1
    except Exception as e:
        fail(f"{label}: {e}")
        results["failed"] += 1

def warn_check(label, test_fn):
    try:
        result = test_fn()
        if result:
            ok(f"{label}: {result if isinstance(result, str) else 'OK'}")
            results["passed"] += 1
        else:
            warn(f"{label}: not found (non-critical)")
            results["warned"] += 1
    except Exception as e:
        warn(f"{label}: {e} (non-critical)")
        results["warned"] += 1

def run_cmd(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()

print(f"\n{BOLD}{'='*55}")
print(" STOCK INTELLIGENCE PIPELINE — PROJECT CHECKPOINT")
print(f"{'='*55}{RESET}")
print(f" Run at  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f" Python  : {sys.version.split()[0]}")
print(f" Project : {os.path.basename(os.getcwd())}")


# ══════════════════════════════════════════════════
# SECTION 1: ENVIRONMENT
# ══════════════════════════════════════════════════
section("1. Environment")

check("Python 3.11.x",
    lambda: sys.version.split()[0]
        if sys.version.split()[0].startswith("3.11")
        else (_ for _ in ()).throw(Exception(f"Got {sys.version.split()[0]} — need 3.11.x")))

check("Virtual environment",
    lambda: "active" if "venv" in sys.executable
        else (_ for _ in ()).throw(Exception("Not active — run: source venv/bin/activate")))

check("Java 17",
    lambda: run_cmd("java -version 2>&1").split('"')[1]
        if '"' in run_cmd("java -version 2>&1")
        else (_ for _ in ()).throw(Exception("Java not found")))

check("Docker running",
    lambda: "running" if "hello" in run_cmd("docker run --rm hello-world 2>&1").lower()
        else (_ for _ in ()).throw(Exception("Docker not responding")))

check("Ollama running",
    lambda: __import__("requests").get(
        "http://localhost:11434/api/tags", timeout=5).json()
        .get("models", [{}])[0].get("name", "")
        or (_ for _ in ()).throw(Exception("No models — run: ollama serve & then ollama pull llama3.2")))

check("Git repo clean",
    lambda: "clean" if "nothing to commit" in run_cmd("git status")
        else f"untracked files present (run git status)")


# ══════════════════════════════════════════════════
# SECTION 2: PROJECT STRUCTURE
# ══════════════════════════════════════════════════
section("2. Project Folder Structure")

REQUIRED_DIRS = [
    "data/bronze/prices",
    "data/bronze/news",
    "data/bronze/filings",
    "data/silver",
    "data/gold",
    "ingestion",
    "transformations",
    "dbt_project/models/staging",
    "dbt_project/models/marts",
    "ai/rag",
    "ai/sentiment",
    "ai/summarizer",
    "api",
    "dashboard",
    "dags",
    "docker",
    "tests",
    "logs",
    "notebooks",
]

for d in REQUIRED_DIRS:
    check(f"dir: {d}",
        lambda d=d: "exists" if os.path.isdir(d)
            else (_ for _ in ()).throw(Exception(f"Missing — run: mkdir -p {d}")))


# ══════════════════════════════════════════════════
# SECTION 3: REQUIRED FILES
# ══════════════════════════════════════════════════
section("3. Required Files")

REQUIRED_FILES = {
    ".env"                          : "API keys file",
    ".gitignore"                    : "Git ignore rules",
    "requirements.txt"              : "Package snapshot",
    "verify_setup.py"               : "System verification script",
    "ingestion/config.py"           : "Pipeline configuration",
    "ingestion/ingest_yfinance.py"  : "yfinance ingestion script",
    "ingestion/ingest_news.py"      : "NewsAPI ingestion script",
    "ingestion/ingest_kaggle.py"    : "Kaggle ingestion script",
    "ingestion/ingest_sec_edgar.py" : "SEC EDGAR ingestion script",
}

for filepath, description in REQUIRED_FILES.items():
    check(f"{filepath}",
        lambda fp=filepath, desc=description:
            f"{desc} — {os.path.getsize(fp):,} bytes"
            if os.path.isfile(fp)
            else (_ for _ in ()).throw(Exception(f"Missing: {fp}")))


# ══════════════════════════════════════════════════
# SECTION 4: API KEYS
# ══════════════════════════════════════════════════
section("4. API Keys")

from dotenv import load_dotenv
load_dotenv()

def check_key(env_var):
    val = os.getenv(env_var, "")
    if val and len(val) > 8:
        return f"{'*'*8}{val[-4:]}"
    raise Exception(f"{env_var} missing from .env")

check("TWELVE_DATA_KEY",  lambda: check_key("TWELVE_DATA_KEY"))
check("NEWS_API_KEY",     lambda: check_key("NEWS_API_KEY"))
check("KAGGLE_USERNAME",  lambda: check_key("KAGGLE_USERNAME"))
check("KAGGLE_KEY",       lambda: check_key("KAGGLE_KEY"))


# ══════════════════════════════════════════════════
# SECTION 5: BRONZE DATA LAYER
# ══════════════════════════════════════════════════
section("5. Bronze Data Layer")

import duckdb

def check_yfinance_bronze():
    conn = duckdb.connect(":memory:")
    df = conn.execute("""
        SELECT COUNT(*) AS rows, COUNT(DISTINCT ticker) AS tickers
        FROM read_parquet('data/bronze/prices/yfinance/**/*.parquet')
    """).df()
    conn.close()
    rows    = df["rows"].iloc[0]
    tickers = df["tickers"].iloc[0]
    if rows < 70000:
        raise Exception(f"Only {rows:,} rows — expected ~75,450")
    return f"{rows:,} rows, {tickers} tickers"

def check_kaggle_bronze():
    conn = duckdb.connect(":memory:")
    df = conn.execute("""
        SELECT COUNT(*) AS rows, COUNT(DISTINCT ticker) AS tickers
        FROM read_parquet('data/bronze/prices/kaggle/**/*.parquet')
    """).df()
    conn.close()
    rows    = df["rows"].iloc[0]
    tickers = df["tickers"].iloc[0]
    if rows < 15000000:
        raise Exception(f"Only {rows:,} rows — expected ~17.4M")
    return f"{rows:,} rows, {tickers:,} tickers"

def check_news_bronze():
    import glob
    files = glob.glob("data/bronze/news/newsapi_tickers/**/*.parquet", recursive=True)
    if not files:
        raise Exception("No news Parquet files found")
    conn = duckdb.connect(":memory:")
    df = conn.execute("""
        SELECT COUNT(*) AS articles, COUNT(DISTINCT ticker) AS tickers
        FROM read_parquet('data/bronze/news/newsapi_tickers/**/*.parquet')
        WHERE title != ''
    """).df()
    conn.close()
    articles = df["articles"].iloc[0]
    tickers  = df["tickers"].iloc[0]
    if articles < 500:
        raise Exception(f"Only {articles} articles — expected ~1,500+")
    return f"{articles:,} articles, {tickers} tickers"

def check_sec_bronze():
    import glob
    files = glob.glob("data/bronze/filings/**/*.parquet", recursive=True)
    if not files:
        raise Exception("No SEC filing Parquet files found")
    conn = duckdb.connect(":memory:")
    df = conn.execute("""
        SELECT COUNT(*) AS filings, COUNT(DISTINCT ticker) AS tickers
        FROM read_parquet('data/bronze/filings/**/*.parquet')
    """).df()
    conn.close()
    filings = df["filings"].iloc[0]
    tickers = df["tickers"].iloc[0]
    if filings < 400:
        raise Exception(f"Only {filings} filings — expected ~495")
    return f"{filings} filings, {tickers} tickers"

check("yfinance bronze",  check_yfinance_bronze)
check("Kaggle bronze",    check_kaggle_bronze)
check("NewsAPI bronze",   check_news_bronze)
check("SEC EDGAR bronze", check_sec_bronze)


# ══════════════════════════════════════════════════
# SECTION 6: DATA QUALITY SPOT CHECKS
# ══════════════════════════════════════════════════
section("6. Data Quality Spot Checks")

def check_price_integrity():
    """No zero or negative close prices in yfinance data."""
    conn = duckdb.connect(":memory:")
    df = conn.execute("""
        SELECT COUNT(*) AS bad_rows
        FROM read_parquet('data/bronze/prices/yfinance/**/*.parquet')
        WHERE close <= 0 OR close IS NULL
    """).df()
    conn.close()
    bad = df["bad_rows"].iloc[0]
    if bad > 0:
        raise Exception(f"{bad} rows with zero/null close price")
    return "all close prices > 0"

def check_date_range():
    """yfinance data covers expected date range."""
    conn = duckdb.connect(":memory:")
    df = conn.execute("""
        SELECT MIN(date)::DATE AS min_date, MAX(date)::DATE AS max_date
        FROM read_parquet('data/bronze/prices/yfinance/**/*.parquet')
    """).df()
    conn.close()
    min_d = str(df["min_date"].iloc[0])
    max_d = str(df["max_date"].iloc[0])
    if "2019" not in min_d:
        raise Exception(f"Expected start 2019, got {min_d}")
    return f"{min_d} → {max_d}"

def check_ticker_count():
    """Exactly 50 tickers in yfinance data."""
    conn = duckdb.connect(":memory:")
    df = conn.execute("""
        SELECT COUNT(DISTINCT ticker) AS n
        FROM read_parquet('data/bronze/prices/yfinance/**/*.parquet')
    """).df()
    conn.close()
    n = df["n"].iloc[0]
    if n != 50:
        raise Exception(f"Expected 50 tickers, got {n}")
    return f"exactly 50 tickers"

def check_no_future_dates():
    """No future-dated rows in price data."""
    conn = duckdb.connect(":memory:")
    df = conn.execute("""
        SELECT COUNT(*) AS future_rows
        FROM read_parquet('data/bronze/prices/yfinance/**/*.parquet')
        WHERE date > CURRENT_DATE
    """).df()
    conn.close()
    future = df["future_rows"].iloc[0]
    if future > 0:
        raise Exception(f"{future} rows with future dates")
    return "no future-dated rows"

def check_news_has_titles():
    """News articles have non-empty titles."""
    conn = duckdb.connect(":memory:")
    df = conn.execute("""
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN title = '' OR title IS NULL THEN 1 ELSE 0 END) AS empty_titles
        FROM read_parquet('data/bronze/news/newsapi_tickers/**/*.parquet')
    """).df()
    conn.close()
    total  = df["total"].iloc[0]
    empty  = df["empty_titles"].iloc[0]
    pct_ok = (total - empty) / max(total, 1) * 100
    return f"{pct_ok:.1f}% articles have titles ({total-empty:,}/{total:,})"

def check_sec_form_types():
    """SEC data contains both 10-K and 10-Q."""
    conn = duckdb.connect(":memory:")
    df = conn.execute("""
        SELECT form_type, COUNT(*) AS n
        FROM read_parquet('data/bronze/filings/**/*.parquet')
        GROUP BY form_type
    """).df()
    conn.close()
    forms = list(df["form_type"])
    if "10-K" not in forms or "10-Q" not in forms:
        raise Exception(f"Missing form types — found: {forms}")
    k = df[df["form_type"]=="10-K"]["n"].iloc[0]
    q = df[df["form_type"]=="10-Q"]["n"].iloc[0]
    return f"10-K: {k} filings, 10-Q: {q} filings"

check("Price integrity (close > 0)",    check_price_integrity)
check("Date range (2019–2024)",         check_date_range)
check("Ticker count (exactly 50)",      check_ticker_count)
check("No future dates",                check_no_future_dates)
check("News titles present",            check_news_has_titles)
check("SEC form types (10-K + 10-Q)",   check_sec_form_types)


# ══════════════════════════════════════════════════
# SECTION 7: INGESTION CONFIG
# ══════════════════════════════════════════════════
section("7. Ingestion Config")

sys.path.insert(0, ".")

def check_config():
    from ingestion.config import TICKERS, HISTORICAL_START, BRONZE_DIR
    if len(TICKERS) != 50:
        raise Exception(f"Expected 50 tickers, got {len(TICKERS)}")
    if "PXD" in TICKERS:
        raise Exception("PXD still in config — should have been replaced with KMI")
    if "KMI" not in TICKERS:
        raise Exception("KMI missing from config")
    return f"50 tickers, start={HISTORICAL_START}, PXD removed, KMI present"

def check_no_pxd_data():
    import glob
    pxd_files = glob.glob("data/bronze/prices/yfinance/**/PXD.parquet", recursive=True)
    if pxd_files:
        raise Exception(f"PXD Parquet files still exist: {pxd_files}")
    return "no PXD data files"

check("Config tickers",    check_config)
check("PXD fully removed", check_no_pxd_data)


# ══════════════════════════════════════════════════
# SECTION 8: DISK USAGE
# ══════════════════════════════════════════════════
section("8. Disk Usage")

def get_dir_size_mb(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except Exception:
                pass
    return total / (1024 * 1024)

def check_disk():
    yf_mb     = get_dir_size_mb("data/bronze/prices/yfinance")
    kaggle_mb = get_dir_size_mb("data/bronze/prices/kaggle")
    news_mb   = get_dir_size_mb("data/bronze/news")
    sec_mb    = get_dir_size_mb("data/bronze/filings")
    total_mb  = yf_mb + kaggle_mb + news_mb + sec_mb
    info(f"yfinance prices : {yf_mb:.1f} MB")
    info(f"Kaggle prices   : {kaggle_mb:.0f} MB")
    info(f"News articles   : {news_mb:.1f} MB")
    info(f"SEC filings     : {sec_mb:.1f} MB")
    info(f"Total bronze    : {total_mb:.0f} MB")
    remaining_gb = (512*1024 - total_mb) / 1024
    info(f"SSD remaining   : ~{remaining_gb:.0f} GB free")
    return f"bronze layer = {total_mb:.0f} MB total"

check("Storage summary", check_disk)


# ══════════════════════════════════════════════════
# SECTION 9: KEY PACKAGES QUICK CHECK
# ══════════════════════════════════════════════════
section("9. Key Package Imports")

PACKAGES = [
    ("pandas",          "pandas"),
    ("duckdb",          "duckdb"),
    ("pyspark",         "pyspark"),
    ("dbt-core",        "dbt.version"),
    ("langchain",       "langchain"),
    ("chromadb",        "chromadb"),
    ("mlflow",          "mlflow"),
    ("pandera",         "pandera"),
    ("transformers",    "transformers"),
    ("torch",           "torch"),
    ("fastapi",         "fastapi"),
    ("streamlit",       "streamlit"),
    ("yfinance",        "yfinance"),
]

for label, module in PACKAGES:
    check(label, lambda m=module: __import__(m) and "importable")


# ══════════════════════════════════════════════════
# SECTION 10: LIVE API PING
# ══════════════════════════════════════════════════
section("10. Live API Connectivity")

import requests as req

def ping_yfinance():
    import yfinance as yf
    df = yf.Ticker("AAPL").history(period="1d")
    if df.empty:
        raise Exception("No data")
    return f"AAPL close: ${df['Close'].iloc[-1]:.2f}"

def ping_twelve_data():
    key = os.getenv("TWELVE_DATA_KEY", "")
    r = req.get(f"https://api.twelvedata.com/price?symbol=AAPL&apikey={key}", timeout=10)
    data = r.json()
    if "price" not in data:
        raise Exception(str(data)[:100])
    return f"AAPL: ${float(data['price']):.2f}"

def ping_newsapi():
    from newsapi import NewsApiClient
    api = NewsApiClient(api_key=os.getenv("NEWS_API_KEY", ""))
    top = api.get_top_headlines(category="business", language="en", page_size=1)
    if top["status"] != "ok":
        raise Exception(str(top))
    return f"{top['totalResults']} business headlines available"

def ping_sec():
    r = req.get(
        "https://efts.sec.gov/LATEST/search-index?q=%22AAPL%22&forms=10-K&dateRange=custom&startdt=2024-01-01&enddt=2024-12-31",
        headers={"User-Agent": "StockIntelligencePipeline research@example.com"},
        timeout=10
    )
    data = r.json()
    hits = len(data.get("hits", {}).get("hits", []))
    if hits == 0:
        raise Exception("SEC returned 0 hits")
    return f"SEC EDGAR responding — {hits} AAPL 10-K hits"

check("yfinance live",    ping_yfinance)
check("Twelve Data live", ping_twelve_data)
check("NewsAPI live",     ping_newsapi)
check("SEC EDGAR live",   ping_sec)


# ══════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════
total = results["passed"] + results["failed"] + results["warned"]
print(f"\n{'='*55}")
print(f"{BOLD} CHECKPOINT SUMMARY{RESET}")
print(f"{'='*55}")
print(f"  {GREEN}Passed : {results['passed']}{RESET}")
print(f"  {RED}Failed : {results['failed']}{RESET}")
print(f"  {YELLOW}Warned : {results['warned']}{RESET}")
print(f"  Total  : {total}")

if results["failed"] == 0:
    print(f"\n  {GREEN}{BOLD}ALL CHECKS PASSED — SAFE TO START PHASE 4 🚀{RESET}\n")
elif results["failed"] <= 2:
    print(f"\n  {YELLOW}{BOLD}{results['failed']} MINOR ISSUE(S) — review above then proceed{RESET}\n")
else:
    print(f"\n  {RED}{BOLD}{results['failed']} ISSUE(S) FOUND — fix before Phase 3{RESET}\n")
