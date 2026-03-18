"""
Stock Intelligence Pipeline — Full System Verification
Python: 3.11.9 | Updated: 2026-03-18
Usage: python verify_setup.py
"""

import sys
import subprocess
import os
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
def section(msg): print(f"\n{BOLD}{BLUE}{'─'*50}{RESET}\n{BOLD} {msg}{RESET}\n{'─'*50}")

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
    except ImportError as e:
        fail(f"{label}: NOT INSTALLED — {e}")
        results["failed"] += 1
    except Exception as e:
        fail(f"{label}: ERROR — {e}")
        results["failed"] += 1

def run_cmd(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.stdout.strip() + r.stderr.strip()

def run_cmd_env(cmd):
    """Run command with venv PATH included — fixes CLI tool detection."""
    venv_bin = os.path.join(os.path.dirname(sys.executable))
    env = os.environ.copy()
    env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    return r.stdout.strip() + r.stderr.strip()

print(f"\n{BOLD}{'='*50}")
print(" STOCK INTELLIGENCE PIPELINE — SYSTEM CHECK")
print(f"{'='*50}{RESET}")
print(f" Run at:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f" Python:  {sys.version.split()[0]}")
print(f" Project: {os.path.basename(os.getcwd())}")

# ── 1. PYTHON ─────────────────────────────────────
section("1. Python Environment")

check("Python 3.11.x",
    lambda: sys.version.split()[0]
        if sys.version.split()[0].startswith("3.11")
        else (_ for _ in ()).throw(
            Exception(f"Expected 3.11.x, got {sys.version.split()[0]}")))

check("Virtual environment active",
    lambda: "venv" if "venv" in sys.executable
        else (_ for _ in ()).throw(
            Exception("Not in venv — run: source venv/bin/activate")))

check("pip",
    lambda: run_cmd("pip --version").split()[1])

# ── 2. SYSTEM TOOLS ───────────────────────────────
section("2. System Tools")

check("Java 17",
    lambda: run_cmd("java -version 2>&1").split('"')[1]
        if '"' in run_cmd("java -version 2>&1")
        else (_ for _ in ()).throw(Exception("Java not found")))

check("Docker",
    lambda: run_cmd("docker --version").split()[2].rstrip(","))

check("Docker Compose",
    lambda: run_cmd("docker compose version").split()[-1])

check("Git",
    lambda: run_cmd("git --version").split()[2])

check("Ollama",
    lambda: run_cmd("ollama --version 2>/dev/null | tail -1").split()[-1])

# ── 3. CORE DATA PACKAGES ─────────────────────────
section("3. Core Data Packages")

check("pandas",        lambda: __import__("pandas").__version__)
check("numpy",         lambda: __import__("numpy").__version__)
check("pyarrow",       lambda: __import__("pyarrow").__version__)
check("duckdb",        lambda: __import__("duckdb").__version__)
check("scikit-learn",  lambda: __import__("sklearn").__version__)
check("scipy",         lambda: __import__("scipy").__version__)
check("matplotlib",    lambda: __import__("matplotlib").__version__)
check("seaborn",       lambda: __import__("seaborn").__version__)
check("sdv",           lambda: __import__("sdv").__version__)

check("ta (indicators)",
    lambda: "ta.trend.EMAIndicator available"
        if hasattr(__import__("ta").trend, "EMAIndicator")
        else (_ for _ in ()).throw(Exception("EMAIndicator missing")))

check("faker",
    lambda: f"live name: {__import__('faker').Faker().name()}")

# ── 4. DATA INGESTION ─────────────────────────────
section("4. Data Ingestion Packages")

check("yfinance",       lambda: __import__("yfinance").__version__)
check("newsapi-python",
    lambda: "installed" if __import__("newsapi")
        else (_ for _ in ()).throw(Exception("not found")))
check("requests",       lambda: __import__("requests").__version__)
check("beautifulsoup4", lambda: __import__("bs4").__version__)
check("kaggle",         lambda: __import__("kaggle").__version__)

# ── 5. PYSPARK ────────────────────────────────────
section("5. PySpark")

check("pyspark",  lambda: __import__("pyspark").__version__)
check("py4j",     lambda: __import__("py4j").__version__)

def check_spark_session():
    from pyspark.sql import SparkSession
    spark = SparkSession.builder \
        .master("local[1]") \
        .appName("verify") \
        .config("spark.driver.memory", "1g") \
        .config("spark.sql.shuffle.partitions", "2") \
        .config("spark.ui.showConsoleProgress", "false") \
        .getOrCreate()
    spark.stop()
    return "SparkSession created and stopped OK"

check("SparkSession init", check_spark_session)

# ── 6. DBT ────────────────────────────────────────
section("6. dbt (SQL Transformations)")

def check_dbt_version():
    import importlib.metadata
    return importlib.metadata.version("dbt-core")

def check_dbt_duckdb():
    import importlib.metadata
    return importlib.metadata.version("dbt-duckdb")

def check_dbt_cli():
    out = run_cmd_env("dbt --version 2>&1")
    if "installed" in out.lower() or "core" in out.lower():
        for line in out.splitlines():
            if "installed" in line.lower():
                return line.strip()
        return "dbt CLI working"
    raise Exception(f"Unexpected output: {out[:100]}")

check("dbt-core",   check_dbt_version)
check("dbt-duckdb", check_dbt_duckdb)
check("dbt CLI",    check_dbt_cli)

# ── 7. DATA QUALITY ───────────────────────────────
section("7. Data Quality — pandera")

check("pandera", lambda: __import__("pandera").__version__)

def check_pandera_works():
    import warnings
    warnings.filterwarnings("ignore")
    import pandera.pandas as pa
    import pandas as pd
    schema = pa.DataFrameSchema({
        "price":  pa.Column(float, pa.Check.gt(0)),
        "volume": pa.Column(int,   pa.Check.gt(0)),
    })
    df = pd.DataFrame({"price": [100.5, 200.3], "volume": [1000, 2000]})
    schema.validate(df)
    return "Schema validation passed on sample stock data"

check("pandera schema validation", check_pandera_works)

# ── 8. AI / GEN AI PACKAGES ───────────────────────
section("8. AI / GenAI Packages")

check("torch",              lambda: __import__("torch").__version__)
check("transformers",       lambda: __import__("transformers").__version__)
check("sentence-transformers",
    lambda: __import__("sentence_transformers").__version__)
check("huggingface-hub",    lambda: __import__("huggingface_hub").__version__)
check("langchain",          lambda: __import__("langchain").__version__)
check("langchain-community",
    lambda: __import__("langchain_community").__version__)
check("langchain-ollama",   lambda: __import__("langchain_ollama").__version__)
check("chromadb",           lambda: __import__("chromadb").__version__)
check("mlflow",             lambda: __import__("mlflow").__version__)

# ── 9. SERVING PACKAGES ───────────────────────────
section("9. Serving Packages")

check("fastapi",    lambda: __import__("fastapi").__version__)
check("uvicorn",    lambda: __import__("uvicorn").__version__)
check("streamlit",  lambda: __import__("streamlit").__version__)
check("python-dotenv",
    lambda: "installed" if __import__("dotenv") else "missing")

# ── 10. OLLAMA ────────────────────────────────────
section("10. Ollama — Local LLM")

def check_ollama_model():
    import requests as req
    try:
        r = req.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        if not models:
            raise Exception("No models — run: ollama pull llama3.2")
        return ", ".join(models)
    except req.exceptions.ConnectionError:
        raise Exception("Ollama not running — run: ollama serve &")

check("Ollama service + models", check_ollama_model)

# ── 11. API KEYS ──────────────────────────────────
section("11. API Keys (.env file)")

from dotenv import load_dotenv
load_dotenv()

def check_key(env_var):
    val = os.getenv(env_var, "")
    if val and len(val) > 8:
        return f"{'*' * 8}{val[-4:]}"
    raise Exception(f"{env_var} missing from .env file")

check("TWELVE_DATA_KEY",  lambda: check_key("TWELVE_DATA_KEY"))
check("NEWS_API_KEY",     lambda: check_key("NEWS_API_KEY"))
check("KAGGLE_USERNAME",  lambda: check_key("KAGGLE_USERNAME"))
check("KAGGLE_KEY",       lambda: check_key("KAGGLE_KEY"))

# ── 12. LIVE API TESTS ────────────────────────────
section("12. Live API Connectivity")

def test_yfinance():
    import yfinance as yf
    hist = yf.Ticker("AAPL").history(period="1d")
    if hist.empty:
        raise Exception("No data returned")
    return f"AAPL close: ${hist['Close'].iloc[-1]:.2f}"

def test_twelve_data():
    import requests as req
    key = os.getenv("TWELVE_DATA_KEY", "")
    r = req.get(
        f"https://api.twelvedata.com/price?symbol=AAPL&apikey={key}",
        timeout=10)
    data = r.json()
    if "price" not in data:
        raise Exception(str(data))
    return f"AAPL: ${float(data['price']):.2f}"

def test_newsapi():
    from newsapi import NewsApiClient
    key = os.getenv("NEWS_API_KEY", "")
    api = NewsApiClient(api_key=key)
    top = api.get_top_headlines(category="business", language="en", page_size=1)
    if top["status"] != "ok":
        raise Exception(str(top))
    return f"{top['totalResults']} business headlines available"

def test_kaggle():
    result = run_cmd_env("kaggle datasets list -s 'stock' --csv 2>&1 | head -2")
    if "error" in result.lower() or "401" in result:
        raise Exception(result)
    return "Kaggle API responding"

check("yfinance — live AAPL",  test_yfinance)
check("Twelve Data — live",    test_twelve_data)
check("NewsAPI — headlines",   test_newsapi)
check("Kaggle CLI",            test_kaggle)

# ── 13. DUCKDB SMOKE TEST ─────────────────────────
section("13. DuckDB Smoke Test")

def test_duckdb():
    import duckdb
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE t AS SELECT * FROM range(5) r(id)")
    n = conn.execute("SELECT COUNT(*) FROM t").fetchone()[0]
    conn.close()
    if n != 5:
        raise Exception(f"Expected 5, got {n}")
    return "In-memory query: 5 rows returned"

check("DuckDB in-memory query", test_duckdb)

# ── SUMMARY ───────────────────────────────────────
total = results["passed"] + results["failed"] + results["warned"]
print(f"\n{'='*50}")
print(f"{BOLD} FINAL SUMMARY{RESET}")
print(f"{'='*50}")
print(f"  {GREEN}Passed:  {results['passed']}{RESET}")
print(f"  {RED}Failed:  {results['failed']}{RESET}")
print(f"  {YELLOW}Warned:  {results['warned']}{RESET}")
print(f"  Total:   {total}")

if results["failed"] == 0:
    print(f"\n  {GREEN}{BOLD}ALL CHECKS PASSED — READY FOR PHASE 1 🚀{RESET}\n")
else:
    print(f"\n  {RED}{BOLD}{results['failed']} CHECK(S) FAILED — FIX BEFORE PROCEEDING{RESET}\n")
