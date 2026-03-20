# Stock Market Intelligence Pipeline

A production-grade GenAI-enabled data engineering portfolio project built on a local development stack.

## Architecture
```
Data Sources → Bronze Layer → Silver Layer → Gold Layer → AI/ML → Serving
```

**Medallion architecture** with Parquet-based data lake, PySpark transformations, dbt SQL models, LangChain RAG, and a Streamlit dashboard.

## Stack

| Layer | Technology |
|---|---|
| Ingestion | yfinance, NewsAPI, Kaggle, SEC EDGAR |
| Storage | Apache Parquet + Hive partitioning |
| Query Engine | DuckDB 1.5.0 |
| Transforms | PySpark 4.1.1, dbt-core 1.11.7 + dbt-duckdb |
| Orchestration | Apache Airflow 2.9.1 (Docker) |
| Sentiment AI | FinBERT (ProsusAI/finbert) on Apple Metal GPU |
| RAG | LangChain + ChromaDB + Llama 3.2 (Ollama) |
| Experiment Tracking | MLflow 3.10.1 |
| Data Quality | pandera 0.30.1 |
| API | FastAPI 0.135.1 |
| Dashboard | Streamlit 1.55.0 + Plotly |
| Language | Python 3.11.9 |

## Data Scale

| Source | Rows | Details |
|---|---|---|
| yfinance | 75,450 | 50 tickers, 2019–2024 daily OHLCV |
| Kaggle | 17.4M | 8,507 tickers, 1962–2017 |
| NewsAPI | 1,528 articles | 49 tickers, FinBERT scored |
| SEC EDGAR | 495 filings | 50 tickers, 10-K + 10-Q, 2022–2024 |

## What's Built

### Phase 1 — Environment
Python 3.11.9, Java 17, Docker, Ollama + Llama 3.2, all packages verified (55/55 checks passing)

### Phase 2 — Data Ingestion (Bronze Layer)
Four ingestion scripts pulling from yfinance, NewsAPI, Kaggle, and SEC EDGAR. All data stored as compressed Parquet with Hive-style year partitioning.

### Phase 3 — Silver Layer
pandera schema validation, data cleaning (null/zero/dupe removal, OHLC integrity checks), feature enrichment: `daily_return`, `price_range`, `typical_price`, `range_pct`.

### Phase 4 — Transforms
- **PySpark**: RSI(14), MACD(12,26,9), Bollinger Bands(20,2), VWAP(20), SMA(50,200) on 17.4M rows
- **dbt**: 3 SQL models (staging + 2 marts), 6 data tests
- **Airflow DAG**: Full pipeline orchestration, 6pm Mon–Fri schedule

### Phase 5 — AI/GenAI Layer
- **FinBERT**: 1,528 headlines scored at 263/sec on Apple Metal GPU
- **LangChain RAG**: ChromaDB vector store (495 SEC filings + 1,000 news docs), Q&A via Llama 3.2
- **MLflow**: Experiment tracking with parameters, metrics, and artifacts

### Phase 6 — Serving
- **FastAPI**: 5 endpoints — `/health`, `/sentiment/{ticker}`, `/indicators/{ticker}`, `/top-movers`, `/performance`, `/ask` (RAG)
- **Streamlit**: 4-page dashboard — Overview, Technical Analysis, Sentiment, AI Chat

## Quick Start
```bash
# Clone and setup
git clone https://github.com/kaarthikmohaan/stock-intelligence-pipeline.git
cd stock-intelligence-pipeline
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Add API keys to .env
cp .env.example .env  # edit with your keys

# Run ingestion
python ingestion/ingest_yfinance.py
python ingestion/ingest_news.py
python ingestion/ingest_sec_edgar.py

# Clean silver layer
python transformations/clean_prices.py --source yfinance

# Compute indicators
python transformations/spark_jobs/compute_indicators.py --source yfinance

# Run dbt models
cd dbt_project && export PIPELINE_BASE_DIR=.. && dbt run && cd ..

# Score sentiment
PYTHONPATH=. python ai/sentiment/finbert_scorer.py

# Build RAG vector store
PYTHONPATH=. python ai/rag/rag_pipeline.py --build

# Start API
PYTHONPATH=. python -m uvicorn api.main:app --port 8000

# Start dashboard
PYTHONPATH=. streamlit run dashboard/app.py --server.port 8501
```

## API Endpoints
```bash
GET  /health                    # Pipeline health check
GET  /sentiment/{ticker}        # FinBERT sentiment scores
GET  /indicators/{ticker}       # RSI, MACD, Bollinger Bands, VWAP
GET  /top-movers                # Daily gainers and losers
GET  /performance               # Best performing tickers 2019-2024
POST /ask  {"question": "..."}  # RAG Q&A via Llama 3.2
```

## Key Results

- NVDA best performer 2019–2024: +0.30% avg daily return (~75% annually)
- FinBERT sentiment: 70.8% neutral, 14.9% bullish, 14.3% bearish
- DuckDB scans 17M rows in ~5 seconds
- FinBERT inference: 263 headlines/second on Apple Metal GPU
- Full pipeline runtime: ~15 minutes end-to-end

## Author

**Karthik Mohan** — [github.com/kaarthikmohaan](https://github.com/kaarthikmohaan)
