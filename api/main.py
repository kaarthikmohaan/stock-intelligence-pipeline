"""
Phase 6 — FastAPI Serving Layer
---------------------------------
REST API exposing pipeline data and AI capabilities.

Endpoints:
    GET  /health                 — API health check
    GET  /sentiment/{ticker}     — FinBERT sentiment for a ticker
    GET  /indicators/{ticker}    — Technical indicators for a ticker
    GET  /top-movers             — Daily top gainers and losers
    POST /ask                    — RAG Q&A via Llama 3.2

Usage:
    uvicorn api.main:app --reload --port 8000
    curl http://localhost:8000/health
    curl http://localhost:8000/sentiment/NVDA
    curl http://localhost:8000/indicators/AAPL
    curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"question":"What is the sentiment around NVIDIA?"}'
"""

import os
import sys
import warnings
from datetime import datetime
from typing import Optional

warnings.filterwarnings("ignore")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import duckdb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingestion.config import GOLD_DIR, SILVER_DIR

app = FastAPI(
    title="Stock Intelligence Pipeline API",
    description="REST API for stock sentiment, technical indicators, and AI Q&A",
    version="1.0.0",
)

# Allow all origins for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Paths ──────────────────────────────────────────────
SENTIMENT_PATH   = os.path.join(GOLD_DIR, "sentiment", "news_sentiment.parquet")
INDICATORS_PATH  = os.path.join(GOLD_DIR, "indicators", "yfinance", "data_source=yfinance", "*.parquet")
SILVER_YFINANCE  = os.path.join(SILVER_DIR, "prices", "yfinance", "**", "*.parquet")


# ── Request/Response Models ────────────────────────────
class AskRequest(BaseModel):
    question: str
    n_results: Optional[int] = 5


class AskResponse(BaseModel):
    question: str
    answer: str
    sources_used: int
    elapsed_ms: float


# ── Routes ─────────────────────────────────────────────

@app.get("/health")
def health():
    """API health check — confirms all data sources are accessible."""
    checks = {}

    try:
        conn = duckdb.connect(":memory:")
        n = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{SENTIMENT_PATH}')").fetchone()[0]
        checks["sentiment_rows"] = n
        conn.close()
    except Exception as e:
        checks["sentiment_error"] = str(e)

    try:
        conn = duckdb.connect(":memory:")
        n = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{INDICATORS_PATH}')").fetchone()[0]
        checks["indicator_rows"] = n
        conn.close()
    except Exception as e:
        checks["indicators_error"] = str(e)

    return {
        "status"    : "healthy",
        "timestamp" : datetime.utcnow().isoformat(),
        "checks"    : checks,
        "version"   : "1.0.0",
    }


@app.get("/sentiment/{ticker}")
def get_sentiment(ticker: str, limit: int = 10):
    """
    Get recent news sentiment for a ticker.
    Returns headlines with bullish/bearish/neutral classification.
    """
    ticker = ticker.upper()
    try:
        conn = duckdb.connect(":memory:")
        df = conn.execute(f"""
            SELECT
                ticker,
                title,
                sentiment,
                ROUND(sentiment_score, 4) AS confidence,
                source,
                published_at
            FROM read_parquet('{SENTIMENT_PATH}')
            WHERE ticker = '{ticker}'
            ORDER BY sentiment_score DESC
            LIMIT {limit}
        """).df()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if df.empty:
        raise HTTPException(status_code=404,
                           detail=f"No sentiment data found for {ticker}")

    # Summary stats
    conn2 = duckdb.connect(":memory:")
    summary = conn2.execute(f"""
        SELECT
            COUNT(*)                                             AS total_articles,
            SUM(CASE WHEN sentiment='positive' THEN 1 ELSE 0 END) AS bullish,
            SUM(CASE WHEN sentiment='negative' THEN 1 ELSE 0 END) AS bearish,
            SUM(CASE WHEN sentiment='neutral'  THEN 1 ELSE 0 END) AS neutral,
            ROUND(AVG(sentiment_score), 4)                       AS avg_confidence
        FROM read_parquet('{SENTIMENT_PATH}')
        WHERE ticker = '{ticker}'
    """).df()
    conn2.close()

    return {
        "ticker"  : ticker,
        "summary" : summary.to_dict(orient="records")[0],
        "articles": df.to_dict(orient="records"),
    }


@app.get("/indicators/{ticker}")
def get_indicators(ticker: str):
    """
    Get latest technical indicators for a ticker.
    Returns RSI, MACD, Bollinger Bands, VWAP, MA signals.
    """
    ticker = ticker.upper()
    try:
        conn = duckdb.connect(":memory:")
        df = conn.execute(f"""
            SELECT
                date::DATE              AS date,
                ticker,
                ROUND(close, 2)         AS close,
                ROUND(rsi_14, 2)        AS rsi_14,
                ROUND(macd_line, 4)     AS macd_line,
                ROUND(macd_signal, 4)   AS macd_signal,
                ROUND(macd_histogram,4) AS macd_histogram,
                ROUND(bb_upper, 2)      AS bb_upper,
                ROUND(bb_middle, 2)     AS bb_middle,
                ROUND(bb_lower, 2)      AS bb_lower,
                ROUND(bb_bandwidth, 4)  AS bb_bandwidth,
                ROUND(vwap, 2)          AS vwap,
                vwap_signal,
                ROUND(sma_50, 2)        AS sma_50,
                ROUND(sma_200, 2)       AS sma_200,
                ma_signal
            FROM read_parquet('{INDICATORS_PATH}')
            WHERE ticker = '{ticker}'
            ORDER BY date DESC
            LIMIT 5
        """).df()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if df.empty:
        raise HTTPException(status_code=404,
                           detail=f"No indicator data found for {ticker}")

    latest = df.iloc[0].to_dict()

    # Interpretation
    rsi  = latest.get("rsi_14", 50)
    macd = latest.get("macd_line", 0)
    interpretation = {
        "rsi_signal" : "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral",
        "macd_signal": "bullish" if macd > 0 else "bearish",
        "ma_signal"  : latest.get("ma_signal", "neutral"),
        "vwap_signal": latest.get("vwap_signal", "neutral"),
    }

    return {
        "ticker"        : ticker,
        "latest"        : latest,
        "interpretation": interpretation,
        "history"       : df.to_dict(orient="records"),
    }


@app.get("/top-movers")
def get_top_movers():
    """Get the top daily gainers and losers from the dbt gold layer."""
    try:
        conn = duckdb.connect("/tmp/stock_intelligence.duckdb")
        df   = conn.execute("""
            SELECT ticker, CAST(price_date AS VARCHAR) AS price_date,
                   close_price, daily_return_pct, mover_type,
                   gainer_rank, loser_rank
            FROM mart_top_movers
            ORDER BY daily_return_pct DESC
        """).df()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    gainers = df[df["mover_type"] == "top_gainer"].to_dict(orient="records")
    losers  = df[df["mover_type"] == "top_loser"].to_dict(orient="records")

    return {
        "date"    : str(df["price_date"].iloc[0]) if not df.empty else None,
        "gainers" : gainers,
        "losers"  : losers,
    }


@app.get("/performance")
def get_performance(limit: int = 10):
    """Get top performing tickers by average daily return from dbt mart."""
    try:
        conn = duckdb.connect("/tmp/stock_intelligence.duckdb")
        df   = conn.execute(f"""
            SELECT ticker, trading_days, avg_daily_return_pct,
                   volatility_pct, best_day_pct, worst_day_pct,
                   return_risk_ratio, avg_close
            FROM mart_daily_returns
            ORDER BY avg_daily_return_pct DESC
            LIMIT {limit}
        """).df()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "top_performers": df.to_dict(orient="records"),
        "period"        : "2019-2024",
    }


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    """
    RAG Q&A endpoint — answer questions using your pipeline data.
    Retrieves relevant documents from ChromaDB and generates
    an answer using Llama 3.2 running locally.
    """
    import time
    t0 = time.time()

    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from ai.rag.rag_pipeline import query_rag
        answer = query_rag(request.question, request.n_results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG error: {str(e)}")

    elapsed_ms = (time.time() - t0) * 1000

    return AskResponse(
        question    = request.question,
        answer      = answer,
        sources_used= request.n_results * 2,
        elapsed_ms  = round(elapsed_ms, 1),
    )
