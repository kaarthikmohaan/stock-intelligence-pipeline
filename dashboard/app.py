"""
Phase 6 — Streamlit Dashboard
--------------------------------
Interactive web dashboard for the Stock Intelligence Pipeline.

Features:
  - Live technical indicators with charts
  - FinBERT sentiment analysis per ticker
  - Top daily movers
  - AI chat powered by RAG + Llama 3.2

Usage:
    streamlit run dashboard/app.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import duckdb
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingestion.config import GOLD_DIR, SILVER_DIR

# ── Page config ────────────────────────────────────────
st.set_page_config(
    page_title="Stock Intelligence Pipeline",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE         = "http://localhost:8000"
INDICATORS_PATH  = os.path.join(GOLD_DIR, "indicators", "yfinance", "data_source=yfinance", "*.parquet")
SENTIMENT_PATH   = os.path.join(GOLD_DIR, "sentiment", "news_sentiment.parquet")
SILVER_PATH      = os.path.join(SILVER_DIR, "prices", "yfinance", "**", "*.parquet")

TICKERS = [
    "AAPL","MSFT","GOOGL","NVDA","META","AMZN","TSM","AVGO","ORCL","AMD",
    "JPM","BAC","WFC","GS","MS","BLK","SCHW","AXP","V","MA",
    "JNJ","UNH","LLY","ABBV","PFE","MRK","TMO","ABT","DHR","BMY",
    "XOM","CVX","COP","SLB","EOG","KMI","MPC","PSX","VLO","OXY",
    "WMT","HD","MCD","NKE","SBUX","TGT","COST","LOW","TJX","DG",
]


@st.cache_data(ttl=300)
def load_indicators(ticker: str) -> pd.DataFrame:
    conn = duckdb.connect(":memory:")
    df = conn.execute(f"""
        SELECT date::DATE AS date, ticker,
               ROUND(close,2) AS close, ROUND(rsi_14,2) AS rsi_14,
               ROUND(macd_line,4) AS macd_line, ROUND(macd_signal,4) AS macd_signal,
               ROUND(bb_upper,2) AS bb_upper, ROUND(bb_middle,2) AS bb_middle,
               ROUND(bb_lower,2) AS bb_lower, ROUND(vwap,2) AS vwap,
               ROUND(sma_50,2) AS sma_50, ROUND(sma_200,2) AS sma_200,
               ma_signal, vwap_signal
        FROM read_parquet('{INDICATORS_PATH}')
        WHERE ticker = '{ticker}'
        ORDER BY date DESC
        LIMIT 252
    """).df()
    conn.close()
    return df.sort_values("date")


@st.cache_data(ttl=300)
def load_sentiment(ticker: str) -> pd.DataFrame:
    conn = duckdb.connect(":memory:")
    df = conn.execute(f"""
        SELECT ticker, title, sentiment,
               ROUND(sentiment_score,4) AS confidence,
               source, published_at
        FROM read_parquet('{SENTIMENT_PATH}')
        WHERE ticker = '{ticker}'
        ORDER BY sentiment_score DESC
    """).df()
    conn.close()
    return df


@st.cache_data(ttl=300)
def load_top_movers() -> dict:
    try:
        r = requests.get(f"{API_BASE}/top-movers", timeout=5)
        return r.json()
    except Exception:
        return {"gainers": [], "losers": []}


@st.cache_data(ttl=300)
def load_performance() -> pd.DataFrame:
    try:
        r = requests.get(f"{API_BASE}/performance?limit=10", timeout=5)
        return pd.DataFrame(r.json()["top_performers"])
    except Exception:
        return pd.DataFrame()


# ── Sidebar ────────────────────────────────────────────
st.sidebar.title("📈 Stock Intelligence")
st.sidebar.markdown("**GenAI-Powered Pipeline**")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📊 Technical Analysis", "📰 Sentiment", "🤖 AI Chat"],
)

selected_ticker = st.sidebar.selectbox("Select Ticker", TICKERS, index=0)
st.sidebar.divider()
st.sidebar.caption(f"Data: yfinance 2019–2024")
st.sidebar.caption(f"AI: Llama 3.2 + FinBERT")
st.sidebar.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d')}")


# ══════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("📈 Stock Intelligence Pipeline")
    st.markdown("**Real-time market intelligence powered by GenAI**")

    # Pipeline stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Price Records", "17.5M+", "yfinance + Kaggle")
    col2.metric("Tickers Tracked", "8,507", "1962–2024")
    col3.metric("News Articles", "1,528", "FinBERT scored")
    col4.metric("SEC Filings", "495", "10-K + 10-Q")

    st.divider()

    # Top movers
    col_l, col_r = st.columns(2)
    movers = load_top_movers()

    with col_l:
        st.subheader("🚀 Top Gainers")
        gainers = movers.get("gainers", [])
        if gainers:
            df_g = pd.DataFrame(gainers)[["ticker", "close_price", "daily_return_pct"]]
            df_g.columns = ["Ticker", "Price ($)", "Return (%)"]
            st.dataframe(df_g, hide_index=True, use_container_width=True)

    with col_r:
        st.subheader("📉 Top Losers")
        losers = movers.get("losers", [])
        if losers:
            df_l = pd.DataFrame(losers)[["ticker", "close_price", "daily_return_pct"]]
            df_l.columns = ["Ticker", "Price ($)", "Return (%)"]
            st.dataframe(df_l, hide_index=True, use_container_width=True)

    st.divider()

    # Performance table
    st.subheader("🏆 Best Performers (2019–2024)")
    perf_df = load_performance()
    if not perf_df.empty:
        perf_df.columns = ["Ticker", "Days", "Avg Return %",
                           "Volatility %", "Best Day %", "Worst Day %",
                           "Risk Ratio", "Avg Price"]
        st.dataframe(perf_df, hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════
# PAGE 2 — TECHNICAL ANALYSIS
# ══════════════════════════════════════════════════════
elif page == "📊 Technical Analysis":
    st.title(f"📊 Technical Analysis — {selected_ticker}")

    df = load_indicators(selected_ticker)

    if df.empty:
        st.error(f"No indicator data for {selected_ticker}")
        st.stop()

    latest = df.iloc[-1]

    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Close", f"${latest['close']:.2f}")
    col2.metric("RSI (14)", f"{latest['rsi_14']:.1f}",
                "Overbought" if latest['rsi_14'] > 70
                else "Oversold" if latest['rsi_14'] < 30 else "Neutral")
    col3.metric("MACD", f"{latest['macd_line']:.3f}",
                "Bullish" if latest['macd_line'] > 0 else "Bearish")
    col4.metric("MA Signal", latest['ma_signal'].upper())
    col5.metric("vs VWAP", latest['vwap_signal'].upper())

    st.divider()

    # Price + Bollinger Bands chart
    st.subheader("Price & Bollinger Bands")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["close"],
                             name="Close", line=dict(color="#2196F3", width=2)))
    fig.add_trace(go.Scatter(x=df["date"], y=df["bb_upper"],
                             name="BB Upper", line=dict(color="#FF9800", dash="dash")))
    fig.add_trace(go.Scatter(x=df["date"], y=df["bb_middle"],
                             name="BB Middle", line=dict(color="#9E9E9E", dash="dot")))
    fig.add_trace(go.Scatter(x=df["date"], y=df["bb_lower"],
                             name="BB Lower", line=dict(color="#FF9800", dash="dash"),
                             fill="tonexty", fillcolor="rgba(255,152,0,0.05)"))
    fig.add_trace(go.Scatter(x=df["date"], y=df["sma_50"],
                             name="SMA 50", line=dict(color="#4CAF50", width=1)))
    fig.add_trace(go.Scatter(x=df["date"], y=df["sma_200"],
                             name="SMA 200", line=dict(color="#F44336", width=1)))
    fig.update_layout(height=400, template="plotly_dark",
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    # RSI chart
    col_rsi, col_macd = st.columns(2)

    with col_rsi:
        st.subheader("RSI (14)")
        fig_rsi = go.Figure()
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5)
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5)
        fig_rsi.add_trace(go.Scatter(x=df["date"], y=df["rsi_14"],
                                     name="RSI", line=dict(color="#9C27B0")))
        fig_rsi.update_layout(height=250, template="plotly_dark",
                              yaxis=dict(range=[0, 100]))
        st.plotly_chart(fig_rsi, use_container_width=True)

    with col_macd:
        st.subheader("MACD")
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df["date"], y=df["macd_line"],
                                      name="MACD", line=dict(color="#2196F3")))
        fig_macd.add_trace(go.Scatter(x=df["date"], y=df["macd_signal"],
                                      name="Signal", line=dict(color="#FF9800")))
        fig_macd.update_layout(height=250, template="plotly_dark")
        st.plotly_chart(fig_macd, use_container_width=True)


# ══════════════════════════════════════════════════════
# PAGE 3 — SENTIMENT
# ══════════════════════════════════════════════════════
elif page == "📰 Sentiment":
    st.title(f"📰 News Sentiment — {selected_ticker}")

    df = load_sentiment(selected_ticker)

    if df.empty:
        st.warning(f"No sentiment data for {selected_ticker}")
        st.stop()

    # Summary metrics
    total    = len(df)
    bullish  = len(df[df["sentiment"] == "positive"])
    bearish  = len(df[df["sentiment"] == "negative"])
    neutral  = len(df[df["sentiment"] == "neutral"])
    avg_conf = df["confidence"].mean()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Articles", total)
    col2.metric("🟢 Bullish", bullish, f"{bullish/total*100:.0f}%")
    col3.metric("🔴 Bearish", bearish, f"{bearish/total*100:.0f}%")
    col4.metric("⚪ Neutral",  neutral,  f"{neutral/total*100:.0f}%")
    col5.metric("Avg Confidence", f"{avg_conf:.2f}")

    # Sentiment distribution chart
    st.divider()
    col_pie, col_bar = st.columns(2)

    with col_pie:
        st.subheader("Sentiment Distribution")
        fig_pie = px.pie(
            values=[bullish, bearish, neutral],
            names=["Bullish", "Bearish", "Neutral"],
            color_discrete_map={"Bullish":"#4CAF50","Bearish":"#F44336","Neutral":"#9E9E9E"},
            template="plotly_dark",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_bar:
        st.subheader("Confidence by Sentiment")
        fig_box = px.box(df, x="sentiment", y="confidence",
                         color="sentiment",
                         color_discrete_map={"positive":"#4CAF50",
                                             "negative":"#F44336",
                                             "neutral":"#9E9E9E"},
                         template="plotly_dark")
        st.plotly_chart(fig_box, use_container_width=True)

    # Article table
    st.divider()
    st.subheader("Recent Headlines")
    display_df = df[["title", "sentiment", "confidence", "source", "published_at"]].copy()
    display_df.columns = ["Title", "Sentiment", "Confidence", "Source", "Published"]

    def color_sentiment(val):
        colors = {"positive": "color: #4CAF50",
                  "negative": "color: #F44336",
                  "neutral" : "color: #9E9E9E"}
        return colors.get(val, "")

    styled = display_df.style.map(color_sentiment, subset=["Sentiment"])
    st.dataframe(styled, hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════
# PAGE 4 — AI CHAT
# ══════════════════════════════════════════════════════
elif page == "🤖 AI Chat":
    st.title("🤖 AI Market Intelligence Chat")
    st.markdown("Ask questions about stocks, filings, and market sentiment. "
                "Powered by **Llama 3.2** + **RAG** over your pipeline data.")

    # Example questions
    st.markdown("**Try asking:**")
    examples = [
        "What is the sentiment around NVIDIA recently?",
        "Which companies filed annual reports in 2024?",
        "What is the news sentiment for AMD?",
        "Tell me about recent Apple SEC filings",
    ]
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        if cols[i % 2].button(ex, use_container_width=True):
            st.session_state["chat_input"] = ex

    st.divider()

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    prompt = st.chat_input("Ask about stocks, filings, or sentiment...")
    if "chat_input" in st.session_state:
        prompt = st.session_state.pop("chat_input")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching pipeline data and generating answer..."):
                try:
                    r = requests.post(
                        f"{API_BASE}/ask",
                        json={"question": prompt, "n_results": 5},
                        timeout=60,
                    )
                    data    = r.json()
                    answer  = data["answer"]
                    elapsed = data["elapsed_ms"] / 1000
                    sources = data["sources_used"]

                    st.markdown(answer)
                    st.caption(f"Sources: {sources} docs | Response time: {elapsed:.1f}s | Model: Llama 3.2")
                except Exception as e:
                    answer = f"Error connecting to API: {e}. Make sure the FastAPI server is running."
                    st.error(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
