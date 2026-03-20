"""
Phase 5 — FinBERT Sentiment Scoring
-------------------------------------
Scores financial news headlines using FinBERT
(ProsusAI/finbert — trained on financial text).

Reads  : data/bronze/news/newsapi_tickers/**/*.parquet
Writes : data/gold/sentiment/news_sentiment.parquet

Usage:
    python ai/sentiment/finbert_scorer.py
    python ai/sentiment/finbert_scorer.py --batch-size 32 --sample 200
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
import duckdb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingestion.config import (
    GOLD_DIR, NEWS_BRONZE,
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
            os.path.join(LOGS_DIR, f"sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        ),
    ],
)
log = logging.getLogger("finbert_scorer")

SENTIMENT_OUTPUT = os.path.join(GOLD_DIR, "sentiment")


def load_model():
    """
    Load FinBERT from HuggingFace.
    First run downloads ~440MB model weights — cached after that.
    Model: ProsusAI/finbert
      - Trained on 4,500 financial news sentences
      - 3 labels: positive (bullish), negative (bearish), neutral
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    model_name = "ProsusAI/finbert"
    log.info(f"Loading FinBERT model: {model_name}")
    log.info("First run downloads ~440MB — cached afterwards...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    # Use Metal GPU on M4 Pro if available
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model  = model.to(device)
    log.info(f"Model loaded — running on: {device.upper()}")

    return tokenizer, model, device


def score_batch(texts: list[str], tokenizer, model, device: str) -> list[dict]:
    """
    Score a batch of texts with FinBERT.
    Returns list of dicts with label and confidence score.
    """
    import torch

    # FinBERT label mapping
    label_map = {0: "positive", 1: "negative", 2: "neutral"}

    # Tokenise — max 512 tokens (BERT limit), truncate longer texts
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs   = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

    results = []
    for prob_row in probs:
        label_idx  = prob_row.argmax()
        results.append({
            "sentiment"       : label_map[label_idx],
            "sentiment_score" : round(float(prob_row[label_idx]), 4),
            "prob_positive"   : round(float(prob_row[0]), 4),
            "prob_negative"   : round(float(prob_row[1]), 4),
            "prob_neutral"    : round(float(prob_row[2]), 4),
        })
    return results


def run(batch_size: int = 16, sample: int = None) -> dict:
    log.info("=" * 60)
    log.info("FinBERT Sentiment Scoring")
    log.info(f"Batch size : {batch_size}")
    log.info(f"Output     : {SENTIMENT_OUTPUT}/")
    log.info("=" * 60)

    start_time = time.time()

    # Load all news headlines
    log.info("Loading news headlines from bronze layer...")
    conn = duckdb.connect(":memory:")
    query = f"""
        SELECT ticker, title, description, published_at, source, url
        FROM read_parquet('{NEWS_BRONZE}/newsapi_tickers/**/*.parquet')
        WHERE title IS NOT NULL AND title != ''
    """
    if sample:
        query += f" USING SAMPLE {sample} ROWS"
        log.info(f"Sample mode: {sample} rows")

    df = conn.execute(query).df()
    conn.close()
    log.info(f"Loaded {len(df):,} headlines")

    if df.empty:
        log.error("No headlines found — run ingest_news.py first")
        return {}

    # Load FinBERT
    tokenizer, model, device = load_model()

    # Score in batches
    log.info(f"Scoring {len(df):,} headlines in batches of {batch_size}...")
    all_scores = []
    total_batches = (len(df) + batch_size - 1) // batch_size

    for i in range(0, len(df), batch_size):
        batch_texts = df["title"].iloc[i:i+batch_size].tolist()
        scores      = score_batch(batch_texts, tokenizer, model, device)
        all_scores.extend(scores)

        batch_num = i // batch_size + 1
        if batch_num % 5 == 0 or batch_num == total_batches:
            elapsed = time.time() - start_time
            rate    = (i + batch_size) / elapsed
            log.info(f"  Batch {batch_num}/{total_batches} | "
                     f"{i+batch_size:,}/{len(df):,} headlines | "
                     f"{rate:.0f} headlines/sec")

    # Merge scores with original data
    scores_df = pd.DataFrame(all_scores)
    result_df = pd.concat([df.reset_index(drop=True), scores_df], axis=1)
    result_df["scored_at"] = datetime.utcnow().isoformat()

    # Summary stats
    dist = result_df["sentiment"].value_counts()
    log.info("\nSentiment distribution:")
    for label, count in dist.items():
        pct = count / len(result_df) * 100
        log.info(f"  {label:10s}: {count:4d} ({pct:.1f}%)")

    # Save to gold layer
    os.makedirs(SENTIMENT_OUTPUT, exist_ok=True)
    out_path = os.path.join(SENTIMENT_OUTPUT, "news_sentiment.parquet")
    result_df.to_parquet(
        out_path,
        engine=PARQUET_ENGINE,
        compression=PARQUET_COMPRESSION,
        index=False,
        coerce_timestamps="us",
    )
    size_kb = os.path.getsize(out_path) / 1024
    log.info(f"\nSaved: {out_path} ({size_kb:.1f} KB, {len(result_df):,} rows)")

    elapsed = time.time() - start_time
    summary = {
        "total_headlines" : len(result_df),
        "sentiment_dist"  : dist.to_dict(),
        "elapsed_sec"     : round(elapsed, 1),
        "output_path"     : out_path,
    }

    log.info("\n" + "=" * 60)
    log.info("SENTIMENT SCORING COMPLETE")
    log.info(f"Total scored: {len(result_df):,} headlines")
    log.info(f"Time taken  : {elapsed:.1f}s")
    log.info("=" * 60)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score news with FinBERT")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for inference (default: 16)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Score only N headlines (for testing)")
    args = parser.parse_args()

    run(args.batch_size, args.sample)
