"""
Phase 5 — LangChain RAG Pipeline
----------------------------------
Builds a vector store from SEC EDGAR filing metadata
and news sentiment, then enables Q&A over your data
using Llama 3.2 running locally via Ollama.

Usage:
    # Build the vector store (run once)
    python ai/rag/rag_pipeline.py --build

    # Ask a question
    python ai/rag/rag_pipeline.py --query "What risks did Apple mention?"

    # Interactive mode
    python ai/rag/rag_pipeline.py --interactive
"""

import os
import sys
import logging
import argparse
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingestion.config import (
    FILINGS_BRONZE, GOLD_DIR, NEWS_BRONZE,
    LOGS_DIR, LOG_LEVEL, LOG_FORMAT,
)

os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("rag_pipeline")

CHROMA_DIR      = os.path.join(GOLD_DIR, "chroma_db")
SENTIMENT_PATH  = os.path.join(GOLD_DIR, "sentiment", "news_sentiment.parquet")


def build_vector_store() -> None:
    """
    Build ChromaDB vector store from:
    1. SEC EDGAR filing metadata (company, form type, period, dates)
    2. News sentiment scores (ticker, headline, sentiment, score)

    Each document is chunked into text and embedded using
    sentence-transformers (all-MiniLM-L6-v2 — fast, accurate).
    """
    import duckdb
    import chromadb
    from sentence_transformers import SentenceTransformer

    log.info("=" * 55)
    log.info("Building RAG Vector Store")
    log.info(f"Output: {CHROMA_DIR}")
    log.info("=" * 55)

    # Load embedding model
    log.info("Loading embedding model (all-MiniLM-L6-v2)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    log.info("Embedding model loaded")

    # Initialise ChromaDB
    client     = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete existing collections if rebuilding
    for name in ["sec_filings", "news_sentiment"]:
        try:
            client.delete_collection(name)
        except Exception:
            pass

    # ── Collection 1: SEC EDGAR filings ──────────────
    log.info("\nLoading SEC EDGAR filings...")
    conn = duckdb.connect(":memory:")
    filings_df = conn.execute(f"""
        SELECT ticker, form_type, entity_name,
               period_of_report, file_date, accession_no
        FROM read_parquet('{FILINGS_BRONZE}/sec_edgar/**/*.parquet')
    """).df()
    conn.close()
    log.info(f"Loaded {len(filings_df):,} filing records")

    filings_col = client.create_collection(
        name="sec_filings",
        metadata={"description": "SEC EDGAR 10-K and 10-Q filing metadata"}
    )

    # Build document text for each filing
    docs, metas, ids = [], [], []
    for i, row in filings_df.iterrows():
        text = (
            f"{row['entity_name']} ({row['ticker']}) filed a "
            f"{row['form_type']} on {row['file_date']} "
            f"for the period ending {row['period_of_report']}. "
            f"Accession number: {row['accession_no']}."
        )
        docs.append(text)
        metas.append({
            "ticker"    : str(row["ticker"]),
            "form_type" : str(row["form_type"]),
            "file_date" : str(row["file_date"]),
            "period"    : str(row["period_of_report"]),
        })
        ids.append(f"filing_{i}")

    # Embed and store
    log.info(f"Embedding {len(docs)} filing documents...")
    embeddings = embedder.encode(docs, show_progress_bar=True).tolist()
    filings_col.add(documents=docs, embeddings=embeddings,
                    metadatas=metas, ids=ids)
    log.info(f"SEC filings collection: {filings_col.count()} documents")

    # ── Collection 2: News sentiment ─────────────────
    log.info("\nLoading news sentiment data...")
    conn = duckdb.connect(":memory:")
    news_df = conn.execute(f"""
        SELECT ticker, title, sentiment, sentiment_score,
               source, published_at
        FROM read_parquet('{SENTIMENT_PATH}')
        WHERE title IS NOT NULL AND title != ''
        LIMIT 1000
    """).df()
    conn.close()
    log.info(f"Loaded {len(news_df):,} sentiment records")

    news_col = client.create_collection(
        name="news_sentiment",
        metadata={"description": "News headlines with FinBERT sentiment scores"}
    )

    docs2, metas2, ids2 = [], [], []
    for i, row in news_df.iterrows():
        text = (
            f"[{row['sentiment'].upper()}] {row['ticker']}: {row['title']} "
            f"(confidence: {row['sentiment_score']:.2f}, "
            f"source: {row['source']}, date: {row['published_at']})"
        )
        docs2.append(text)
        metas2.append({
            "ticker"    : str(row["ticker"]),
            "sentiment" : str(row["sentiment"]),
            "score"     : float(row["sentiment_score"]),
            "source"    : str(row["source"]),
        })
        ids2.append(f"news_{i}")

    log.info(f"Embedding {len(docs2)} news documents...")
    embeddings2 = embedder.encode(docs2, show_progress_bar=True).tolist()
    news_col.add(documents=docs2, embeddings=embeddings2,
                 metadatas=metas2, ids=ids2)
    log.info(f"News sentiment collection: {news_col.count()} documents")

    log.info("\n" + "=" * 55)
    log.info("Vector store built successfully")
    log.info(f"  SEC filings : {filings_col.count()} documents")
    log.info(f"  News        : {news_col.count()} documents")
    log.info(f"  Location    : {CHROMA_DIR}")
    log.info("=" * 55)


def query_rag(question: str, n_results: int = 5) -> str:
    """
    Answer a question using RAG:
    1. Embed the question
    2. Search ChromaDB for relevant documents
    3. Pass documents + question to Llama 3.2
    4. Return grounded answer
    """
    import chromadb
    from sentence_transformers import SentenceTransformer
    from langchain_ollama import OllamaLLM

    log.info(f"\nQuestion: {question}")

    # Load vector store
    client   = chromadb.PersistentClient(path=CHROMA_DIR)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Search both collections
    q_embedding = embedder.encode([question]).tolist()
    context_parts = []

    for col_name in ["sec_filings", "news_sentiment"]:
        try:
            col     = client.get_collection(col_name)
            results = col.query(query_embeddings=q_embedding,
                                n_results=min(n_results, col.count()))
            docs    = results["documents"][0]
            context_parts.extend(docs)
            log.info(f"  Retrieved {len(docs)} docs from {col_name}")
        except Exception as e:
            log.warning(f"  Could not query {col_name}: {e}")

    if not context_parts:
        return "No relevant documents found in the vector store."

    context = "\n\n".join(context_parts)

    # Build prompt
    prompt = f"""You are a financial analyst assistant for a stock market intelligence pipeline.
Answer the user's question based ONLY on the context provided below.
If the context doesn't contain enough information, say so clearly.
Be concise and factual.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    # Call Llama 3.2 via Ollama
    log.info("  Calling Llama 3.2 via Ollama...")
    llm    = OllamaLLM(model="llama3.2", temperature=0.1)
    answer = llm.invoke(prompt)

    return answer


def run_interactive() -> None:
    """Interactive Q&A session."""
    print("\n" + "=" * 55)
    print(" Stock Intelligence RAG — Interactive Mode")
    print(" Type 'quit' to exit")
    print("=" * 55)

    example_questions = [
        "Which companies filed 10-K reports in 2024?",
        "What is the sentiment around NVIDIA recently?",
        "Which tickers have the most bearish news?",
        "Tell me about Apple's recent SEC filings",
    ]

    print("\nExample questions:")
    for q in example_questions:
        print(f"  • {q}")
    print()

    while True:
        question = input("Your question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        answer = query_rag(question)
        print(f"\nAnswer: {answer}\n")
        print("-" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LangChain RAG pipeline")
    parser.add_argument("--build", action="store_true",
                        help="Build/rebuild the vector store")
    parser.add_argument("--query", type=str, default=None,
                        help="Ask a single question")
    parser.add_argument("--interactive", action="store_true",
                        help="Start interactive Q&A session")
    args = parser.parse_args()

    if args.build:
        build_vector_store()
    elif args.query:
        answer = query_rag(args.query)
        print(f"\nAnswer: {answer}")
    elif args.interactive:
        run_interactive()
    else:
        # Default: build then run a test query
        build_vector_store()
        test_q = "Which companies filed annual reports in 2024?"
        print(f"\nTest query: {test_q}")
        answer = query_rag(test_q)
        print(f"Answer: {answer}")
