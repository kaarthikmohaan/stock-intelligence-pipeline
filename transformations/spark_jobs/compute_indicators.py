"""
Phase 4 — PySpark Technical Indicators
----------------------------------------
Computes RSI, MACD, Bollinger Bands, and VWAP
on silver price data using PySpark window functions.

Reads from : data/silver/prices/SOURCE/
Writes to  : data/gold/indicators/SOURCE/

Usage:
    python transformations/spark_jobs/compute_indicators.py --source yfinance
    python transformations/spark_jobs/compute_indicators.py --source kaggle
    python transformations/spark_jobs/compute_indicators.py --source all
"""

import os
import sys
import logging
import argparse
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from ingestion.config import (
    SILVER_DIR, GOLD_DIR,
    PARQUET_COMPRESSION, LOGS_DIR, LOG_LEVEL, LOG_FORMAT,
)

os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(LOGS_DIR, f"indicators_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        ),
    ],
)
log = logging.getLogger("compute_indicators")

# Suppress verbose Spark logs — only show warnings and errors
os.environ["PYSPARK_PYTHON"] = sys.executable


def create_spark_session(app_name: str = "StockIndicators"):
    """
    Create a PySpark session configured for local development.
    Uses all available CPU cores on your M4 Pro.
    """
    from pyspark.sql import SparkSession

    spark = SparkSession.builder \
        .master("local[*]") \
        .appName(app_name) \
        .config("spark.driver.memory", "16g") \
        .config("spark.executor.memory", "16g") \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.memory.fraction", "0.8") \
        .config("spark.memory.storageFraction", "0.3") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.ansi.enabled", "false") \
        .config("spark.ui.showConsoleProgress", "false") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    return spark


def compute_rsi(df, price_col: str = "close", period: int = 14):
    """
    RSI — Relative Strength Index (14-day default)

    Formula:
        gain = average of positive daily returns over period
        loss = average of negative daily returns over period
        RS   = gain / loss
        RSI  = 100 - (100 / (1 + RS))

    Interpretation:
        RSI > 70 → overbought (consider selling)
        RSI < 30 → oversold  (consider buying)
    """
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    # Window: for each ticker, ordered by date
    ticker_window = Window.partitionBy("ticker").orderBy("date")

    # Previous close price
    df = df.withColumn("prev_close", F.lag(price_col, 1).over(ticker_window))

    # Daily change
    df = df.withColumn("price_change",
        F.col(price_col) - F.col("prev_close"))

    # Separate gains and losses
    df = df.withColumn("gain",
        F.when(F.col("price_change") > 0, F.col("price_change")).otherwise(0.0))
    df = df.withColumn("loss",
        F.when(F.col("price_change") < 0, -F.col("price_change")).otherwise(0.0))

    # Rolling average over period window
    rolling = Window.partitionBy("ticker").orderBy("date") \
        .rowsBetween(-period + 1, 0)

    df = df.withColumn("avg_gain", F.avg("gain").over(rolling))
    df = df.withColumn("avg_loss", F.avg("loss").over(rolling))

    # RSI calculation
    df = df.withColumn("rs",
        F.when(F.col("avg_loss") == 0, 100.0)
         .otherwise(F.col("avg_gain") / F.col("avg_loss")))

    df = df.withColumn(f"rsi_{period}",
        F.round(100 - (100 / (1 + F.col("rs"))), 2))

    # Clean up intermediate columns
    df = df.drop("prev_close", "price_change", "gain", "loss",
                 "avg_gain", "avg_loss", "rs")
    return df


def compute_macd(df, price_col: str = "close",
                 fast: int = 12, slow: int = 26, signal: int = 9):
    """
    MACD — Moving Average Convergence Divergence

    Formula:
        EMA_fast   = 12-day exponential moving average
        EMA_slow   = 26-day exponential moving average
        MACD_line  = EMA_fast - EMA_slow
        Signal     = 9-day EMA of MACD_line
        Histogram  = MACD_line - Signal

    Interpretation:
        MACD crosses above signal → bullish momentum
        MACD crosses below signal → bearish momentum
        Histogram above 0 → bulls winning
    """
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    # We use simple moving averages as EMA approximation
    # (true EMA requires iterative calculation, harder in Spark)
    fast_window = Window.partitionBy("ticker").orderBy("date") \
        .rowsBetween(-fast + 1, 0)
    slow_window = Window.partitionBy("ticker").orderBy("date") \
        .rowsBetween(-slow + 1, 0)

    df = df.withColumn(f"sma_{fast}", F.avg(price_col).over(fast_window))
    df = df.withColumn(f"sma_{slow}", F.avg(price_col).over(slow_window))

    # MACD line
    df = df.withColumn("macd_line",
        F.round(F.col(f"sma_{fast}") - F.col(f"sma_{slow}"), 4))

    # Signal line — moving average of MACD
    signal_window = Window.partitionBy("ticker").orderBy("date") \
        .rowsBetween(-signal + 1, 0)
    df = df.withColumn("macd_signal",
        F.round(F.avg("macd_line").over(signal_window), 4))

    # Histogram — difference between MACD and signal
    df = df.withColumn("macd_histogram",
        F.round(F.col("macd_line") - F.col("macd_signal"), 4))

    df = df.drop(f"sma_{fast}", f"sma_{slow}")
    return df


def compute_bollinger_bands(df, price_col: str = "close",
                             period: int = 20, num_std: float = 2.0):
    """
    Bollinger Bands — volatility indicator

    Formula:
        middle_band = 20-day simple moving average
        upper_band  = middle + (2 × 20-day standard deviation)
        lower_band  = middle - (2 × 20-day standard deviation)
        bandwidth   = (upper - lower) / middle × 100

    Interpretation:
        Price near upper band → statistically expensive
        Price near lower band → statistically cheap
        Narrow bands → low volatility (breakout likely soon)
        Wide bands  → high volatility
    """
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    rolling = Window.partitionBy("ticker").orderBy("date") \
        .rowsBetween(-period + 1, 0)

    df = df.withColumn("bb_middle",
        F.round(F.avg(price_col).over(rolling), 4))
    df = df.withColumn("bb_std",
        F.stddev(price_col).over(rolling))
    df = df.withColumn("bb_upper",
        F.round(F.col("bb_middle") + (num_std * F.col("bb_std")), 4))
    df = df.withColumn("bb_lower",
        F.round(F.col("bb_middle") - (num_std * F.col("bb_std")), 4))

    # Bandwidth — how wide the bands are as % of middle
    df = df.withColumn("bb_bandwidth",
        F.round((F.col("bb_upper") - F.col("bb_lower"))
                / F.col("bb_middle") * 100, 4))

    # %B — where is price within the bands (0=lower, 1=upper)
    df = df.withColumn("bb_pct_b",
        F.round((F.col(price_col) - F.col("bb_lower"))
                / (F.col("bb_upper") - F.col("bb_lower")), 4))

    df = df.drop("bb_std")
    return df


def compute_vwap(df):
    """
    VWAP — Volume Weighted Average Price (daily rolling 20-day)

    Formula:
        VWAP = SUM(typical_price × volume) / SUM(volume)

    Interpretation:
        Price > VWAP → bullish (buyers in control)
        Price < VWAP → bearish (sellers in control)
        Used by institutional traders as execution benchmark
    """
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    rolling = Window.partitionBy("ticker").orderBy("date") \
        .rowsBetween(-19, 0)  # 20-day rolling

    # typical_price already computed in silver layer
    df = df.withColumn("vwap",
        F.round(
            F.try_divide(
                F.sum(F.col("typical_price") * F.col("volume")).over(rolling),
                F.sum("volume").over(rolling)
            ),
            4
        ))

    # Price position relative to VWAP
    df = df.withColumn("vwap_signal",
        F.when(F.col("close") > F.col("vwap"), "above")
         .when(F.col("close") < F.col("vwap"), "below")
         .otherwise("at"))

    return df


def compute_moving_averages(df, price_col: str = "close"):
    """
    Simple and Exponential Moving Averages — trend indicators

    SMA_50  = 50-day simple moving average (medium-term trend)
    SMA_200 = 200-day simple moving average (long-term trend)

    Golden Cross: SMA_50 crosses above SMA_200 → strong buy signal
    Death Cross:  SMA_50 crosses below SMA_200 → strong sell signal
    """
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    w50  = Window.partitionBy("ticker").orderBy("date").rowsBetween(-49, 0)
    w200 = Window.partitionBy("ticker").orderBy("date").rowsBetween(-199, 0)

    df = df.withColumn("sma_50",  F.round(F.avg(price_col).over(w50),  4))
    df = df.withColumn("sma_200", F.round(F.avg(price_col).over(w200), 4))

    # Golden/Death cross signal
    df = df.withColumn("ma_signal",
        F.when(F.col("sma_50") > F.col("sma_200"), "golden")
         .when(F.col("sma_50") < F.col("sma_200"), "death")
         .otherwise("neutral"))

    return df


def process_source(spark, source: str) -> dict:
    """Read silver → compute indicators → write gold for one source."""
    from pyspark.sql import functions as F

    silver_path = os.path.join(SILVER_DIR, "prices", source)
    gold_path   = os.path.join(GOLD_DIR,   "indicators", source)
    os.makedirs(gold_path, exist_ok=True)

    log.info(f"\nSource : {source}")
    log.info(f"Silver : {silver_path}")
    log.info(f"Gold   : {gold_path}")

    # Read silver Parquet
    log.info("Reading silver data...")
    t0 = time.time()
    df = spark.read.parquet(silver_path)
    total = df.count()
    log.info(f"Loaded {total:,} rows in {time.time()-t0:.1f}s")

    # Ensure correct types
    df = df.withColumn("date",   F.col("date").cast("timestamp"))
    df = df.withColumn("close",  F.col("close").cast("double"))
    df = df.withColumn("high",   F.col("high").cast("double"))
    df = df.withColumn("low",    F.col("low").cast("double"))
    df = df.withColumn("volume", F.col("volume").cast("double"))
    df = df.withColumn("typical_price", F.col("typical_price").cast("double"))

    # Compute all indicators
    log.info("Computing RSI(14)...")
    df = compute_rsi(df)

    log.info("Computing MACD(12,26,9)...")
    df = compute_macd(df)

    log.info("Computing Bollinger Bands(20,2)...")
    df = compute_bollinger_bands(df)

    log.info("Computing VWAP(20)...")
    df = compute_vwap(df)

    log.info("Computing Moving Averages(50,200)...")
    df = compute_moving_averages(df)

    # Add metadata
    df = df.withColumn("indicators_computed_at",
        F.lit(datetime.utcnow().isoformat()))

    # Write to gold layer partitioned by year
    log.info("Writing gold Parquet...")
    t1 = time.time()
    df.write \
      .mode("overwrite") \
      .partitionBy("data_source") \
      .parquet(gold_path)

    elapsed = time.time() - t1
    log.info(f"Gold write done in {elapsed:.1f}s")

    return {"source": source, "rows": total}


def run(sources: list) -> None:
    log.info("=" * 60)
    log.info("Phase 4 — PySpark Technical Indicators")
    log.info(f"Sources : {sources}")
    log.info(f"Output  : {GOLD_DIR}/indicators/")
    log.info("=" * 60)

    spark = create_spark_session()
    log.info(f"Spark version : {spark.version}")
    log.info(f"Master        : {spark.sparkContext.master}")

    start_time = time.time()

    for source in sources:
        process_source(spark, source)

    spark.stop()
    elapsed = time.time() - start_time
    log.info("\n" + "=" * 60)
    log.info("INDICATORS COMPLETE")
    log.info(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    log.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute technical indicators")
    parser.add_argument("--source", choices=["yfinance", "kaggle", "all"],
                        default="yfinance")
    args = parser.parse_args()

    sources = ["yfinance", "kaggle"] if args.source == "all" else [args.source]
    run(sources)
