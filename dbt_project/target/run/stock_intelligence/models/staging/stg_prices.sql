
  
  create view "stock_intelligence"."main"."stg_prices__dbt_tmp" as (
    

SELECT
    date::DATE                          AS price_date,
    ticker,
    ROUND(open::DOUBLE, 4)              AS open_price,
    ROUND(high::DOUBLE, 4)              AS high_price,
    ROUND(low::DOUBLE, 4)               AS low_price,
    ROUND(close::DOUBLE, 4)             AS close_price,
    volume::BIGINT                      AS volume,
    ROUND(daily_return::DOUBLE, 6)      AS daily_return,
    ROUND(price_range::DOUBLE, 4)       AS price_range,
    ROUND(typical_price::DOUBLE, 4)     AS typical_price,
    ROUND(range_pct::DOUBLE, 4)         AS range_pct,
    data_source
FROM read_parquet(
    '/Users/karthikmohan/Desktop/stock-intelligence-pipeline/data/silver/prices/yfinance/**/*.parquet'
)
WHERE close::DOUBLE > 0
  AND date IS NOT NULL
  AND ticker IS NOT NULL
  );
