{{ config(materialized='table') }}

WITH base AS (
    SELECT * FROM {{ ref('stg_prices') }}
    WHERE daily_return IS NOT NULL
)

SELECT
    ticker,
    COUNT(*)                                    AS trading_days,
    ROUND(AVG(daily_return) * 100, 4)           AS avg_daily_return_pct,
    ROUND(STDDEV(daily_return) * 100, 4)        AS volatility_pct,
    ROUND(MAX(daily_return) * 100, 4)           AS best_day_pct,
    ROUND(MIN(daily_return) * 100, 4)           AS worst_day_pct,
    ROUND(AVG(close_price), 2)                  AS avg_close,
    ROUND(MAX(close_price), 2)                  AS max_close,
    ROUND(MIN(close_price), 2)                  AS min_close,
    ROUND(AVG(volume), 0)                       AS avg_volume,
    MIN(price_date)                             AS first_date,
    MAX(price_date)                             AS last_date,
    ROUND(
        CASE WHEN STDDEV(daily_return) > 0
        THEN AVG(daily_return) / STDDEV(daily_return)
        ELSE 0 END, 4
    )                                           AS return_risk_ratio
FROM base
GROUP BY ticker
ORDER BY avg_daily_return_pct DESC
