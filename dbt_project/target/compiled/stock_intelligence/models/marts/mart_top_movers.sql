

WITH latest_date AS (
    SELECT MAX(price_date) AS max_date
    FROM "stock_intelligence"."main"."stg_prices"
),
latest_prices AS (
    SELECT
        p.ticker,
        p.price_date,
        p.close_price,
        p.daily_return,
        p.volume,
        p.range_pct
    FROM "stock_intelligence"."main"."stg_prices" p
    CROSS JOIN latest_date ld
    WHERE p.price_date = ld.max_date
      AND p.daily_return IS NOT NULL
),
ranked AS (
    SELECT
        ticker,
        price_date,
        close_price,
        ROUND(daily_return * 100, 4)    AS daily_return_pct,
        volume,
        ROUND(range_pct, 4)             AS range_pct,
        RANK() OVER (ORDER BY daily_return DESC) AS gainer_rank,
        RANK() OVER (ORDER BY daily_return ASC)  AS loser_rank
    FROM latest_prices
)
SELECT
    ticker,
    price_date,
    close_price,
    daily_return_pct,
    volume,
    range_pct,
    gainer_rank,
    loser_rank,
    CASE
        WHEN gainer_rank <= 10 THEN 'top_gainer'
        WHEN loser_rank  <= 10 THEN 'top_loser'
        ELSE 'neutral'
    END AS mover_type
FROM ranked
WHERE gainer_rank <= 10 OR loser_rank <= 10
ORDER BY daily_return_pct DESC