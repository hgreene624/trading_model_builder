# OHLCV Fixture Layout

Place vendor-exported Parquet bars inside the provider/ticker/timeframe folders.

```
alpaca/
  AAPL/
    1D/
      2024-01-02__2024-01-24.parquet
  MSFT/
    1D/
      2024-01-02__2024-01-24.parquet
  TSLA/
    1D/
      2024-01-02__2024-01-24.parquet
```

You can provide additional date ranges or tickers if needed—just mirror the same nesting pattern.
