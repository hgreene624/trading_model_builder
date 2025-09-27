# Evolutionary Algorithm & Walk-Forward Evaluation

_Date:_ 2025-09-27T01:11:20Z

## Environment
- Python: Python 3.12.10
- Key dependencies installed: requirements.txt, requirements-dev.txt
- Tests executed with synthetic OHLCV data via temporary monkey patch of `src.data.loader.get_ohlcv`

## Evolutionary Search Smoke Test
- Command:
  ```bash
  DATA_PROVIDER=yahoo python - <<'PY'
  from datetime import datetime
  import pandas as pd
  import src.data.loader as loader
  from src.optimization.evolutionary import evolutionary_search

  def fake_get_ohlcv(symbol, start, end, timeframe="1D", force_provider=None):
      idx = pd.date_range(start, end, freq="B", tz="UTC")
      if idx.empty:
          return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
      base = hash(symbol) % 50 + 50
      trend = pd.Series(range(len(idx)), index=idx) * 0.2
      df = pd.DataFrame(index=idx)
      df["open"] = base + trend
      df["high"] = df["open"] + 1
      df["low"] = df["open"] - 1
      df["close"] = df["open"] + 0.5
      df["volume"] = 1_000_000
      return df

  loader.get_ohlcv = fake_get_ohlcv

  space = {
      "breakout_n": (10, 20),
      "exit_n": (5, 15),
      "atr_n": (10, 20),
      "atr_multiple": (1.0, 3.0),
      "tp_multiple": (0.0, 1.5),
      "holding_period_limit": (3, 10),
  }

  evolutionary_search(
      "src.models.atr_breakout",
      ["AAPL", "MSFT"],
      datetime(2020, 1, 1),
      datetime(2021, 1, 1),
      100_000,
      space,
      generations=2,
      pop_size=4,
      seed=42,
      min_trades=1,
      log_file="ea_fake.log",
  )
  PY
  ```
- Parameters: 2 generations, population 4, tickers AAPL/MSFT, 2020-2021 period
- Observations:
  - Fitness scores produced without runtime errors when data supplied.
  - Without patch, loader raised `RuntimeError: No data returned ...` because remote providers blocked (missing Alpaca keys, Yahoo 403).
  - Training log recorded metrics for each individual; high Sharpe/Calmar values due to synthetic data lacking drawdowns.

## Walk-Forward Validation Smoke Test
- Used same synthetic data patch; 2 splits, 126-day train, 63-day test, EA enabled (2 gens/pop=4).
- Reproduction snippet:
  ```bash
  DATA_PROVIDER=yahoo python - <<'PY'
  from datetime import datetime
  import pandas as pd
  import src.data.loader as loader
  from src.optimization.walkforward import walk_forward

  def fake_get_ohlcv(symbol, start, end, timeframe="1D", force_provider=None):
      idx = pd.date_range(start, end, freq="B", tz="UTC")
      if idx.empty:
          return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
      base = hash(symbol) % 50 + 50
      trend = pd.Series(range(len(idx)), index=idx) * 0.1
      df = pd.DataFrame(index=idx)
      df["open"] = base + trend
      df["high"] = df["open"] + 1
      df["low"] = df["open"] - 1
      df["close"] = df["open"] + 0.5
      df["volume"] = 1_000_000
      return df

  loader.get_ohlcv = fake_get_ohlcv

  base_params = {
      "breakout_n": 14,
      "exit_n": 6,
      "atr_n": 14,
      "atr_multiple": 2.0,
      "tp_multiple": 0.5,
      "holding_period_limit": 5,
  }

  walk_forward(
      "src.models.atr_breakout",
      ["AAPL", "MSFT"],
      datetime(2020, 1, 1),
      datetime(2021, 1, 1),
      100_000,
      base_params,
      splits=2,
      train_days=126,
      test_days=63,
      use_ea=True,
      ea_generations=2,
      ea_pop=4,
      seed=42,
      min_trades=1,
      log_file="walkforward_fake.log",
  )
  PY
  ```
- Resulted in consistent OOS metrics across splits (identical due to deterministic synthetic prices).
- JSONL `walkforward_fake.log` captured per-split telemetry, confirming logging pipeline.

## Issues Encountered
1. **Data acquisition failures**: `src.data.loader.get_ohlcv` attempts Alpaca then Yahoo. Without credentials/network the EA aborts. Need offline fixtures or provider toggle for local runs.
2. **Degenerate metrics**: Synthetic or flat data yields zero drawdown → infinite Calmar. Consider clamping or guarding to avoid misleading UI stats.

## Recommendations
- Provide cached OHLCV fixtures under `storage/data/ohlcv` for demo tickers/periods so EA can run offline without monkey patching.
- Allow loader to honour an `OFFLINE_ONLY` flag that skips external fetches and falls back to deterministic synthetic data (useful for unit tests/CI).
- In `_clamped_fitness`, cap Calmar when `max_drawdown` is ~0 to prevent huge scores that swamp other metrics.
- For walk-forward, surface EA-selected parameters per split in UI (already present in logs) and highlight if OOS trades=0 to detect degenerate strategies.

