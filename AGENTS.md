# Project Guidelines for `alpaca_trader`

Welcome! This document gives AI agents the context they need to safely extend the project.

## Tech stack & entry points
- Streamlit drives the UI. `Home.py` is the main entry point; additional pages live under `pages/`.
- Core logic sits under `src/`, split into subpackages for data ingestion (`src/data`), backtesting (`src/backtest`), and persistence (`src/storage`).
- Pytest-based tests live under `tests/` and rely on pandas/pyarrow for data handling.

## Code style expectations
- Python formatting follows the standard library style with 4-space indentation and descriptive docstrings where practical.
- Prefer dataclasses or typed dictionaries for structured data; type hints should be added/maintained when editing existing modules.
- Avoid adding bare `print` debugging in library code—use the existing logging hooks instead.
- Never wrap import statements in try/except blocks.

## Testing & verification
- Install dev dependencies with `pip install -r requirements-dev.txt` before running tests.
- The canonical test command is `pytest` from the repository root.
- When contributing data-related features, add deterministic fixtures so tests can run offline.

## Offline model test fixture
Comprehensive placeholder folders for an offline validation fixture live under `tests/test_data/full_model/`. You must supply the Parquet files locally (they are not tracked in git) before running end-to-end tests.

Directory layout:
```
full_model/
  AGENTS.md (detailed usage for the fixture subtree)
  data/
    portfolios/model_validation.json
  storage/
    data/ohlcv/alpaca/<SYMBOL>/1D/<DATE_RANGE>.parquet  # provide locally
    portfolios/model_validation.json
```

How to use the fixture end-to-end once Parquet shards are present:
1. From the repo root, export the data dir so `src/storage` looks at the fixture portfolio:
   ```bash
   export DATA_DIR="$(pwd)/tests/test_data/full_model/data"
   ```
2. Point the OHLCV cache to the fixture by temporarily swapping the storage directories (symlink or copy):
   ```bash
   rm -rf storage/data/ohlcv && ln -s "$(pwd)/tests/test_data/full_model/storage/data/ohlcv" storage/data/ohlcv
   rm -rf storage/portfolios && ln -s "$(pwd)/tests/test_data/full_model/storage/portfolios" storage/portfolios
   ```
   (If you cannot create symlinks, copy the directories instead.)
3. Add the vendor-exported Parquet files for each ticker/timeframe into the appropriate directories (see the fixture `AGENTS.md`).
4. Run any portfolio or simulation code (Streamlit pages, smoke tests, or unit tests); they will consume the cached OHLCV shards and the `model_validation` portfolio without hitting external APIs.
5. After finishing, restore the original `storage/*` directories as needed.

Following these steps ensures all parts of the model—portfolio discovery, cached data loading, and simulation—can be validated entirely offline while keeping large binary assets out of version control.
