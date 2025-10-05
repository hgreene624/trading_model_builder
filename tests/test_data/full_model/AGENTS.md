# Full Model Fixture Instructions

This directory contains an offline snapshot scaffold that covers the minimum data needed to exercise the full breakout workflow without network requests. Only metadata files are tracked in git; you must drop the Parquet bars locally before running tests.

## Contents
- `data/portfolios/model_validation.json` – canonical portfolio definition consumed by `src/storage` when `DATA_DIR` points here.
- `storage/portfolios/model_validation.json` – legacy location mirror for components that still scan `storage/portfolios`.
- `storage/data/ohlcv/alpaca/<SYMBOL>/1D/<DATE_RANGE>.parquet` – deterministic OHLCV bars for AAPL, MSFT, and TSLA generated with UTC business-day timestamps. Columns must match the loader contract: `open`, `high`, `low`, `close`, `volume`, and `vwap`. Provide these files locally; they are intentionally omitted from version control.

## Usage patterns
1. Export `DATA_DIR=$(pwd)/tests/test_data/full_model/data` before invoking code that calls `src.storage.get_data_dir()`.
2. Populate `storage/data/ohlcv/alpaca/<SYMBOL>/1D/` with the vendor-exported Parquet shards (for example, `2024-01-02__2024-01-24.parquet`). Remove the placeholder `.gitkeep` files as you add real data.
3. Point the cache readers at this fixture by symlinking or copying the `storage/` subdirectories into the project root (see root-level `AGENTS.md` for commands).
4. When running tests, prefer temporary symlinks so cleanup is just `rm storage/data/ohlcv storage/portfolios && git checkout -- storage`.
5. Avoid editing the parquet files manually—regenerate them with a script if new date ranges or columns are required.

## Extending the fixture
- Keep timelines short (≤ one month) to limit repository size.
- Preserve UTC-indexed parquet files; tests assume timezone-aware timestamps.
- Update both the canonical and legacy portfolio paths when adding new portfolios here.
