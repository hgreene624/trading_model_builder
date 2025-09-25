# Alpaca Trader Research Dashboard

## Overview
This repository packages a multi-page Streamlit application for researching daily breakout strategies on US equities. The home page surfaces connection status for your Alpaca paper account, recent simulation artifacts, and quick metrics that link the rest of the workflow together.【F:Home.py†L112-L194】

## Features
- **Unified market data ingestion** – `src/data/loader.get_ohlcv` automatically chooses Alpaca as the primary provider, falls back to Yahoo Finance when needed, and normalizes timestamps/columns. A lightweight parquet cache keeps previously downloaded bars on disk for reuse.【F:src/data/loader.py†L15-L134】【F:src/data/cache.py†L9-L81】
- **Portfolio construction workflows** – The Portfolios page guides you through fetching index constituents from Wikipedia, filtering by metadata, enriching with liquidity stats, and saving curated universes to storage.【F:pages/1_Portfolios.py†L35-L199】
- **Strategy tuning and backtesting** – The Ticker Selector & Tuning page drives an evolutionary search over ATR breakout parameters and stores the best runs, backed by the standalone ATR backtest engine under `src/backtest`.【F:pages/3_Ticker_Selector_and_Tuning.py†L360-L460】【F:src/backtest/engine.py†L1-L200】
- **Portfolio-level simulation** – The Simulate Portfolio page replays saved strategies across many tickers, schedules entries/exits with budget constraints, and renders interactive Plotly charts for both equity curves and per-symbol diagnostics.【F:pages/4_Simulate_Portfolio.py†L1-L200】
- **Artifact & portfolio storage** – Helper utilities in `src/storage.py` unify portfolio persistence, migrate legacy files, and expose helpers used throughout the UI and tests.【F:src/storage.py†L14-L198】

## Repository layout
| Path | Description |
| --- | --- |
| `Home.py` | Streamlit entry point with the dashboard shell and Alpaca connectivity checks.【F:Home.py†L112-L194】 |
| `pages/` | Streamlit multipage views for portfolios, tuning, adapters, and simulations.【F:pages/1_Portfolios.py†L35-L199】【F:pages/3_Ticker_Selector_and_Tuning.py†L360-L460】【F:pages/4_Simulate_Portfolio.py†L1-L200】 |
| `src/` | Core Python packages covering data ingestion, backtesting engines, storage helpers, and utilities.【F:src/data/loader.py†L15-L134】【F:src/backtest/engine.py†L1-L200】【F:src/storage.py†L14-L198】 |
| `tests/` | Pytest suite that exercises the data pipeline, walk-forward logic, and adapters.【F:tests/test_data_pipeline.py†L1-L160】 |
| `requirements*.txt` | Dependency pins for the application runtime and optional developer tooling.【F:requirements.txt†L1-L19】【F:requirements-dev.txt†L1-L8】 |

## Getting started
1. Create and activate a virtual environment (recommended).
2. Install the application dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Install developer and testing tools:
   ```bash
   pip install -r requirements-dev.txt
   ```

## Configuration
Provide Alpaca API credentials via environment variables, a local `.env`, or Streamlit secrets. The app reads `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `ALPACA_BASE_URL`, and `ALPACA_DATA_URL`, and will also hydrate `APCA_*` variables for SDK compatibility when secrets are present.【F:Home.py†L120-L139】【F:pages/1_Portfolios.py†L35-L51】 If credentials are missing, the data loader automatically falls back to Yahoo Finance, so you can still explore workflows with public data.【F:src/data/loader.py†L76-L134】

## Running the app
Start the Streamlit UI from the project root:
```bash
streamlit run Home.py
```
The Streamlit sidebar lists all pages in the workflow once the server starts.【F:Home.py†L112-L194】

## Running tests
Install the dev requirements and execute the test suite with Pytest:
```bash
pip install -r requirements-dev.txt
pytest
```
The tests cover data loading legs, walk-forward helpers, and logging utilities to help guard against regressions.【F:tests/test_data_pipeline.py†L1-L160】

## Data & storage locations
Downloaded OHLCV bars are cached under `storage/data/ohlcv` as parquet files so repeated simulations can reuse local data.【F:src/data/cache.py†L9-L81】 Portfolio definitions and related metadata live under `data/portfolios` (with automatic upgrades from legacy `storage/portfolios` files).【F:src/storage.py†L45-L198】 Simulation outputs and reports are written to `storage/simulations` and `storage/reports`, which drive the recents list on the home dashboard.【F:Home.py†L8-L82】

