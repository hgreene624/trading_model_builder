# Specify: Alpaca Trader Research Dashboard

## Summary
- Multi-page Streamlit research workspace for designing, tuning, and auditing ATR-based breakout strategies across user-defined US equity universes.
- Wraps reusable data ingestion, storage, backtesting, and evolutionary optimization packages that can run headless for scripting or testing.
- Persists artifacts (portfolios, logs, parameter sets, simulations) under a predictable storage tree so UI pages, CLI utilities, and automated jobs can share state.

## Product context
- **Purpose**: Provide a self-hosted sandbox for quants to curate stock universes, evolve breakout strategy parameters, and review historical performance before automating live trading.
- **Personas**: Quant researchers building daily breakout systems; engineers integrating Alpaca market data; PMs reviewing optimizer telemetry and simulation outputs.
- **Environments**: Local Streamlit server for analysts, optional CI for pytest verification, and CLI scripts for offline log analysis.

## Primary workflows
1. **Environment overview (Home page)**
   - Loads `.env`/environment secrets and reports Alpaca trading API connectivity, falling back to Yahoo-only workflows when credentials are missing.【F:Home.py†L14-L43】
   - Surfaces counts and tables for saved portfolios, recent simulations, and base model specs discovered under `storage/` to guide navigation.【F:Home.py†L96-L147】
2. **Portfolio curation (`pages/1_Portfolios.py`)**
   - Imports index constituents from cached Wikipedia scrapes, applies name/sector filters, enforces a portfolio cap, and normalizes tickers before saving.【F:pages/1_Portfolios.py†L44-L204】【F:pages/1_Portfolios.py†L296-L331】
   - Prefetches OHLCV data via the unified loader, computes median price and dollar volume over a configurable priors window, and records per-symbol coverage plus shard metadata for reproducibility.【F:pages/1_Portfolios.py†L332-L415】【F:pages/1_Portfolios.py†L424-L472】
   - Persists curated universes with provenance (filters, coverage ranges, cache roots) through `storage.save_portfolio`, upgrading legacy payloads automatically.【F:pages/1_Portfolios.py†L473-L520】【F:src/storage.py†L92-L198】
3. **Strategy adaptation & training (`pages/2_Model_Builder.py`)**
   - Loads saved portfolios, infers overlapping historical coverage, and proposes train/holdout splits anchored to cached shards.【F:pages/2_Model_Builder.py†L404-L520】【F:src/data/portfolio_prefetch.py†L54-L139】
   - Exposes base ATR breakout and dip-overlay parameters, editable EA bounds, and profile presets that hydrate Streamlit controls from JSON templates.【F:pages/2_Model_Builder.py†L210-L359】【F:pages/2_Model_Builder.py†L400-L486】
   - Runs `optimization.evolutionary_search` with live progress callbacks, renders holdout equity via Plotly, and logs telemetry/metadata through `TrainingLogger` for downstream inspection.【F:pages/2_Model_Builder.py†L1203-L1400】【F:src/optimization/evolutionary.py†L629-L856】【F:src/utils/training_logger.py†L9-L56】
   - Saves winning genomes and training summaries back to `storage` (portfolios, params, logs) so future sessions and automation can reuse tuned configurations.【F:pages/2_Model_Builder.py†L1438-L1613】【F:src/storage.py†L206-L284】【F:src/storage.py†L831-L859】
4. **EA log inspection (`pages/3_Model_Inspector.py`)**
   - Discovers JSONL logs under `storage/logs/ea`, normalizes `session_meta`, `holdout_meta`, and per-generation events into cached DataFrames, and lets users scrub by generation.【F:pages/3_Model_Inspector.py†L82-L205】【F:pages/3_Model_Inspector.py†L663-L805】
   - Reconstructs equity curves via `models.general_trainer`/holdout helpers, overlays benchmarks using the tri-panel widget, and highlights cost drag, trade counts, and rolling performance.【F:pages/3_Model_Inspector.py†L453-L867】【F:src/models/general_trainer.py†L90-L200】【F:src/utils/tri_panel.py†L1-L138】
   - Provides debug expanders with raw metadata and loader breadcrumbs to diagnose missing telemetry or flat equity plots.【F:pages/3_Model_Inspector.py†L869-L879】
5. **Data ingestion & caching**
   - `src.data.loader.get_ohlcv` normalizes bar data, prioritizes Alpaca’s historical API with disk/RAM caches, and falls back to Yahoo Finance when the primary provider fails or credentials are absent.【F:src/data/loader.py†L15-L134】【F:src/data/loader.py†L173-L271】
   - Provider adapters handle authentication, retries, and normalization independently so downstream workflows can rely on consistent OHLCV frames.【F:src/data/alpaca_data.py†L18-L93】【F:src/data/yf.py†L14-L136】
6. **Backtesting & simulation primitives**
   - `src/backtest/engine` implements the ATR breakout engine, warmup caching, and detailed trade cost accounting; results feed both the EA optimizer and manual scripts.【F:src/backtest/engine.py†L1-L158】【F:src/backtest/engine.py†L334-L420】
   - `models/general_trainer` aggregates per-symbol equity curves, computes metrics, and optionally trains probability gates for trade filtering across portfolios.【F:src/models/general_trainer.py†L1-L174】
7. **Artifact management & tooling**
   - `src/storage` unifies portfolio, parameter, simulation, and metadata persistence with atomic writes and legacy upgrade helpers; `list_*` utilities power discovery throughout the app.【F:src/storage.py†L21-L200】【F:src/storage.py†L593-L859】
   - CLI utilities under `scripts/` replicate inspector flows (log review, weight tuning, smoke backtests) for offline or automated environments.【F:scripts/ea_log_inspect.py†L1-L75】【F:scripts/ea_weight_tuner.py†L1-L49】

## Integrations & dependencies
- **Data providers**: Alpaca (primary) via `alpaca-py` and Yahoo Finance (fallback) with configurable retries and provider overrides.【F:src/data/alpaca_data.py†L18-L93】【F:src/data/yf.py†L71-L136】
- **Frameworks**: Streamlit for UI, Plotly for visualization, pandas/numpy for analytics, and scikit-learn compatible interfaces inside the evolutionary optimizer.
- **Secrets/config**: Environment variables or Streamlit secrets supply Alpaca credentials; additional JSON configs (EA fitness weights, parameter profiles) live under `storage/config` and `storage/params`.

## Storage & data layout
- `storage/data/ohlcv/<provider>/<SYMBOL>/<TF>/YYYY-MM-DD__YYYY-MM-DD.parquet` — cached bars shared across workflows.【F:src/data/loader.py†L191-L236】
- `storage/portfolios/*.json` — curated universes with filters, coverage, and shard metadata; legacy `.txt`/`.parquet` files auto-upgraded on load.【F:src/storage.py†L92-L198】
- `storage/logs/ea/*.jsonl` — append-only optimizer telemetry consumed by the inspector and CLI tools.【F:src/utils/training_logger.py†L9-L56】
- `storage/params/ea/*.json` — saved EA genomes for reuse; `storage/simulations` and `storage/reports` capture downstream scenario runs surfaced on Home.【F:src/storage.py†L593-L859】【F:Home.py†L102-L141】

## Observability & analytics
- `TrainingLogger` standardizes structured logging (session meta, generation events, leaderboard summaries) and timestamps for downstream analysis.【F:src/utils/training_logger.py†L9-L56】
- Streamlit dashboards compute KPI chips (Sharpe, CAGR, alpha retention, trade hit rates) and provide rolling-return heatmaps for holdout analysis.【F:pages/2_Model_Builder.py†L1203-L1525】【F:pages/3_Model_Inspector.py†L559-L867】
- CLI scripts mirror inspector capabilities to allow Git-based reviews or automated anomaly detection on EA runs.【F:scripts/ea_log_inspect.py†L1-L75】

## Testing & quality
- Pytest suite validates loader normalization, provider selection, and date handling under `tests/test_data_pipeline.py`; additional fixtures support offline validation once parquet shards are supplied.【F:tests/test_data_pipeline.py†L1-L132】【F:tests/test_data/full_model/AGENTS.md†L1-L60】
- `pytest.ini` configures test discovery; developer dependencies (pytest, black, mypy, etc.) are declared in `requirements-dev.txt` to support linting and static checks.【F:pytest.ini†L1-L8】【F:requirements-dev.txt†L1-L8】

## Known limitations & assumptions
- Holdout chart initialization currently stubs loader callbacks, so equity visualizations rely on strategy modules returning complete curves; wiring real data loaders would improve fidelity.【F:pages/2_Model_Builder.py†L1240-L1281】
- Advanced EA controls include legacy fields (`folds`, duplicate `min_trades`) that no longer influence optimizer runs but remain in the UI, potentially confusing operators.【F:pages/2_Model_Builder.py†L1076-L1111】
- Inspector fallbacks emit flat equity lines when logs lack curve data, signaling missing simulator plumbing between EA outputs and holdout reconstruction helpers.【F:pages/3_Model_Inspector.py†L453-L555】

## Appendices
- **Automation hooks**: `scripts/repro_env_reset.py` and `scripts/run_smoke_backtest.py` simplify refreshing local caches and validating backtest engines outside Streamlit.【F:scripts/repro_env_reset.py†L1-L123】【F:scripts/run_smoke_backtest.py†L1-L200】
- **Future extensions**: Probability-gated training paths exist in `models.general_trainer`, but UI toggles are not yet exposed; integrating them would require surfacing gate parameters and model serialization policies.【F:src/models/general_trainer.py†L117-L166】
