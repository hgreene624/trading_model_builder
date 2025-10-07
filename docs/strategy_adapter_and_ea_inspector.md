# Strategy Adapter & EA Train/Test Inspector Overview

## Purpose and system context

The trading model builder exposes two Streamlit pages that work together to tune and audit the "EA" (evolutionary algorithm) workflow around the ATR breakout strategy.  The **Strategy Adapter** drives portfolio-level parameter search with an evolutionary optimizer, while the **EA Train/Test Inspector** visualizes the JSONL telemetry the optimizer emits.  Together they orchestrate:

- Discovering the training/holdout windows for a saved portfolio, including cached OHLCV coverage metadata.【F:pages/2_Strategy_Adapter.py†L44-L158】【F:src/storage.py†L206-L234】
- Launching `src.optimization.evolutionary.evolutionary_search` with UI-configured bounds and capturing live progress into Streamlit session state and holdout equity charts.【F:pages/2_Strategy_Adapter.py†L1203-L1400】【F:src/utils/holdout_chart.py†L128-L215】【F:src/optimization/evolutionary.py†L1035-L1445】
- Persisting the results (JSONL logs, best parameter sets, portfolio metadata) so the inspector page can reconstruct training details later.【F:pages/2_Strategy_Adapter.py†L1438-L1613】【F:src/storage.py†L206-L284】【F:src/storage.py†L831-L859】
- Replaying evolutionary generations, metrics, and holdout curves from those logs for post-run analysis, with optional benchmark comparisons through the tri-panel widget.【F:pages/3_EA_Train_Inspector.py†L682-L886】【F:src/utils/tri_panel.py†L1-L138】

This document walks a new project manager through the full flow, key parameters, saved artifacts, and gaps that merit follow-up.

## Strategy Adapter page

### Step 1 — Portfolio & data setup

1. **Portfolio selection**: the page only enumerates previously-saved portfolios (no ad-hoc lists) via `storage.list_portfolios`.  Loading uses `storage.load_portfolio`, which transparently upgrades legacy JSON/TXT payloads to the canonical schema.【F:pages/2_Strategy_Adapter.py†L404-L475】【F:src/storage.py†L206-L234】
2. **Ticker normalization**: `_normalize_symbols` strips headers/invalid tickers and deduplicates to build the working symbol list.【F:pages/2_Strategy_Adapter.py†L161-L193】
3. **Coverage inference**:
   - If portfolio metadata contains `per_symbol_ranges`, the helper `intersection_range` finds the overlapping ISO dates.  Otherwise the page falls back to 365 days ending today and warns the operator.【F:pages/2_Strategy_Adapter.py†L216-L317】
   - Daily cache shards and root cache path are surfaced for visibility, using `_extract_daily_shards` to parse flexible metadata structures.【F:pages/2_Strategy_Adapter.py†L54-L113】
4. **Train/holdout split**: the train percentage slider (50–95%) drives a deterministic recomputation of train/holdout boundaries, ensuring both windows are at least one day when possible.  The logic guards against metadata gaps (e.g., no overlap) and recomputes the holdout start if the proposed training end collides with coverage end.【F:pages/2_Strategy_Adapter.py†L318-L400】
5. **Session state**: the actual split fractions and dates are cached in `st.session_state` for later reuse (e.g., when refreshing the page or saving metadata back to storage).【F:pages/2_Strategy_Adapter.py†L318-L399】

### Step 2 — Strategy & evolutionary configuration

1. **Strategy choice**: the UI is currently hard-wired to `src.models.atr_breakout`.  Adding new strategies would require populating this selectbox dynamically and ensuring `_filter_params_for_strategy` exposes each strategy's accepted parameter set.【F:pages/2_Strategy_Adapter.py†L404-L520】
2. **Base parameters**: defaults (e.g., `breakout_n`, `atr_multiple`, `risk_per_trade`) are stored in session state.  Several knobs (`risk_per_trade`, `use_trend_filter`, SMA settings) are not consumed by the evolutionary run—only the keys that `_filter_params_for_strategy` retains or the EA mutates matter—so they currently serve as documentation or stubs for future strategy variants.【F:pages/2_Strategy_Adapter.py†L411-L515】【F:pages/2_Strategy_Adapter.py†L196-L236】
3. **EA meta-configuration**:
   - Population, generation count, selection method, and mutation/crossover controls live in `ea_cfg`.  These defaults match the dataclass `EAConfig` consumed by the optimizer when the page passes `ea_config_payload`.【F:pages/2_Strategy_Adapter.py†L449-L520】【F:src/optimization/evolutionary.py†L629-L733】
   - Guardrails include `min_trades`, `n_jobs`, and evaluation shuffling.  `min_trades` also governs the optimizer's gating of low-activity individuals.【F:pages/2_Strategy_Adapter.py†L640-L712】【F:src/optimization/evolutionary.py†L828-L856】
   - Parameter bounds for breakout, ATR, TP, and hold duration are fully user-editable.  When the base `entry_mode` is "dip", the UI exposes additional ranges for trend MA and dip filters; otherwise those features stay fixed at their defaults.【F:pages/2_Strategy_Adapter.py†L713-L877】
4. **Advanced section caveat**: the expander labelled "Advanced run settings" collects CV folds, starting equity, min trades, and job count again, but only `n_jobs` is reused later.  `folds` and the second `min_trades` value are legacy fields that no longer flow into the run and could confuse users.【F:pages/2_Strategy_Adapter.py†L1098-L1111】

### Step 3 — Train & monitor

1. **Progress callback**: pressing the "Train" button initializes Streamlit progress widgets and resets cached tables.  The page wires a callback `_cb` into the optimizer so that generation lifecycle events update the UI and the holdout chart in real time.【F:pages/2_Strategy_Adapter.py†L1203-L1400】
2. **Holdout chart**: `src.utils.holdout_chart.init_chart` is called with a synthetic loader (lambda returning `{}`) and an engine function that aggregates per-symbol equity series using `_portfolio_equity_curve`.  Because the loader is a stub, the chart assumes the strategy module can operate without additional cached data; integrating a proper loader would allow richer chart tooltips or instrumentation.【F:pages/2_Strategy_Adapter.py†L1240-L1261】【F:src/utils/holdout_chart.py†L128-L215】
3. **Parameter space**: the slider inputs are converted into `(min, max)` tuples for each tunable parameter, with optional dip-mode extensions.  The resulting dictionary is handed to `evolutionary_search` together with the EA config payload.【F:pages/2_Strategy_Adapter.py†L1263-L1319】【F:src/optimization/evolutionary.py†L629-L733】
4. **EA invocation and logging**:
   - Before launching, the page seeds a JSONL file under `storage/logs/ea` and writes an initial `holdout_meta` event via `TrainingLogger`.  This event captures the actual training/holdout dates, coverage statistics, tickers, and cost assumptions for later inspection.【F:pages/2_Strategy_Adapter.py†L1438-L1468】【F:src/utils/training_logger.py†L9-L41】
   - The optimizer returns up to five top candidates.  The page normalizes dip-mode payloads, renders a leaderboard, updates session state, and surfaces success messaging with train/test share summaries.  Portfolio metadata is augmented via `append_to_portfolio` so the newest run is discoverable elsewhere in the app.【F:pages/2_Strategy_Adapter.py†L1473-L1569】【F:src/storage.py†L206-L234】
5. **Persisting parameters**: an always-available "Save EA Best Params" panel writes the best genome to `storage/params/ea/<portfolio>__<strategy>.json`.  This lets downstream tooling or automation pick up EA-tuned defaults without rerunning the optimizer.【F:pages/2_Strategy_Adapter.py†L1575-L1613】【F:src/storage.py†L831-L859】

### Evolutionary search internals (for context)

Although `evolutionary_search` lives outside the Streamlit page, understanding its event model helps interpret the inspector charts:

- Fitness weights can be overridden via `storage/config/ea_fitness.json`.  When present, the optimizer logs a `fitness_config` event summarizing the weights actually used and a normalized `holdout_policy` that blends train/test scores according to configured tolerances.【F:src/optimization/evolutionary.py†L740-L801】
- Each generation emits `generation_start`, per-individual `individual_evaluated` payloads (including train/test metrics, penalties, and elapsed time), and `generation_end` aggregates (best scores, penalty stats, trade rates).  Failing evaluations produce `holdout_eval_failed`, `under_min_trades`, or `degenerate_fitness` events so the inspector can flag anomalies.【F:src/optimization/evolutionary.py†L1035-L1200】
- The run concludes with a `session_end` event that includes best-by-score and best-by-return genomes, total elapsed time, and the fitness weights in effect.【F:src/optimization/evolutionary.py†L1362-L1445】

## EA Train/Test Inspector page

### Log acquisition & normalization

1. **Discovery**: by default the page looks under `storage/logs/ea` for `*_ea.jsonl` files.  Users can pick a specific log or fall back to the most recent file.  Missing directories are created on the fly, but if the folder is empty the page aborts with an error message.【F:pages/3_EA_Train_Inspector.py†L663-L680】
2. **Parsing**: `load_ea_log` reads the JSONL file into a DataFrame, while `_eval_table` and `_gen_end_table` denormalize `individual_evaluated` and `generation_end` payloads, JSON-encoding nested dict columns to keep Streamlit caching happy.【F:pages/3_EA_Train_Inspector.py†L88-L157】
3. **Metadata fallback**: `session_meta` events are preferred for strategy/ticker/start-end context, but the page gracefully falls back to the initial `holdout_meta` emitted by the Strategy Adapter when newer logs are missing session data.  Defaults (e.g., `starting_equity`) come from these payloads.【F:pages/3_EA_Train_Inspector.py†L686-L701】

### Dashboard & playback controls

1. **Fixed windows**: The training and test date inputs are read-only and display ISO Y-M-D values derived from the log.  If the holdout window was not logged, the inspector infers a test start one day after `train_end`.【F:pages/3_EA_Train_Inspector.py†L703-L753】
2. **Generation navigation**: arrows and a slider control `ea_inspect_gen` in session state, bounded by the maximum generation discovered in the log.  The chosen generation drives the metrics dashboard and charts below.【F:pages/3_EA_Train_Inspector.py†L754-L805】
3. **Metrics dashboard**: `_render_metric_dashboard` builds a two-column layout (performance metrics + cost impact chips) when data is available.  Cost KPIs such as "Alpha Retention %" compute ratios between pre- and post-cost Sharpe/CAGR to contextualize slippage drag.【F:pages/3_EA_Train_Inspector.py†L200-L450】

### Equity curve reconstruction

1. **run_equity_curve pipeline**: a cached helper attempts to recreate the EA's equity curve in three passes:
   - Call `train_general_model` to obtain aggregate equity curves from the general trainer (ideally returning `aggregate['equity_curve']`).
   - Fallback to functions exported by `src.utils.holdout_chart` (`holdout_equity`, `simulate_holdout`, `run_holdout`), matching signature permutations dynamically.
   - As a last resort, emit a flat two-point line (triggering a UI warning about missing simulator integration).
   Debug breadcrumbs accumulate in session state for troubleshooting.【F:pages/3_EA_Train_Inspector.py†L453-L555】【F:src/models/general_trainer.py†L1-L200】【F:src/utils/holdout_chart.py†L218-L359】
2. **Charts**: the page renders two Plotly charts—the top-K individuals within the current generation and the best-by-return leader from each generation up to the selected index.  Both charts splice train and test windows at the appropriate boundary and annotate the train end.  Completely flat curves raise warnings so teams know to wire actual simulator output.【F:pages/3_EA_Train_Inspector.py†L559-L832】
3. **Tri-panel**: when a best row is available, the inspector combines train/test equity into a single Series and feeds it to `render_tri_panel`, which normalizes the curve, overlays benchmark TRI/PRI data (if available), and computes CAGR-style statistics.  Missing data is handled gracefully with logging inside `tri_panel`.【F:pages/3_EA_Train_Inspector.py†L832-L867】【F:src/utils/tri_panel.py†L1-L138】
4. **Debug expander**: the raw holdout metadata and equity provider breadcrumbs appear under an expander for quick root-cause analysis when charts fail to populate.【F:pages/3_EA_Train_Inspector.py†L869-L879】

## Produced artifacts & downstream consumers

- **EA logs**: `storage/logs/ea/<timestamp>_ea.jsonl` contains the full telemetry stream, including fitness weights, per-individual metrics, and summary payloads for each generation.  Both the Strategy Adapter and Inspector rely on `TrainingLogger`'s append-only format.【F:pages/2_Strategy_Adapter.py†L1438-L1468】【F:src/utils/training_logger.py†L9-L41】
- **Portfolio metadata**: each successful run updates `portfolio.meta['latest_training']` with coverage dates, split fractions, and a pointer to the EA log.  Other pages (e.g., Walkforward) can read this to pre-populate their own defaults.【F:pages/2_Strategy_Adapter.py†L1548-L1569】【F:src/storage.py†L206-L234】
- **Parameter files**: saving best params drops JSON payloads in `storage/params/ea`, enabling automated deployment or future EA warm starts by reading `load_strategy_params`.【F:pages/2_Strategy_Adapter.py†L1575-L1613】【F:src/storage.py†L831-L859】

## Dubious assumptions, gaps, and opportunities

1. **Stubbed data loader for holdout chart**: the Strategy Adapter initializes the holdout chart with `loader_fn=lambda **kwargs: {}`, so the equity engine never receives real price data.  This works only because `_portfolio_equity_curve` calls each strategy directly; wiring a real loader would allow consistent data hydration and facilitate multi-strategy support.【F:pages/2_Strategy_Adapter.py†L1240-L1261】
2. **Unused session hooks**: `set_config` from `holdout_chart` is imported but never called, which means late changes to equity/holdout windows after initialization would not propagate to the chart.  Either remove the import or update the callback to synchronize configuration changes.【F:pages/2_Strategy_Adapter.py†L13-L16】【F:src/utils/holdout_chart.py†L155-L181】
3. **Legacy Advanced controls**: the "CV folds" and duplicate "Min trades" inputs in the advanced section are vestiges of earlier trainers.  They risk confusing users because their values do not influence the EA run.  Refactoring should either repurpose or remove them.【F:pages/2_Strategy_Adapter.py†L1098-L1111】
4. **Single-strategy assumption**: the UI and helper `_filter_params_for_strategy` assume ATR breakout-style parameters.  Generalizing the page would require exposing per-strategy schemas (e.g., via dataclasses) and validating the saved EA params against those schemas when reloaded.【F:pages/2_Strategy_Adapter.py†L196-L236】【F:pages/2_Strategy_Adapter.py†L404-L520】
5. **Probability gate training**: `train_general_model` contains optional probability gating that trains an auxiliary model when `prob_gate_enabled` is set.  The Strategy Adapter UI never surfaces these controls, so the feature remains hidden unless parameters are preseeded manually.  Consider adding explicit toggles and validation flows if gating is part of the roadmap.【F:src/models/general_trainer.py†L117-L157】
6. **Equity curve gaps**: the inspector frequently falls back to flat curves when the trainer fails to return equity data.  This indicates missing plumbing between the EA run and the general trainer or holdout helpers.  Aligning the EA's result schema with `run_equity_curve` would eliminate confusing flat-line charts.【F:pages/3_EA_Train_Inspector.py†L453-L555】
7. **Mutation guardrails**: Dip-mode bounds default to the base params when the entry mode is "breakout", effectively locking those genes.  If the intent is to disable dip evolution entirely in breakout runs, consider hiding those controls to reduce noise; otherwise expose separate toggles to re-enable them.【F:pages/2_Strategy_Adapter.py†L449-L515】【F:pages/2_Strategy_Adapter.py†L781-L1297】

## Suggested next steps

- Integrate real data loading into the holdout chart and inspector to avoid flat-line fallbacks and to validate that the EA's results remain reproducible outside the Streamlit session.
- Audit the Advanced settings panel, removing dead inputs and clarifying which values impact the EA run.
- Add strategy discovery (e.g., scanning `src/models`) so the Strategy Adapter can target multiple models without code changes, coupled with schema-aware parameter filtering.
- Enhance the inspector to flag logs missing `session_meta` events or to prompt regeneration when telemetry is incomplete.
- Document the EA log schema formally (JSON Schema or table) so third-party tooling can consume it reliably without reverse-engineering column names from the inspector implementation.

