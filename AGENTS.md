# AGENTS.md

This file is for AI coding agents (OpenAI Codex/Copilot agent/Cursor) and human contributors. It describes how to work safely in this repo and what *not* to break. When in doubt: keep changes small, ask for the exact files/tracebacks, and preserve the workflow from data → strategy → optimization → UI.

---

## 1) Project purpose & entrypoints
- **Goal:** Streamlit-based research environment for an **ATR breakout** strategy with backtests, evolutionary tuning (EA), and walk-forward validation.
- **Root app:** `Home.py` (sets page config, renders dashboards, checks Alpaca connectivity). Only call Streamlit APIs when the script is executed by Streamlit.
- **Pages (workflow order):**
  - `pages/1_Portfolios.py` – index/universe filters, OHLCV fetching, liquidity metrics, curated-portfolio save.
  - `pages/2_Strategy_Adapter.py` – **EA scope only** (evolutionary bounds, param save/load). *Walk-forward is not here.*
  - `pages/3_Ticker_Selector_and_Tuning.py` – per‑ticker tuning helpers.
  - `pages/3_Walkforward.py` – rolling train/test splits, optional “Use EA inside each split,” parameter re‑optimization.
  - `pages/4_Simulate_Portfolio.py` – multi-asset scheduling, allocations, diagnostics/Plotly charts.
- **Core libs:**
  - `src/backtest/` – `engine.py` (entries/exits incl. `wilder_atr`), `metrics.py` (CAGR, Sharpe, DD, expectancy, edge ratio).
  - `src/models/` – `atr_breakout.py` wrapper, `general_trainer.py` orchestrator.
  - `src/optimization/` – `evolutionary.py`, `walkforward.py`, `tuning/evolve.py`.
  - `src/data/` – loaders, caching, memory cache.
  - `src/storage.py` – JSON artifacts (portfolios, params, runs).

## 2) Python & dependencies
- **Python:** 3.12 (project uses 3.10+ syntax like `str | Path`).
- **Install:**
  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install -U pip
  pip install -r requirements.txt
  # optional dev extras
  pip install -r requirements-dev.txt || true
  ```

## 3) Run & test
```bash
# App
streamlit run Home.py

# Static checks
ruff check .
ruff format --check .
mypy src || true   # advisory unless CI enforces

# Tests / smoke
pytest -q || true
python scripts/smoke_data.py || true
```

## 4) Environment configuration
- **Credentials lookup (Alpaca):** Streamlit `secrets.toml` → environment variables. *Never* hardcode.
- **Endpoints:** default to Alpaca **paper** endpoint unless told otherwise.
- **Provider toggles:** `ALPACA_FEED`, `DATA_PROVIDER` influence loader choice (Alpaca first, Yahoo fallback). If missing/invalid, degrade gracefully and log a single-line warning.

## 5) Data loader behavior & caching
- **Layers:** in‑RAM memo + on-disk cache under
  `storage/data/ohlcv/<provider>/<SYMBOL>/<TF>/YYYY-MM-DD__YYYY-MM-DD.parquet`.
- **Provider order:** **Alpaca → Yahoo** fallback. Preserve semantics when extending.
- **Cache invalidation:** only when symbol params/timeframe/provider changes or data coverage is stale; do not spam deletes.
- **Path hygiene:** ensure parent dirs exist **before** writes to avoid `FileNotFoundError`.
- **Logging:** emit one concise summary per batch, avoid per‑symbol spam during EA/WF.

## 6) Persistent storage conventions
- Centralize JSON writes/reads in `src/storage.py`. It normalizes portfolio files and upgrades legacy schemas.
- Data root can be overridden by `DATA_DIR`.
- Artifacts:
  - **Portfolios** (curated universes, liquidity metadata).
  - **Strategy params** (EA best params per portfolio).
  - **Simulations/backtests** (run metadata, metrics, curves).
- Do **not** change on-disk schema or filenames without an explicit migration plan.

## 7) Streamlit page patterns
- **Page ordering:** numeric filename prefixes.
- **Caching:** use `st.cache_data` for expensive calls (clear only if inputs change).
- **Widget keys:** all interactive widgets must have **unique** keys; avoid collisions across reruns.
- **State resets:** “Save” buttons must persist via `storage.py` **without** wiping `st.session_state`.
- **Charts:** Plotly in UI; precompute heavy work in `src/*` modules.
- **EA Train Inspector:** ensure equity‑curve provider is wired; flat line means trainer/holdout curve wasn’t returned.

## 8) Simulation & analytics helpers
- Simulation expects cached OHLCV already pulled by Portfolios.
- Technical overlays are computed locally prior to scheduling entries/exits.
- Preserve this contract—don’t fetch data inside tight UI loops.

## 9) Backtest & optimization modules
- **ATR engine** contract: input OHLCV with required columns; pure‑function outputs (trades, pnl, equity, MFE/MAE). Do **not** change signatures of `wilder_atr` or `backtest_atr_breakout`.
- **Evolutionary search:** penalty‑aware fitness, multiprocessing, `TrainingLogger` emits append‑only JSONL; CLI inspectors depend on this.
- **Walk‑forward:** `walkforward.py` handles rolling splits; optional per‑split EA. Keep public function args stable (e.g., `walk_forward(...)`—no surprise kwargs).

## 10) Utilities & scripts
- Inspectors: `scripts/ea_log_inspect.py`, `scripts/wf_result_inspect.py` (assume JSON/JSONL shapes from loggers).
- Streamlit theme overrides: `.streamlit/config.toml`.

## 11) Test coverage expectations
- Cover: loader fallbacks, cache path creation, timezone conversions, optimization penalties, and logging throttling.
- Prefer fast unit tests; long EA/WF can be sampled or mocked.

---

# CODING SAFEGUARDS & PERSONAL PREFERENCES (MUST READ)

## A) Top‑down, incremental approach
1. **Start at the symptom:** reproduce with the exact page/script the user ran.
2. **Trace callers:** data → engine → metrics → models/trainer → optimization → UI.
3. **Minimal diffs:** fix the narrowest layer possible; avoid repo‑wide refactors.

## B) Change safety for every edit (PR template)
- **Why** the change is needed.
- **Impact map:** files/functions touched + who calls them.
- **Minimal diff / code patch.**
- **Test/check commands** to validate.
- **Rollback**: how to revert (config flag or one‑line revert).

## C) Debug protocol (when an error is shared)
- Identify the likely cause and **the exact function/line** to inspect.
- Provide a **minimal patch**.
- Include a **quick sanity check** command to run locally.

## D) My preferences (do these)
- **No blind changes.** Ask for the **specific files** or **full tracebacks** before proposing edits.
- **No giant rewrites.** If a file must be replaced, keep it under ~100 lines or give *precise* patch instructions.
- **Silence noisy logs** (e.g., repetitive `[loader] ...` per‑symbol lines); retain a single batch summary.
- **UTC & tz rules:** never call `pd.Timestamp(..., tz="UTC")` when value already has tzinfo; use `tz_convert("UTC")` instead.
- **Paths:** always `mkdir(parents=True, exist_ok=True)` before writing under `storage/data/...`.
- **Streamlit state:** “Save” actions must not reset the page; don’t clear `st.session_state` unless requested.
- **EA vs Walk‑forward split:** keep walk‑forward code **out of** `2_Strategy_Adapter.py`. Save EA params per‑portfolio; load them later in `3_Walkforward.py`.
- **Graphs/windows:** on strategy charts, exclude pre‑trade dead zones from equity curve visuals **only** if they are truly outside the feasible trading window; otherwise show where trades were possible but absent.
- **Public API stability:** don’t rename `evolutionary_search`, `walk_forward`, or change their argument contracts without coordination.

## E) “Do‑Not‑Touch” list (without explicit approval)
- Function signatures: `wilder_atr`, `backtest_atr_breakout`, `evolutionary_search`, `walk_forward`.
- On‑disk artifact schema under `storage/` (file names/keys).
- Existing `st.session_state` keys and page routing.
- Trainer/logger JSONL shapes consumed by CLI tools and UI inspectors.

## F) Common foot‑guns & their fixes
- **Flat equity curves in EA inspector:** wire the **holdout/trainer curve provider** in `run_equity_curve()`; verify trainer returns a curve for the selected run.
- **Parquet write errors:** ensure directories exist **before** saving cache files.
- **Duplicate Streamlit keys:** add explicit `key="..."` everywhere and dedupe across conditional branches.
- **Walk‑forward zero results:** confirm “Use EA inside each split” flag is read and params forwarded; don’t pass unsupported kwargs (e.g., `ea_kwargs`) unless `walk_forward(...)` accepts them.

## G) Commit/PR rules (for agents)
- Keep diffs small (≤ ~200 lines).
- Include the safety checklist from **B** in the PR description.
- Link to the console error/traceback you addressed.
- If you changed logs, paste before/after examples.

---

## 12) Optional nested AGENTS.md
Agents read the **closest** AGENTS.md. You may add:
- `pages/AGENTS.md` – UI‑only rules (widgets, state, charts; no business logic).
- `src/optimization/AGENTS.md` – EA vs Walk‑forward responsibilities, stable function signatures, logging expectations.

## 13) Quick commands (copy/paste)
```bash
# App
streamlit run Home.py

# Format & lint
ruff format . && ruff check .

# Create cache dirs if missing (safe)
python - <<'PY'
from pathlib import Path
for p in ["storage/data/ohlcv", "storage/logs/wf", "storage/logs/ea"]:
    Path(p).mkdir(parents=True, exist_ok=True)
print("ok")
PY
```

---

**Contact & expectations:** When proposing non‑trivial edits, ask for the exact files and full tracebacks first. Keep it incremental, keep it reversible, and do not break working pages.
