# pages/2_Strategy_Adapter.py
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from importlib import import_module
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

# --- Page chrome ---
st.set_page_config(page_title="Strategy Adapter", layout="wide")
st.title("🧪 Strategy Adapter")

# --- Helpers ---
def _utc_now():
    return datetime.now(timezone.utc)

def _safe_import(dotted: str):
    return import_module(dotted)

def _ss_get_dict(key: str, default: Dict[str, Any]) -> Dict[str, Any]:
    if key not in st.session_state or not isinstance(st.session_state[key], dict):
        st.session_state[key] = dict(default)
    return st.session_state[key]

def _write_kv_table(d: Dict[str, Any], title: str = ""):
    if title:
        st.markdown(f"**{title}**")
    df = pd.DataFrame({"key": list(d.keys()), "value": [d[k] for k in d.keys()]})
    st.dataframe(df, width="stretch", height=min(360, 40 + 28 * len(df)))

# --- Helper to filter params for strategy ---
def _filter_params_for_strategy(strategy_dotted: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only keys the current strategy accepts (e.g., ATRParams fields)."""
    try:
        mod = _safe_import(strategy_dotted)
        keys = []
        if hasattr(mod, "ATRParams"):
            try:
                keys = list(getattr(mod, "ATRParams").__annotations__.keys())
            except Exception:
                keys = []
        if not keys and hasattr(mod, "PARAMS_ALLOWED"):
            try:
                keys = list(getattr(mod, "PARAMS_ALLOWED"))
            except Exception:
                keys = []
        if not keys:
            # sensible fallback for current ATR breakout
            keys = ["breakout_n", "exit_n", "atr_n", "atr_multiple", "tp_multiple", "holding_period_limit", "allow_short"]
        return {k: params[k] for k in keys if k in params}
    except Exception:
        # final fallback: safest minimal subset
        safe = ["breakout_n", "exit_n", "atr_n", "atr_multiple", "tp_multiple", "holding_period_limit"]
        return {k: params[k] for k in safe if k in params}

# ---------- LEFT SIDEBAR-LIKE COLUMN: configuration ----------
left, right = st.columns([1.05, 1.55], gap="large")

with left:
    st.subheader("Portfolio & Strategy")

    # Portfolio selection
    # Expecting a loader that can tell us portfolios & tickers. Fallback to text box if none.
    tickers: List[str] = []
    try:
        P = _safe_import("src.portfolios.store")
        portfolios = sorted(P.list_portfolios())  # implement in your store (already used earlier)
    except Exception:
        portfolios = []

    port_name = st.selectbox("Portfolio", options=(["<custom>"] + portfolios) if portfolios else ["<custom>"], index=0)
    if port_name == "<custom>":
        tickers_csv = st.text_input("Tickers (CSV)", "AAPL,MSFT")
        tickers = [t.strip().upper() for t in tickers_csv.split(",") if t.strip()]
    else:
        try:
            tickers = list(_safe_import("src.portfolios.store").load_portfolio(port_name))
        except Exception as e:
            st.error(f"Failed to load portfolio '{port_name}': {e}")
            st.stop()

    if not tickers:
        st.warning("No tickers selected. Add tickers or choose a portfolio.")
        st.stop()

    st.info(f"Selected **{len(tickers)}** symbols.")

    # Strategy module (kept as before)
    strategy_dotted = st.selectbox("Strategy", ["src.models.atr_breakout"], index=0)

    # Base params (kept as before)
    st.caption("Base parameters passed to the strategy’s run_strategy().")
    base = _ss_get_dict(
        "adapter_base_params",
        {
            "breakout_n": 14,
            "exit_n": 6,
            "atr_n": 14,
            "atr_multiple": 2.0,
            "tp_multiple": 0.5,
            "holding_period_limit": 5,
            # extras (the strategy ignores unknowns; we keep them for future work)
            "risk_per_trade": 0.005,
            "use_trend_filter": False,
            "sma_fast": 20,
            "sma_slow": 50,
            "sma_long": 200,
            "long_slope_len": 20,
            "cost_bps": 1.0,
            "execution": "close",
        },
    )

    with st.expander("Base params", expanded=False):
        base["breakout_n"] = st.number_input("breakout_n", 5, 300, base["breakout_n"], 1, help="Lookback used for breakout entry signal.")
        base["exit_n"] = st.number_input("exit_n", 4, 300, base["exit_n"], 1, help="Lookback used for breakout exit/stop logic.")
        base["atr_n"] = st.number_input("atr_n", 5, 60, base["atr_n"], 1, help="ATR window length.")
        base["atr_multiple"] = st.number_input("atr_multiple", 0.5, 10.0, float(base["atr_multiple"]), 0.1, help="ATR multiple for stop distance.")
        base["tp_multiple"] = st.number_input("tp_multiple", 0.2, 10.0, float(base["tp_multiple"]), 0.1, help="Take-profit multiple (vs ATR or entry logic).")
        base["holding_period_limit"] = st.number_input("holding_period_limit", 1, 400, base["holding_period_limit"], 1, help="Max bars to hold a position.")
        # Keep the extras so future strategies/UI can reuse
        base["risk_per_trade"] = st.number_input("risk_per_trade", 0.0005, 0.05, float(base["risk_per_trade"]), 0.0005, format="%.4f", help="Fraction of equity risked per trade.")
        base["use_trend_filter"] = st.checkbox("use_trend_filter", value=bool(base["use_trend_filter"]), help="Optional trend filter gate.")
        base["sma_fast"] = st.number_input("sma_fast", 5, 100, base["sma_fast"], 1, help="Fast MA length (if trend filter used).")
        base["sma_slow"] = st.number_input("sma_slow", 10, 200, base["sma_slow"], 1, help="Slow MA length (if trend filter used).")
        base["sma_long"] = st.number_input("sma_long", 100, 400, base["sma_long"], 1, help="Long MA length (if trend filter used).")
        base["long_slope_len"] = st.number_input("long_slope_len", 5, 60, base["long_slope_len"], 1, help="Slope window for long MA trend check.")
        base["cost_bps"] = st.number_input("cost_bps", 0.0, 20.0, float(base["cost_bps"]), 0.1, help="Per-trade cost (basis points).")
        base["execution"] = st.selectbox("Execution", ["close"], index=0, help="Execution price proxy used in backtest.")

    # General training knobs
    folds = st.number_input("CV folds", 2, 10, 4, 1, help="Cross-validation splits for the base model trainer.")
    equity = st.number_input("Starting equity ($)", 1000.0, 1_000_000.0, 10_000.0, 100.0, help="Starting equity for per-symbol runs.")
    min_trades = st.number_input("Min trades (valid)", 0, 200, 2, 1, help="Minimum total trades needed for a run to be considered valid.")

    # Parallelism controls (kept visible near the top)
    max_procs = os.cpu_count() or 8
    n_jobs = st.slider("Jobs (processes)", 1, max(1, max_procs - 1), min(8, max(1, max_procs - 1))),  # noqa
    n_jobs = int(n_jobs[0])

    st.caption("Tip: If Alpaca SIP limits or YF rate limits bite, reduce Jobs to 1–4.")

# ---------- RIGHT COLUMN: actions + results ----------
with right:
    # ------------------------ BASE TRAINING ------------------------
    st.subheader("Train Base Model")
    run_btn = st.button("🚀 Train (portfolio)", type="primary", help="Runs the portfolio-level base trainer with the config on the left.", use_container_width=False)

    st.divider()
    st.subheader("Results")

    if run_btn:
        if not tickers:
            st.error("This portfolio has no tickers.")
            st.stop()

        # resolve modules
        try:
            loader = _safe_import("src.data.loader")
            trainer = _safe_import("src.models.general_trainer")
            metrics = _safe_import("src.backtest.metrics")
        except Exception as e:
            st.error(f"Import error: {e}")
            st.stop()

        prog = st.progress(0.0, text="Starting training…")
        status = st.empty()

        try:
            # For the base trainer we’ll do a simple portfolio CV using your existing general_trainer
            # Over the past N years (controlled below), but keep minimal knobs here — your prior flow.
            end = _utc_now()
            start = end - timedelta(days=365)  # keep same horizon as your earlier page; tweak if you prefer

            status.write("Loading data…")
            prog.progress(0.1)

            # The general trainer consumes tickers, period, base params; it aggregates per-symbol metrics.
            status.write("Running trainer…")
            prog.progress(0.4)

            clean_base = _filter_params_for_strategy(strategy_dotted, base)
            res = trainer.train_general_model(
                strategy_dotted=strategy_dotted,
                tickers=tickers,
                start=start,
                end=end,
                starting_equity=float(equity),
                base_params=clean_base,
            )

            prog.progress(0.8)
            st.session_state["base_train_res"] = res

            # Leaderboard-like view
            agg = (res or {}).get("aggregate", {}) or {}
            agg_metrics = agg.get("metrics", {}) or {}
            per = (res or {}).get("results", []) or []

            status.write("Rendering results…")
            if per:
                rows = []
                for r in per:
                    sym = r.get("symbol")
                    m = (r.get("metrics") or {})
                    rows.append({
                        "symbol": sym,
                        "trades": m.get("trades", 0),
                        "total_return": round(float(m.get("total_return", 0.0)), 4),
                        "cagr": round(float(m.get("cagr", 0.0)), 4),
                        "sharpe": round(float(m.get("sharpe", 0.0)), 3),
                        "calmar": round(float(m.get("calmar", 0.0)), 3),
                        "max_dd": round(float(m.get("max_drawdown", 0.0)), 4),
                        "expectancy": round(float(m.get("expectancy", 0.0)), 4),
                    })
                lb = pd.DataFrame(rows).sort_values(["total_return", "sharpe"], ascending=[False, False])
                st.dataframe(lb, width="stretch", height=360)
            else:
                st.warning("No per-symbol result rows generated.")

            # Aggregate metrics summary
            if agg_metrics:
                _write_kv_table(agg_metrics, title="Aggregate metrics (equal-weight portfolio curve)")
            else:
                st.info("No aggregate metrics computed.")

            status.write("Done.")
            prog.progress(1.0)

        except Exception as e:
            st.error(f"Training failed: {e}")
            st.stop()

    # ------------------------ WALK-FORWARD ------------------------
    st.markdown("---")
    st.subheader("Walk-Forward Validation")

    wf_cfg = _ss_get_dict(
        "wf_cfg",
        {
            "days_back": 365 * 3,  # 3y default
            "splits": 3,
            "train_days": 252,
            "test_days": 63,
            "step_days": 0,       # 0 => use test_days
            "use_ea": False,
            "ea_generations": 4,
            "ea_pop": 12,
            "ea_min_trades": 3,
        },
    )
    with st.expander("Walk-Forward params", expanded=False):
        wf_cfg["days_back"] = st.number_input("History window (days)", 252, 365 * 10, wf_cfg["days_back"], 1,
                                              help="Total lookback used to build WF splits (train+test across splits).")
        wf_cfg["splits"] = st.number_input("Splits", 1, 12, wf_cfg["splits"], 1,
                                           help="Number of walk-forward segments.")
        wf_cfg["train_days"] = st.number_input("Train window (days)", 60, 600, wf_cfg["train_days"], 1,
                                               help="In-sample training window per split.")
        wf_cfg["test_days"] = st.number_input("Test window (days)", 21, 252, wf_cfg["test_days"], 1,
                                              help="Out-of-sample evaluation window per split.")
        wf_cfg["step_days"] = st.number_input("Step size (days)", 0, 600, wf_cfg["step_days"], 1,
                                              help="Advance between splits. 0 = same as Test window.")
        wf_cfg["use_ea"] = st.checkbox("Use EA inside each split (opt params on IS)", value=bool(wf_cfg["use_ea"]))
        col_ea1, col_ea2, col_ea3 = st.columns(3)
        with col_ea1:
            wf_cfg["ea_generations"] = st.number_input("EA generations", 1, 50, wf_cfg["ea_generations"], 1)
        with col_ea2:
            wf_cfg["ea_pop"] = st.number_input("EA population", 2, 200, wf_cfg["ea_pop"], 1)
        with col_ea3:
            wf_cfg["ea_min_trades"] = st.number_input("EA min trades", 0, 200, wf_cfg["ea_min_trades"], 1)

    run_wf = st.button("🧭 Run Walk-Forward", type="secondary", help="Runs walk-forward validation on the selected tickers.", use_container_width=False)

    if run_wf:
        try:
            from src.optimization.walkforward import walk_forward
            end = _utc_now()
            start = end - timedelta(days=int(wf_cfg["days_back"] or 365))
            step_days = int(wf_cfg["step_days"] or 0) or int(wf_cfg["test_days"])

            # Use best available params:
            # 1) if you just trained in this session, pull that
            # 2) otherwise fall back to current base params
            use_params = _filter_params_for_strategy(strategy_dotted, base)
            res = st.session_state.get("base_train_res")
            if isinstance(res, dict):
                # If your trainer later includes a suggested best param-set, plug it here.
                # For now we keep the base as the walk-forward starting point.
                pass

            st.info("Starting walk-forward…")
            prog_wf = st.progress(0.05, text="Loading splits & data…")

            out = walk_forward(
                strategy_dotted=strategy_dotted,
                tickers=tickers,
                start=start,
                end=end,
                starting_equity=float(equity),
                base_params=use_params,
                splits=int(wf_cfg["splits"]),
                train_days=int(wf_cfg["train_days"]),
                test_days=int(wf_cfg["test_days"]),
                step_days=int(step_days),
                use_ea=bool(wf_cfg["use_ea"]),
                ea_kwargs=dict(
                    generations=int(wf_cfg["ea_generations"]),
                    pop_size=int(wf_cfg["ea_pop"]),
                    min_trades=int(wf_cfg["ea_min_trades"]),
                    n_jobs=int(n_jobs),
                ),
                n_jobs=int(n_jobs),
                progress_cb=lambda evt, ctx: None,  # keep console quiet; UI below
            )

            prog_wf.progress(0.7, text="Rendering results…")

            # Expecting the structure returned by your tested walkforward.py
            # { "splits": [...], "summary": {...}, "artifacts": {... optional ...} }
            summary = (out or {}).get("summary", {}) or {}
            splits = (out or {}).get("splits", []) or []
            artifacts = (out or {}).get("artifacts", {}) or {}

            # Summary block
            if summary:
                st.markdown("**Walk-Forward (OOS mean) summary**")
                st.dataframe(
                    pd.DataFrame({"metric": list(summary.keys()), "value": [summary[k] for k in summary.keys()]}),
                    width="stretch",
                    height=320,
                )
            else:
                st.warning("No summary metrics were returned.")

            # First split preview (mirrors your test output)
            if splits:
                first = splits[0]
                st.markdown("**First split preview**")
                meta = first.get("meta", {})
                st.code(
                    f"dates: {meta.get('train_start')} → {meta.get('train_end')} | "
                    f"OOS: {meta.get('test_start')} → {meta.get('test_end')}\n"
                    f"params: {first.get('params')}\n"
                    f"oos trades: {first.get('metrics', {}).get('trades', 'N/A')}",
                    language="text",
                )

            # Artifacts / anomalies (if your walkforward attaches them)
            if artifacts:
                st.markdown("**WF Artifacts / Anomalies**")
                _write_kv_table(artifacts)

            prog_wf.progress(1.0, text="Walk-Forward complete.")

        except Exception as e:
            st.error(f"Walk-forward failed: {e}")
            st.stop()