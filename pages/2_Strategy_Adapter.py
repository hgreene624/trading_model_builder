# pages/2_Strategy_Adapter.py
# ---------------------------------------------------------------------
# STREAMLIT BASE MODEL TRAINER PAGE
# This page is intentionally self-contained and documented so a new GPT
# instance can understand the project contracts without other context.
#
# It drives the "base model" training loop by:
#  - letting the user pick a portfolio (list of tickers from src.storage)
#  - choose the strategy (ATR breakout for now)
#  - set core params (the 80/20 set only, to avoid mismatch)
#  - calling train_general_model(...) and rendering results
#  - optionally persisting results via src.storage
#
# Dependencies on project contracts:
#   - src.storage: list_portfolios(), load_portfolio(), save_portfolio_model()
#   - src.models.general_trainer: train_general_model(...)  # or src.training.general_trainer
#   - strategy dotted path: "src.models.atr_breakout" which exposes run_strategy(...)
#   - backtest/metrics expect the BacktestResult & Trade dict shapes as defined in engine.py
# ---------------------------------------------------------------------

from __future__ import annotations

# --- Ensure project root (that contains /src and /pages) is importable ---
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ------------------------------------------------------------------------

import json
from datetime import date, timedelta

import pandas as pd
import streamlit as st

# Storage helpers (must exist in your repo)
from src.storage import (
    list_portfolios,
    load_portfolio,
    save_portfolio_model,
)

# NOTE: If you placed general_trainer in src/training/ instead, swap the import:
# from src.training.general_trainer import train_general_model
from src.models.general_trainer import train_general_model  # <- keep if trainer lives in src/models/


# ----------------------------- UI HELPERS -------------------------------

def _default_dates(years: int = 3) -> tuple[date, date]:
    """Return sensible default date range: last N years to today."""
    end = date.today()
    start = end - timedelta(days=int(365.25 * years))
    return start, end


def _format_metrics_table(per_symbol_results: list[dict]) -> pd.DataFrame:
    """
    Build a flat leaderboard DataFrame from the trainer response:
      results: [
        {"symbol": "...", "metrics": {...}, "meta": {...}, "trade_count": int},
        ...
      ]
    """
    rows = []
    for r in per_symbol_results:
        sym = r.get("symbol")
        m = r.get("metrics", {})
        rows.append({
            "symbol": sym,
            "trades": m.get("trades", 0),
            "win_rate": m.get("win_rate", 0.0),
            "expectancy": m.get("expectancy", 0.0),
            "profit_factor": m.get("profit_factor", 0.0),
            "cagr": m.get("cagr", 0.0),
            "max_drawdown": m.get("max_drawdown", 0.0),
            "calmar": m.get("calmar", 0.0),
            "sharpe": m.get("sharpe", 0.0),
            "edge_ratio": m.get("edge_ratio", 0.0),
            "entry_eff": m.get("entry_efficiency", 0.0),
            "exit_eff": m.get("exit_efficiency", 0.0),
            "avg_hold_days": m.get("avg_holding_days", 0.0),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["calmar", "profit_factor", "expectancy"], ascending=[False, False, False])
    return df


# ----------------------------- PAGE BODY --------------------------------

st.title("Base Model Trainer (ATR Breakout)")

# Left column: configuration; Right column: results
cfg_col, out_col = st.columns([1, 2], gap="large")

with cfg_col:
    st.subheader("Configuration")

    # Portfolios
    try:
        portfolios = list_portfolios()
    except Exception as e:
        portfolios = []
        st.error(f"storage.list_portfolios() failed: {e}")

    if not portfolios:
        st.warning("No portfolios found. Add one via your storage module.")
        st.stop()

    port_name = st.selectbox("Portfolio", options=portfolios, index=0)
    try:
        port = load_portfolio(port_name)
        tickers = list(port.get("tickers", []))
    except Exception as e:
        tickers = []
        st.error(f"storage.load_portfolio({port_name!r}) failed: {e}")
        st.stop()

    st.caption(f"{port_name}: {len(tickers)} tickers")
    if not tickers:
        st.warning("Selected portfolio has no tickers.")
        st.stop()

    # Strategy (dotted import path). Add more strategies over time.
    strategy_dotted = st.selectbox(
        "Strategy",
        options=["src.models.atr_breakout"],
        index=0,
        help="Module must expose run_strategy(symbol, start, end, starting_equity, params).",
    )

    # Dates & equity
    d_start, d_end = _default_dates(3)
    c1, c2 = st.columns(2)
    start = c1.date_input("Start", value=d_start)
    end = c2.date_input("End", value=d_end)
    starting_equity = st.number_input("Starting equity per symbol", min_value=100.0, max_value=10_000_000.0, value=10_000.0, step=100.0)

    st.divider()
    st.subheader("ATR Parameters (80/20)")

    # IMPORTANT: keep only params supported by ATRParams and engine to avoid mismatches
    breakout_n = st.number_input("breakout_n (lookback highs)", min_value=5, max_value=300, value=20, step=1)
    exit_n = st.number_input("exit_n (lookback lows)", min_value=4, max_value=300, value=10, step=1)
    atr_n = st.number_input("atr_n", min_value=5, max_value=60, value=14, step=1)
    atr_multiple = st.number_input("atr_multiple", min_value=0.5, max_value=10.0, value=2.0, step=0.1)
    tp_multiple = st.number_input("tp_multiple (0=disabled)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
    holding_limit = st.number_input("holding_period_limit (days, 0=disabled)", min_value=0, max_value=252, value=0, step=1)

    st.caption("Execution & costs")
    execution = st.selectbox("execution", options=["close", "next_open"], index=0)
    commission_bps = st.number_input("commission_bps (one-way)", min_value=0.0, max_value=50.0, value=0.0, step=0.1)
    slippage_bps = st.number_input("slippage_bps (one-way)", min_value=0.0, max_value=50.0, value=0.0, step=0.1)

    # Aggregate params dict passed to trainer → strategy wrapper → engine
    params = {
        "breakout_n": int(breakout_n),
        "exit_n": int(exit_n),
        "atr_n": int(atr_n),
        "atr_multiple": float(atr_multiple),
        "tp_multiple": float(tp_multiple),
        "holding_period_limit": int(holding_limit),
        "execution": execution,  # the wrapper splits this off for the engine
    }

    st.divider()
    go = st.button("Train base (portfolio) model", type="primary", use_container_width=True)

with out_col:
    st.subheader("Results")

    if go:
        with st.spinner("Running backtests across portfolio…"):
            # Train & compute metrics per symbol + aggregate
            summary = train_general_model(
                strategy_dotted=strategy_dotted,
                tickers=tickers,
                start=start,
                end=end,
                starting_equity=starting_equity,
                base_params=params,
            )

        # Cache last run in session for quick recall
        st.session_state["base_train_res"] = summary

        # Flatten to leaderboard
        lb = _format_metrics_table(summary.get("results", []))
        if lb.empty:
            st.warning("No results produced. Check data loader and date range.")
        else:
            # Pretty formatting
            num_cols = [c for c in lb.columns if c not in ("symbol",)]
            fmt = {c: "{:.4f}" for c in num_cols}
            fmt.update({"trades": "{:.0f}", "avg_hold_days": "{:.1f}"})
            st.dataframe(lb.style.format(fmt), use_container_width=True, height=420)

        # Show aggregate metrics if present
        agg = summary.get("aggregate", {}).get("metrics", {})
        if agg:
            st.caption("Aggregate (equal-weight normalized curves):")
            agg_df = pd.DataFrame([agg])
            agg_fmt = {c: "{:.4f}" for c in agg_df.columns if c not in ("trades",)}
            if "trades" in agg_df.columns:
                agg_fmt["trades"] = "{:.0f}"
            st.dataframe(agg_df.style.format(agg_fmt), use_container_width=True, height=120)

        # Raw JSON expander for debugging
        with st.expander("Raw run summary (JSON)"):
            st.code(json.dumps(summary, default=str, indent=2))

        # Optional: persist
        if st.button("Save Result to Portfolio", use_container_width=True):
            try:
                save_portfolio_model(port_name, strategy_dotted, summary)
                st.success("Saved result.")
            except Exception as e:
                st.error(f"save_portfolio_model failed: {e}")

    else:
        # If page just opened (or after previous run), show last result if present
        last = st.session_state.get("base_train_res")
        if last:
            lb = _format_metrics_table(last.get("results", []))
            n_eval = int(lb["symbol"].nunique()) if ("symbol" in lb.columns and not lb.empty) else 0
            st.info(f"Loaded last run: {n_eval} symbols evaluated. Click **Train** to run again.")
            if not lb.empty:
                num_cols = [c for c in lb.columns if c not in ("symbol",)]
                fmt = {c: "{:.4f}" for c in num_cols}
                fmt.update({"trades": "{:.0f}", "avg_hold_days": "{:.1f}"})
                st.dataframe(lb.style.format(fmt), use_container_width=True, height=360)
        else:
            st.info("Configure settings on the left, then click **Train base (portfolio) model**.")