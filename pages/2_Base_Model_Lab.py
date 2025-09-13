# pages/2_Base_Model_Lab.py
from __future__ import annotations

import os
from datetime import datetime
import io
import json

import pandas as pd
import streamlit as st

from src.storage import (
    list_portfolios,
    load_portfolio,
    list_portfolio_models,
    save_portfolio_model,
    save_training_log,
)
from src.models.general_trainer import train_general_model


st.set_page_config(page_title="Base Model Lab", layout="wide")
st.title("📦 Base Model Lab (Portfolio-level)")

# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────
def _default_params() -> dict:
    """
    Sane defaults for the ATR breakout-style strategy. These are the *base model*
    params used for CV evaluation. You can tweak them here or in the UI.
    """
    return {
        "breakout_n": 55,
        "exit_n": 35,
        "atr_n": 14,
        "atr_multiple": 2.5,
        "tp_multiple": 2.2,
        "holding_period_limit": 120,
        "risk_per_trade": 0.008,  # 0.8%
        "use_trend_filter": True,
        "sma_fast": 20,
        "sma_slow": 60,
        "sma_long": 200,
        "long_slope_len": 20,
        "cost_bps": 2.0,
        # optional gates if supported by your engine:
        "chop_max": 55,
        "atr_ratio_max": 1.8,
        # execution policy (engine reads this); default = "close"
        "execution": "close",
    }


def _strategy_choices() -> list[str]:
    # Add more dotted strategy modules as you add adapters.
    return ["src.models.atr_breakout"]


# ────────────────────────────────────────────────────────────────────────────────
# Layout
# ────────────────────────────────────────────────────────────────────────────────
left, right = st.columns([1.0, 1.4], gap="large")

with left:
    st.subheader("Portfolio & Training")

    ports = list_portfolios()
    if not ports:
        st.warning("No portfolios found. Create one on the **Portfolios** page first.")
        st.stop()

    port_name = st.selectbox("Portfolio", options=ports, index=0, key="bm_port")
    port_obj = load_portfolio(port_name) or {}
    tickers = list(port_obj.get("tickers", []))
    st.info(f"Selected portfolio **{port_name}** with **{len(tickers)}** tickers.")

    # Strategy + base params
    strategy_dotted = st.selectbox("Strategy", _strategy_choices(), index=0, key="bm_strategy")

    st.markdown("**Base Model Parameters**")
    with st.expander("Edit base params (used for CV evaluation)", expanded=False):
        p = _default_params()
        # Integers
        p["breakout_n"] = st.number_input("breakout_n (entry lookback)", 5, 300, p["breakout_n"], 1)
        p["exit_n"] = st.number_input("exit_n (exit lookback)", 4, 300, p["exit_n"], 1)
        p["atr_n"] = st.number_input("atr_n", 5, 60, p["atr_n"], 1)
        p["sma_fast"] = st.number_input("sma_fast", 5, 100, p["sma_fast"], 1)
        p["sma_slow"] = st.number_input("sma_slow", 10, 200, p["sma_slow"], 1)
        p["sma_long"] = st.number_input("sma_long", 100, 400, p["sma_long"], 1)
        p["long_slope_len"] = st.number_input("long_slope_len", 5, 60, p["long_slope_len"], 1)
        p["holding_period_limit"] = st.number_input("holding_period_limit (days)", 5, 400, p["holding_period_limit"], 1)

        # Floats
        p["atr_multiple"] = st.number_input("atr_multiple", 0.5, 10.0, float(p["atr_multiple"]), 0.1)
        p["tp_multiple"] = st.number_input("tp_multiple", 0.5, 10.0, float(p["tp_multiple"]), 0.1)
        p["risk_per_trade"] = st.number_input(
            "risk_per_trade (fraction of equity)",
            0.0005, 0.05, float(p["risk_per_trade"]), 0.0005, format="%.4f"
        )
        p["cost_bps"] = st.number_input("cost_bps (one-way)", 0.0, 20.0, float(p["cost_bps"]), 0.1)

        # Booleans
        p["use_trend_filter"] = st.checkbox("use_trend_filter", value=bool(p["use_trend_filter"]))

        # Optional gates if supported
        p["chop_max"] = st.number_input("chop_max (optional)", 0, 100, int(p.get("chop_max", 55)), 1)
        p["atr_ratio_max"] = st.number_input("atr_ratio_max (optional)", 0.1, 5.0, float(p.get("atr_ratio_max", 1.8)), 0.1)

        # Execution policy (close is default per your decision)
        p["execution"] = st.selectbox("Execution policy", ["close"], index=0)

    # CV / runtime knobs
    st.markdown("**Cross-Validation & Runtime**")
    folds = st.number_input("CV folds", 2, 10, 4, 1, key="bm_folds")
    equity = st.number_input("Starting equity ($)", 1000.0, 1_000_000.0, 10_000.0, 100.0, key="bm_equity")
    min_trades = st.number_input("Min trades (valid) to keep a symbol", 0, 200, 4, 1, key="bm_min_trades")

    max_w = os.cpu_count() or 4
    workers = st.number_input("CPU workers", 1, max_w, min(4, max_w), 1, key="bm_workers")

    run_btn = st.button("🚀 Train base (portfolio) model", type="primary", use_container_width=True)

with right:
    st.subheader("Results")

    if run_btn:
        if not tickers:
            st.error("This portfolio has no tickers.")
            st.stop()

        # UI elements for live progress
        bar = st.progress(0)
        line = st.empty()
        subline = st.empty()

        total_syms = len(tickers)
        state = {"done": 0, "current": None, "fold_i": 0, "fold_n": 0}


        def hook(ev: dict):
            phase = ev.get("phase")
            if phase == "prefetch_start":
                line.markdown(
                    f"**Prefetching OHLCV** · {ev.get('symbols_total', 0)} symbols · workers={ev.get('workers')}")
                subline.write("")
                bar.progress(0)
            elif phase == "prefetch_progress":
                i = int(ev.get("i", 0));
                n = int(ev.get("n", 1))
                sym = ev.get("symbol")
                ok = ev.get("ok", False)
                bar.progress(int(i / max(1, n) * 100))
                subline.markdown(f"• Cached `{sym}`  ✅" if ok else f"• Cached `{sym}`  ⚠️ failed")
            elif phase == "prefetch_done":
                bar.progress(100)
                line.markdown(f"**Prefetch complete** · ok={ev.get('ok', 0)} · errors={ev.get('errors', 0)}")
                subline.write("")
                bar.progress(0)  # reset for training

            elif phase == "start":
                bar.progress(0)
                line.markdown(f"**Starting training** · {total_syms} symbols · workers={ev.get('workers')}")
                subline.write("")
            elif phase == "queued":
                i = int(ev.get("i", 0));
                n = int(ev.get("n", 1))
                subline.markdown(f"• Queued `{ev.get('symbol')}` ({i}/{n})")
            elif phase == "fold_start":
                state["fold_i"] = int(ev.get("i", 0))
                state["fold_n"] = int(ev.get("n", 0))
                subline.markdown(f"• Fold {state['fold_i']}/{state['fold_n']} [{ev.get('start')} → {ev.get('end')}]")
            elif phase == "symbol_done":
                state["done"] = int(ev.get("i", state["done"]))
                done = state["done"]
                n = int(ev.get("n", total_syms))
                bar.progress(min(100, int(done / max(1, n) * 100)))
                msg = f"Finished `{ev.get('symbol')}` · trades={ev.get('cv_trades', 0)} · sharpe={ev.get('cv_sharpe', 0):.2f}"
                line.markdown(msg)
                subline.write("")
            elif phase == "done":
                bar.progress(100)
                line.markdown(f"**Completed** · {ev.get('rows')} rows across {ev.get('symbols')} symbols")
                subline.write("")
        # Run training with progress hook
        res = train_general_model(
            portfolio=port_name,
            strategy_dotted=strategy_dotted,
            params=p,
            folds=int(folds),
            starting_equity=float(equity),
            min_trades=int(min_trades),
            workers=int(workers),
            progress_hook=hook,     # ← live updates
        )

        st.session_state["base_train_res"] = res

        # Build leaderboard + show results (unchanged from your version)
        lb = pd.DataFrame(res.get("leaderboard", []))
        n_eval = int(lb["symbol"].nunique()) if ("symbol" in lb.columns and not lb.empty) else 0

        errs_top = res.get("errors", []) or (res.get("log", {}) or {}).get("errors", []) or []
        st.success(f"Done: {n_eval} symbols evaluated. Errors: {len(errs_top)}")

        if lb.empty:
            st.warning("No leaderboard rows (maybe min_trades too high or CV returned no trades).")
        else:
            sort_cols = [c for c in ["cv_sharpe", "cv_cagr", "cv_trades"] if c in lb.columns]
            if sort_cols:
                lb = lb.sort_values(sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)
            numeric_cols = [c for c, dt in zip(lb.columns, lb.dtypes) if pd.api.types.is_numeric_dtype(dt)]
            fmt_map = {c: "{:.4f}" for c in numeric_cols if c != "cv_trades"}
            if "cv_trades" in lb.columns:
                fmt_map["cv_trades"] = "{:.0f}"
            st.dataframe(lb.style.format(fmt_map), use_container_width=True, height=420)

        # (keep your Export log / Save model UI below)