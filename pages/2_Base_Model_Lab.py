# pages/2_Base_Model_Lab.py
from __future__ import annotations

import io
import json
import os
from datetime import datetime
import inspect

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

    # Workers is kept for API compatibility but we’ll run single-thread in trainer for now.
    max_w = os.cpu_count() or 4
    workers = st.number_input("CPU workers (trainer may ignore for now)", 1, max_w, min(4, max_w), 1, key="bm_workers")

    run_btn = st.button("🚀 Train base (portfolio) model", type="primary", use_container_width=True)

with right:
    st.subheader("Results")

    if run_btn:
        if not tickers:
            st.error("This portfolio has no tickers.")
            st.stop()

        # Live progress placeholders (no nonlocal; we keep state in a dict)
        progress = st.progress(0, text="Starting…")
        status_box = st.empty()
        lines_box = st.empty()
        prog_state = {"n": len(tickers), "i": 0}

        def _on_progress(evt: dict):
            """Optional callback if general_trainer supports `on_progress`."""
            try:
                # Update counters if present
                i = int(evt.get("i", prog_state["i"]))
                n = int(evt.get("n", prog_state["n"]))
                prog_state["i"], prog_state["n"] = i, n

                # Compose message
                sym = evt.get("symbol")
                phase = evt.get("phase", "")
                msg = evt.get("msg", "")
                txt = " ".join(x for x in [phase, sym, msg] if x)

                # Progress bar
                ratio = 0.0 if n <= 0 else min(1.0, max(0.0, i / n))
                progress.progress(int(ratio * 100), text=txt or "Working…")

                # Status lines
                now_row = evt.get("row")
                if now_row:
                    lines_box.write(pd.DataFrame([now_row]))
                if txt:
                    status_box.info(txt)
            except Exception:
                # Fail-safe: never break training due to UI updates
                pass

        # Build kwargs dynamically so we don't pass unknown args
        sig = inspect.signature(train_general_model)
        kwargs = dict(
            portfolio=port_name,
            strategy_dotted=strategy_dotted,
            params=p,
            folds=int(folds),
            starting_equity=float(equity),
            min_trades=int(min_trades),
            workers=int(workers),
        )
        if "on_progress" in sig.parameters:
            kwargs["on_progress"] = _on_progress

        with st.spinner("Training…"):
            res = train_general_model(**kwargs)

        # Clear progress UI
        progress.progress(100, text="Done")
        status_box.empty()
        lines_box.empty()

        # Persist immediately to avoid None on first render
        st.session_state["base_train_res"] = res or {}

        # Build leaderboard DF and compute how many unique symbols were evaluated
        lb = pd.DataFrame((res or {}).get("leaderboard", []))
        n_eval = int(lb["symbol"].nunique()) if ("symbol" in lb.columns and not lb.empty) else 0

        # Prefer top-level errors; fallback to log.errors
        errs_top = (res or {}).get("errors", [])
        if not errs_top:
            errs_top = ((res or {}).get("log", {}) or {}).get("errors", []) or []

        st.success(f"Done: {n_eval} symbols evaluated. Errors: {len(errs_top)}")

        # Leaderboard table
        if lb.empty:
            st.warning("No leaderboard rows (maybe min_trades too high or CV returned no trades).")
        else:
            # Try to sort if metrics exist
            sort_cols = [c for c in ["cv_sharpe", "cv_cagr", "cv_trades"] if c in lb.columns]
            if sort_cols:
                lb = lb.sort_values(sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)

            # Pretty display: format numeric columns
            numeric_cols = [c for c, dt in zip(lb.columns, lb.dtypes) if pd.api.types.is_numeric_dtype(dt)]
            fmt_map = {c: "{:.4f}" for c in numeric_cols if c not in ("cv_trades",)}
            if "cv_trades" in lb.columns:
                fmt_map["cv_trades"] = "{:.0f}"

            st.dataframe(
                lb.style.format(fmt_map),
                use_container_width=True,
                height=420,
            )

        # Export Log
        log_res = st.session_state.get("base_train_res", res or {})
        if log_res:
            colA, colB = st.columns([1, 3])
            with colA:
                if st.button("Export training log", use_container_width=True):
                    path = save_training_log(port_name, (log_res.get("log", {}) or {}))
                    st.success(f"Saved log to: {path}")
            with colB:
                buf = io.StringIO()
                json.dump((log_res.get("log", {}) or {}), buf, indent=2)
                st.download_button(
                    label="Download log JSON",
                    data=buf.getvalue(),
                    file_name="base_model_training_log.json",
                    mime="application/json",
                )

        # Errors (if any)
        errs = errs_top
        if errs:
            with st.expander("Errors", expanded=False):
                st.write(pd.DataFrame(errs))

        # Save trained portfolio model
        st.markdown("---")
        st.markdown("**Save this portfolio model**")
        default_name = f"{port_name}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_name = st.text_input("Model name", value=default_name, key="bm_model_name")
        if st.button("💾 Save model", use_container_width=True):
            payload = {
                "meta": {
                    "portfolio": port_name,
                    "strategy": strategy_dotted,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "workers": int(workers),
                    "folds": int(folds),
                    "starting_equity": float(equity),
                    "min_trades": int(min_trades),
                    "n_symbols": n_eval,
                },
                "base_params": p,
                "leaderboard": (res or {}).get("leaderboard", []),
                "errors": errs_top,
            }
            path = save_portfolio_model(port_name, model_name, payload)
            st.success(f"Saved portfolio model → `{path}`")

        # Existing models list
        models = list_portfolio_models(port_name)
        if models:
            with st.expander("Existing saved models for this portfolio", expanded=False):
                st.write(pd.DataFrame({"model": models}))

    else:
        # If page just opened (or after previous run), show last result if present
        last = st.session_state.get("base_train_res")
        if last:
            lb = pd.DataFrame(last.get("leaderboard", []))
            n_eval = int(lb["symbol"].nunique()) if ("symbol" in lb.columns and not lb.empty) else 0
            st.info(f"Loaded last run: {n_eval} symbols evaluated. Click **Train** to run again.")
            if not lb.empty:
                numeric_cols = [c for c, dt in zip(lb.columns, lb.dtypes) if pd.api.types.is_numeric_dtype(dt)]
                fmt_map = {c: "{:.4f}" for c in numeric_cols if c not in ("cv_trades",)}
                if "cv_trades" in lb.columns:
                    fmt_map["cv_trades"] = "{:.0f}"
                st.dataframe(lb.style.format(fmt_map), use_container_width=True, height=360)
        else:
            st.info("Configure settings on the left, then click **Train base (portfolio) model**.")