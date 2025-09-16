from __future__ import annotations

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
)
from src.models.general_trainer import train_general_model

st.set_page_config(page_title="Base Model Lab", layout="wide")
st.title("📦 Base Model Lab (Portfolio-level)")


def _default_params() -> dict:
    return {
        "breakout_n": 55,
        "exit_n": 35,
        "atr_n": 14,
        "atr_multiple": 2.5,
        "tp_multiple": 2.0,
        "holding_period_limit": 120,
        "risk_per_trade": 0.008,
        "use_trend_filter": True,
        "sma_fast": 20,
        "sma_slow": 60,
        "sma_long": 200,
        "long_slope_len": 20,
        "cost_bps": 2.0,
        "execution": "close",
    }

left, right = st.columns([1.0, 1.6], gap="large")

with left:
    st.subheader("Portfolio & Training")

    ports = list_portfolios()
    if not ports:
        st.warning("No portfolios found. Create one on the **Portfolios** page first.")
        st.stop()

    port_name = st.selectbox("Portfolio", options=ports, index=0)
    tickers = list(load_portfolio(port_name).get("tickers", []))
    st.info(f"Selected portfolio **{port_name}** with **{len(tickers)}** tickers.")

    strategy_dotted = st.selectbox("Strategy", ["src.models.atr_breakout"], index=0)

    with st.expander("Base params", expanded=False):
        p = _default_params()
        p["breakout_n"] = st.number_input("breakout_n", 5, 300, p["breakout_n"], 1)
        p["exit_n"] = st.number_input("exit_n", 4, 300, p["exit_n"], 1)
        p["atr_n"] = st.number_input("atr_n", 5, 60, p["atr_n"], 1)
        p["atr_multiple"] = st.number_input("atr_multiple", 0.5, 10.0, float(p["atr_multiple"]), 0.1)
        p["tp_multiple"] = st.number_input("tp_multiple", 0.5, 10.0, float(p["tp_multiple"]), 0.1)
        p["holding_period_limit"] = st.number_input("holding_period_limit", 5, 400, p["holding_period_limit"], 1)
        p["risk_per_trade"] = st.number_input("risk_per_trade", 0.0005, 0.05, float(p["risk_per_trade"]), 0.0005, format="%.4f")
        p["use_trend_filter"] = st.checkbox("use_trend_filter", value=bool(p["use_trend_filter"]))
        p["sma_fast"] = st.number_input("sma_fast", 5, 100, p["sma_fast"], 1)
        p["sma_slow"] = st.number_input("sma_slow", 10, 200, p["sma_slow"], 1)
        p["sma_long"] = st.number_input("sma_long", 100, 400, p["sma_long"], 1)
        p["long_slope_len"] = st.number_input("long_slope_len", 5, 60, p["long_slope_len"], 1)
        p["cost_bps"] = st.number_input("cost_bps", 0.0, 20.0, float(p["cost_bps"]), 0.1)
        p["execution"] = st.selectbox("Execution", ["close"], index=0)

    folds = st.number_input("CV folds", 2, 10, 4, 1)
    equity = st.number_input("Starting equity ($)", 1000.0, 1_000_000.0, 10_000.0, 100.0)
    min_trades = st.number_input("Min trades (valid)", 0, 200, 2, 1)

    run_btn = st.button("🚀 Train base (portfolio) model", type="primary", use_container_width=True)

with right:
    st.subheader("Results")

    if run_btn:
        if not tickers:
            st.error("This portfolio has no tickers.")
            st.stop()

        # Progress UI
        prog = st.progress(0.0, text="Starting training…")
        status = st.empty()
        status.write(f"Training {len(tickers)} symbols with {int(folds)} folds…")

        res = train_general_model(
            portfolio=port_name,
            strategy_dotted=strategy_dotted,
            params=p,
            folds=int(folds),
            starting_equity=float(equity),
            min_trades=int(min_trades),
            workers=1,
        )

        st.session_state["base_train_res"] = res
        prog.progress(1.0, text="Done")

        lb = pd.DataFrame(res.get("leaderboard", []))
        errs = res.get("errors", [])

        if lb.empty:
            st.warning("No leaderboard rows (no trades met the min filter). Try loosening params.")
        else:
            sort_cols = [c for c in ["cv_sharpe", "cv_cagr", "cv_trades"] if c in lb.columns]
            if sort_cols:
                lb = lb.sort_values(sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)
            st.dataframe(lb, use_container_width=True, height=420)

        if errs:
            with st.expander("Errors", expanded=False):
                st.write(pd.DataFrame(errs))

        st.markdown("---")
        st.markdown("**Save this portfolio model**")
        default_name = f"{port_name}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_name = st.text_input("Model name", value=default_name)
        if st.button("💾 Save model", use_container_width=True):
            payload = {
                "meta": {
                    "portfolio": port_name,
                    "strategy": strategy_dotted,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "folds": int(folds),
                    "starting_equity": float(equity),
                    "min_trades": int(min_trades),
                },
                "base_params": p,
                "leaderboard": res.get("leaderboard", []),
                "errors": errs,
            }
            path = save_portfolio_model(port_name, model_name, payload)
            st.success(f"Saved → {path}")

    else:
        last = st.session_state.get("base_train_res")
        if last:
            st.info("Loaded last run. Click **Train** to run again.")
            lb = pd.DataFrame(last.get("leaderboard", []))
            if not lb.empty:
                st.dataframe(lb, use_container_width=True, height=360)
        else:
            st.info("Configure on the left, then click **Train**.")