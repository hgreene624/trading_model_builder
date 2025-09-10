# pages/4_Simulate_Portfolio.py
import streamlit as st
import pandas as pd
from datetime import date, timedelta

from src.storage import (
    list_portfolios, add_simulation, list_strategies, get_strategy, get_default_strategy
)
from src.models.atr_breakout import backtest_single
from src.utils.plotting import equity_chart

st.set_page_config(page_title="Simulate Portfolio", page_icon="🧮")

st.title("🧮 Simulate Portfolio")

portfolios = list_portfolios()
if not portfolios:
    st.info("Create a portfolio first (Ticker Tuning → Save).")
    st.stop()

names = [p["name"] for p in portfolios]
choice = st.selectbox("Portfolio", names, index=0)
p = next((x for x in portfolios if x["name"] == choice), None)
items = p.get("items", [])

if not items:
    st.warning("This portfolio has no items yet.")
    st.stop()

colA, colB, colC = st.columns(3)
with colA:
    start = st.date_input("Start Date", value=date.today() - timedelta(days=365*3))
with colB:
    end = st.date_input("End Date", value=date.today())
with colC:
    starting_equity = st.number_input("Starting Equity ($)", min_value=1000, value=50_000, step=1000)

st.subheader(f"Tickers in '{p['name']}'")

strategy_selections = {}
for it in items:
    symbol = it["symbol"]
    model = it["model"]

    # Build options: default strategy (if exists) first, then others, then a "Use portfolio params" fallback
    strategies = list_strategies(symbol=symbol, model=model)
    default_strat = get_default_strategy(symbol, model)

    opts = []
    ids = []

    if default_strat:
        opts.append(f"★ Default: {default_strat['name']}  ({default_strat['id'][:8]})")
        ids.append(default_strat["id"])

    for s in strategies:
        if not default_strat or s["id"] != default_strat["id"]:
            opts.append(f"{s['name']}  ({s['id'][:8]})")
            ids.append(s["id"])

    # Always include the portfolio-saved params option
    opts.append("Use portfolio params")
    ids.append("PORTFOLIO_DEFAULT")

    default_idx = 0  # show default strategy first if exists, otherwise first list item; user can pick "Use portfolio params"
    sel = st.selectbox(f"{symbol} — strategy", opts, index=default_idx, key=f"strategy_{p['id']}_{symbol}")
    strategy_selections[symbol] = ids[opts.index(sel)]

st.write("---")

if st.button("Run Simulation", type="primary"):
    with st.spinner("Simulating..."):
        try:
            per_ticker = []
            curves = []
            per_capital = starting_equity / max(len(items), 1)

            for it in items:
                symbol = it["symbol"]
                chosen = strategy_selections.get(symbol, "PORTFOLIO_DEFAULT")
                if chosen != "PORTFOLIO_DEFAULT":
                    srec = get_strategy(chosen)
                    params = srec["params"] if srec else it["params"]
                else:
                    params = it["params"]

                res = backtest_single(
                    symbol,
                    start.isoformat(),
                    end.isoformat(),
                    breakout_n=int(params.get("breakout_n", 55)),
                    exit_n=int(params.get("exit_n", 20)),
                    atr_n=int(params.get("atr_n", 14)),
                    starting_equity=float(per_capital),
                    atr_multiple=float(params.get("atr_multiple", 3.0)),
                    risk_per_trade=float(params.get("risk_per_trade", 0.01)),
                    allow_fractional=bool(params.get("allow_fractional", True)),
                )
                eq = res["equity"].rename(symbol)
                curves.append(eq)
                per_ticker.append({"symbol": symbol, **res["metrics"]})

            aligned = pd.concat(curves, axis=1).fillna(method="ffill").dropna(how="all")
            total_eq = aligned.sum(axis=1)
            st.plotly_chart(equity_chart(total_eq, title=f"Total Equity — {p['name']}"), use_container_width=True)

            out = pd.DataFrame(per_ticker).sort_values("total_return", ascending=False)
            out["total_return_%"] = (out["total_return"] * 100).round(2)
            st.subheader("Per-Ticker Performance")
            st.dataframe(out[["symbol", "total_return_%", "sharpe", "max_drawdown", "final_equity", "start", "end"]], use_container_width=True)

            add_simulation({
                "portfolio_id": p["id"],
                "portfolio_name": p["name"],
                "start": start.isoformat(),
                "end": end.isoformat(),
                "starting_equity": float(starting_equity),
                "final_equity": float(total_eq.iloc[-1]),
            })
            st.success("Simulation saved.")
        except Exception as e:
            st.error(f"Error: {e}")