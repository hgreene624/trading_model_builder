import streamlit as st
import pandas as pd
from datetime import date, timedelta
from src.storage import list_portfolios, add_simulation
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

st.write(f"**Tickers in '{p['name']}'**: " + ", ".join(it["symbol"] for it in items))

if st.button("Run Simulation", type="primary"):
    with st.spinner("Simulating..."):
        try:
            per_ticker = []
            curves = []
            per_capital = starting_equity / max(len(items),1)
            for it in items:
                res = backtest_single(it["symbol"], start.isoformat(), end.isoformat(),
                                      it["params"].get("breakout_n",55),
                                      it["params"].get("exit_n",20),
                                      it["params"].get("atr_n",14),
                                      per_capital)
                eq = res["equity"].rename(it["symbol"])
                curves.append(eq)
                per_ticker.append({
                    "symbol": it["symbol"],
                    **res["metrics"]
                })
            aligned = pd.concat(curves, axis=1).fillna(method="ffill").dropna(how="all")
            total_eq = aligned.sum(axis=1)
            st.plotly_chart(equity_chart(total_eq, title=f"Total Equity — {p['name']}"), use_container_width=True)
            out = pd.DataFrame(per_ticker).sort_values("total_return", ascending=False)
            out["total_return_%"] = (out["total_return"]*100).round(2)
            st.subheader("Per-Ticker Performance")
            st.dataframe(out[["symbol","total_return_%","sharpe","max_drawdown","final_equity","start","end"]], use_container_width=True)

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
