# pages/2_Ticker_Selector_and_Tuning.py
import pandas as pd
import streamlit as st
from datetime import date, timedelta

from src.models.atr_breakout import backtest_single
from src.tuning.evolve import evolve_params, Bounds
from src.utils.plotting import equity_chart
from src.storage import list_portfolios, create_portfolio, add_item


st.set_page_config(page_title="Ticker Selector & Tuning", page_icon="🧪")
st.title("🧪 Ticker Selector & Model Tuning")

tabs = st.tabs(["Manual Backtest", "Evolutionary Tuning"])

# ------------------------ Manual Backtest ------------------------
with tabs[0]:
    with st.sidebar:
        st.markdown("**Manual Params (ATR Breakout)**")
        breakout_n = st.slider("Breakout lookback (days)", 10, 200, 55, 1, key="mb_breakout")
        exit_n = st.slider("Exit lookback (days)", 5, 100, 20, 1, key="mb_exit")
        atr_n = st.slider("ATR lookback (days)", 5, 50, 14, 1, key="mb_atr")
        starting_equity_manual = st.number_input("Starting Equity ($)", min_value=1000, value=10_000, step=500, key="mb_eq")

    colA, colB = st.columns(2)
    with colA:
        symbol_manual = st.text_input("Ticker", value="AAPL", key="mb_symbol").upper().strip()
    with colB:
        end_manual = st.date_input("End Date", value=date.today(), key="mb_end")
        start_manual = st.date_input("Start Date", value=end_manual - timedelta(days=365*3), key="mb_start")

    if st.button("Run Manual Backtest", type="primary", key="btn_manual"):
        with st.spinner("Backtesting..."):
            try:
                res = backtest_single(
                    symbol_manual,
                    start_manual.isoformat(),
                    end_manual.isoformat(),
                    breakout_n,
                    exit_n,
                    atr_n,
                    starting_equity_manual,
                )
                equity = res["equity"]
                metrics = res["metrics"]
                st.subheader(f"Manual Results — {symbol_manual}")
                st.plotly_chart(equity_chart(equity, title=f"Equity — {symbol_manual}"), use_container_width=True)
                st.write(pd.DataFrame([metrics]))
                st.success("Done.")
                st.session_state["last_result"] = {
                    "symbol": symbol_manual,
                    "params": {"breakout_n": breakout_n, "exit_n": exit_n, "atr_n": atr_n},
                }
            except Exception as e:
                st.error(f"Error: {e}")

# --------------------- Evolutionary Tuning -----------------------
with tabs[1]:
    st.markdown("Use an evolutionary algorithm to **maximize Sharpe** by tuning breakout/exit/ATR windows.")

    c1, c2, c3 = st.columns(3)
    with c1:
        symbol = st.text_input("Ticker", value="AAPL", key="ev_symbol").upper().strip()
    with c2:
        end = st.date_input("End Date", value=date.today(), key="ev_end")
    with c3:
        start = st.date_input("Start Date", value=end - timedelta(days=365*3), key="ev_start")

    c4, c5, c6 = st.columns(3)
    with c4:
        starting_equity = st.number_input("Starting Equity ($)", min_value=1000, value=10_000, step=500, key="ev_eq")
    with c5:
        pop_size = st.slider("Population size", 10, 120, 40, 5, key="ev_pop")
        generations = st.slider("Generations", 5, 50, 20, 1, key="ev_gen")
    with c6:
        crossover_rate = st.slider("Crossover rate", 0.0, 1.0, 0.7, 0.05, key="ev_cr")
        mutation_rate = st.slider("Mutation rate", 0.0, 1.0, 0.3, 0.05, key="ev_mr")

    st.markdown("**Parameter bounds**")
    b1, b2, b3 = st.columns(3)
    with b1:
        breakout_min = st.number_input("Breakout min", 10, 300, 20, 1, key="ev_bmin")
        breakout_max = st.number_input("Breakout max", breakout_min + 1, 400, 120, 1, key="ev_bmax")
    with b2:
        exit_min = st.number_input("Exit min", 5, 200, 10, 1, key="ev_emin")
        exit_max = st.number_input("Exit max", exit_min + 1, 300, 60, 1, key="ev_emax")
    with b3:
        atr_min = st.number_input("ATR min", 5, 100, 7, 1, key="ev_amin")
        atr_max = st.number_input("ATR max", atr_min + 1, 150, 30, 1, key="ev_amax")

    # Progress elements
    prog = st.progress(0, text="Idle")
    status = st.empty()

    if st.button("Run Evolutionary Tuning", type="primary", key="btn_ev"):
        bounds = Bounds(
            breakout_min=int(breakout_min),
            breakout_max=int(breakout_max),
            exit_min=int(exit_min),
            exit_max=int(exit_max),
            atr_min=int(atr_min),
            atr_max=int(atr_max),
        )

        def _cb(done: int, total: int, best_fit: float):
            prog.progress(min(int(100 * done / total), 100), text=f"Generation {done}/{total} — best Sharpe: {best_fit:.2f}")

        with st.spinner("Evolving…"):
            try:
                best_params, best_metrics, history = evolve_params(
                    symbol=symbol,
                    start=start.isoformat(),
                    end=end.isoformat(),
                    starting_equity=float(starting_equity),
                    bounds=bounds,
                    pop_size=int(pop_size),
                    generations=int(generations),
                    crossover_rate=float(crossover_rate),
                    mutation_rate=float(mutation_rate),
                    random_seed=42,
                    progress_cb=_cb,
                )
                prog.progress(100, text="Done")

                st.subheader(f"Best Parameters — {symbol}")
                st.write(best_params)
                st.write(pd.DataFrame([best_metrics]))

                # Run a full backtest with the best params to display equity curve
                res_best = backtest_single(
                    symbol,
                    start.isoformat(),
                    end.isoformat(),
                    best_params["breakout_n"],
                    best_params["exit_n"],
                    best_params["atr_n"],
                    float(starting_equity),
                )
                st.plotly_chart(
                    equity_chart(res_best["equity"], title=f"Equity — {symbol} (Best Params)"),
                    use_container_width=True,
                )

                st.session_state["last_result"] = {
                    "symbol": symbol,
                    "params": best_params,
                }

                # Show simple history table
                hist_df = pd.DataFrame(history)
                st.caption("Tuning history")
                st.dataframe(hist_df, use_container_width=True)
                st.success("Evolution complete.")

            except Exception as e:
                st.error(f"Error during evolution: {e}")

# ------------------- Save to Portfolio (shared) -------------------
st.divider()
st.subheader("Save this configuration to a Portfolio")

portfolios = list_portfolios()
names = ["— Create new —"] + [p["name"] for p in portfolios]
choice = st.selectbox("Target Portfolio", names, index=1 if len(portfolios) > 0 else 0)
new_name = ""
if choice == "— Create new —":
    new_name = st.text_input("New Portfolio Name", value="My Portfolio")

if st.button("Save to Portfolio", key="btn_save"):
    last = st.session_state.get("last_result")
    if not last:
        st.warning("Run a backtest or tuning first.")
    else:
        if choice == "— Create new —":
            if not new_name.strip():
                st.warning("Please provide a name for the new portfolio.")
            else:
                p = create_portfolio(new_name.strip())
                add_item(p["id"], last["symbol"], "atr_breakout", last["params"])
                st.success(f"Saved {last['symbol']} to portfolio '{p['name']}'.")
        else:
            pid = next((p["id"] for p in portfolios if p["name"] == choice), None)
            if pid:
                add_item(pid, last["symbol"], "atr_breakout", last["params"])
                st.success(f"Saved {last['symbol']} to portfolio '{choice}'.")
            else:
                st.error("Portfolio not found.")