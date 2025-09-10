# pages/2_Ticker_Selector_and_Tuning.py
import pandas as pd
import streamlit as st
from datetime import date, timedelta

from src.models.atr_breakout import backtest_single
from src.tuning.evolve import evolve_params, Bounds
from src.utils.plotting import equity_chart
from src.storage import list_portfolios, create_portfolio, add_item, list_strategies, save_strategy


st.set_page_config(page_title="Ticker Selector & Tuning", page_icon="🧪")
st.title("🧪 Ticker Selector & Model Tuning")

tabs = st.tabs(["Manual Backtest", "Evolutionary Tuning"])

# ------------------------ Manual Backtest ------------------------
with tabs[0]:
    with st.sidebar:
        st.markdown("**Manual Params (ATR Breakout + Risk)**")
        breakout_n = st.slider("Breakout lookback (days)", 10, 300, 55, 1, key="mb_breakout")
        exit_n = st.slider("Exit lookback (days)", 5, 200, 20, 1, key="mb_exit")
        atr_n = st.slider("ATR lookback (days)", 5, 100, 14, 1, key="mb_atr")
        atr_multiple = st.slider("ATR multiple (stop distance)", 1.0, 6.0, 3.0, 0.1, key="mb_am")
        risk_per_trade = st.slider("Risk per trade (% of equity)", 0.1, 3.0, 1.0, 0.1, key="mb_rpt") / 100.0
        allow_fractional = st.toggle("Allow fractional shares", value=True, key="mb_frac")
        starting_equity_manual = st.number_input("Starting Equity ($)", min_value=1000, value=10_000, step=500, key="mb_eq")

    colA, colB = st.columns(2)
    with colA:
        symbol_manual = st.text_input("Ticker", value="AAPL", key="mb_symbol").upper().strip()
    with colB:
        end_manual = st.date_input("End Date", value=date.today(), key="mb_end")
        start_manual = st.date_input("Start Date", value=end_manual - timedelta(days=365*3), key="mb_start")

    # ---- Load existing strategy for this ticker ----
    st.markdown("**Load saved strategy for this ticker**")
    existing = list_strategies(symbol_manual, model="atr_breakout")
    label_to_id = {"— Select a strategy —": ""}
    for s in existing:
        label_to_id[f"{s['name']}  ({s['id'][:8]})"] = s["id"]
    sel_label = st.selectbox("Saved strategies", list(label_to_id.keys()), index=0, key="mb_sel_strat")
    if st.button("Load Strategy", key="btn_load_strat"):
        sid = label_to_id.get(sel_label, "")
        if sid:
            srec = next((x for x in existing if x["id"] == sid), None)
            if srec:
                p = srec["params"]
                # Set widget state then rerun to refresh sliders
                st.session_state["mb_breakout"] = int(p.get("breakout_n", 55))
                st.session_state["mb_exit"] = int(p.get("exit_n", 20))
                st.session_state["mb_atr"] = int(p.get("atr_n", 14))
                st.session_state["mb_am"] = float(p.get("atr_multiple", 3.0))
                st.session_state["mb_rpt"] = float(p.get("risk_per_trade", 0.01)) * 100.0
                st.session_state["mb_frac"] = bool(p.get("allow_fractional", True))
                st.rerun()
        else:
            st.info("Select a saved strategy first.")

    # ---- Run manual backtest ----
    if st.button("Run Manual Backtest", type="primary", key="btn_manual"):
        with st.spinner("Backtesting..."):
            try:
                res = backtest_single(
                    symbol_manual,
                    start_manual.isoformat(),
                    end_manual.isoformat(),
                    breakout_n=st.session_state["mb_breakout"],
                    exit_n=st.session_state["mb_exit"],
                    atr_n=st.session_state["mb_atr"],
                    starting_equity=starting_equity_manual,
                    atr_multiple=st.session_state["mb_am"],
                    risk_per_trade=st.session_state["mb_rpt"] / 100.0,
                    allow_fractional=st.session_state["mb_frac"],
                )
                equity = res["equity"]
                metrics = res["metrics"]
                st.subheader(f"Manual Results — {symbol_manual}")
                st.plotly_chart(equity_chart(equity, title=f"Equity — {symbol_manual}"), use_container_width=True)
                st.write(pd.DataFrame([metrics]))
                st.success("Done.")
                st.session_state["last_result"] = {
                    "symbol": symbol_manual,
                    "params": {
                        "breakout_n": st.session_state["mb_breakout"],
                        "exit_n": st.session_state["mb_exit"],
                        "atr_n": st.session_state["mb_atr"],
                        "atr_multiple": st.session_state["mb_am"],
                        "risk_per_trade": st.session_state["mb_rpt"] / 100.0,
                        "allow_fractional": st.session_state["mb_frac"],
                    },
                }
            except Exception as e:
                st.error(f"Error: {e}")

    # ---- Save current settings as strategy ----
    st.markdown("**Save current settings as a Strategy**")
    default_name = f"{symbol_manual}-Manual"
    strategy_name = st.text_input("Strategy name", value=default_name, key="mb_save_name")
    if st.button("Save Strategy", key="btn_save_strategy"):
        params_now = {
            "breakout_n": int(st.session_state["mb_breakout"]),
            "exit_n": int(st.session_state["mb_exit"]),
            "atr_n": int(st.session_state["mb_atr"]),
            "atr_multiple": float(st.session_state["mb_am"]),
            "risk_per_trade": float(st.session_state["mb_rpt"]) / 100.0,
            "allow_fractional": bool(st.session_state["mb_frac"]),
        }
        rec = save_strategy(symbol_manual, "atr_breakout", params_now, name=strategy_name)
        st.success(f"Saved strategy: {rec['name']}  ({rec['id'][:8]})")


# --------------------- Evolutionary Tuning -----------------------
with tabs[1]:
    st.markdown("Use an evolutionary algorithm to **maximize Sharpe** by tuning breakout/exit/ATR **and** risk params.")

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
        pop_size = st.slider("Population size", 10, 200, 40, 5, key="ev_pop")
        generations = st.slider("Generations", 5, 100, 20, 1, key="ev_gen")
    with c6:
        crossover_rate = st.slider("Crossover rate", 0.0, 1.0, 0.7, 0.05, key="ev_cr")
        mutation_rate = st.slider("Mutation rate", 0.0, 1.0, 0.35, 0.05, key="ev_mr")

    st.markdown("**Parameter bounds**")
    b1, b2, b3 = st.columns(3)
    with b1:
        breakout_min = st.number_input("Breakout min", 10, 400, 20, 1, key="ev_bmin")
        breakout_max = st.number_input("Breakout max", breakout_min + 1, 500, 120, 1, key="ev_bmax")
    with b2:
        exit_min = st.number_input("Exit min", 5, 300, 10, 1, key="ev_emin")
        exit_max = st.number_input("Exit max", exit_min + 1, 300, 60, 1, key="ev_emax")
    with b3:
        atr_min = st.number_input("ATR min", 5, 150, 7, 1, key="ev_amin")
        atr_max = st.number_input("ATR max", atr_min + 1, 200, 30, 1, key="ev_amax")

    r1, r2 = st.columns(2)
    with r1:
        atr_multiple_min = st.number_input("ATR multiple min", 0.5, 10.0, 1.5, 0.1, key="ev_am_min")
        atr_multiple_max = st.number_input("ATR multiple max", atr_multiple_min + 0.1, 15.0, 5.0, 0.1, key="ev_am_max")
    with r2:
        risk_per_trade_min = st.number_input("Risk per trade min (%)", 0.05, 5.0, 0.2, 0.05, key="ev_rpt_min") / 100.0
        risk_per_trade_max = st.number_input("Risk per trade max (%)", (risk_per_trade_min * 100) + 0.05, 10.0, 2.0, 0.05, key="ev_rpt_max") / 100.0

    # Progress elements
    prog = st.progress(0, text="Idle")

    if st.button("Run Evolutionary Tuning", type="primary", key="btn_ev"):
        bounds = Bounds(
            breakout_min=int(breakout_min),
            breakout_max=int(breakout_max),
            exit_min=int(exit_min),
            exit_max=int(exit_max),
            atr_min=int(atr_min),
            atr_max=int(atr_max),
            atr_multiple_min=float(atr_multiple_min),
            atr_multiple_max=float(atr_multiple_max),
            risk_per_trade_min=float(risk_per_trade_min),
            risk_per_trade_max=float(risk_per_trade_max),
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

                # Run full backtest with best params to show equity
                res_best = backtest_single(
                    symbol,
                    start.isoformat(),
                    end.isoformat(),
                    best_params["breakout_n"],
                    best_params["exit_n"],
                    best_params["atr_n"],
                    float(starting_equity),
                    atr_multiple=best_params["atr_multiple"],
                    risk_per_trade=best_params["risk_per_trade"],
                    allow_fractional=True,
                )
                st.plotly_chart(
                    equity_chart(res_best["equity"], title=f"Equity — {symbol} (Best Params)"),
                    use_container_width=True,
                )

                st.session_state["last_result"] = {"symbol": symbol, "params": best_params}

                # Save best as strategy
                st.markdown("**Save best as a Strategy**")
                default_name = f"{symbol}-Best"
                best_name = st.text_input("Strategy name", value=default_name, key="ev_save_name")
                if st.button("Save Best Strategy", key="btn_save_best"):
                    rec = save_strategy(symbol, "atr_breakout", best_params, name=best_name)
                    st.success(f"Saved strategy: {rec['name']}  ({rec['id'][:8]})")

                # History table
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