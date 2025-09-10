# pages/2_Ticker_Selector_and_Tuning.py
import pandas as pd
import streamlit as st
from datetime import date, timedelta

from src.models.atr_breakout import backtest_single
from src.tuning.evolve import evolve_params, Bounds
from src.utils.plotting import equity_chart
from src.storage import (
    list_portfolios, create_portfolio, add_item,
    list_strategies, save_strategy, set_default_strategy,
    list_param_bounds, save_param_bounds, get_param_bounds, get_default_param_bounds
)

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

    # Load saved strategies for this ticker (model params)
    st.markdown("**Load saved strategy for this ticker**")
    existing = list_strategies(symbol_manual, model="atr_breakout")
    label_to_id = {"— Select a strategy —": ""}
    for s in existing:
        star = "★ " if s.get("is_default") else ""
        label_to_id[f"{star}{s['name']}  ({s['id'][:8]})"] = s["id"]

    sel_label = st.selectbox("Saved strategies", list(label_to_id.keys()), index=0, key="mb_sel_strat")
    if st.button("Load Strategy", key="btn_load_strat"):
        sid = label_to_id.get(sel_label, "")
        if sid:
            srec = next((x for x in existing if x["id"] == sid), None)
            if srec:
                p = srec["params"]
                st.session_state["mb_breakout"] = int(p.get("breakout_n", 55))
                st.session_state["mb_exit"] = int(p.get("exit_n", 20))
                st.session_state["mb_atr"] = int(p.get("atr_n", 14))
                st.session_state["mb_am"] = float(p.get("atr_multiple", 3.0))
                st.session_state["mb_rpt"] = float(p.get("risk_per_trade", 0.01)) * 100.0
                st.session_state["mb_frac"] = bool(p.get("allow_fractional", True))
                st.rerun()
        else:
            st.info("Select a saved strategy first.")

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

    # Save current settings as a strategy
    st.markdown("**Save current settings as a Strategy**")
    default_name = f"{symbol_manual}-Manual"
    strategy_name = st.text_input("Strategy name", value=default_name, key="mb_save_name")
    set_default = st.checkbox("Set as default for this ticker", value=False, key="mb_set_default")
    if st.button("Save Strategy", key="btn_save_strategy"):
        params_now = {
            "breakout_n": int(st.session_state["mb_breakout"]),
            "exit_n": int(st.session_state["mb_exit"]),
            "atr_n": int(st.session_state["mb_atr"]),
            "atr_multiple": float(st.session_state["mb_am"]),
            "risk_per_trade": float(st.session_state["mb_rpt"]) / 100.0,
            "allow_fractional": bool(st.session_state["mb_frac"]),
        }
        rec = save_strategy(symbol_manual, "atr_breakout", params_now, name=strategy_name, is_default=set_default)
        if set_default:
            set_default_strategy(symbol_manual, "atr_breakout", rec["id"])
        st.success(f"Saved strategy: {rec['name']}  ({rec['id'][:8]})")


# --------------------- Evolutionary Tuning -----------------------
with tabs[1]:
    st.markdown("Use an evolutionary algorithm to **maximize Sharpe**; load or save **Parameter Profiles** (bounds + run settings).")

    # ---------- 1) Ticker & Parameter Profile ----------
    c1, c2, c3 = st.columns(3)
    with c1:
        symbol = st.text_input("Ticker", value="AAPL", key="ev_symbol").upper().strip()
    with c2:
        end = st.date_input("End Date", value=date.today(), key="ev_end")
    with c3:
        start = st.date_input("Start Date", value=end - timedelta(days=365*3), key="ev_start")

    st.markdown("**Parameter Profile (bounds + run settings)**")
    profiles = list_param_bounds(symbol=symbol, model="atr_breakout")
    prof_labels = ["Default (system)"]
    prof_ids = ["__DEFAULT__"]
    default_bounds = get_default_param_bounds(symbol, "atr_breakout")
    if default_bounds:
        prof_labels.append(f"★ Default for {symbol}  ({default_bounds['id'][:8]})")
        prof_ids.append(default_bounds["id"])
    for pb in profiles:
        if default_bounds and pb["id"] == default_bounds["id"]:
            continue
        prof_labels.append(f"{pb['name']}  ({pb['id'][:8]})")
        prof_ids.append(pb["id"])

    sel_prof_idx = st.selectbox("Profiles", list(range(len(prof_labels))),
                                format_func=lambda i: prof_labels[i], index=0, key="ev_prof_idx")

    def _apply_profile_to_state(rec):
        pr = rec["profile"]
        st.session_state["ev_start"] = date.fromisoformat(pr["start"])
        st.session_state["ev_end"] = date.fromisoformat(pr["end"])
        st.session_state["ev_eq"] = float(pr.get("starting_equity", 10000))
        st.session_state["ev_pop"] = int(pr.get("pop_size", 40))
        st.session_state["ev_gen"] = int(pr.get("generations", 20))
        st.session_state["ev_cr"] = float(pr.get("crossover_rate", 0.7))
        st.session_state["ev_mr"] = float(pr.get("mutation_rate", 0.35))
        b = pr["bounds"]
        st.session_state["ev_bmin"] = int(b["breakout_min"])
        st.session_state["ev_bmax"] = int(b["breakout_max"])
        st.session_state["ev_emin"] = int(b["exit_min"])
        st.session_state["ev_emax"] = int(b["exit_max"])
        st.session_state["ev_amin"] = int(b["atr_min"])
        st.session_state["ev_amax"] = int(b["atr_max"])
        st.session_state["ev_am_min"] = float(b["atr_multiple_min"])
        st.session_state["ev_am_max"] = float(b["atr_multiple_max"])
        st.session_state["ev_rpt_min"] = float(b["risk_per_trade_min"]) * 100.0
        st.session_state["ev_rpt_max"] = float(b["risk_per_trade_max"]) * 100.0

    if st.button("Load Profile", key="btn_load_prof"):
        chosen_id = prof_ids[sel_prof_idx]
        if chosen_id != "__DEFAULT__":
            rec = get_param_bounds(chosen_id)
            if rec:
                _apply_profile_to_state(rec)
                st.success(f"Loaded profile: {rec['name']}")
                st.rerun()
            else:
                st.warning("Profile not found.")
        else:
            st.info("Loaded default system values.")

    # ---------- 2) Editable run settings + bounds ----------
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
        risk_per_trade_min = st.number_input("Risk per trade min (%)", 0.05, 5.0, 0.2, 0.05, key="ev_rpt_min")
        risk_per_trade_max = st.number_input("Risk per trade max (%)", risk_per_trade_min + 0.05, 10.0, 2.0, 0.05, key="ev_rpt_max")

    # ---------- Helper: persistent results renderer ----------
    def render_ev_results():
        """Show best params/metrics if they exist in session; include SAVE forms."""
        if "ev_best_params" not in st.session_state or "ev_best_metrics" not in st.session_state:
            return

        symbol_ss = st.session_state.get("ev_symbol", "TICKER")
        st.subheader(f"Best Parameters — {symbol_ss}")
        st.write(pd.DataFrame([st.session_state["ev_best_metrics"]]))

        # Equity curve (already computed and stored), fallback to recompute if missing
        eq = st.session_state.get("ev_best_equity")
        if eq is None:
            bp = st.session_state["ev_best_params"]
            res_best = backtest_single(
                symbol_ss,
                st.session_state["ev_start"].isoformat(),
                st.session_state["ev_end"].isoformat(),
                bp["breakout_n"], bp["exit_n"], bp["atr_n"],
                float(st.session_state["ev_eq"]),
                atr_multiple=bp["atr_multiple"],
                risk_per_trade=bp["risk_per_trade"],
                allow_fractional=True,
            )
            eq = res_best["equity"]
        st.plotly_chart(
            equity_chart(eq, title=f"Equity — {symbol_ss} (Best Params)"),
            use_container_width=True,
        )

        # -------- Save Parameter Bounds (FORM) --------
        with st.form("save_bounds_form", clear_on_submit=False):
            st.markdown("**Save Parameter Bounds Profile**")
            auto_name = (
                f"{symbol_ss}_{st.session_state['ev_start']}_{st.session_state['ev_end']}"
                f"_pop{st.session_state['ev_pop']}_gen{st.session_state['ev_gen']}"
                f"_b{int(st.session_state['ev_bmin'])}-{int(st.session_state['ev_bmax'])}"
                f"_e{int(st.session_state['ev_emin'])}-{int(st.session_state['ev_emax'])}"
                f"_atr{int(st.session_state['ev_amin'])}-{int(st.session_state['ev_amax'])}"
                f"_am{st.session_state['ev_am_min']:.1f}-{st.session_state['ev_am_max']:.1f}"
                f"_rpt{st.session_state['ev_rpt_min']:.2f}-{st.session_state['ev_rpt_max']:.2f}"
            )
            prof_name = st.text_input("Profile name", value=auto_name, key="ev_prof_name")
            save_bounds = st.form_submit_button("Save Parameter Bounds")
            if save_bounds:
                profile = {
                    "start": st.session_state["ev_start"].isoformat(),
                    "end": st.session_state["ev_end"].isoformat(),
                    "starting_equity": float(st.session_state["ev_eq"]),
                    "pop_size": int(st.session_state["ev_pop"]),
                    "generations": int(st.session_state["ev_gen"]),
                    "crossover_rate": float(st.session_state["ev_cr"]),
                    "mutation_rate": float(st.session_state["ev_mr"]),
                    "bounds": {
                        "breakout_min": int(st.session_state["ev_bmin"]),
                        "breakout_max": int(st.session_state["ev_bmax"]),
                        "exit_min": int(st.session_state["ev_emin"]),
                        "exit_max": int(st.session_state["ev_emax"]),
                        "atr_min": int(st.session_state["ev_amin"]),
                        "atr_max": int(st.session_state["ev_amax"]),
                        "atr_multiple_min": float(st.session_state["ev_am_min"]),
                        "atr_multiple_max": float(st.session_state["ev_am_max"]),
                        "risk_per_trade_min": float(st.session_state["ev_rpt_min"]) / 100.0,
                        "risk_per_trade_max": float(st.session_state["ev_rpt_max"]) / 100.0,
                    },
                }
                rec = save_param_bounds(symbol_ss, "atr_breakout", profile, name=prof_name, is_default=False)
                st.success(f"Saved profile: {rec['name']}  ({rec['id'][:8]})")

        # -------- Save Best Strategy (FORM) --------
        with st.form("save_best_form", clear_on_submit=False):
            st.markdown("**Save Best Model Params**")
            default_best_name = f"{symbol_ss}-Best"
            best_name = st.text_input("Strategy name", value=default_best_name, key="ev_best_name")
            set_default_model = st.checkbox("Set as default model params for this ticker", value=False, key="ev_best_default")
            save_best = st.form_submit_button("Save Best Strategy")
            if save_best:
                srec = save_strategy(symbol_ss, "atr_breakout", st.session_state["ev_best_params"], name=best_name, is_default=set_default_model)
                if set_default_model:
                    set_default_strategy(symbol_ss, "atr_breakout", srec["id"])
                st.success(f"Saved best params as: {srec['name']}  ({srec['id'][:8]})")

        # Also update portfolio save hook
        st.session_state["last_result"] = {"symbol": symbol_ss, "params": st.session_state["ev_best_params"]}

    # ---------- 3) Run evolutionary tuning ----------
    prog = st.progress(0, text="Idle")

    if st.button("Run Evolutionary Tuning", type="primary", key="btn_ev"):
        bounds = Bounds(
            breakout_min=int(st.session_state["ev_bmin"]),
            breakout_max=int(st.session_state["ev_bmax"]),
            exit_min=int(st.session_state["ev_emin"]),
            exit_max=int(st.session_state["ev_emax"]),
            atr_min=int(st.session_state["ev_amin"]),
            atr_max=int(st.session_state["ev_amax"]),
            atr_multiple_min=float(st.session_state["ev_am_min"]),
            atr_multiple_max=float(st.session_state["ev_am_max"]),
            risk_per_trade_min=float(st.session_state["ev_rpt_min"]) / 100.0,
            risk_per_trade_max=float(st.session_state["ev_rpt_max"]) / 100.0,
        )

        def _cb(done: int, total: int, best_fit: float):
            prog.progress(min(int(100 * done / total), 100), text=f"Generation {done}/{total} — best Sharpe: {best_fit:.2f}")

        with st.spinner("Evolving…"):
            try:
                best_params, best_metrics, _history = evolve_params(
                    symbol=st.session_state["ev_symbol"],
                    start=st.session_state["ev_start"].isoformat(),
                    end=st.session_state["ev_end"].isoformat(),
                    starting_equity=float(st.session_state["ev_eq"]),
                    bounds=bounds,
                    pop_size=int(st.session_state["ev_pop"]),
                    generations=int(st.session_state["ev_gen"]),
                    crossover_rate=float(st.session_state["ev_cr"]),
                    mutation_rate=float(st.session_state["ev_mr"]),
                    random_seed=42,
                    progress_cb=_cb,
                )
                prog.progress(100, text="Done")

                # Compute and persist results so they survive reruns
                res_best = backtest_single(
                    st.session_state["ev_symbol"],
                    st.session_state["ev_start"].isoformat(),
                    st.session_state["ev_end"].isoformat(),
                    best_params["breakout_n"],
                    best_params["exit_n"],
                    best_params["atr_n"],
                    float(st.session_state["ev_eq"]),
                    atr_multiple=best_params["atr_multiple"],
                    risk_per_trade=best_params["risk_per_trade"],
                    allow_fractional=True,
                )

                st.session_state["ev_best_params"] = best_params
                st.session_state["ev_best_metrics"] = res_best["metrics"]  # use backtested metrics for consistency
                st.session_state["ev_best_equity"] = res_best["equity"]

                st.success("Evolution complete.")
            except Exception as e:
                st.error(f"Error during evolution: {e}")

    # Render results if available (persists across reruns/checkbox toggles)
    render_ev_results()

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