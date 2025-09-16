# pages/2_Strategy_Adapter.py
# ---------------------------------------------------------------------
# Strategy Adapter — Evolutionary Trainer (with tooltips)
# - Runs evolutionary search over strategy params across a portfolio
# - Live progress UI + JSONL log download
# - Every control has a help tooltip explaining impact
# ---------------------------------------------------------------------

from __future__ import annotations
from pathlib import Path
import sys
from datetime import date, timedelta, datetime

import pandas as pd
import streamlit as st

# --- sys.path bootstrap so `import src.*` works when running from /pages ---
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# --------------------------------------------------------------------------

from src.storage import list_portfolios, load_portfolio
from src.optimization.evolutionary import evolutionary_search


# ------------------------------- UI HELPERS -------------------------------

def _default_dates(years: int = 3) -> tuple[date, date]:
    end = date.today()
    start = end - timedelta(days=int(365.25 * years))
    return start, end


def _top_table(scored):
    """scored = List[(params: dict, score: float)] -> DataFrame"""
    rows = []
    for params, score in scored:
        row = {"score": float(score)}
        row.update({k: params.get(k) for k in params.keys()})
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("score", ascending=False)
    return df


def _make_progress_cb(status_box, indiv_box, gen_box):
    """Return a progress callback that updates Streamlit placeholders."""
    history = []

    def cb(event: str, payload: dict):
        nonlocal history

        if event == "generation_start":
            gen = payload["gen"]
            pop = payload["pop_size"]
            status_box.info(f"Generation {gen} starting … (population={pop})")

        elif event == "individual_evaluated":
            gen = payload["gen"]
            idx = payload["idx"]
            score = payload.get("score", 0.0)
            m = payload.get("metrics", {}) or {}
            trades = int(m.get("trades", 0) or 0)
            hold = float(m.get("avg_holding_days", 0.0) or 0.0)
            cagr = float(m.get("cagr", 0.0) or 0.0)
            calmar = float(m.get("calmar", 0.0) or 0.0)
            sharpe = float(m.get("sharpe", 0.0) or 0.0)

            history.append({
                "gen": gen, "idx": idx, "score": score,
                "trades": trades, "avg_hold_days": hold,
                "cagr": cagr, "calmar": calmar, "sharpe": sharpe
            })
            history = history[-100:]  # keep last 100 to avoid huge UI
            indiv_box.dataframe(pd.DataFrame(history), use_container_width=True, height=260)

        elif event == "generation_end":
            gen = payload["gen"]
            best = payload.get("best_score", 0.0)
            avg = payload.get("avg_score", 0.0)
            avg_tr = payload.get("avg_trades", 0.0)
            no_tr_pct = 100.0 * payload.get("pct_no_trades", 0.0)
            elite_n = payload.get("elite_n")
            breed_n = payload.get("breed_n")
            inject_n = payload.get("inject_n")

            df = pd.DataFrame([{
                "generation": gen,
                "best_score": best,
                "avg_score": avg,
                "avg_trades": avg_tr,
                "no_trades_%": no_tr_pct,
                "elite_n": elite_n,
                "breed_n": breed_n,
                "inject_n": inject_n,
            }])
            gen_box.dataframe(df, use_container_width=True, height=88)

        elif event == "done":
            secs = payload.get("elapsed_sec", 0.0)
            status_box.success(f"EA completed in {secs:.1f}s")
        else:
            status_box.write({"event": event, "payload": payload})

    return cb


def _ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------- PAGE ----------------------------------

st.title("Strategy Adapter — Evolutionary Trainer")

cfg, out = st.columns([1, 2], gap="large")

with cfg:
    st.subheader("Portfolio & Data")

    # Portfolios
    try:
        portfolios = list_portfolios()
    except Exception as e:
        portfolios = []
        st.error(f"storage.list_portfolios() failed: {e}")

    if not portfolios:
        st.warning("No portfolios found. Add one via your storage module.")
        st.stop()

    port_name = st.selectbox(
        "Portfolio",
        options=portfolios,
        index=0,
        help="Select which saved portfolio (list of tickers) to optimize across. "
             "The EA evaluates parameter sets on this entire ticker set."
    )
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

    # Strategy (add more dotted paths as you build them)
    strategy_dotted = st.selectbox(
        "Strategy module",
        options=["src.models.atr_breakout"],
        index=0,
        help="The Python module implementing your strategy. It must expose "
             "`run_strategy(symbol, start, end, starting_equity, params)` and return "
             "an equity curve, daily returns, and trades."
    )

    # Dates & equity
    d_start, d_end = _default_dates(3)
    c1, c2 = st.columns(2)
    start = c1.date_input(
        "Start",
        value=d_start,
        help="Backtest start date. Ensure it’s long enough for your lookback windows to warm up."
    )
    end = c2.date_input(
        "End",
        value=d_end,
        help="Backtest end date. The EA evaluates each candidate over this full period."
    )
    starting_equity = st.number_input(
        "Starting equity per symbol",
        min_value=100.0, max_value=10_000_000.0,
        value=10_000.0, step=100.0,
        help="Capital allocated to each symbol at the start of the test. "
             "Aggregate portfolio metrics are computed across equal-weighted symbol curves."
    )

    st.divider()
    st.subheader("Parameter Space (bounds for EA sampling)")
    c1, c2 = st.columns(2)

    breakout_n = c1.slider(
        "breakout_n (low, high)", 5, 100, (10, 30),
        help="Lookback window (days) for the rolling high breakout. "
             "Higher → fewer signals, potentially longer holds. Lower → more signals, more whipsaws."
    )
    exit_n = c1.slider(
        "exit_n (low, high)", 4, 60, (5, 15),
        help="Exit lookback (days), often a trailing low or opposite channel. "
             "Higher → exits later (longer holds). Lower → exits quicker (shorter holds)."
    )
    atr_n = c1.slider(
        "atr_n (low, high)", 5, 60, (10, 20),
        help="ATR window (days). Larger smooths ATR (fewer trades). Smaller reacts faster (more trades)."
    )

    atr_multiple = c2.slider(
        "atr_multiple (low, high)", 0.5, 5.0, (1.2, 2.5),
        help="How many ATRs below the breakout high to allow as a cushion. "
             "Higher → stricter entries (fewer trades). Lower → looser entries (more trades)."
    )
    tp_multiple = c2.slider(
        "tp_multiple (low, high)", 0.0, 5.0, (0.0, 1.5),
        help="Optional profit target in multiples of ATR. 0 disables the TP. "
             "Higher → earlier profit-taking. Lower → let winners run."
    )
    holding_limit = c2.slider(
        "holding_period_limit (low, high)", 0, 252, (0, 60),
        help="Max days to hold a position. 0 disables. Useful to prevent capital being locked for too long."
    )

    param_space = {
        "breakout_n": (int(breakout_n[0]), int(breakout_n[1])),
        "exit_n": (int(exit_n[0]), int(exit_n[1])),
        "atr_n": (int(atr_n[0]), int(atr_n[1])),
        "atr_multiple": (float(atr_multiple[0]), float(atr_multiple[1])),
        "tp_multiple": (float(tp_multiple[0]), float(tp_multiple[1])),
        "holding_period_limit": (int(holding_limit[0]), int(holding_limit[1])),
    }

    st.divider()
    st.subheader("EA Controls")
    c1, c2, c3 = st.columns(3)

    generations = c1.number_input(
        "generations", 1, 200, 10, 1,
        help="How many evolutionary steps to run. More generations → more exploration and refinement."
    )
    pop_size = c1.number_input(
        "population size", 2, 200, 30, 1,
        help="Number of parameter sets evaluated per generation. Larger pops explore more in parallel."
    )

    mutation_rate = c2.slider(
        "mutation_rate", 0.0, 1.0, 0.4, 0.05,
        help="Chance a given parameter is randomly perturbed in a child. "
             "Higher → more exploration (risk of noise). Lower → more exploitation (risk of stagnation)."
    )
    elite_frac = c2.slider(
        "elite_frac", 0.0, 0.95, 0.5, 0.05,
        help="Fraction of top performers kept each generation (elitism). "
             "Higher preserves best solutions. Too high can reduce diversity."
    )
    random_inject_frac = c2.slider(
        "random_inject_frac", 0.0, 0.95, 0.2, 0.05,
        help="Fraction of fresh random individuals injected each gen. "
             "Maintains diversity and helps escape local optima."
    )

    min_trades = c3.number_input(
        "min_trades (gate)", 0, 1000, 10, 1,
        help="Hard gate: individuals with fewer trades than this score 0 and can’t survive selection. "
             "Prevents degenerate, no-trade ‘winners’."
    )
    require_hold_days = c3.checkbox(
        "require avg_holding_days > 0", value=False,
        help="If on, any individual with non-positive average holding days is discarded (score=0)."
    )
    eps_mdd = c3.number_input(
        "eps_mdd", 0.0, 0.01, 1e-4, step=1e-4, format="%.4f",
        help="Minimum |Max Drawdown| used to avoid Calmar blow-ups. "
             "If |MDD| < eps, Calmar is treated as 0."
    )
    eps_sharpe = c3.number_input(
        "eps_sharpe", 0.0, 0.01, 1e-4, step=1e-4, format="%.4f",
        help="Minimum |Sharpe| used to avoid tiny-std explosions. "
             "If |Sharpe| < eps, Sharpe is treated as 0."
    )

    st.divider()
    st.subheader("Fitness Weights & Holding Window (Swing-friendly)")
    c1, c2 = st.columns(2)

    alpha_cagr = c1.number_input(
        "α (CAGR weight)", 0.0, 5.0, 1.0, 0.1,
        help="Weight on CAGR (growth). Increasing α chases higher terminal equity."
    )
    beta_calmar = c1.number_input(
        "β (Calmar weight)", 0.0, 5.0, 1.0, 0.1,
        help="Weight on Calmar (CAGR / MaxDD). Increasing β favors better risk-adjusted growth."
    )
    gamma_sharpe = c1.number_input(
        "γ (Sharpe weight)", 0.0, 5.0, 0.25, 0.05,
        help="Weight on Sharpe (mean/vol). Increasing γ favors smoother equity curves."
    )

    min_holding_days = c2.number_input(
        "min_holding_days", 0.0, 100.0, 3.0, 0.5,
        help="Preferred lower bound for average holding period. Helps avoid day-trading behavior."
    )
    max_holding_days = c2.number_input(
        "max_holding_days", 0.0, 365.0, 30.0, 1.0,
        help="Preferred upper bound for average holding period. Helps avoid tying up capital for too long."
    )
    holding_penalty_weight = c2.number_input(
        "holding_penalty_weight (λ)", 0.0, 5.0, 0.1, 0.05,
        help="Strength of the penalty when average holding days falls outside the preferred band. "
             "Higher λ enforces swing-trade behavior more strictly."
    )

    # Run setup
    log_dir = ROOT / "logs"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = _ensure_dir(log_dir / f"ea_{port_name}_{ts}.jsonl").as_posix()

    go = st.button(
        "Run Evolutionary Search",
        type="primary",
        use_container_width=True,
        help="Start optimizing parameters across the selected portfolio with the settings above."
    )


with out:
    st.subheader("Run Status & Results")
    status_box = st.empty()
    indiv_box = st.empty()
    gen_box = st.empty()
    results_box = st.container()
    download_box = st.container()

    if go:
        progress_cb = _make_progress_cb(status_box, indiv_box, gen_box)

        with st.spinner("Running EA…"):
            scored = evolutionary_search(
                strategy_dotted=strategy_dotted,
                tickers=tickers,
                start=start,
                end=end,
                starting_equity=starting_equity,
                param_space=param_space,
                generations=int(generations),
                pop_size=int(pop_size),
                mutation_rate=float(mutation_rate),
                elite_frac=float(elite_frac),
                random_inject_frac=float(random_inject_frac),
                min_trades=int(min_trades),
                require_hold_days=bool(require_hold_days),
                eps_mdd=float(eps_mdd),
                eps_sharpe=float(eps_sharpe),
                alpha_cagr=float(alpha_cagr),
                beta_calmar=float(beta_calmar),
                gamma_sharpe=float(gamma_sharpe),
                min_holding_days=float(min_holding_days),
                max_holding_days=float(max_holding_days),
                holding_penalty_weight=float(holding_penalty_weight),
                progress_cb=progress_cb,
                log_file=log_file,
            )

        # Render top results table
        df = _top_table(scored)
        if df.empty:
            results_box.warning("No results returned. Check data, dates, or parameter space.")
        else:
            results_box.dataframe(df, use_container_width=True, height=360)

        # Download log
        try:
            txt = Path(log_file).read_text(encoding="utf-8")
            download_box.download_button(
                "Download EA log (JSONL)",
                data=txt,
                file_name=Path(log_file).name,
                mime="application/json",
                use_container_width=True,
                help="Download the full run log as JSON Lines for troubleshooting or audit."
            )
            st.caption(f"Log file: {log_file}")
        except Exception as e:
            download_box.error(f"Could not read log file: {e}")

    else:
        st.info("Fill in the configuration on the left, then press **Run Evolutionary Search**.")