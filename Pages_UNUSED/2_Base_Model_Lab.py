# pages/2_Strategy_Adapter.py
from __future__ import annotations
from pathlib import Path
import sys
from datetime import date, timedelta, datetime
import os

# Avoid BLAS/OMP oversubscription per worker
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.storage import list_portfolios, load_portfolio
from src.optimization.evolutionary import evolutionary_search


def _default_dates(years: int = 3) -> tuple[date, date]:
    end = date.today()
    start = end - timedelta(days=int(365.25 * years))
    return start, end


def _top_table(scored):
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
            history = history[-100:]
            # CHANGED: width="stretch" replaces deprecated use_container_width
            indiv_box.dataframe(pd.DataFrame(history), width="stretch", height=260)

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
            # CHANGED: width="stretch"
            gen_box.dataframe(df, width="stretch", height=88)

        elif event == "done":
            secs = payload.get("elapsed_sec", 0.0)
            status_box.success(f"EA completed in {secs:.1f}s")
        else:
            status_box.write({"event": event, "payload": payload})

    return cb


def _ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


st.title("Strategy Adapter — Evolutionary Trainer")

# Performance at top
st.subheader("Performance", help="Control CPU parallelism for faster runs.")
cpu_cnt = os.cpu_count() or 1
default_workers = 16 if cpu_cnt >= 24 else max(1, min(8, cpu_cnt - 1))
n_jobs = st.slider(
    "n_jobs (parallel workers)",
    min_value=1,
    max_value=max(2, cpu_cnt),
    value=min(default_workers, cpu_cnt),
    help=(
        "Number of worker processes to evaluate candidates in parallel. "
        "On Apple Silicon, 12–16 is a good starting point. "
        "Each worker is restricted to 1 BLAS/OMP thread to avoid oversubscription."
    ),
)
st.caption(f"Detected CPU cores: {cpu_cnt} • Defaulted workers: {min(default_workers, cpu_cnt)}")

cfg, out = st.columns([1, 2], gap="large")

with cfg:
    st.subheader("Portfolio & Data")

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
        help="Choose which saved portfolio (list of tickers) to optimize across."
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

    strategy_dotted = st.selectbox(
        "Strategy module",
        options=["src.models.atr_breakout"],
        index=0,
        help="Module must expose run_strategy(symbol, start, end, starting_equity, params)."
    )

    d_start, d_end = _default_dates(3)
    c1, c2 = st.columns(2)
    start = c1.date_input(
        "Start", value=d_start,
        help="Backtest start date. Ensure it’s long enough for your lookback windows to warm up."
    )
    end = c2.date_input(
        "End", value=d_end,
        help="Backtest end date. The EA evaluates each candidate over this full period."
    )
    starting_equity = st.number_input(
        "Starting equity per symbol",
        min_value=100.0, max_value=10_000_000.0,
        value=10_000.0, step=100.0,
        help="Capital allocated to each symbol at the start of the test."
    )

    st.divider()
    st.subheader("Parameter Space (bounds for EA sampling)")
    c1, c2 = st.columns(2)

    breakout_n = c1.slider(
        "breakout_n (low, high)", 5, 100, (10, 30),
        help="Lookback for breakout high. Higher → fewer signals, longer holds; lower → more signals."
    )
    exit_n = c1.slider(
        "exit_n (low, high)", 4, 60, (5, 15),
        help="Exit lookback window. Higher → exits later; lower → exits quicker."
    )
    atr_n = c1.slider(
        "atr_n (low, high)", 5, 60, (10, 20),
        help="ATR smoothing window. Larger → smoother ATR, fewer signals; smaller → more reactive."
    )

    atr_multiple = c2.slider(
        "atr_multiple (low, high)", 0.5, 5.0, (1.2, 2.5),
        help="Entry cushion in ATRs. Higher → stricter entries (fewer trades)."
    )
    tp_multiple = c2.slider(
        "tp_multiple (low, high)", 0.0, 5.0, (0.0, 1.5),
        help="Profit target in ATRs. 0 disables TP. Higher → take profits earlier."
    )
    holding_limit = c2.slider(
        "holding_period_limit (low, high)", 0, 252, (0, 60),
        help="Max days to hold a position. 0 disables."
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
        "generations", 1, 500, 10, 1,
        help="How many evolutionary steps to run."
    )
    pop_size = c1.number_input(
        "population size", 2, 500, 30, 1,
        help="Number of parameter sets evaluated per generation."
    )
    st.caption("Tip: set population ≥ n_jobs for full CPU utilization.")

    mutation_rate = c2.slider(
        "mutation_rate", 0.0, 1.0, 0.4, 0.05,
        help="Chance a given parameter is randomly perturbed in a child."
    )
    elite_frac = c2.slider(
        "elite_frac", 0.0, 0.95, 0.5, 0.05,
        help="Fraction of top performers kept each generation (elitism)."
    )
    random_inject_frac = c2.slider(
        "random_inject_frac", 0.0, 0.95, 0.2, 0.05,
        help="Fraction of fresh random individuals injected each generation."
    )

    min_trades = c3.number_input(
        "min_trades (gate)", 0, 10000, 10, 1,
        help="Hard gate: individuals with fewer trades than this score 0."
    )
    min_hold_gate = c3.number_input(
        "min_avg_holding_days_gate", 0.0, 30.0, 1.0, 0.5,
        help="Hard gate: score=0 if avg holding days is below this (avoid day-trading)."
    )
    require_hold_days = c3.checkbox(
        "require avg_holding_days > 0", value=False,
        help="If on, any individual with non-positive avg holding days is discarded."
    )

    st.divider()
    st.subheader("Fitness Weights & Preferences")
    c1, c2 = st.columns(2)

    alpha_cagr = c1.number_input(
        "α (CAGR weight)", 0.0, 5.0, 1.0, 0.1,
        help="Weight on CAGR (growth)."
    )
    beta_calmar = c1.number_input(
        "β (Calmar weight)", 0.0, 5.0, 1.0, 0.1,
        help="Weight on Calmar (growth vs drawdown)."
    )
    gamma_sharpe = c1.number_input(
        "γ (Sharpe weight)", 0.0, 5.0, 0.25, 0.05,
        help="Weight on Sharpe (smoothness)."
    )

    min_holding_days = c2.number_input(
        "min_holding_days", 0.0, 100.0, 3.0, 0.5,
        help="Preferred lower bound for average holding period."
    )
    max_holding_days = c2.number_input(
        "max_holding_days", 0.0, 365.0, 30.0, 1.0,
        help="Preferred upper bound for average holding period."
    )
    holding_penalty_weight = c2.number_input(
        "holding_penalty_weight (λ)", 0.0, 5.0, 1.0, 0.05,
        help="Penalty weight for falling outside the holding period band."
    )

    st.divider()
    st.subheader("Trade-Rate Preference (per symbol per year)")
    c1, c2, c3 = st.columns(3)
    trade_rate_min = c1.number_input(
        "trade_rate_min", 0.0, 365.0, 5.0, 1.0,
        help="Preferred lower bound for trades per symbol per year."
    )
    trade_rate_max = c2.number_input(
        "trade_rate_max", 0.0, 365.0, 50.0, 1.0,
        help="Preferred upper bound for trades per symbol per year."
    )
    trade_rate_penalty_weight = c3.number_input(
        "trade_rate_penalty_weight (λ)", 0.0, 5.0, 0.5, 0.05,
        help="Penalty weight for being outside the trade-rate band."
    )

    log_dir = ROOT / "logs"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = _ensure_dir(log_dir / f"ea_{port_name}_{ts}.jsonl").as_posix()

    go = st.button(
        "Run Evolutionary Search",
        type="primary",
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
        n_jobs_eff = int(min(max(1, n_jobs), int(pop_size)))

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
                n_jobs=n_jobs_eff,
                min_trades=int(min_trades),
                min_avg_holding_days_gate=float(min_hold_gate),
                require_hold_days=bool(require_hold_days),
                eps_mdd=1e-4,
                eps_sharpe=1e-4,
                alpha_cagr=float(alpha_cagr),
                beta_calmar=float(beta_calmar),
                gamma_sharpe=float(gamma_sharpe),
                min_holding_days=float(min_holding_days),
                max_holding_days=float(max_holding_days),
                holding_penalty_weight=float(holding_penalty_weight),
                trade_rate_min=float(trade_rate_min),
                trade_rate_max=float(trade_rate_max),
                trade_rate_penalty_weight=float(trade_rate_penalty_weight),
                progress_cb=progress_cb,
                log_file=log_file,
            )

        # Top results table (CHANGED: width="stretch")
        df = _top_table(scored)
        if df.empty:
            results_box.warning("No results returned. Check data, dates, or parameter space.")
        else:
            results_box.dataframe(df, width="stretch", height=360)

        # Log download (CHANGED: width="stretch")
        try:
            txt = Path(log_file).read_text(encoding="utf-8")
            download_box.download_button(
                "Download EA log (JSONL)",
                data=txt,
                file_name=Path(log_file).name,
                mime="application/json",
                width="stretch",
                help="Download the full run log as JSON Lines for troubleshooting or audit."
            )
            st.caption(f"Log file: {log_file}")
        except Exception as e:
            download_box.error(f"Could not read log file: {e}")

    else:
        st.info("Set **n_jobs** at the top, fill in the configuration on the left, then press **Run Evolutionary Search**.")