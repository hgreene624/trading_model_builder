# pages/5_EA_Train_Test_Inspector.py
from __future__ import annotations
import json, os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ---------- settings ----------
LOG_DIR = Path("storage/logs/ea")
DEFAULT_PAGE_TITLE = "EA Train/Test Inspector"

# ---------- helpers ----------

def _latest_log_file(dirpath: Path) -> Optional[Path]:
    if not dirpath.exists():
        return None
    files = sorted(dirpath.glob("*_ea.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None

@st.cache_data(show_spinner=False)
def load_ea_log(log_path: str) -> pd.DataFrame:
    rows = []
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return pd.DataFrame(rows)

def _get_session_meta(df: pd.DataFrame) -> Dict[str, Any]:
    # Prefer explicit session_meta if present
    smeta_rows = df[df["event"]=="session_meta"]["payload"].apply(pd.Series) if ("event" in df.columns and (df["event"]=="session_meta").any()) else pd.DataFrame()
    if not smeta_rows.empty:
        return smeta_rows.iloc[-1].to_dict()
    # Fallback: infer train window from evolutionary_search args is not possible here; leave None
    return {}

@st.cache_data(show_spinner=False)
def _eval_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return tidy table of individual evaluations with metrics."""
    evals = df[df["event"]=="individual_evaluated"]["payload"].apply(pd.Series).reset_index(drop=True)
    if evals.empty:
        return pd.DataFrame()
    metrics = pd.json_normalize(evals["metrics"]).reset_index(drop=True)
    out = pd.concat([evals.drop(columns=["metrics"]), metrics], axis=1)
    return out

@st.cache_data(show_spinner=False)
def _gen_end_table(df: pd.DataFrame) -> pd.DataFrame:
    if "event" not in df.columns:
        return pd.DataFrame()
    g = df[df["event"]=="generation_end"]["payload"].apply(pd.Series)
    return g.reset_index(drop=True) if not g.empty else pd.DataFrame()

# --- equity simulators (pluggable) ---

def _hashable_params(d: Dict[str, Any]) -> Tuple:
    # makes a stable key for cache
    return tuple(sorted(d.items()))

@st.cache_data(show_spinner=True)
def run_equity_curve(
    strategy_dotted: str,
    tickers: List[str],
    start_iso: str,
    end_iso: str,
    starting_equity: float,
    params: Dict[str, Any],
) -> pd.DataFrame:
    """
    Try to produce an equity curve [date,equity] for the window.
    We attempt train_general_model first; if that does not return a curve,
    users can adapt here to call their holdout_chart runner.
    """
    # 1) primary: general_trainer
    try:
        from src.models.general_trainer import train_general_model
        res = train_general_model(strategy_dotted, tickers, start_iso, end_iso, starting_equity, params)
        # Try common shapes:
        # res["aggregate"]["equity_curve"] -> list of [ts, equity] or dict with 'date','equity'
        agg = (res or {}).get("aggregate", {})
        curve = agg.get("equity_curve") or agg.get("curve") or None
        if isinstance(curve, list) and curve and isinstance(curve[0], (list, tuple)):
            df = pd.DataFrame(curve, columns=["date","equity"])
            df["date"] = pd.to_datetime(df["date"])
            return df
        # sometimes a dict of arrays
        if isinstance(curve, dict) and "date" in curve and "equity" in curve:
            df = pd.DataFrame({"date": pd.to_datetime(curve["date"]), "equity": curve["equity"]})
            return df
        # Fallback: try day-level portfolio in res
        if "portfolio" in res and isinstance(res["portfolio"], dict):
            p = res["portfolio"]
            if "date" in p and "equity" in p:
                return pd.DataFrame({"date": pd.to_datetime(p["date"]), "equity": p["equity"]})
    except Exception:
        pass

    # 2) optional: project-specific holdout runner (uncomment/adjust if you have one)
    try:
        import holdout_chart as hc  # adjust if module path differs
        if hasattr(hc, "simulate_holdout"):
            ec, _stats = hc.simulate_holdout(params, start_iso, end_iso)  # signature may differ in your project
            # expecting ec as list of [date,equity] or a DataFrame
            if isinstance(ec, pd.DataFrame):
                df = ec.copy()
            else:
                df = pd.DataFrame(ec, columns=["date","equity"])
            df["date"] = pd.to_datetime(df["date"])
            return df
    except Exception:
        pass

    # 3) last resort: return a 2-point flat line to avoid crashing the UI
    return pd.DataFrame({"date": pd.to_datetime([start_iso, end_iso]), "equity": [starting_equity, starting_equity]})

def _plot_gen_topK(
    eval_df: pd.DataFrame,
    gen_idx: int,
    k: int,
    strategy: str,
    tickers: List[str],
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    starting_equity: float,
) -> go.Figure:
    G = eval_df[eval_df["gen"]==gen_idx].copy()
    if G.empty:
        return go.Figure()
    # rank by final return in that gen
    G = G.sort_values(by="total_return", ascending=False).head(min(k, len(G)))
    # plot
    fig = go.Figure()
    for i, row in G.iterrows():
        params = row["params"]
        ec_train = run_equity_curve(strategy, tickers, train_start, train_end, starting_equity, params)
        ec_test  = run_equity_curve(strategy, tickers, test_start,  test_end,  ec_train["equity"].iloc[-1] if not ec_train.empty else starting_equity, params)
        # combine for seamless line (train then test)
        ec = pd.concat([ec_train, ec_test], ignore_index=True)
        name = f"gen{gen_idx} idx{int(row['idx'])} (ret {row['total_return']:.3f})"
        fig.add_trace(go.Scatter(x=ec["date"], y=ec["equity"], mode="lines", name=name, line=dict(width=1)))
    # demarcation line at train_end
    fig.add_vline(x=pd.to_datetime(train_end), line_width=2, line_dash="dot", line_color="#888")
    fig.update_layout(
        title=f"Gen {gen_idx}: Top-{k} by final return (train + test)",
        xaxis_title="Date",
        yaxis_title="Equity",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=420,
    )
    return fig

def _plot_leaders_through_gen(
    eval_df: pd.DataFrame,
    upto_gen: int,
    strategy: str,
    tickers: List[str],
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    starting_equity: float,
) -> go.Figure:
    fig = go.Figure()
    all_gens = sorted(eval_df["gen"].unique())
    for g in [g for g in all_gens if g <= upto_gen]:
        G = eval_df[eval_df["gen"]==g]
        if G.empty:
            continue
        row = G.loc[G["total_return"].idxmax()]  # best by return in that gen
        params = row["params"]
        ec_train = run_equity_curve(strategy, tickers, train_start, train_end, starting_equity, params)
        ec_test  = run_equity_curve(strategy, tickers, test_start,  test_end,  ec_train["equity"].iloc[-1] if not ec_train.empty else starting_equity, params)
        ec = pd.concat([ec_train, ec_test], ignore_index=True)
        name = f"Gen {g} (ret {row['total_return']:.3f})"
        fig.add_trace(go.Scatter(x=ec["date"], y=ec["equity"], mode="lines", name=name, line=dict(width=1)))
    fig.add_vline(x=pd.to_datetime(train_end), line_width=2, line_dash="dot", line_color="#888")
    fig.update_layout(
        title=f"Leaders up to Gen {upto_gen} (best-by-return per gen | train + test)",
        xaxis_title="Date", yaxis_title="Equity",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=420,
    )
    return fig

# ---------- UI ----------

def _file_picker():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(LOG_DIR.glob("*_ea.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    labels = [f.name for f in files]
    if not files:
        st.error(f"No EA logs found in {LOG_DIR}")
        return None
    idx = st.selectbox("EA log", options=list(range(len(files))), format_func=lambda i: labels[i], index=0)
    return files[idx]

def main():
    st.set_page_config(page_title=DEFAULT_PAGE_TITLE, layout="wide")
    st.title(DEFAULT_PAGE_TITLE)

    # pick a log (default to latest)
    log_file = _file_picker() or _latest_log_file(LOG_DIR)
    if not log_file:
        st.stop()

    df = load_ea_log(str(log_file))
    eval_df = _eval_table(df)
    gen_df = _gen_end_table(df)

    # session meta (train window, etc.)
    smeta = _get_session_meta(df)
    strategy = smeta.get("strategy", "src.models.atr_breakout:Strategy")
    tickers = smeta.get("tickers", [])
    starting_equity = float(smeta.get("starting_equity", 10000.0))
    train_start = smeta.get("train_start")
    train_end   = smeta.get("train_end")

    # Controls row
    c1, c2, c3, c4, c5 = st.columns([2,2,2,2,2])

    # Train/Test pickers (train defaults from log; test you can set)
    with c1:
        st.markdown("**Training window**")
        train_start = st.text_input("Train start (ISO)", value=str(train_start) if train_start else "")
        train_end   = st.text_input("Train end (ISO)", value=str(train_end) if train_end else "")
    with c2:
        st.markdown("**Test window**")
        # default: next day after train_end → today
        default_test_start = ""
        if train_end:
            try:
                default_test_start = (pd.to_datetime(train_end) + pd.Timedelta(days=1)).date().isoformat()
            except Exception:
                default_test_start = ""
        test_start = st.text_input("Test start (ISO)", value=default_test_start)
        test_end   = st.text_input("Test end (ISO)", value="")
    with c3:
        st.markdown("**Top-K (current gen)**")
        top_k = st.number_input("K", min_value=1, max_value=50, value=5, step=1)
    with c4:
        st.markdown("**Playback**")
        max_gen = int(eval_df["gen"].max()) if not eval_df.empty else 0
        if "ea_inspect_gen" not in st.session_state:
            st.session_state.ea_inspect_gen = 0
        # stepper
        cols = st.columns(3)
        if cols[0].button("⟵ Prev", use_container_width=True):
            st.session_state.ea_inspect_gen = max(0, st.session_state.ea_inspect_gen - 1)
        if cols[2].button("Next ⟶", use_container_width=True):
            st.session_state.ea_inspect_gen = min(max_gen, st.session_state.ea_inspect_gen + 1)
        # play controls
        play = st.checkbox("Play", value=False)
        speed = st.slider("Speed (gens/sec)", min_value=0.2, max_value=5.0, value=1.0, step=0.1)
        if play:
            # use autorefresh to tick forward
            interval_ms = int(1000.0 / max(0.2, float(speed)))
            st.experimental_rerun()  # first rerun to apply; below we use autorefresh
    with c5:
        st.markdown("**Generation**")
        st.session_state.ea_inspect_gen = st.slider("Gen", 0, int(eval_df["gen"].max() if not eval_df.empty else 0), int(st.session_state.ea_inspect_gen))

    # safety
    if not train_start or not train_end:
        st.warning("Training dates missing. Enter train_start/train_end (ISO) or run a new EA with session_meta logging.")
        st.stop()
    if not test_start or not test_end:
        st.info("Tip: set a test window to visualize out-of-sample performance (e.g., next 60–90 days).")

    # ---- Chart 1: top-K of current generation (by return) ----
    st.subheader("Chart 1 — Top-K of current generation (by final return)")
    fig1 = _plot_gen_topK(
        eval_df=eval_df,
        gen_idx=int(st.session_state.ea_inspect_gen),
        k=int(top_k),
        strategy=strategy,
        tickers=tickers,
        train_start=train_start, train_end=train_end,
        test_start=test_start,   test_end=test_end,
        starting_equity=starting_equity,
    )
    st.plotly_chart(fig1, use_container_width=True)

    # ---- Chart 2: leaders up to the current generation ----
    st.subheader("Chart 2 — Best-by-return per generation (up to current gen)")
    fig2 = _plot_leaders_through_gen(
        eval_df=eval_df,
        upto_gen=int(st.session_state.ea_inspect_gen),
        strategy=strategy,
        tickers=tickers,
        train_start=train_start, train_end=train_end,
        test_start=test_start,   test_end=test_end,
        starting_equity=starting_equity,
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.caption("A vertical dotted line marks the end of the training window. Curves continue into the test window on the same scale.")

if __name__ == "__main__":
    main()