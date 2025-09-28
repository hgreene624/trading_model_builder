# pages/5_EA_Train_Test_Inspector.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import warnings

# ---------- settings ----------
LOG_DIR = Path("storage/logs/ea")
DEFAULT_PAGE_TITLE = "EA Train/Test Inspector"

# ---------- small debug buffer ----------
def _dbg(msg: str):
    try:
        st.session_state.setdefault("ea_inspector_debug", []).append(str(msg))
    except Exception:
        pass

# ---------- helpers ----------

def _latest_log_file(dirpath: Path) -> Optional[Path]:
    if not dirpath.exists():
        return None
    files = sorted(dirpath.glob("*_ea.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None

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
    smeta_rows = (
        df[df["event"] == "session_meta"]["payload"].apply(pd.Series)
        if ("event" in df.columns and (df["event"] == "session_meta").any())
        else pd.DataFrame()
    )
    return smeta_rows.iloc[-1].to_dict() if not smeta_rows.empty else {}

def _get_holdout_meta(df: pd.DataFrame) -> Dict[str, Any]:
    hmeta_rows = (
        df[df["event"] == "holdout_meta"]["payload"].apply(pd.Series)
        if ("event" in df.columns and (df["event"] == "holdout_meta").any())
        else pd.DataFrame()
    )
    # last one wins
    return hmeta_rows.iloc[-1].to_dict() if not hmeta_rows.empty else {}

def _sanitize_obj_cols(df: pd.DataFrame) -> pd.DataFrame:
    """JSON-encode any dict/list columns (prevents hashing errors later)."""
    if df.empty:
        return df
    out = df.copy()
    for col in list(out.columns):
        if out[col].map(lambda x: isinstance(x, (dict, list))).any():
            out[col] = out[col].apply(lambda x: json.dumps(x, sort_keys=True) if isinstance(x, (dict, list)) else x)
    return out

def _eval_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return tidy table of individual evaluations with metrics.
    NOTE: JSON-encode 'params' and any dict/list columns to avoid caching/hashing issues.
    """
    if "event" not in df.columns:
        return pd.DataFrame()
    evals = df[df["event"] == "individual_evaluated"]["payload"].apply(pd.Series).reset_index(drop=True)
    if evals.empty:
        return pd.DataFrame()
    metrics = pd.json_normalize(evals["metrics"]).reset_index(drop=True)
    out = pd.concat([evals.drop(columns=["metrics"]), metrics], axis=1)
    # Avoid dicts; keep a JSON copy of params
    if "params" in out.columns:
        out["params_json"] = out["params"].apply(
            lambda d: json.dumps(d, sort_keys=True) if isinstance(d, dict) else (d if isinstance(d, str) else "{}")
        )
        out = out.drop(columns=["params"])
    out = _sanitize_obj_cols(out)
    # Make sure 'gen' is numeric
    if "gen" in out.columns:
        out["gen"] = pd.to_numeric(out["gen"], errors="coerce").fillna(0).astype(int)
    return out

def _gen_end_table(df: pd.DataFrame) -> pd.DataFrame:
    if "event" not in df.columns:
        return pd.DataFrame()
    g = df[df["event"] == "generation_end"]["payload"].apply(pd.Series)
    g = _sanitize_obj_cols(g)
    return g.reset_index(drop=True) if not g.empty else pd.DataFrame()

def _row_params(row: pd.Series) -> Dict[str, Any]:
    pj = row.get("params_json")
    try:
        return json.loads(pj) if isinstance(pj, str) else {}
    except Exception:
        return {}

# ---------- equity provider ----------

@st.cache_data(show_spinner=True, hash_funcs={dict: lambda d: json.dumps(d, sort_keys=True)})
def run_equity_curve(
    strategy_dotted: str,
    tickers: List[str],
    start_iso: str,
    end_iso: str,
    starting_equity: float,
    params: Dict[str, Any],
) -> pd.DataFrame:
    """
    Produce an equity curve [date,equity] for the window using the same path the EA uses.
    Tries general_trainer first. If no curve is found, tries multiple signatures against
    src.utils.holdout_chart (or fallback holdout_chart). Emits debug breadcrumbs.
    """
    # Normalize tickers
    if isinstance(tickers, str):
        tickers = [t.strip() for t in tickers.split(",") if t.strip()]
    tickers = list(tickers)

    # 1) general_trainer
    try:
        from src.models.general_trainer import train_general_model
        _dbg("trainer: calling train_general_model(...)")
        res = train_general_model(strategy_dotted, tickers, start_iso, end_iso, starting_equity, params)
        if isinstance(res, dict):
            agg = res.get("aggregate") or {}
            for key in ("equity_curve", "curve", "equity"):
                curve = agg.get(key)
                # list-of-pairs
                if isinstance(curve, list) and curve and isinstance(curve[0], (list, tuple)):
                    df = pd.DataFrame(curve, columns=["date", "equity"])
                    df["date"] = pd.to_datetime(df["date"])
                    _dbg(f"trainer: aggregate.{key} list-of-pairs")
                    return df
                # dict-of-arrays
                if isinstance(curve, dict) and {"date", "equity"}.issubset(curve.keys()):
                    df = pd.DataFrame({"date": pd.to_datetime(curve["date"]), "equity": curve["equity"]})
                    _dbg(f"trainer: aggregate.{key} dict-of-arrays")
                    return df
            p = res.get("portfolio")
            if isinstance(p, dict) and {"date", "equity"}.issubset(p.keys()):
                df = pd.DataFrame({"date": pd.to_datetime(p["date"]), "equity": p["equity"]})
                _dbg("trainer: portfolio{date,equity}")
                return df
        _dbg("trainer: no curve fields found")
    except Exception as e:
        _dbg(f"trainer: exception {type(e).__name__}: {e}")

    # 2) holdout runner (same module Strategy Adapter uses)
    hc = None
    try:
        try:
            from src.utils import holdout_chart as hc_mod
            hc = hc_mod
            _dbg("holdout: using src.utils.holdout_chart")
        except Exception:
            import holdout_chart as hc_mod  # root module fallback
            hc = hc_mod
            _dbg("holdout: using root holdout_chart")
    except Exception as e:
        _dbg(f"holdout: import failed {type(e).__name__}: {e}")

    if hc is not None:
        # Candidate function names and signature variants
        candidates = [
            ("holdout_equity", ("params", "start", "end", "tickers", "starting_equity", "strategy")),
            ("simulate_holdout", ("params", "start", "end", "tickers", "starting_equity", "strategy")),
            ("run_holdout", ("params", "start", "end", "tickers", "starting_equity", "strategy")),
        ]
        args = {
            "params": params,
            "start": start_iso,
            "end": end_iso,
            "tickers": tickers,
            "starting_equity": starting_equity,
            "strategy": strategy_dotted,
        }
        for fn_name, _sig in candidates:
            if hasattr(hc, fn_name):
                fn = getattr(hc, fn_name)
                try:
                    from inspect import signature
                    sig = signature(fn)
                    kwargs = {k: v for k, v in args.items() if k in sig.parameters}
                    _dbg(f"holdout: calling {fn_name} with {list(kwargs.keys())}")
                    ec = fn(**kwargs)
                    if isinstance(ec, tuple):
                        ec = ec[0]
                    if isinstance(ec, pd.DataFrame) and {"date", "equity"}.issubset(ec.columns):
                        df = ec[["date", "equity"]].copy()
                        df["date"] = pd.to_datetime(df["date"])
                        return df
                    if isinstance(ec, list) and ec and isinstance(ec[0], (list, tuple)):
                        df = pd.DataFrame(ec, columns=["date", "equity"])
                        df["date"] = pd.to_datetime(df["date"])
                        return df
                    _dbg(f"holdout: {fn_name} returned no usable curve")
                except Exception as e:
                    _dbg(f"holdout: {fn_name} raised {type(e).__name__}: {e}")

    # 3) last resort: flat line (UI warns)
    _dbg("fallback: flat 2-point line returned")
    return pd.DataFrame({"date": pd.to_datetime([start_iso, end_iso]), "equity": [starting_equity, starting_equity]})

# ---------- plotting ----------

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
    G = eval_df[eval_df["gen"] == gen_idx].copy()
    if G.empty:
        return go.Figure()
    # rank by final return in that gen
    G = G.sort_values(by="total_return", ascending=False).head(min(k, len(G)))

    fig = go.Figure()
    flat_warn = False
    for _, row in G.iterrows():
        params = _row_params(row)
        ec_train = run_equity_curve(strategy, tickers, train_start, train_end, starting_equity, params)
        end_equity = ec_train["equity"].iloc[-1] if not ec_train.empty else starting_equity
        ec_test = run_equity_curve(strategy, tickers, test_start, test_end, end_equity, params)
        ec = pd.concat([ec_train, ec_test], ignore_index=True)
        if len(ec) <= 2 or ec["equity"].nunique() <= 1:
            flat_warn = True
        name = f"gen{gen_idx} idx{int(row['idx'])} (ret {row['total_return']:.3f})"
        fig.add_trace(
            go.Scatter(x=ec["date"], y=ec["equity"], mode="lines", name=name, line=dict(width=1))
        )

    if flat_warn:
        warnings.warn(
            "Equity curve(s) appear flat; the trainer didn't return a curve. Wire your holdout/trainer curve provider in run_equity_curve()."
        )

    # all-flat annotation
    if len(fig.data) > 0:
        all_flat = all((len(t.y) <= 2 or (np.max(t.y) - np.min(t.y) == 0)) for t in fig.data)
        if all_flat:
            fig.add_annotation(
                text="No equity curves returned by simulator. Check run_equity_curve wiring.",
                showarrow=False, xref="paper", yref="paper", x=0.01, y=0.95,
                bgcolor="#ffeeee", bordercolor="#cc0000",
            )

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
        G = eval_df[eval_df["gen"] == g]
        if G.empty:
            continue
        row = G.loc[G["total_return"].idxmax()]  # best by return in that gen
        params = _row_params(row)
        ec_train = run_equity_curve(strategy, tickers, train_start, train_end, starting_equity, params)
        end_equity = ec_train["equity"].iloc[-1] if not ec_train.empty else starting_equity
        ec_test = run_equity_curve(strategy, tickers, test_start, test_end, end_equity, params)
        ec = pd.concat([ec_train, ec_test], ignore_index=True)
        name = f"Gen {g} (ret {row['total_return']:.3f})"
        fig.add_trace(go.Scatter(x=ec["date"], y=ec["equity"], mode="lines", name=name, line=dict(width=1)))

    # all-flat annotation
    if len(fig.data) > 0:
        all_flat = all((len(t.y) <= 2 or (np.max(t.y) - np.min(t.y) == 0)) for t in fig.data)
        if all_flat:
            fig.add_annotation(
                text="No equity curves returned by simulator. Check run_equity_curve wiring.",
                showarrow=False, xref="paper", yref="paper", x=0.01, y=0.95,
                bgcolor="#ffeeee", bordercolor="#cc0000",
            )

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

    # session + holdout meta
    smeta = _get_session_meta(df)
    hmeta = _get_holdout_meta(df)

    # Session meta is emitted by the EA run (preferred source).  When unavailable,
    # fall back to anything provided in the holdout metadata payload so the page
    # still has reasonable defaults even for older/partial logs.
    strategy = smeta.get("strategy") or hmeta.get("strategy") or "src.models.atr_breakout:Strategy"
    tickers = smeta.get("tickers") or hmeta.get("tickers") or []
    starting_equity = float(
        smeta.get("starting_equity")
        or hmeta.get("starting_equity")
        or 10000.0
    )
    train_start = smeta.get("train_start") or hmeta.get("train_start")
    train_end = smeta.get("train_end") or hmeta.get("train_end")

    # Controls row
    c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 2, 2])

    with c1:
        st.markdown("**Training window**")
        train_start = st.text_input("Train start (ISO)", value=str(train_start) if train_start else "")
        train_end = st.text_input("Train end (ISO)", value=str(train_end) if train_end else "")

    with c2:
        st.markdown("**Test window**")
        # default from holdout_meta, else next day after train_end
        default_test_start = hmeta.get("holdout_start")
        if not default_test_start and train_end:
            try:
                default_test_start = (pd.to_datetime(train_end) + pd.Timedelta(days=1)).date().isoformat()
            except Exception:
                default_test_start = ""
        test_start = st.text_input("Test start (ISO)", value=str(default_test_start) if default_test_start else "")
        default_test_end = hmeta.get("holdout_end")
        test_end = st.text_input("Test end (ISO)", value=str(default_test_end) if default_test_end else "")
        # Ensure empty user inputs fall back to holdout metadata when possible
        if not test_start:
            test_start = str(default_test_start) if default_test_start else ""
        if not test_end:
            test_end = str(default_test_end) if default_test_end else ""

    with c3:
        st.markdown("**Top-K (current gen)**")
        top_k = st.number_input("K", min_value=1, max_value=50, value=5, step=1)

    with c4:
        st.markdown("**Playback**")
        max_gen = int(eval_df["gen"].max()) if not eval_df.empty else 0
        if "ea_inspect_gen" not in st.session_state:
            st.session_state.ea_inspect_gen = 0
        cols = st.columns(3)
        if cols[0].button("⟵ Prev", use_container_width=True):
            st.session_state.ea_inspect_gen = max(0, st.session_state.ea_inspect_gen - 1)
        if cols[2].button("Next ⟶", use_container_width=True):
            st.session_state.ea_inspect_gen = min(max_gen, st.session_state.ea_inspect_gen + 1)
        st.caption(f"Max gen in log: {max_gen}")

    with c5:
        st.markdown("**Generation**")
        st.session_state.ea_inspect_gen = st.slider(
            "Gen",
            0,
            int(eval_df["gen"].max() if not eval_df.empty else 0),
            int(st.session_state.ea_inspect_gen),
        )

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
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
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
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        starting_equity=starting_equity,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Debug trace from the equity provider
    with st.expander("Debug: equity provider trace", expanded=False):
        msgs = st.session_state.get("ea_inspector_debug", [])
        if msgs:
            st.code("\n".join(msgs))
        else:
            st.caption("No debug messages yet.")
        if hmeta:
            st.caption("Holdout metadata snapshot from EA log:")
            st.json(hmeta)

    st.caption(
        "A vertical dotted line marks the end of the training window. "
        "Curves continue into the test window on the same scale."
    )

if __name__ == "__main__":
    main()