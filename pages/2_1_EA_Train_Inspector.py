# pages/5_EA_Train_Test_Inspector.py
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import time
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

def _coerce_date(value: Any) -> Optional[date]:
    """Parse *value* into a ``date`` (day precision) when possible."""
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    try:
        coerced = pd.to_datetime(value, errors="coerce")
    except Exception:
        return None
    if isinstance(coerced, pd.Series):
        coerced = coerced.iloc[0] if not coerced.empty else pd.NaT
    if pd.isna(coerced):
        return None
    if isinstance(coerced, pd.Timestamp):
        return coerced.date()
    if isinstance(coerced, datetime):
        return coerced.date()
    if isinstance(coerced, date):
        return coerced
    return None

def _deep_get(mapping: Dict[str, Any], *keys: str) -> Any:
    cur: Any = mapping
    for key in keys:
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            return None
    return cur

def _infer_log_windows(df: pd.DataFrame) -> Dict[str, Optional[date]]:
    """Infer train/test date bounds from an EA log."""
    train_starts: List[date] = []
    train_ends: List[date] = []
    dataset_starts: List[date] = []
    dataset_ends: List[date] = []

    if "event" not in df.columns or "payload" not in df.columns:
        return {"train_start": None, "train_end": None, "test_start": None, "test_end": None}

    for _, row in df.iterrows():
        payload = row.get("payload", {})
        if not isinstance(payload, dict):
            continue

        event = row.get("event")
        def _push(bucket: List[date], value: Any) -> None:
            d = _coerce_date(value)
            if d is not None:
                bucket.append(d)

        if event == "session_meta":
            _push(train_starts, payload.get("train_start"))
            _push(train_ends, payload.get("train_end"))
        if event == "holdout_meta":
            _push(dataset_starts, payload.get("holdout_start"))
            _push(dataset_ends, payload.get("holdout_end"))
            _push(train_ends, payload.get("train_end"))

        # Generic nested fallbacks
        for key in ("train_start", "start"):
            _push(train_starts, payload.get(key))
        for key in ("train_end", "end"):
            _push(train_ends, payload.get(key))

        _push(train_starts, _deep_get(payload, "train", "start"))
        _push(train_ends, _deep_get(payload, "train", "end"))
        _push(dataset_starts, _deep_get(payload, "test", "start"))
        _push(dataset_ends, _deep_get(payload, "test", "end"))
        _push(train_starts, _deep_get(payload, "period", "start"))
        _push(train_ends, _deep_get(payload, "period", "end"))
        _push(dataset_starts, _deep_get(payload, "period", "start"))
        _push(dataset_ends, _deep_get(payload, "period", "end"))

    train_start = min(train_starts) if train_starts else (min(dataset_starts) if dataset_starts else None)
    train_end = max(train_ends) if train_ends else None
    if train_start and train_end and train_start > train_end:
        train_start, train_end = train_end, train_start
    dataset_end = max(dataset_ends) if dataset_ends else train_end
    if dataset_end and train_end and dataset_end < train_end:
        dataset_end = train_end

    test_start = None
    if train_end is not None:
        test_start = train_end + timedelta(days=1)
        if dataset_end is not None and test_start > dataset_end:
            test_start = dataset_end
    elif dataset_starts:
        # Fallback: use dataset start if train_end missing
        test_start = min(dataset_starts)

    return {
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": dataset_end,
    }

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

    strategy = smeta.get("strategy", "src.models.atr_breakout:Strategy")
    tickers = smeta.get("tickers", [])
    starting_equity = float(smeta.get("starting_equity", 10000.0))

    windows = _infer_log_windows(df)

    train_start_date = windows.get("train_start") or _coerce_date(smeta.get("train_start"))
    train_end_date = windows.get("train_end") or _coerce_date(smeta.get("train_end"))
    test_start_date = windows.get("test_start") or _coerce_date(hmeta.get("holdout_start"))
    test_end_date = windows.get("test_end") or _coerce_date(hmeta.get("holdout_end"))

    def _fmt(d: Optional[date]) -> str:
        return d.isoformat() if isinstance(d, date) else ""

    max_gen = int(eval_df["gen"].max()) if not eval_df.empty else 0
    st.session_state.setdefault("ea_inspect_gen", 0)
    st.session_state.setdefault("ea_playback_active", False)
    st.session_state.setdefault("ea_playback_speed", 1.0)
    st.session_state.ea_inspect_gen = int(min(max_gen, max(0, st.session_state.ea_inspect_gen)))

    # Controls row
    c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 2, 2])

    with c1:
        st.markdown("**Training window**")
        train_start = st.text_input("Train start (ISO)", value=_fmt(train_start_date)).strip()
        train_end = st.text_input("Train end (ISO)", value=_fmt(train_end_date)).strip()

    with c2:
        st.markdown("**Test window**")
        test_start = st.text_input("Test start (ISO)", value=_fmt(test_start_date)).strip()
        test_end = st.text_input("Test end (ISO)", value=_fmt(test_end_date)).strip()

    with c3:
        st.markdown("**Top-K (current gen)**")
        top_k = st.number_input("K", min_value=1, max_value=50, value=5, step=1)

    with c4:
        st.markdown("**Playback**")
        control_cols = st.columns(4)
        if control_cols[0].button("◀ Prev", use_container_width=True):
            st.session_state.ea_inspect_gen = max(0, st.session_state.ea_inspect_gen - 1)
            st.session_state.ea_playback_active = False
        if control_cols[1].button("▶ Play", use_container_width=True):
            if st.session_state.ea_inspect_gen >= max_gen:
                st.session_state.ea_inspect_gen = 0
            st.session_state.ea_playback_active = True
        if control_cols[2].button("⏸ Pause", use_container_width=True):
            st.session_state.ea_playback_active = False
        if control_cols[3].button("Next ▶", use_container_width=True):
            st.session_state.ea_inspect_gen = min(max_gen, st.session_state.ea_inspect_gen + 1)
            st.session_state.ea_playback_active = False

        speed_options = [0.25, 0.5, 1.0, 1.5, 2.0]
        current_speed = float(st.session_state.get("ea_playback_speed", 1.0))
        if current_speed not in speed_options:
            current_speed = 1.0
        speed = st.select_slider(
            "Speed (sec/step)",
            options=speed_options,
            value=current_speed,
            format_func=lambda x: f"{x:.2f}s" if x < 1.0 else f"{x:.1f}s",
        )
        st.session_state.ea_playback_speed = float(speed)
        st.caption(f"Max gen in log: {max_gen}")

    with c5:
        st.markdown("**Generation**")
        st.session_state.ea_inspect_gen = st.slider(
            "Gen",
            0,
            int(eval_df["gen"].max() if not eval_df.empty else 0),
            int(st.session_state.ea_inspect_gen),
        )

    # Auto-advance playback if active
    if st.session_state.get("ea_playback_active", False):
        if st.session_state.ea_inspect_gen < max_gen:
            delay = max(0.0, float(st.session_state.get("ea_playback_speed", 1.0)))
            if delay:
                time.sleep(delay)
            st.session_state.ea_inspect_gen = min(max_gen, st.session_state.ea_inspect_gen + 1)
            try:
                st.experimental_rerun()
            except AttributeError:
                st.rerun()
        else:
            st.session_state.ea_playback_active = False

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

    st.caption(
        "A vertical dotted line marks the end of the training window. "
        "Curves continue into the test window on the same scale."
    )

if __name__ == "__main__":
    main()