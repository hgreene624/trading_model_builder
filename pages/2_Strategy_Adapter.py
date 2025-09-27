# pages/2_Strategy_Adapter.py
from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from importlib import import_module
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

from src.utils.holdout_chart import init_chart, on_generation_end, set_config  # package path fallback
from src.storage import list_portfolios, load_portfolio, save_strategy_params

# --- Page chrome ---
st.set_page_config(page_title="Strategy Adapter", layout="wide")
st.title("🧪 Strategy Adapter")


# --- Helpers ---
def _utc_now():
    return datetime.now(timezone.utc)


def _safe_import(dotted: str):
    return import_module(dotted)


def _ss_get_dict(key: str, default: Dict[str, Any]) -> Dict[str, Any]:
    if key not in st.session_state or not isinstance(st.session_state[key], dict):
        st.session_state[key] = dict(default)
    return st.session_state[key]


def _write_kv_table(d: Dict[str, Any], title: str = ""):
    if title:
        st.markdown(f"**{title}**")
    df = pd.DataFrame({"key": list(d.keys()), "value": [d[k] for k in d.keys()]})
    st.dataframe(df, width="stretch", height=min(360, 40 + 28 * len(df)))


def _portfolio_equity_curve(
        strategy_dotted: str,
        tickers: List[str],
        start,
        end,
        starting_equity: float,
        params: Dict[str, Any],
) -> pd.Series:
    """Simulate aggregated portfolio equity for the given params on [start, end)."""

    try:
        mod = _safe_import(strategy_dotted)
        run = getattr(mod, "run_strategy")
    except Exception:
        return pd.Series(dtype=float)

    curves: Dict[str, pd.Series] = {}
    for sym in tickers:
        try:
            result = run(sym, start, end, starting_equity, params)
        except Exception:
            continue
        eq = result.get("equity")
        if eq is None or len(eq) == 0:
            continue
        if isinstance(eq, pd.DataFrame):
            if "equity" in eq.columns:
                eq = eq["equity"]
            else:
                eq = eq.iloc[:, 0]
        elif not isinstance(eq, pd.Series):
            try:
                eq = pd.Series(eq)
            except Exception:
                continue
        eq = eq.dropna()
        if eq.empty:
            continue
        eq = eq[~eq.index.duplicated(keep="last")]
        try:
            if not isinstance(eq.index, pd.DatetimeIndex):
                eq.index = pd.to_datetime(eq.index, errors="coerce")
        except Exception:
            continue
        eq = eq[~eq.index.isna()].sort_index()
        if eq.empty:
            continue
        try:
            eq = eq.astype(float)
        except Exception:
            continue
        first_valid = None
        for val in eq.values:
            if pd.isna(val):
                continue
            try:
                fv = float(val)
            except (TypeError, ValueError):
                continue
            if abs(fv) > 1e-9:
                first_valid = fv
                break
        if first_valid is None:
            continue
        norm = (eq / first_valid).astype(float)
        curves[sym] = norm

    if not curves:
        return pd.Series(dtype=float)

    df = pd.DataFrame(curves).sort_index()
    df = df.ffill().dropna(how="all")
    if df.empty:
        return pd.Series(dtype=float)

    portfolio = df.mean(axis=1, skipna=True) * float(starting_equity)
    portfolio.name = "portfolio_equity"
    return portfolio


def _normalize_symbols(seq) -> list[str]:
    out: list[str] = []
    for x in (seq or []):
        s = None
        if isinstance(x, str):
            s = x
        elif isinstance(x, dict):
            s = x.get("symbol") or x.get("ticker") or x.get("Symbol") or x.get("Ticker")
        elif hasattr(x, "get"):
            try:
                s = x.get("symbol") or x.get("ticker")
            except Exception:
                s = None
        else:
            s = str(x)
        if not s:
            continue
        s = s.strip().upper()
        # Drop obvious headers/placeholders
        if s in {"SYMBOL", "SYMBOLS", "TICKER", "TICKERS", "NAME", "SECURITY", "COMPANY", "N/A", ""}:
            continue
        # Basic ticker sanity: letters/digits/.- up to 10 chars
        if not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", s):
            continue
        out.append(s)
    # de-dup while preserving rough order
    seen = set()
    dedup = []
    for s in out:
        if s not in seen:
            seen.add(s)
            dedup.append(s)
    return dedup


# --- Helper to filter params for strategy ---
def _filter_params_for_strategy(strategy_dotted: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only keys the current strategy accepts (e.g., ATRParams fields)."""
    try:
        mod = _safe_import(strategy_dotted)
        keys = []
        if hasattr(mod, "ATRParams"):
            try:
                keys = list(getattr(mod, "ATRParams").__annotations__.keys())
            except Exception:
                keys = []
        if not keys and hasattr(mod, "PARAMS_ALLOWED"):
            try:
                keys = list(getattr(mod, "PARAMS_ALLOWED"))
            except Exception:
                keys = []
        if not keys:
            # sensible fallback for current ATR breakout
            keys = ["breakout_n", "exit_n", "atr_n", "atr_multiple", "tp_multiple", "holding_period_limit",
                    "allow_short"]
        return {k: params[k] for k in keys if k in params}
    except Exception:
        # final fallback: safest minimal subset
        safe = ["breakout_n", "exit_n", "atr_n", "atr_multiple", "tp_multiple", "holding_period_limit"]
        return {k: params[k] for k in safe if k in params}


# ---------- LEFT SIDEBAR-LIKE COLUMN: configuration ----------
left, right = st.columns([1.05, 1.55], gap="large")

with left:
    st.subheader("Portfolio & Strategy")

    # Portfolio selection
    # Use src.storage to list/load saved portfolios; fall back to manual CSV entry.
    tickers: List[str] = []
    try:
        portfolios = sorted(list_portfolios())
    except Exception as e:
        st.warning(f"Could not list portfolios: {e}")
        portfolios = []

    # ---- Portfolio selection (EA uses saved portfolios only) ----
    tickers: List[str] = []
    try:
        portfolios = sorted(list_portfolios())
    except Exception as e:
        st.warning(f"Could not list portfolios: {e}")
        portfolios = []

    if not portfolios:
        st.error("No saved portfolios found. Create one on the Portfolios page first.")
        st.stop()

    default_idx = portfolios.index("Default") if "Default" in portfolios else 0
    port_name = st.selectbox(
        "Portfolio",
        options=portfolios,
        index=default_idx,
        help="EA runs over the selected portfolio's symbols."
    )

    try:
        obj = load_portfolio(port_name)
        # Accept either a plain list of symbols or a dict that contains them
        if isinstance(obj, dict):
            raw = obj.get("tickers") or obj.get("symbols") or obj.get("items") or obj.get("data") or []
        else:
            raw = obj
        tickers = _normalize_symbols(raw)
    except Exception as e:
        st.error(f"Failed to load portfolio '{port_name}': {e}")
        st.stop()

    if not tickers:
        st.warning("No tickers selected. Add tickers or choose a portfolio.")
        st.stop()

    st.info(f"Selected **{len(tickers)}** symbols: {', '.join(tickers[:12])}{'…' if len(tickers) > 12 else ''}")

    # Strategy module (kept as before)
    strategy_dotted = st.selectbox("Strategy", ["src.models.atr_breakout"], index=0)

    # Base params (kept as before)
    st.caption("Base parameters passed to the strategy’s run_strategy().")
    base = _ss_get_dict(
        "adapter_base_params",
        {
            "breakout_n": 70,
            "exit_n": 16,
            "atr_n": 8,
            "atr_multiple": 2.20,
            "tp_multiple": 1.78,
            "holding_period_limit": 20,
            # extras (the strategy ignores unknowns; we keep them for future work)
            "risk_per_trade": 0.005,
            "use_trend_filter": False,
            "sma_fast": 20,
            "sma_slow": 50,
            "sma_long": 200,
            "long_slope_len": 20,
            "cost_bps": 1.0,
            "execution": "close",
        },
    )

    # === EA (Base model) knobs & param bounds ===
    ea_cfg = _ss_get_dict(
        "ea_cfg",
        {
            # search controls
            "generations": 12,
            "pop_size": 100,
            "min_trades": 12,
            "n_jobs": max(1, min(8, (os.cpu_count() or 2) - 1)),
            # param bounds (inclusive ints; floats as (min, max))
            "breakout_n_min": 8, "breakout_n_max": 80,
            "exit_n_min": 4, "exit_n_max": 40,
            "atr_n_min": 7, "atr_n_max": 35,
            "atr_multiple_min": 0.8, "atr_multiple_max": 4.0,
            "tp_multiple_min": 0.8, "tp_multiple_max": 4.0,
            "hold_min": 5, "hold_max": 60,
        },
    )

    with st.expander("EA (Base model) — search knobs & bounds", expanded=False):
        col0, col1, col2, col3 = st.columns(4)
        with col0:
            ea_cfg["generations"] = st.number_input("Generations", 1, 200, int(ea_cfg["generations"]), 1)
        with col1:
            ea_cfg["pop_size"] = st.number_input("Population", 2, 400, int(ea_cfg["pop_size"]), 1)
        with col2:
            ea_cfg["min_trades"] = st.number_input("Min trades (gate)", 0, 200, int(ea_cfg["min_trades"]), 1)
        with col3:
            ea_cfg["n_jobs"] = st.number_input("Jobs (EA)", 1, max(1, (os.cpu_count() or 2)), int(ea_cfg["n_jobs"]), 1)

        st.caption("Parameter search bounds")


        # Dynamically tighten slider track to the configured bounds (better UX)
        def _pad_int_range(lo: int, hi: int, pad_ratio: float = 0.25, hard_lo: int = 1, hard_hi: int = 600) -> tuple[
            int, int]:
            span = max(1, hi - lo)
            pad = max(1, int(span * pad_ratio))
            return max(hard_lo, lo - pad), min(hard_hi, hi + pad)


        def _pad_float_range(lo: float, hi: float, pad_ratio: float = 0.25, hard_lo: float = 0.05,
                             hard_hi: float = 50.0) -> tuple[float, float]:
            span = max(1e-9, hi - lo)
            pad = span * pad_ratio
            return max(hard_lo, lo - pad), min(hard_hi, hi + pad)


        # Compute padded domains for each param based on current bounds
        _br_min, _br_max = _pad_int_range(int(ea_cfg["breakout_n_min"]), int(ea_cfg["breakout_n_max"]))
        _ex_min, _ex_max = _pad_int_range(int(ea_cfg["exit_n_min"]), int(ea_cfg["exit_n_max"]))
        _atrn_min, _atrn_max = _pad_int_range(int(ea_cfg["atr_n_min"]), int(ea_cfg["atr_n_max"]))
        _hold_min, _hold_max = _pad_int_range(int(ea_cfg["hold_min"]), int(ea_cfg["hold_max"]))

        _atrm_min, _atrm_max = _pad_float_range(float(ea_cfg["atr_multiple_min"]), float(ea_cfg["atr_multiple_max"]))
        _tpm_min, _tpm_max = _pad_float_range(float(ea_cfg["tp_multiple_min"]), float(ea_cfg["tp_multiple_max"]))

        # ---- INT ranges via sliders ----
        c1, c2 = st.columns(2)
        with c1:
            br_lo, br_hi = st.slider(
                "breakout_n range",
                min_value=_br_min, max_value=_br_max,
                value=(int(ea_cfg["breakout_n_min"]), int(ea_cfg["breakout_n_max"])),
                step=1,
                help="Bars for the entry breakout lookback.")
            ex_lo, ex_hi = st.slider(
                "exit_n range",
                min_value=_ex_min, max_value=_ex_max,
                value=(int(ea_cfg["exit_n_min"]), int(ea_cfg["exit_n_max"])),
                step=1,
                help="Bars for exit/stop lookback.")
            atrn_lo, atrn_hi = st.slider(
                "atr_n range",
                min_value=_atrn_min, max_value=_atrn_max,
                value=(int(ea_cfg["atr_n_min"]), int(ea_cfg["atr_n_max"])),
                step=1,
                help="ATR window length.")
            hold_lo, hold_hi = st.slider(
                "holding_period_limit range",
                min_value=_hold_min, max_value=_hold_max,
                value=(int(ea_cfg["hold_min"]), int(ea_cfg["hold_max"])),
                step=1,
                help="Max bars a trade may be held.")

        # ---- FLOAT ranges via sliders ----
        with c2:
            atrm_lo, atrm_hi = st.slider(
                "atr_multiple range",
                min_value=_atrm_min, max_value=_atrm_max,
                value=(float(ea_cfg["atr_multiple_min"]), float(ea_cfg["atr_multiple_max"])),
                step=0.05,
                help="Stop distance as multiple of ATR.")
            tpm_lo, tpm_hi = st.slider(
                "tp_multiple range",
                min_value=_tpm_min, max_value=_tpm_max,
                value=(float(ea_cfg["tp_multiple_min"]), float(ea_cfg["tp_multiple_max"])),
                step=0.05,
                help="Take-profit multiple.")

        # Persist back to session config
        ea_cfg["breakout_n_min"], ea_cfg["breakout_n_max"] = int(br_lo), int(br_hi)
        ea_cfg["exit_n_min"], ea_cfg["exit_n_max"] = int(ex_lo), int(ex_hi)
        ea_cfg["atr_n_min"], ea_cfg["atr_n_max"] = int(atrn_lo), int(atrn_hi)
        ea_cfg["hold_min"], ea_cfg["hold_max"] = int(hold_lo), int(hold_hi)

        ea_cfg["atr_multiple_min"], ea_cfg["atr_multiple_max"] = float(atrm_lo), float(atrm_hi)
        ea_cfg["tp_multiple_min"], ea_cfg["tp_multiple_max"] = float(tpm_lo), float(tpm_hi)

    with st.expander("Base params", expanded=False):
        base["breakout_n"] = st.number_input("breakout_n", 5, 300, base["breakout_n"], 1,
                                             help="Lookback used for breakout entry signal.")
        base["exit_n"] = st.number_input("exit_n", 4, 300, base["exit_n"], 1,
                                         help="Lookback used for breakout exit/stop logic.")
        base["atr_n"] = st.number_input("atr_n", 5, 60, base["atr_n"], 1, help="ATR window length.")
        base["atr_multiple"] = st.number_input("atr_multiple", 0.5, 10.0, float(base["atr_multiple"]), 0.1,
                                               help="ATR multiple for stop distance.")
        base["tp_multiple"] = st.number_input("tp_multiple", 0.2, 10.0, float(base["tp_multiple"]), 0.1,
                                              help="Take-profit multiple (vs ATR or entry logic).")
        base["holding_period_limit"] = st.number_input("holding_period_limit", 1, 400, base["holding_period_limit"], 1,
                                                       help="Max bars to hold a position.")
        # Keep the extras so future strategies/UI can reuse
        base["risk_per_trade"] = st.number_input("risk_per_trade", 0.0005, 0.05, float(base["risk_per_trade"]), 0.0005,
                                                 format="%.4f", help="Fraction of equity risked per trade.")
        base["use_trend_filter"] = st.checkbox("use_trend_filter", value=bool(base["use_trend_filter"]),
                                               help="Optional trend filter gate.")
        base["sma_fast"] = st.number_input("sma_fast", 5, 100, base["sma_fast"], 1,
                                           help="Fast MA length (if trend filter used).")
        base["sma_slow"] = st.number_input("sma_slow", 10, 200, base["sma_slow"], 1,
                                           help="Slow MA length (if trend filter used).")
        base["sma_long"] = st.number_input("sma_long", 100, 400, base["sma_long"], 1,
                                           help="Long MA length (if trend filter used).")
        base["long_slope_len"] = st.number_input("long_slope_len", 5, 60, base["long_slope_len"], 1,
                                                 help="Slope window for long MA trend check.")
        base["cost_bps"] = st.number_input("cost_bps", 0.0, 20.0, float(base["cost_bps"]), 0.1,
                                           help="Per-trade cost (basis points).")
        base["execution"] = st.selectbox("Execution", ["close"], index=0,
                                         help="Execution price proxy used in backtest.")

    # General training knobs
    folds = st.number_input("CV folds", 2, 10, 4, 1, help="Cross-validation splits for the base model trainer.")
    equity = st.number_input("Starting equity ($)", 1000.0, 1_000_000.0, 10_000.0, 100.0,
                             help="Starting equity for per-symbol runs.")
    min_trades = st.number_input("Min trades (valid)", 0, 200, 2, 1,
                                 help="Minimum total trades needed for a run to be considered valid.")

    # Parallelism controls (kept visible near the top)
    max_procs = os.cpu_count() or 8
    n_jobs = st.slider("Jobs (processes)", 1, max(1, max_procs - 1), min(8, max(1, max_procs - 1)))

    st.caption("Tip: If Alpaca SIP limits or YF rate limits bite, reduce Jobs to 1–4.")

# ---------- RIGHT COLUMN: actions + results ----------
with right:
    # ------------------------ BASE TRAINING ------------------------
    st.subheader("Train Base Model")
    run_btn = st.button(
        "🚀 Train (portfolio)",
        type="primary",
        help="Runs the portfolio-level base trainer with the config on the left.",
        width="stretch",
    )

    st.divider()
    st.subheader("Results")

    # --- Persistent result placeholders (rehydrate from session if available) ---
    st.caption("Recent EA evaluations (rolling window)")
    eval_table_placeholder = st.empty()

    st.markdown("**Best candidate so far**")
    best_score_col, best_params_col = st.columns([1, 1.8], gap="large")
    with best_score_col:
        best_score_placeholder = st.empty()
    with best_params_col:
        best_params_placeholder = st.empty()

    st.markdown("**Holdout equity (outside training window)**")
    holdout_chart_placeholder = st.empty()
    holdout_status_placeholder = st.empty()

    st.caption("Generation summary")
    gen_summary_placeholder = st.empty()

    # --- Pull any previous run artifacts back into the UI ---
    live_rows_state = st.session_state.get("adapter_live_rows") or []
    if live_rows_state:
        eval_table_placeholder.dataframe(pd.DataFrame(live_rows_state), width="stretch", height=380)
    else:
        eval_table_placeholder.info("No evaluations yet. Run the EA to populate this table.")

    best_tracker_state = st.session_state.get("adapter_best_tracker") or {}
    best_score_val = best_tracker_state.get("score")
    best_delta_val = best_tracker_state.get("delta")
    if isinstance(best_score_val, (int, float)) and best_score_val not in (float("-inf"), float("inf")):
        best_score_placeholder.metric(
            "Best score",
            f"{best_score_val:.3f}",
            delta=None if best_delta_val is None else f"{best_delta_val:+.3f}",
        )
    else:
        best_score_placeholder.metric("Best score", "—")

    best_params_state = best_tracker_state.get("params") or {}
    if best_params_state:
        df_params_state = pd.DataFrame(
            {"param": list(best_params_state.keys()), "value": [best_params_state[k] for k in best_params_state.keys()]}
        )
        best_params_placeholder.dataframe(df_params_state.set_index("param"), width="stretch", height=220)
    else:
        best_params_placeholder.info("Waiting for evaluations…")

    holdout_history_state = st.session_state.get("adapter_holdout_history") or []
    # (chart rendered by holdout_chart helper)
holdout_status_state = st.session_state.get("adapter_holdout_status") or ("info",
                                                                          "Holdout equity will appear when a best candidate is found.")
status_kind, status_msg = holdout_status_state
if status_kind == "success":
    holdout_status_placeholder.success(status_msg)
elif status_kind == "warning":
    holdout_status_placeholder.warning(status_msg)
elif status_kind == "error":
    holdout_status_placeholder.error(status_msg)
else:
    holdout_status_placeholder.info(status_msg)

gen_history_state = st.session_state.get("adapter_gen_history") or []
if gen_history_state:
    gen_summary_placeholder.dataframe(pd.DataFrame(gen_history_state[-12:]), width="stretch", height=180)
else:
    gen_summary_placeholder.info("No generations have completed yet.")

# ------------------------ EA RUN (refactored) ------------------------
if run_btn:
    if not tickers:
        st.error("This portfolio has no tickers.")
        st.stop()

    # Resolve modules used during EA
    try:
        evo = _safe_import("src.optimization.evolutionary")
        progmod = _safe_import("src.utils.progress")  # optional UI progress sink
    except Exception as e:
        st.error(f"Import error: {e}")
        st.stop()

    prog = st.progress(0.0, text="Preparing EA search…")
    status = st.empty()

    # Reset per-run UI/session state
    live_rows: list[dict[str, Any]] = []
    gen_history: list[dict[str, Any]] = []
    best_tracker: dict[str, Any] = {"score": float("-inf"), "params": {}, "delta": None}

    st.session_state["adapter_live_rows"] = []
    st.session_state["adapter_gen_history"] = []
    st.session_state["adapter_best_tracker"] = dict(best_tracker)

    eval_table_placeholder.empty()
    best_score_placeholder.metric("Best score", "—")
    best_params_placeholder.info("Waiting for evaluations…")

    end = _utc_now()
    start = end - timedelta(days=365)

    # ---- Initialize holdout chart (blank at start). The helper will update per improving generation. ----
    holdout_span = max(30, min(180, (end - start).days // 3 or 90))
    holdout_end = start
    holdout_start = holdout_end - timedelta(days=int(holdout_span))

    def _hc_engine(params, data, starting_equity):
        series = _portfolio_equity_curve(
            strategy_dotted,
            tickers,
            holdout_start,
            holdout_end,
            float(starting_equity),
            params,
        )
        return pd.DataFrame({"equity": series})

    init_chart(
        placeholder=holdout_chart_placeholder,
        starting_equity=float(equity),
        holdout_start=holdout_start,
        holdout_end=holdout_end,
        loader_fn=lambda **kwargs: {},  # unused by _hc_engine
        engine_fn=_hc_engine,
        symbols=tickers,
        max_curves=8,
    )
    st.session_state["_hc_last_score"] = None

    # ---- Build EA param space from UI bounds ----
    cfg = ea_cfg

    def _clamp_int(lo, hi):
        return (int(lo), int(max(lo, hi)))

    def _clamp_float(lo, hi):
        return (float(lo), float(max(float(lo), float(hi))))

    param_space = {
        "breakout_n": _clamp_int(cfg["breakout_n_min"], cfg["breakout_n_max"]),
        "exit_n": _clamp_int(cfg["exit_n_min"], cfg["exit_n_max"]),
        "atr_n": _clamp_int(cfg["atr_n_min"], cfg["atr_n_max"]),
        "atr_multiple": _clamp_float(cfg["atr_multiple_min"], cfg["atr_multiple_max"]),
        "tp_multiple": _clamp_float(cfg["tp_multiple_min"], cfg["tp_multiple_max"]),
        "holding_period_limit": _clamp_int(cfg["hold_min"], cfg["hold_max"]),
    }

    # Optional richer progress sink
    ui_cb = getattr(progmod, "ui_progress", lambda *_args, **_kw: (lambda *_a, **_k: None))(st)
    # Track best-of-generation and last plotted score
    gen_best = {"score": float("-inf"), "params": {}}
    st.session_state.setdefault("_hc_last_score", None)

    def _cb(evt, ctx):
        # evt: "generation_start", "individual_evaluated", "generation_end", "done"
        try:
            ui_cb(evt, ctx)
        except Exception:
            pass
        if evt == "generation_start":
            gen = int(ctx.get("gen", 0))
            # Reset generation tracker for best-of-generation
            gen_best["score"] = float("-inf")
            gen_best["params"] = {}
            status.info(f"Generation {gen + 1} starting (population={ctx.get('pop_size', 'n/a')})")
            prog.progress(
                min(0.9, 0.1 + (gen / max(1, cfg['generations'])) * 0.8),
                text=f"EA generation {gen + 1}/{cfg['generations']}…",
            )
        elif evt == "individual_evaluated":
            metrics = ctx.get("metrics", {}) or {}
            row = {
                "gen": ctx.get("gen"),
                "idx": ctx.get("idx"),
                "score": ctx.get("score"),
                "trades": int(metrics.get("trades", 0) or 0),
                "cagr": metrics.get("cagr"),
                "calmar": metrics.get("calmar"),
                "sharpe": metrics.get("sharpe"),
            }
            live_rows.append(row)
            live_rows[:] = live_rows[-60:]
            eval_table_placeholder.dataframe(pd.DataFrame(live_rows), width="stretch", height=380)
            st.session_state["adapter_live_rows"] = list(live_rows)

            score = ctx.get("score")
            if isinstance(score, (int, float)):
                # Track best within this generation
                try:
                    cur_best = gen_best.get("score")
                except Exception:
                    cur_best = float("-inf")
                if cur_best in (None, float("-inf")) or float(score) > float(cur_best):
                    gen_best["score"] = float(score)
                    gen_best["params"] = dict(ctx.get("params") or {})
                prev = best_tracker.get("score")
                if prev in (None, float("-inf")) or score > prev:
                    delta = None if prev in (None, float("-inf")) else float(score) - float(prev)
                    best_tracker.update({"score": float(score), "params": dict(ctx.get("params") or {}), "delta": delta})
                    best_score_placeholder.metric(
                        "Best score",
                        f"{best_tracker['score']:.3f}",
                        delta=None if delta is None else f"{delta:+.3f}",
                    )
                    if best_tracker["params"]:
                        dfp = pd.DataFrame({"param": list(best_tracker["params"].keys()),
                                            "value": [best_tracker["params"][k] for k in best_tracker["params"].keys()]})
                        best_params_placeholder.dataframe(dfp.set_index("param"), width="stretch", height=220)
                    st.session_state["adapter_best_tracker"] = dict(best_tracker)

        elif evt == "generation_end":
            gen_history.append({
                "generation": ctx.get("gen"),
                "best_score": ctx.get("best_score"),
                "avg_score": ctx.get("avg_score"),
                "avg_trades": ctx.get("avg_trades"),
                "no_trades_pct": ctx.get("pct_no_trades"),
            })
            gen_summary_placeholder.dataframe(pd.DataFrame(gen_history[-12:]), width="stretch", height=180)
            st.session_state["adapter_gen_history"] = list(gen_history)
            # Determine gen-best from ctx first, then fallback to accumulator
            best_score_gen = ctx.get("best_score")
            if not isinstance(best_score_gen, (int, float)):
                best_score_gen = gen_best.get("score", float("-inf"))
            best_params_gen = ctx.get("best_params") or gen_best.get("params") or dict(ctx.get("params") or {})
            last_plotted = st.session_state.get("_hc_last_score")

            # Always push Gen 0; otherwise only if score improves
            should_push = (last_plotted is None) or (isinstance(best_score_gen, (int, float)) and best_score_gen > last_plotted)
            if should_push and isinstance(best_score_gen, (int, float)) and best_params_gen:
                try:
                    on_generation_end(int(ctx.get("gen", 0)), float(best_score_gen), dict(best_params_gen))
                    st.session_state["_hc_last_score"] = float(best_score_gen)
                except Exception:
                    pass

            # Progress update
            gen_idx = int(ctx.get("gen", 0))
            total_gens = int(cfg.get("generations", 1)) or 1
            p = min(0.95, 0.1 + ((gen_idx + 1) / total_gens) * 0.8)
            label = (
                f"EA gen {gen_idx} done; best={best_score_gen:.3f}"
                if isinstance(best_score_gen, (int, float)) else
                "EA evolving…"
            )
            prog.progress(p, text=label)

        elif evt == "done":
            elapsed = ctx.get("elapsed_sec")
            if isinstance(elapsed, (int, float)):
                status.success(f"EA completed in {elapsed:.1f}s")
                prog.progress(1.0, text="EA completed")

    # EA logging: timestamped JSONL under storage/logs/ea
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("storage", "logs", "ea")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{ts}_ea.jsonl")

    # Run EA search (helper updates chart live via on_generation_end)
    try:
        top = evo.evolutionary_search(
            strategy_dotted=strategy_dotted,
            tickers=tickers,
            start=start,
            end=end,
            starting_equity=float(equity),
            param_space=param_space,
            generations=int(cfg["generations"]),
            pop_size=int(cfg["pop_size"]),
            min_trades=int(cfg["min_trades"]),
            n_jobs=int(n_jobs),
            progress_cb=_cb,
            log_file=log_file,
        )
    except Exception as e:
        st.error(f"Training failed: {e}")
        st.stop()

    if not top:
        st.error("EA returned no candidates.")
        st.stop()

    best_params, best_score = top[0]
    st.session_state["ea_best_params"] = dict(best_params)
    st.session_state["ea_portfolio"] = port_name
    st.session_state["ea_strategy"] = strategy_dotted
    st.session_state["ea_top_results"] = list(top)

    prog.progress(0.95, text="Rendering EA leaderboard…")

    rows = []
    for params, score in top[: min(50, len(top))]:
        r = {"score": float(score)}
        r.update({k: params.get(k) for k in ("breakout_n", "exit_n", "atr_n", "atr_multiple", "tp_multiple", "holding_period_limit")})
        rows.append(r)
    lb = pd.DataFrame(rows).sort_values("score", ascending=False)
    st.markdown("**EA leaderboard (top candidates)**")
    st.dataframe(lb, width="stretch", height=360)

    st.session_state["ea_log_file"] = log_file
    st.success(f"EA complete. Best score={best_score:.3f}.")



# --- Always-available Save EA Best Params section ---
with right:
    st.divider()
    st.subheader("Save EA Best Params")

    ea_best = st.session_state.get("ea_best_params") or {}
    if not ea_best:
        st.info("Run training to produce EA params, then save them here.")
    else:
        # show what will be saved
        with st.expander("EA best parameters", expanded=False):
            st.json(ea_best)

        # default to the portfolio/strategy that produced these params
        portfolio_to_save = st.session_state.get("ea_portfolio") or port_name
        strategy_to_save = st.session_state.get("ea_strategy") or strategy_dotted

        col_s1, col_s2 = st.columns([1, 3])
        with col_s1:
            do_save_always = st.button(
                "💾 Save EA Best Params",
                type="primary",
                use_container_width=True,
                key="save_ea_btn"
            )

        if do_save_always:
            try:
                saved_path = save_strategy_params(
                    portfolio=portfolio_to_save,
                    strategy=strategy_to_save,
                    params=ea_best,
                    scope="ea",
                )
                st.success(f"Saved EA params for '{portfolio_to_save}' → {saved_path or '(path not returned)'}")
            except Exception as e:
                st.error(f"Save failed: {e}")