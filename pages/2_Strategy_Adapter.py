# pages/2_Strategy_Adapter.py
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from importlib import import_module
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

from src.storage import list_portfolios, load_portfolio

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

# --- Symbol normalization (defensive against headers/objects) ---
import re

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
            keys = ["breakout_n", "exit_n", "atr_n", "atr_multiple", "tp_multiple", "holding_period_limit", "allow_short"]
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

    port_name = st.selectbox("Portfolio", options=(["<custom>"] + portfolios) if portfolios else ["<custom>"], index=0)

    if port_name == "<custom>":
        tickers_csv = st.text_input("Tickers (CSV)", "AAPL,MSFT")
        raw = [t for t in tickers_csv.split(",")]
        tickers = _normalize_symbols(raw)
    else:
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
            "breakout_n": 20,
            "exit_n": 10,
            "atr_n": 14,
            "atr_multiple": 2.0,
            "tp_multiple": 1.5,
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
            "generations": 8,
            "pop_size": 24,
            "min_trades": 5,
            "n_jobs": max(1, min(8, (os.cpu_count() or 2) - 1)),
            # param bounds (inclusive ints; floats as (min, max))
            "breakout_n_min": 8,   "breakout_n_max": 80,
            "exit_n_min": 4,       "exit_n_max": 40,
            "atr_n_min": 7,        "atr_n_max": 35,
            "atr_multiple_min": 0.8, "atr_multiple_max": 4.0,
            "tp_multiple_min": 0.8,  "tp_multiple_max": 4.0,
            "hold_min": 5,         "hold_max": 60,
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
        def _pad_int_range(lo: int, hi: int, pad_ratio: float = 0.25, hard_lo: int = 1, hard_hi: int = 600) -> tuple[int, int]:
            span = max(1, hi - lo)
            pad = max(1, int(span * pad_ratio))
            return max(hard_lo, lo - pad), min(hard_hi, hi + pad)

        def _pad_float_range(lo: float, hi: float, pad_ratio: float = 0.25, hard_lo: float = 0.05, hard_hi: float = 50.0) -> tuple[float, float]:
            span = max(1e-9, hi - lo)
            pad = span * pad_ratio
            return max(hard_lo, lo - pad), min(hard_hi, hi + pad)

        # Compute padded domains for each param based on current bounds
        _br_min, _br_max = _pad_int_range(int(ea_cfg["breakout_n_min"]), int(ea_cfg["breakout_n_max"]))
        _ex_min, _ex_max = _pad_int_range(int(ea_cfg["exit_n_min"]), int(ea_cfg["exit_n_max"]))
        _atrn_min, _atrn_max = _pad_int_range(int(ea_cfg["atr_n_min"]), int(ea_cfg["atr_n_max"]))
        _hold_min, _hold_max = _pad_int_range(int(ea_cfg["hold_min"]), int(ea_cfg["hold_max"]))

        _atrm_min, _atrm_max = _pad_float_range(float(ea_cfg["atr_multiple_min"]), float(ea_cfg["atr_multiple_max"]))
        _tpm_min, _tpm_max   = _pad_float_range(float(ea_cfg["tp_multiple_min"]),  float(ea_cfg["tp_multiple_max"]))

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
        ea_cfg["exit_n_min"],     ea_cfg["exit_n_max"]     = int(ex_lo), int(ex_hi)
        ea_cfg["atr_n_min"],      ea_cfg["atr_n_max"]      = int(atrn_lo), int(atrn_hi)
        ea_cfg["hold_min"],       ea_cfg["hold_max"]       = int(hold_lo), int(hold_hi)

        ea_cfg["atr_multiple_min"], ea_cfg["atr_multiple_max"] = float(atrm_lo), float(atrm_hi)
        ea_cfg["tp_multiple_min"],  ea_cfg["tp_multiple_max"]  = float(tpm_lo), float(tpm_hi)

    with st.expander("Base params", expanded=False):
        base["breakout_n"] = st.number_input("breakout_n", 5, 300, base["breakout_n"], 1, help="Lookback used for breakout entry signal.")
        base["exit_n"] = st.number_input("exit_n", 4, 300, base["exit_n"], 1, help="Lookback used for breakout exit/stop logic.")
        base["atr_n"] = st.number_input("atr_n", 5, 60, base["atr_n"], 1, help="ATR window length.")
        base["atr_multiple"] = st.number_input("atr_multiple", 0.5, 10.0, float(base["atr_multiple"]), 0.1, help="ATR multiple for stop distance.")
        base["tp_multiple"] = st.number_input("tp_multiple", 0.2, 10.0, float(base["tp_multiple"]), 0.1, help="Take-profit multiple (vs ATR or entry logic).")
        base["holding_period_limit"] = st.number_input("holding_period_limit", 1, 400, base["holding_period_limit"], 1, help="Max bars to hold a position.")
        # Keep the extras so future strategies/UI can reuse
        base["risk_per_trade"] = st.number_input("risk_per_trade", 0.0005, 0.05, float(base["risk_per_trade"]), 0.0005, format="%.4f", help="Fraction of equity risked per trade.")
        base["use_trend_filter"] = st.checkbox("use_trend_filter", value=bool(base["use_trend_filter"]), help="Optional trend filter gate.")
        base["sma_fast"] = st.number_input("sma_fast", 5, 100, base["sma_fast"], 1, help="Fast MA length (if trend filter used).")
        base["sma_slow"] = st.number_input("sma_slow", 10, 200, base["sma_slow"], 1, help="Slow MA length (if trend filter used).")
        base["sma_long"] = st.number_input("sma_long", 100, 400, base["sma_long"], 1, help="Long MA length (if trend filter used).")
        base["long_slope_len"] = st.number_input("long_slope_len", 5, 60, base["long_slope_len"], 1, help="Slope window for long MA trend check.")
        base["cost_bps"] = st.number_input("cost_bps", 0.0, 20.0, float(base["cost_bps"]), 0.1, help="Per-trade cost (basis points).")
        base["execution"] = st.selectbox("Execution", ["close"], index=0, help="Execution price proxy used in backtest.")

    # General training knobs
    folds = st.number_input("CV folds", 2, 10, 4, 1, help="Cross-validation splits for the base model trainer.")
    equity = st.number_input("Starting equity ($)", 1000.0, 1_000_000.0, 10_000.0, 100.0, help="Starting equity for per-symbol runs.")
    min_trades = st.number_input("Min trades (valid)", 0, 200, 2, 1, help="Minimum total trades needed for a run to be considered valid.")

    # Parallelism controls (kept visible near the top)
    max_procs = os.cpu_count() or 8
    n_jobs = st.slider("Jobs (processes)", 1, max(1, max_procs - 1), min(8, max(1, max_procs - 1))),  # noqa
    n_jobs = int(n_jobs[0])

    st.caption("Tip: If Alpaca SIP limits or YF rate limits bite, reduce Jobs to 1–4.")

# ---------- RIGHT COLUMN: actions + results ----------
with right:
    # ------------------------ BASE TRAINING ------------------------
    st.subheader("Train Base Model")
    run_btn = st.button("🚀 Train (portfolio)", type="primary", help="Runs the portfolio-level base trainer with the config on the left.", width="stretch")

    st.divider()
    st.subheader("Results")

    if run_btn:
        if not tickers:
            st.error("This portfolio has no tickers.")
            st.stop()

        # resolve modules
        try:
            loader = _safe_import("src.data.loader")
            evo = _safe_import("src.optimization.evolutionary")
            metrics = _safe_import("src.backtest.metrics")
            progmod = _safe_import("src.utils.progress")  # EA UI progress sink
        except Exception as e:
            st.error(f"Import error: {e}")
            st.stop()

        prog = st.progress(0.0, text="Preparing EA search…")
        status = st.empty()

        # EA logging: timestamped JSONL under storage/logs/ea
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join("storage", "logs", "ea")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{ts}_ea.jsonl")

        # Rich EA progress sink for Streamlit UI (fallback to no-op if missing)
        ui_cb = getattr(progmod, "ui_progress", lambda *_args, **_kw: (lambda *_a, **_k: None))(st)

        try:
            end = _utc_now()
            start = end - timedelta(days=365)

            # Build param space from UI bounds
            cfg = ea_cfg
            def _clamp_int(lo, hi): return (int(lo), int(max(lo, hi)))
            def _clamp_float(lo, hi): return (float(lo), float(max(float(lo), float(hi))))

            param_space = {
                "breakout_n": _clamp_int(cfg["breakout_n_min"], cfg["breakout_n_max"]),
                "exit_n": _clamp_int(cfg["exit_n_min"], cfg["exit_n_max"]),
                "atr_n": _clamp_int(cfg["atr_n_min"], cfg["atr_n_max"]),
                "atr_multiple": _clamp_float(cfg["atr_multiple_min"], cfg["atr_multiple_max"]),
                "tp_multiple": _clamp_float(cfg["tp_multiple_min"], cfg["tp_multiple_max"]),
                "holding_period_limit": _clamp_int(cfg["hold_min"], cfg["hold_max"]),
            }

            clean_base = _filter_params_for_strategy(strategy_dotted, base)

            status.write("Running EA (base model training)…")
            prog.progress(0.1)

            # Streamlit progress callback (wrap rich UI sink + keep existing bar)
            def _cb(evt, ctx):
                # evt: e.g., "generation_start", "evaluation", "generation_end"
                # ctx: dict with gen/pop etc. (best-effort)
                try:
                    ui_cb(evt, ctx)  # forward to rich UI sink
                except Exception:
                    pass
                try:
                    if evt == "generation_start":
                        gen = ctx.get("gen", 0)
                        prog.progress(min(0.9, 0.1 + (gen / max(1, cfg['generations'])) * 0.8),
                                      text=f"EA generation {gen+1}/{cfg['generations']}…")
                    elif evt == "generation_end":
                        best = ctx.get("best_score")
                        prog.progress(None, text=f"EA gen {ctx.get('gen')} done; best={best:.3f}" if best is not None else "EA evolving…")
                except Exception:
                    pass

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
                n_jobs=int(cfg["n_jobs"]),
                progress_cb=_cb,
                log_file=log_file,
                # keep other kwargs at defaults (risk weights etc.) unless you decide to expose them
            )

            if not top:
                raise RuntimeError("EA returned no candidates.")

            # Save best and show leaderboard
            best_params, best_score = top[0]
            st.session_state["ea_best_params"] = dict(best_params)
            st.session_state["ea_top_results"] = list(top)

            prog.progress(0.95, text="Rendering EA leaderboard…")

            rows = []
            for params, score in top[: min(50, len(top))]:
                r = {"score": float(score)}
                r.update({k: params.get(k) for k in ("breakout_n","exit_n","atr_n","atr_multiple","tp_multiple","holding_period_limit")})
                rows.append(r)
            lb = pd.DataFrame(rows).sort_values("score", ascending=False)
            st.markdown("**EA leaderboard (top candidates)**")
            st.dataframe(lb, width="stretch", height=360)

            st.session_state["ea_log_file"] = log_file
            st.success(f"EA complete. Best score={best_score:.3f}. A Walk-Forward button is now available below.")

            # Offer EA log download (if file exists)
            try:
                if os.path.exists(log_file):
                    with open(log_file, "rb") as _fh:
                        st.download_button(
                            "⬇️ Download EA log (JSONL)",
                            data=_fh.read(),
                            file_name=os.path.basename(log_file),
                            mime="application/json",
                            use_container_width=True,
                        )
            except Exception:
                pass

            prog.progress(1.0)

        except Exception as e:
            st.error(f"Training failed: {e}")
            st.stop()

    # ------------------------ WALK-FORWARD ------------------------
    st.markdown("---")
    st.subheader("Walk-Forward Validation")

    wf_cfg = _ss_get_dict(
        "wf_cfg",
        {
            "days_back": 365 * 3,  # 3y default
            "splits": 3,
            "train_days": 252,
            "test_days": 63,
            "step_days": 0,       # 0 => use test_days
            "use_ea": False,
            "ea_generations": 4,
            "ea_pop": 12,
            "ea_min_trades": 3,
        },
    )
    with st.expander("Walk-Forward params", expanded=False):
        wf_cfg["days_back"] = st.number_input("History window (days)", 252, 365 * 10, wf_cfg["days_back"], 1,
                                              help="Total lookback used to build WF splits (train+test across splits).")
        wf_cfg["splits"] = st.number_input("Splits", 1, 12, wf_cfg["splits"], 1,
                                           help="Number of walk-forward segments.")
        wf_cfg["train_days"] = st.number_input("Train window (days)", 60, 600, wf_cfg["train_days"], 1,
                                               help="In-sample training window per split.")
        wf_cfg["test_days"] = st.number_input("Test window (days)", 21, 252, wf_cfg["test_days"], 1,
                                              help="Out-of-sample evaluation window per split.")
        wf_cfg["step_days"] = st.number_input("Step size (days)", 0, 600, wf_cfg["step_days"], 1,
                                              help="Advance between splits. 0 = same as Test window.")
        wf_cfg["use_ea"] = st.checkbox("Use EA inside each split (opt params on IS)", value=bool(wf_cfg["use_ea"]))
        col_ea1, col_ea2, col_ea3 = st.columns(3)
        with col_ea1:
            wf_cfg["ea_generations"] = st.number_input("EA generations", 1, 50, wf_cfg["ea_generations"], 1)
        with col_ea2:
            wf_cfg["ea_pop"] = st.number_input("EA population", 2, 200, wf_cfg["ea_pop"], 1)
        with col_ea3:
            wf_cfg["ea_min_trades"] = st.number_input("EA min trades", 0, 200, wf_cfg["ea_min_trades"], 1)

    # Gate WF behind EA completion (best params discovered)
    ea_best = st.session_state.get("ea_best_params")
    if not ea_best:
        st.info("Run **Train (EA)** first. The Walk-Forward button will appear here after EA finishes.")
        run_wf = False
    else:
        st.markdown(f"Using EA-best params for WF: `{ea_best}`")
        run_wf = st.button("🧭 Run Walk-Forward", type="secondary",
                           help="Runs walk-forward validation using EA-best params.",
                           width="stretch")

    if run_wf:
        try:
            from src.optimization.walkforward import walk_forward
            end = _utc_now()
            start = end - timedelta(days=int(wf_cfg["days_back"] or 365))
            step_days = int(wf_cfg["step_days"] or 0) or int(wf_cfg["test_days"])

            # Prefer EA-best params; otherwise fall back to current base params
            use_params = _filter_params_for_strategy(strategy_dotted, st.session_state.get("ea_best_params") or base)

            st.info("Starting walk-forward…")
            prog_wf = st.progress(0.05, text="Loading splits & data…")

            out = walk_forward(
                strategy_dotted=strategy_dotted,
                tickers=tickers,
                start=start,
                end=end,
                starting_equity=float(equity),
                base_params=use_params,
                splits=int(wf_cfg["splits"]),
                train_days=int(wf_cfg["train_days"]),
                test_days=int(wf_cfg["test_days"]),
                step_days=int(step_days),
                use_ea=bool(wf_cfg["use_ea"]),
                ea_kwargs=dict(
                    generations=int(wf_cfg["ea_generations"]),
                    pop_size=int(wf_cfg["ea_pop"]),
                    min_trades=int(wf_cfg["ea_min_trades"]),
                    n_jobs=int(n_jobs),
                ),
                n_jobs=int(n_jobs),
                progress_cb=lambda evt, ctx: None,  # keep console quiet; UI below
            )

            # --- Adapt to new walkforward.py shape ---
            prog_wf.progress(0.7, text="Rendering results…")

            out = out or {}
            splits = out.get("splits", []) or []

            # Aggregate → show both mean & median OOS metrics if available
            aggregate = out.get("aggregate", {}) or {}
            oos_mean = aggregate.get("oos_mean") or {}
            oos_median = aggregate.get("oos_median") or {}

            if oos_mean or oos_median:
                st.markdown("**Walk-Forward OOS aggregate (equal-weight across splits)**")
                import pandas as _pd
                # Build a two-column table: metric | mean | median
                metrics = sorted(set(list(oos_mean.keys()) + list(oos_median.keys())))
                df_sum = _pd.DataFrame(
                    {
                        "metric": metrics,
                        "mean": [oos_mean.get(m) for m in metrics],
                        "median": [oos_median.get(m) for m in metrics],
                    }
                )
                st.dataframe(df_sum, width="stretch", height=340)
            else:
                st.warning("No aggregate OOS metrics were returned.")

            # First split preview: dates, params_used, and key OOS metrics
            if splits:
                first = splits[0] or {}
                train_start = first.get("train_start")
                train_end = first.get("train_end")
                test_start = first.get("test_start")
                test_end = first.get("test_end")
                params_used = first.get("params_used") or {}
                oos = first.get("out_sample") or {}

                st.markdown("**First split preview**")
                st.code(
                    "train: {ts} → {te} | OOS: {os} → {oe}\nparams_used: {p}\n"
                    "oos trades: {trades}".format(
                        ts=train_start, te=train_end, os=test_start, oe=test_end,
                        p=params_used,
                        trades=oos.get("trades", "N/A"),
                    ),
                    language="text",
                )
            else:
                st.info("No split details were returned.")

            # Optional artifacts (older UI expected this; walkforward.py may not emit it)
            artifacts = out.get("artifacts", {}) or {}
            if artifacts:
                st.markdown("**WF Artifacts / Anomalies**")
                _write_kv_table(artifacts)

            prog_wf.progress(1.0, text="Walk-Forward complete.")

        except Exception as e:
            st.error(f"Walk-forward failed: {e}")
            st.stop()