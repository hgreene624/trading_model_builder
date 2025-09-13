# pages/2_Base_Model_Lab.py
from __future__ import annotations
from datetime import date, timedelta
from typing import Dict, List, Tuple
import inspect
import numpy as np
import pandas as pd
import streamlit as st

from src.storage import (
    list_portfolios, load_portfolio, base_model_path, write_json,
    save_portfolio_model, list_portfolio_models, load_portfolio_model,
)
from src.data.alpaca_data import load_ohlcv
# from src.data.cache import get_ohlcv_cached as load_ohlcv  # if you prefer caching
from src.models.general_trainer import train_general_model, TrainConfig

from src.models.base_model_utils import (
    compute_block_stats,
    suggest_priors_from_metrics,
)
from src.storage import save_base_metrics_ctx, load_latest_base_metrics

st.set_page_config(page_title="Base-Model Lab", layout="wide")
st.title("🧪 Base-Model Lab (priors from portfolio)")

# ------------------------- helpers -------------------------
CTX_KEY = "bm_ctx"          # holds computed priors/metrics/tickers/windows
GM_RES_KEY = "gm_results"   # holds general model training results

def _get_ctx() -> dict | None:
    return st.session_state.get(CTX_KEY)

def _set_ctx(d: dict) -> None:
    st.session_state[CTX_KEY] = d

def _clear_ctx() -> None:
    for k in (CTX_KEY, GM_RES_KEY):
        if k in st.session_state:
            del st.session_state[k]

# -------- UI controls (global) --------
c0, c1, c2 = st.columns([1.2, 1, 1])
with c0:
    portfolios = list_portfolios()
    port = st.selectbox("Portfolio", portfolios, index=0 if portfolios else None)
with c1:
    priors_years = st.number_input("Priors window (years)", value=8, min_value=3, max_value=20, step=1)
with c2:
    select_years = st.number_input("Selection window (years, OOS)", value=2, min_value=1, max_value=5, step=1)

today = date.today()
priors_start = date(today.year - priors_years, today.month, today.day)
priors_end = date(today.year - select_years, today.month, today.day) - timedelta(days=1)
select_start = date(today.year - select_years, today.month, today.day)
select_end = today

st.caption(f"Priors: **{priors_start} → {priors_end}** (long history).  Selection (OOS): **{select_start} → {select_end}**.")

# Show ticker count for the selected portfolio
try:
    _obj0 = load_portfolio(port) if port else None
    _tick0 = _obj0.get("tickers", []) if _obj0 else []
    st.metric("Tickers in portfolio", len(_tick0))
except Exception:
    pass


# --- Param sampler -----------------------------------------------------------
def _sample_params(priors: Dict[str, Dict], rng: np.random.Generator) -> Dict:
    out = {}
    for k, spec in priors.items():
        lo = spec.get("low")
        hi = spec.get("high")
        seed = spec.get("seed", {})
        dist = seed.get("dist", "uniform")
        if k == "use_trend_filter":
            p = float(seed.get("p", 0.5))
            out[k] = bool(rng.random() < p)
            continue
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            if dist == "gamma":
                kshape = float(seed.get("k", 2.0))
                theta = float(seed.get("theta", 5.0))
                val = float(rng.gamma(kshape, theta))
                val = lo + (val % max(1e-9, (hi - lo)))
            else:
                val = float(lo) + float(rng.random()) * (float(hi) - float(lo))
            if all(isinstance(x, (int, np.integer)) for x in (lo, hi)) and k not in {"atr_multiple", "risk_per_trade", "tp_multiple", "cost_bps"}:
                val = int(round(val))
            out[k] = val
        else:
            out[k] = lo
    return out

# --- Safe backtest wrapper: filters unsupported params so runs never "error" ---
def _run_backtest_safe(sym: str, start: str, end: str, params: Dict, starting_equity: float = 10_000.0) -> Dict:
    try:
        from src.models.atr_breakout import backtest_single  # type: ignore
    except Exception as e:
        return {"error": f"import backtest_single failed: {e}"}

    # Filter kwargs to what the strategy actually accepts
    try:
        sig = inspect.signature(backtest_single)
        allowed = set(sig.parameters.keys()) - {"symbol", "start", "end", "starting_equity"}
        filtered = {k: v for k, v in params.items() if k in allowed}
    except Exception:
        filtered = params

    try:
        return backtest_single(sym, start, end, starting_equity=starting_equity, **filtered)
    except Exception as e:
        # Last resort: run with no params beyond required
        try:
            return backtest_single(sym, start, end, starting_equity=starting_equity)
        except Exception:
            return {"error": str(e)}

# --- CV evaluator (robust, never returns 'error') -----------------------------
def _cv_eval(sym: str, params: Dict, folds: List[Tuple[str, str]]) -> Dict:
    sh_list, tr_list = [], []
    for fs, fe in folds:
        m = _run_backtest_safe(sym, fs, fe, params, starting_equity=10_000.0)
        sh_list.append(float(m.get("sharpe", 0.0) or 0.0))
        tr_list.append(int(m.get("trades", 0) or 0))
    return {
        "cv_sharpe": float(np.mean(sh_list)) if sh_list else -1e9,
        "cv_trades": int(np.sum(tr_list)),
        "fold_sharpes": sh_list,
        "fold_trades": tr_list,
    }

# ===================== STAGE 1: COMPUTE METRICS & PRIORS =====================
with st.expander("📂 Load previously saved metrics (skip recompute)", expanded=False):
    if port and st.button("Load latest saved metrics for this portfolio", use_container_width=True, key="bm_load_latest"):
        loaded = load_latest_base_metrics(port)
        if loaded:
            # Rebuild ctx from saved blob
            import pandas as pd
            pri_df = pd.DataFrame(loaded.get("pri_df", []))
            sel_df = pd.DataFrame(loaded.get("sel_df", []))
            if not pri_df.empty and "ticker" in pri_df.columns:
                pri_df = pri_df.set_index("ticker")
            if not sel_df.empty and "ticker" in sel_df.columns:
                sel_df = sel_df.set_index("ticker")

            ctx = {
                "port": loaded["meta"]["port"],
                "tickers": loaded["meta"].get("tickers", []),
                "pri_df": pri_df,
                "sel_df": sel_df,
                "priors": loaded.get("priors", {}),
                "windows": loaded["meta"].get("windows", {}),
                "errors": loaded["meta"].get("errors", []),
                "created": loaded["meta"].get("created"),
            }
            _set_ctx(ctx)
            st.success("Loaded saved metrics/priors. Scroll down to continue.")
            # Auto-persist the computed metrics so you can reload later
            try:
                outp = save_base_metrics_ctx(ctx)
                st.caption(f"Saved computed metrics → `{outp}`")
            except Exception as e:
                st.warning(f"Could not save metrics cache: {e}")

        else:
            st.info("No saved metrics found for this portfolio yet.")

with st.form("compute_form"):
    compute_btn = st.form_submit_button("📊 Compute metrics & suggest priors", use_container_width=True)
    reset_btn = st.form_submit_button("🔄 Reset page state")

if reset_btn:
    _clear_ctx()
    st.experimental_rerun()

if compute_btn:
    if not port:
        st.error("Pick a portfolio first.")
        st.stop()

    obj = load_portfolio(port)
    if not obj:
        st.error(f"Portfolio '{port}' not found or empty.")
        st.stop()

    tickers: List[str] = obj.get("tickers", [])
    if not tickers:
        st.warning("This portfolio has no tickers yet.")
        st.stop()

    st.write(f"Found **{len(tickers)}** tickers in portfolio **{port}**.")
    prog = st.progress(0.0)
    pri_rows, sel_rows = [], []
    err_log: List[str] = []

    for i, sym in enumerate(tickers):
        try:
            with st.spinner(f"Loading {sym} …"):
                df_p = load_ohlcv(sym, priors_start.isoformat(), priors_end.isoformat())
                df_s = load_ohlcv(sym, select_start.isoformat(), select_end.isoformat())
        except Exception as e:
            err_log.append(f"{sym}: {e}")
            df_p, df_s = pd.DataFrame(), pd.DataFrame()

        if not df_p.empty:
            stats_p = compute_block_stats(df_p)
            stats_p["ticker"] = sym
            pri_rows.append(stats_p)

        if not df_s.empty:
            stats_s = compute_block_stats(df_s)
            stats_s["ticker"] = sym
            sel_rows.append(stats_s)

        prog.progress((i + 1) / len(tickers))

    pri_df = pd.DataFrame(pri_rows).set_index("ticker") if pri_rows else pd.DataFrame()
    sel_df = pd.DataFrame(sel_rows).set_index("ticker") if sel_rows else pd.DataFrame()

    if pri_df.empty:
        st.error("No usable data in the Priors window. Try widening the dates or adjusting the portfolio.")
        st.stop()

    priors = suggest_priors_from_metrics(pri_df)

    ctx = {
        "port": port,
        "tickers": tickers,
        "pri_df": pri_df,
        "sel_df": sel_df,
        "priors": priors,
        "windows": {
            "priors_start": priors_start.isoformat(),
            "priors_end": priors_end.isoformat(),
            "select_start": select_start.isoformat(),
            "select_end": select_end.isoformat(),
        },
        "errors": err_log,
        "created": today.isoformat(),
    }
    _set_ctx(ctx)
    st.success("Metrics & priors computed. You can now save the spec or train the general model.")

ctx = _get_ctx()

# ===================== STAGE 2: DISPLAY + SAVE SPEC =====================
if ctx:
    pri_df: pd.DataFrame = ctx["pri_df"]
    sel_df: pd.DataFrame = ctx["sel_df"]
    priors: Dict[str, Dict] = ctx["priors"]
    tickers: List[str] = ctx["tickers"]

    if ctx.get("errors"):
        with st.expander("⚠️ Data load warnings (click to expand)"):
            for line in ctx["errors"]:
                st.write("- ", line)

    cA, cB = st.columns([1, 1])
    with cA:
        with st.expander("Priors window metrics (portfolio)", expanded=False):
            st.dataframe(pri_df, use_container_width=True, height=300)
            if not pri_df.empty:
                med = pri_df.median(numeric_only=True)
                st.caption(
                    f"Median Sharpe≈{med.get('sharpe_ann', float('nan')):.2f} | "
                    f"CAGR≈{(med.get('cagr', float('nan')) or 0)*100:.1f}% | "
                    f"MaxDD≈{med.get('max_dd', float('nan')):.2%} | "
                    f"ATR%≈{(med.get('median_atr_pct', float('nan')) or 0)*100:.2f}% | "
                    f"trend R²≈{med.get('trend_r2', float('nan')):.2f}"
                )
    with cB:
        with st.expander("Selection window metrics (OOS, for ranking)", expanded=False):
            if not sel_df.empty:
                st.dataframe(sel_df, use_container_width=True, height=300)
            else:
                st.info("No selection-window data for these tickers (or window too short).")

    st.subheader("Suggested priors (Trend/Breakout v2)")
    _pri_rows = [{"param": k, "low": v.get("low"), "high": v.get("high"), "seed": v.get("seed")} for k, v in priors.items()]
    _pri_df = pd.DataFrame(_pri_rows).set_index("param")
    with st.expander("Show suggested priors", expanded=False):
        st.dataframe(_pri_df, use_container_width=True, height=320)

    st.caption("Priors computed. Proceed to training below. (Spec saving removed to reduce clutter.)")

# ============ STAGE 3: Train General Base Model (CV random search) ============
# ============ STAGE 3: Train General Base Model (CV random search) ============
if ctx:
    st.divider()
    st.subheader("🧠 Train General Base Model (CV random search)")

    with st.expander("Configure training", expanded=True):
        form = st.form("gm_form")
        with form:
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            with c1: K = st.number_input("K per ticker", 8, 4096, 256, 8)
            with c2: n_folds = st.number_input("CV folds", 2, 8, 3, 1)
            with c3: min_tr = st.number_input("Min trades/fold", 0, 50, 2, 1)
            with c4: enforce = st.checkbox("Enforce trades", value=False)
            with c5: seed = st.number_input("Seed", 0, 999_999, 2025, 1)
            with c6: max_t = st.number_input("Max tickers", 1, len(ctx["tickers"]), min(50, len(ctx["tickers"])), 1)
            go = st.form_submit_button("🎯 Train")

    if go:
        # progress callback to render status
        prog = st.progress(0.0, text="Starting…")
        def cb(pct: float, msg: str):
            prog.progress(min(max(pct, 0.0), 1.0), text=f"Training… {msg}")

        cfg = TrainConfig(K=K, n_folds=n_folds, min_trades_fold=min_tr, enforce_trades=enforce, seed=seed)
        tickers = ctx["tickers"][: int(max_t)]
        res = train_general_model(
            tickers=tickers,
            priors=ctx["priors"],
            priors_start=date.fromisoformat(ctx["windows"]["priors_start"]),
            priors_end=date.fromisoformat(ctx["windows"]["priors_end"]),
            strategy_dotted="src.models.atr_breakout",  # adjust if different
            cfg=cfg,
            progress_cb=cb,
        )
        st.success(f"Done. Evaluated {len(tickers)} tickers.")
        st.dataframe(res["leaderboard"], use_container_width=True, height=320)

        # Show smoke test & a couple of per-ticker diagnostics
        with st.expander("Debug / smoke test"):
            st.json(res.get("debug", {}))
            if isinstance(res.get("cv_summary"), dict):
                st.caption("Cross‑validation summary:")
                st.json(res["cv_summary"])

        st.session_state["gm_results"] = res

        # --- Save / compare portfolio models ---
        st.subheader("💾 Save & compare portfolio models")

        default_pm_name = f"{ctx['port']}__{today.isoformat()}__GM"
        pm_name = st.text_input("Portfolio model name", value=default_pm_name, key="gm_pm_name")
        if st.button("Save portfolio model", use_container_width=True, key="gm_save_btn"):
            try:
                save_portfolio_model(ctx["port"], pm_name, res)
                st.success(f"Saved portfolio model '{pm_name}'.")
            except Exception as e:
                st.error(f"Save failed: {e}")

        # List & compare saved models for this portfolio
        try:
            saved_names = list_portfolio_models(ctx["port"]) or []
        except Exception:
            saved_names = []

        if saved_names:
            rows = []
            for nm in saved_names:
                try:
                    blob = load_portfolio_model(ctx["port"], nm) or {}
                    # Try to summarize with leaderboard if present
                    cv_sh_mean = np.nan
                    cv_tr_sum = 0
                    if isinstance(blob.get("leaderboard"), (list, tuple)):
                        _lb = pd.DataFrame(blob["leaderboard"]) if blob["leaderboard"] else pd.DataFrame()
                        if not _lb.empty:
                            if "cv_sharpe" in _lb.columns:
                                cv_sh_mean = float(_lb["cv_sharpe"].astype(float).mean())
                            if "cv_trades" in _lb.columns:
                                cv_tr_sum = int(_lb["cv_trades"].fillna(0).astype(int).sum())
                    rows.append({
                        "model": nm,
                        "tickers": len((blob.get("per_ticker") or {})),
                        "cv_sharpe_mean": cv_sh_mean,
                        "cv_trades_total": cv_tr_sum,
                    })
                except Exception:
                    rows.append({"model": nm, "tickers": np.nan, "cv_sharpe_mean": np.nan, "cv_trades_total": 0})

            comp_df = pd.DataFrame(rows)
            if not comp_df.empty:
                comp_df = comp_df.sort_values(["cv_sharpe_mean", "cv_trades_total"], ascending=[False, False])
                st.dataframe(comp_df, use_container_width=True, height=260)
        else:
            st.caption("No saved portfolio models yet for this portfolio.")

        # Show skipped tickers/errors if any
        _skipped = res.get("skipped", {}) if isinstance(res, dict) else {}
        if _skipped:
            with st.expander("Skipped tickers / errors", expanded=False):
                st.json(_skipped)

    # Save button stays the same but now uses session_state["gm_results"]
# ---------------- Baseline Strategy Filter (optional) -------------------------
st.divider()
with st.expander("⚡ Quick baseline filter (optional)", expanded=False):
    st.subheader("Random K-sample filter")
    st.markdown(
        "We randomly sample **K** parameter sets per ticker from the suggested priors, "
        "backtest them on the **Priors** window (train) and the **Selection** window (validation), "
        "then keep tickers whose **validation Sharpe** and **trade count** meet your thresholds. "
        "This yields a cleaner universe for the Evolution tuner."
    )

    ctx = _get_ctx()
    if ctx:
        tickers = ctx["tickers"]
        priors = ctx["priors"]
        priors_start = date.fromisoformat(ctx["windows"]["priors_start"])
        priors_end = date.fromisoformat(ctx["windows"]["priors_end"])
        select_start = date.fromisoformat(ctx["windows"]["select_start"])
        select_end = date.fromisoformat(ctx["windows"]["select_end"])

        cX, cY, cZ, cW, cR = st.columns([1, 1, 1, 1, 1])
        with cX:
            K = st.number_input("Samples per ticker (K)", min_value=4, max_value=200, value=24, step=4, key="bm_K")
        with cY:
            min_sh = st.number_input("Min validation Sharpe", min_value=-2.0, max_value=5.0, value=0.30, step=0.05, key="bm_sh")
        with cZ:
            min_tr = st.number_input("Min validation trades", min_value=0, max_value=200, value=8, step=1, key="bm_tr")
        with cW:
            max_ticks = st.number_input("Max tickers to test", min_value=1, max_value=len(tickers), value=min(50, len(tickers)), step=1, key="bm_max")
        with cR:
            seed_ui = st.number_input("Random seed", min_value=0, max_value=999_999, value=1337, step=1, key="bm_seed")

        go = st.button("🚀 Run baseline filter", use_container_width=True, key="bm_run")

        if go:
            rng = np.random.default_rng(int(seed_ui))
            tested = 0
            kept = []
            dropped = []
            prog2 = st.progress(0.0, text="Running baseline filter…")
            test_list = tickers[: int(max_ticks)]
            for i, sym in enumerate(test_list):
                best = None
                for s in range(int(K)):
                    p = _sample_params(priors, rng)
                    res_train = _run_backtest_safe(sym, priors_start.isoformat(), priors_end.isoformat(), p)
                    res_valid = _run_backtest_safe(sym, select_start.isoformat(), select_end.isoformat(), p)
                    sh = float(res_valid.get("sharpe", 0.0) or 0.0)
                    tr = int(res_valid.get("trades", 0) or 0)
                    score = sh
                    if (best is None) or (score > best.get("score", -1e9)):
                        best = {
                            "symbol": sym,
                            "score": score,
                            "sharpe_v": sh,
                            "trades_v": tr,
                            "params": p,
                            "metrics_train": res_train,
                            "metrics_valid": res_valid,
                        }
                tested += 1
                if best is None:
                    dropped.append({"symbol": sym, "reason": "no successful runs"})
                else:
                    if (best["sharpe_v"] >= float(min_sh)) and (best["trades_v"] >= int(min_tr)):
                        kept.append(best)
                    else:
                        best["reason"] = f"val Sharpe {best['sharpe_v']:.2f}, trades {best['trades_v']}"
                        dropped.append(best)
                prog2.progress((i + 1) / max(1, len(test_list)))

            st.success(f"Baseline done. Tested {tested} tickers. Kept {len(kept)}; Dropped {len(dropped)}.")
            if kept:
                kept_df_rows = []
                for r in kept:
                    row = {"ticker": r["symbol"], "val_sharpe": r["sharpe_v"], "val_trades": r["trades_v"]}
                    row.update({f"p_{k}": v for k, v in r["params"].items()})
                    kept_df_rows.append(row)
                st.dataframe(pd.DataFrame(kept_df_rows), use_container_width=True, height=280)
            if dropped:
                with st.expander("Dropped (didn't meet thresholds)"):
                    st.dataframe(pd.DataFrame([{"ticker": r.get("symbol"), "reason": r.get("reason", "")} for r in dropped]), use_container_width=True, height=220)
            st.session_state["bm_candidates"] = kept
    else:
        st.info("Run **Compute metrics & suggest priors** first to enable this section.")
