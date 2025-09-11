# pages/2_Base_Model_Lab.py
from __future__ import annotations
from datetime import date, timedelta
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import streamlit as st

from src.storage import list_portfolios, load_portfolio, base_model_path, write_json
from src.data.alpaca_data import load_ohlcv
# from src.data.cache import get_ohlcv_cached as load_ohlcv  # if you prefer caching

st.set_page_config(page_title="Base-Model Lab", layout="wide")
st.title("🧪 Base-Model Lab (priors from portfolio)")

# -------- UI controls
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

run = st.button("📊 Compute metrics & suggest priors", type="primary", use_container_width=True)
st.divider()

def tr_atr_pct(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    return (atr / df["close"]).replace([np.inf, -np.inf], np.nan)

def momentum(df: pd.DataFrame, n: int) -> float:
    return float(df["close"].pct_change(n).iloc[-1]) if len(df) > n else np.nan

def sharpe_like(rets: pd.Series) -> float:
    if rets.std(ddof=0) == 0 or rets.dropna().empty:
        return 0.0
    # daily sharpe; approximate annual ~ sqrt(252) scaling can be done outside
    return float(np.sqrt(252) * rets.mean() / (rets.std(ddof=0) + 1e-12))

def compute_block_stats(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {}
    df = df.copy()
    df["ret"] = df["close"].pct_change()
    r = df["ret"].dropna()
    sharpe_d = sharpe_like(r)
    cagr = float((df["close"].iloc[-1] / df["close"].iloc[0]) ** (252 / max(1, len(df))) - 1.0)
    dd = float((df["close"] / df["close"].cummax() - 1.0).min())
    atrp = tr_atr_pct(df)
    r2 = np.nan
    # trendiness via log-linear R²
    y = np.log(df["close"].dropna())
    if len(y) >= 60:
        x = np.arange(len(y))
        A = np.vstack([x, np.ones_like(x)]).T
        beta, alpha = np.linalg.lstsq(A, y.values, rcond=None)[0]
        yhat = alpha + beta * x
        ss_res = np.sum((y.values - yhat) ** 2)
        ss_tot = np.sum((y.values - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "sharpe_ann": sharpe_d,         # already annualized-ish
        "cagr": cagr,
        "max_dd": dd,
        "median_atr_pct": float(np.nanmedian(atrp)) if len(atrp.dropna()) else np.nan,
        "mom_63": momentum(df, 63),
        "mom_252": momentum(df, 252),
        "trend_r2": r2,
        "rows": int(len(df)),
        "start": str(df.index[0].date()),
        "end": str(df.index[-1].date()),
    }

def suggest_priors_from_metrics(df_metrics: pd.DataFrame) -> Dict[str, Dict]:
    """
    Heuristics that map portfolio metrics to Trend/Breakout v2 priors.
    You can refine these later or replace with CV sampling.
    """
    # Portfolio medians
    med_atr = float(df_metrics["median_atr_pct"].median())
    med_trend = float(df_metrics["trend_r2"].median())
    # Heuristic breakout windows: more trendiness → longer lookback; more vol → shorter.
    # We clamp to sensible ranges; EA will search within these bounds.
    if not np.isfinite(med_trend):
        med_trend = 0.2
    base_lo = int(np.clip(12 - 8 * med_trend + 100 * med_atr, 8, 40))
    base_hi = int(np.clip(36 + 12 * med_trend + 160 * med_atr, 25, 80))

    priors = {
        "breakout_n": {"low": base_lo, "high": base_hi, "seed": {"dist": "gamma", "k": 2.2, "theta": 10}},
        "exit_n": {"low": max(4, int(base_lo*0.4)), "high": max(8, int(base_hi*0.8)), "seed": {"dist": "gamma", "k": 1.8, "theta": 8}},
        "atr_n": {"low": 10, "high": 22, "seed": {"dist": "uniform"}},
        "atr_multiple": {"low": 2.0, "high": 3.5, "seed": {"dist": "uniform"}},
        "tp_multiple": {"low": 1.2, "high": 3.2, "seed": {"dist": "uniform"}},
        "holding_period_limit": {"low": 20, "high": 160, "seed": {"dist": "uniform"}},
        "risk_per_trade": {"low": 0.006, "high": 0.015, "seed": {"dist": "uniform"}},  # 0.6%–1.5% to chase higher CAGR
        "use_trend_filter": {"low": 0, "high": 1, "seed": {"dist": "bernoulli", "p": 0.7}},
        "sma_fast": {"low": 10, "high": 40, "seed": {"dist": "uniform"}},
        "sma_slow": {"low": 50, "high": 120, "seed": {"dist": "uniform"}},
        "sma_long": {"low": 180, "high": 280, "seed": {"dist": "uniform"}},
        "long_slope_len": {"low": 10, "high": 30, "seed": {"dist": "uniform"}},
        "cost_bps": {"low": 1.0, "high": 6.0, "seed": {"dist": "uniform"}},
        # optional volatility/CHOP gates you can implement in strategy:
        "chop_max": {"low": 38, "high": 60, "seed": {"dist": "uniform"}},
        "atr_ratio_max": {"low": 1.2, "high": 2.2, "seed": {"dist": "uniform"}},  # ATR vs its MA
    }
    return priors

if run and port:
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
    err_log = []

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

    if err_log:
        with st.expander("⚠️ Data load warnings (click to expand)"):
            for line in err_log:
                st.write("- ", line)

    pri_df = pd.DataFrame(pri_rows).set_index("ticker") if pri_rows else pd.DataFrame()
    sel_df = pd.DataFrame(sel_rows).set_index("ticker") if sel_rows else pd.DataFrame()

    if pri_df.empty:
        st.error("No usable data in the Priors window. Try widening the dates or adjusting the portfolio.")
        st.stop()

    # ---- Suggest priors from the portfolio's Priors-window metrics
    priors = suggest_priors_from_metrics(pri_df)

    # ---- UI: show metrics & priors
    cA, cB = st.columns([1, 1])
    with cA:
        st.subheader("Priors window metrics (portfolio)")
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
        st.subheader("Selection window metrics (OOS, for ranking)")
        if not sel_df.empty:
            st.dataframe(sel_df, use_container_width=True, height=300)
        else:
            st.info("No selection-window data for these tickers (or window too short).")

    st.subheader("Suggested priors (Trend/Breakout v2)")
    st.json(priors)

    # ---- Save spec
    archetype = "trend_breakout_v2"
    default_name = f"{archetype}__{port}__{today.isoformat()}"
    model_name = st.text_input("Base model name", value=default_name, help="File will be saved under storage/base_models/")
    save_btn = st.button("💾 Save Base Model Spec", use_container_width=True)

    if save_btn:
        spec = {
            "meta": {
                "archetype": archetype,
                "portfolio": port,
                "created": today.isoformat(),
                "priors_window": {"start": priors_start.isoformat(), "end": priors_end.isoformat()},
                "selection_window": {"start": select_start.isoformat(), "end": select_end.isoformat()},
                "tickers": tickers,
                "counts": {"priors_nonempty": int(len(pri_df)), "selection_nonempty": int(len(sel_df))},
            },
            "priors": priors,
            "metrics": {
                "priors": pri_df.reset_index().to_dict(orient="records"),
                "selection": sel_df.reset_index().to_dict(orient="records"),
            },
        }

        # Try using your helper; fall back to a sane path if signature differs.
        try:
            out_path = base_model_path(archetype, port, today.isoformat())
        except TypeError:
            from pathlib import Path
            out_path = Path(f"storage/base_models/{model_name}.json").as_posix()

        try:
            write_json(out_path, spec)
            st.success(f"Saved base model spec → `{out_path}`")
        except Exception as e:
            st.error(f"Failed to save spec: {e}")