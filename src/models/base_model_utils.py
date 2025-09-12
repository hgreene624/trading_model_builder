# src/models/base_model_utils.py
from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd

def tr_atr_pct(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    return (atr / df["close"]).replace([np.inf, -np.inf], np.nan)

def _sharpe_like(rets: pd.Series) -> float:
    r = rets.dropna()
    if r.empty:
        return 0.0
    vol = r.std(ddof=0)
    if vol == 0:
        return 0.0
    return float(np.sqrt(252) * r.mean() / (vol + 1e-12))

def compute_block_stats(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {}
    df = df.copy()
    df["ret"] = df["close"].pct_change()
    sharpe_d = _sharpe_like(df["ret"])
    cagr = float((df["close"].iloc[-1] / df["close"].iloc[0]) ** (252 / max(1, len(df))) - 1.0)
    dd = float((df["close"] / df["close"].cummax() - 1.0).min())
    atrp = tr_atr_pct(df)
    r2 = np.nan
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
        "sharpe_ann": sharpe_d,
        "cagr": cagr,
        "max_dd": dd,
        "median_atr_pct": float(np.nanmedian(atrp)) if len(atrp.dropna()) else np.nan,
        "mom_63": float(df["close"].pct_change(63).iloc[-1]) if len(df) > 63 else np.nan,
        "mom_252": float(df["close"].pct_change(252).iloc[-1]) if len(df) > 252 else np.nan,
        "trend_r2": r2,
        "rows": int(len(df)),
        "start": str(df.index[0].date()),
        "end": str(df.index[-1].date()),
    }

def suggest_priors_from_metrics(df_metrics: pd.DataFrame) -> Dict[str, Dict]:
    med_atr = float(df_metrics["median_atr_pct"].median())
    med_trend = float(df_metrics["trend_r2"].median())
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
        "risk_per_trade": {"low": 0.006, "high": 0.015, "seed": {"dist": "uniform"}},
        "use_trend_filter": {"low": 0, "high": 1, "seed": {"dist": "bernoulli", "p": 0.7}},
        "sma_fast": {"low": 10, "high": 40, "seed": {"dist": "uniform"}},
        "sma_slow": {"low": 50, "high": 120, "seed": {"dist": "uniform"}},
        "sma_long": {"low": 180, "high": 280, "seed": {"dist": "uniform"}},
        "long_slope_len": {"low": 10, "high": 30, "seed": {"dist": "uniform"}},
        "cost_bps": {"low": 1.0, "high": 6.0, "seed": {"dist": "uniform"}},
        "chop_max": {"low": 38, "high": 60, "seed": {"dist": "uniform"}},
        "atr_ratio_max": {"low": 1.2, "high": 2.2, "seed": {"dist": "uniform"}},
    }
    return priors