# src/backtest/metrics.py
"""
Core performance & timing metrics for model selection.

This module intentionally implements the "80/20" set:
- CAGR, Max Drawdown, Calmar
- Profit Factor, Expectancy per trade
- MFE/MAE Edge Ratio
- Entry/Exit efficiency (relative to entry/exit day's range)
- Sharpe (for continuity), Win-rate, Avg holding days

Input contracts:
- equity: pd.Series (cumulative equity; index=DatetimeIndex)
- daily_returns: pd.Series (pct change of equity, daily)
- trades: list[dict] as defined in the project CONTRACT in engine.py

All functions are dependency-light (numpy/pandas only) and safe on empty inputs.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

TRADING_DAYS = 252

# ---------- Path/shape independent helpers ----------

def _to_float(x, default=0.0):
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except Exception:
        return default

# ---------- Baseline performance ----------

def sharpe_ratio(daily_returns: pd.Series, risk_free_daily: float = 0.0) -> float:
    r = daily_returns.dropna().astype(float) - risk_free_daily
    sd = r.std(ddof=0)
    if len(r) == 0 or sd == 0:
        return 0.0
    return float((r.mean() / sd) * np.sqrt(TRADING_DAYS))

def total_return(equity: pd.Series) -> float:
    if equity is None or len(equity) < 2:
        return 0.0
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)

def cagr(equity: pd.Series) -> float:
    if equity is None or len(equity) < 2:
        return 0.0
    start_val = _to_float(equity.iloc[0], default=np.nan)
    end_val = _to_float(equity.iloc[-1], default=np.nan)
    if not np.isfinite(start_val) or not np.isfinite(end_val) or start_val <= 0:
        return 0.0
    days = (equity.index[-1] - equity.index[0]).days
    if days <= 0:
        return 0.0
    years = days / 365.25
    return float((end_val / start_val) ** (1 / years) - 1.0)

def max_drawdown(equity: pd.Series) -> float:
    if equity is None or len(equity) == 0:
        return 0.0
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(dd.min())

def calmar_ratio(equity: pd.Series) -> float:
    mdd = abs(max_drawdown(equity))
    if mdd == 0:
        return 0.0
    return float(cagr(equity) / mdd)

# ---------- Trade summaries ----------

def summarize_trades(trades: list[dict]) -> dict:
    if not trades:
        return {"trades": 0, "win_rate": 0.0, "avg_return": 0.0, "avg_holding_days": 0.0}
    rets = np.array([_to_float(t.get("return_pct", 0.0)) for t in trades], dtype=float)
    holds = np.array([_to_float(t.get("holding_days", 0.0)) for t in trades], dtype=float)
    wins = (rets > 0).sum()
    return {
        "trades": int(len(trades)),
        "win_rate": float(wins / len(trades)),
        "avg_return": float(np.nanmean(rets)) if len(rets) else 0.0,
        "avg_holding_days": float(np.nanmean(holds)) if len(holds) else 0.0,
    }

def profit_factor(trades: list[dict]) -> float:
    if not trades:
        return 0.0
    rets = np.array([_to_float(t.get("return_pct", 0.0)) for t in trades], dtype=float)
    gp = rets[rets > 0].sum()
    gl = -rets[rets < 0].sum()
    if gl == 0:
        return float(gp) if gp > 0 else 0.0
    return float(gp / gl)

def expectancy_per_trade(trades: list[dict]) -> float:
    if not trades:
        return 0.0
    rets = np.array([_to_float(t.get("return_pct", 0.0)) for t in trades], dtype=float)
    return float(np.nanmean(rets)) if len(rets) else 0.0

# ---------- Timing diagnostics ----------

def mfe_mae_edge_ratio(trades: list[dict]) -> dict:
    """
    Edge Ratio is median(MFE) / median(|MAE|).
    Requires per-trade 'mfe' and 'mae' in *return terms* (e.g., +0.03, -0.02).
    """
    if not trades:
        return {"median_mfe": 0.0, "median_mae": 0.0, "edge_ratio": 0.0}
    mfes = [t.get("mfe", np.nan) for t in trades]
    maes = [abs(t.get("mae", np.nan)) for t in trades]
    mfes = np.array([_to_float(x, np.nan) for x in mfes], dtype=float)
    maes = np.array([_to_float(x, np.nan) for x in maes], dtype=float)
    mfes = mfes[np.isfinite(mfes)]
    maes = maes[np.isfinite(maes)]
    if len(mfes) == 0 or len(maes) == 0:
        return {"median_mfe": 0.0, "median_mae": 0.0, "edge_ratio": 0.0}
    med_mfe = float(np.median(mfes))
    med_mae = float(np.median(maes))
    er = float(med_mfe / med_mae) if med_mae != 0 else 0.0
    return {"median_mfe": med_mfe, "median_mae": med_mae, "edge_ratio": er}

def entry_exit_efficiency(trades: list[dict]) -> dict:
    """
    Entry efficiency (long): (day_high - entry_price) / (day_high - day_low)
    Exit efficiency  (long): (exit_price - day_low_exit) / (day_high_exit - day_low_exit)
    For shorts, invert appropriately.
    """
    entries, exits = [], []
    for t in trades:
        side = t.get("side", "long")
        e = t.get("entry_price"); lo = t.get("day_low"); hi = t.get("day_high")
        x = t.get("exit_price"); lo2 = t.get("day_low_exit"); hi2 = t.get("day_high_exit")
        if None not in (e, lo, hi) and hi > lo:
            if side == "long":
                eff = (hi - e) / (hi - lo)
            else:
                eff = (e - lo) / (hi - lo)
            entries.append(np.clip(_to_float(eff, 0.0), 0.0, 1.0))
        if None not in (x, lo2, hi2) and hi2 > lo2:
            if side == "long":
                eff2 = (x - lo2) / (hi2 - lo2)
            else:
                eff2 = (hi2 - x) / (hi2 - lo2)
            exits.append(np.clip(_to_float(eff2, 0.0), 0.0, 1.0))
    return {
        "entry_efficiency": float(np.mean(entries)) if entries else 0.0,
        "exit_efficiency": float(np.mean(exits)) if exits else 0.0,
    }

# ---------- One-shot summary ----------

def compute_core_metrics(equity: pd.Series, daily_returns: pd.Series, trades: list[dict]) -> dict:
    base = summarize_trades(trades)
    out = {
        **base,
        "total_return": total_return(equity),
        "cagr": cagr(equity),
        "max_drawdown": max_drawdown(equity),
        "calmar": calmar_ratio(equity),
        "sharpe": sharpe_ratio(daily_returns),
        "profit_factor": profit_factor(trades),
        "expectancy": expectancy_per_trade(trades),
    }
    out.update(mfe_mae_edge_ratio(trades))
    out.update(entry_exit_efficiency(trades))
    return out