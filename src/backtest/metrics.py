# src/backtest/metrics.py
from __future__ import annotations
import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


def sharpe_ratio(daily_returns: pd.Series, risk_free_rate_daily: float = 0.0) -> float:
    """Annualized Sharpe from daily returns."""
    r = daily_returns.dropna().astype(float) - risk_free_rate_daily
    if r.std(ddof=0) == 0 or len(r) == 0:
        return 0.0
    return float((r.mean() / r.std(ddof=0)) * np.sqrt(252))


def max_drawdown(equity: pd.Series) -> float:
    """Return the minimum drawdown (negative number)."""
    if len(equity) == 0:
        return 0.0
    running_max = equity.cummax()
    dd = (equity / running_max) - 1.0
    return float(dd.min())


def cagr(equity: pd.Series) -> float:
    """Compound annual growth rate based on index span (calendar days)."""
    if len(equity) < 2:
        return 0.0
    start_val = float(equity.iloc[0])
    end_val = float(equity.iloc[-1])
    if start_val <= 0:
        return 0.0
    days = (equity.index[-1] - equity.index[0]).days
    years = max(days / 365.25, 1e-9)
    return float((end_val / start_val) ** (1.0 / years) - 1.0)


def volatility(daily_returns: pd.Series) -> float:
    r = daily_returns.dropna().astype(float)
    return float(r.std(ddof=0) * np.sqrt(252)) if len(r) else 0.0


def summarize_equity(
    equity: pd.Series,
    starting_equity: float,
) -> Dict[str, float]:
    ret = (equity.iloc[-1] / starting_equity) - 1.0 if len(equity) else 0.0
    daily = equity.pct_change().fillna(0.0)
    return {
        "total_return": float(ret),
        "sharpe": sharpe_ratio(daily),
        "max_drawdown": max_drawdown(equity),
        "volatility": volatility(daily),
        "cagr": cagr(equity),
        "final_equity": float(equity.iloc[-1]) if len(equity) else float(starting_equity),
        "start": str(equity.index[0].date()) if len(equity) else "",
        "end": str(equity.index[-1].date()) if len(equity) else "",
    }


def summarize_trades(trades: List[Dict]) -> Dict[str, float]:
    """Win rate and average trade metrics."""
    if not trades:
        return {"trades": 0, "win_rate": 0.0, "avg_return": 0.0, "avg_holding_days": 0.0}
    rets = [t.get("return_pct", 0.0) for t in trades]
    win_rate = float(sum(1 for x in rets if x > 0) / len(rets))
    hold = [t.get("holding_days", 0) for t in trades]
    return {
        "trades": len(trades),
        "win_rate": win_rate,
        "avg_return": float(np.mean(rets)) if rets else 0.0,
        "avg_holding_days": float(np.mean(hold)) if hold else 0.0,
    }