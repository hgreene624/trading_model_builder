# src/training/general_trainer.py
"""
General trainer orchestrates:
- iterate tickers
- call strategy wrapper
- compute core metrics
- return JSON-serializable summary for UI/storage

This module has NO Streamlit code.
"""

from __future__ import annotations
from importlib import import_module
from typing import Any, Dict, List

import pandas as pd

from src.backtest.metrics import compute_core_metrics

def import_callable(dotted: str):
    """
    Import run_strategy from a dotted path, e.g., 'src.models.atr_breakout'.
    Strategy module MUST expose run_strategy(symbol, start, end, starting_equity, params) -> BacktestResult
    """
    mod = import_module(dotted)
    if not hasattr(mod, "run_strategy"):
        raise ImportError(f"{dotted} missing run_strategy(...)")
    return getattr(mod, "run_strategy")

def train_general_model(
    strategy_dotted: str,
    tickers: List[str],
    start,
    end,
    starting_equity: float,
    base_params: Dict[str, Any],
) -> Dict:
    run = import_callable(strategy_dotted)
    per_symbol = []
    aggregate_equity = None

    for sym in tickers:
        result = run(sym, start, end, starting_equity, base_params)
        metrics = compute_core_metrics(result["equity"], result["daily_returns"], result["trades"])

        per_symbol.append({
            "symbol": sym,
            "metrics": metrics,
            "meta": result.get("meta", {}),
            "trade_count": metrics["trades"],
        })

        # naive aggregate equity: equal-weighted average of normalized curves
        eq = result["equity"]
        if eq is not None and len(eq) > 0:
            norm = eq / float(eq.iloc[0])
            aggregate_equity = norm if aggregate_equity is None else aggregate_equity.add(norm, fill_value=0.0)

    # finalize aggregate to average if present
    agg_metrics = {}
    if aggregate_equity is not None:
        aggregate_equity = aggregate_equity / len(per_symbol)
        daily = aggregate_equity.pct_change().fillna(0.0)
        agg_metrics = compute_core_metrics(aggregate_equity, daily, trades=[])

    return {
        "strategy": strategy_dotted,
        "params": dict(base_params),
        "period": {"start": str(start), "end": str(end)},
        "results": per_symbol,
        "aggregate": {"metrics": agg_metrics},
    }