# src/models/atr_breakout.py
"""
Pluggable strategy wrapper for ATR breakout.

This keeps the UI/trainer decoupled: they call run_strategy(...) without caring
about how ATRParams works internally.
"""

from __future__ import annotations
from typing import Any, Dict, Tuple

from src.backtest.engine import ATRParams, backtest_atr_breakout

def _split_execution(p: Dict[str, Any] | Any) -> Tuple[Dict[str, Any] | Any, str]:
    if isinstance(p, dict):
        q = dict(p)
        exec_mode = str(q.pop("execution", "close"))
        return q, exec_mode
    return p, "close"

def run_strategy(
    symbol: str,
    start,
    end,
    starting_equity: float,
    params: Dict[str, Any] | ATRParams,
) -> Dict:
    p, exec_mode = _split_execution(params)
    if isinstance(p, dict):
        p = ATRParams(**p)
    return backtest_atr_breakout(
        symbol=symbol,
        start=start,
        end=end,
        starting_equity=starting_equity,
        params=p,
        execution=exec_mode,
    )