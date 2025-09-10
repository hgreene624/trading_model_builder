# src/models/atr_breakout.py
from __future__ import annotations
from typing import Dict, Any

import pandas as pd

from src.backtest.engine import ATRParams, backtest_atr_breakout


def backtest_single(
    symbol: str,
    start: str,
    end: str,
    breakout_n: int = 55,
    exit_n: int = 20,
    atr_n: int = 14,
    starting_equity: float = 10_000,
    atr_multiple: float = 3.0,
    risk_per_trade: float = 0.01,
    allow_fractional: bool = True,
    slippage_bp: float = 5.0,
    fee_per_trade: float = 0.0,
) -> Dict[str, Any]:
    """
    Wrapper preserving your previous signature (with added risk params).
    Returns {"equity": pd.Series, "trades": List[Dict], "metrics": Dict}
    """
    params = ATRParams(
        breakout_n=int(breakout_n),
        exit_n=int(exit_n),
        atr_n=int(atr_n),
        atr_multiple=float(atr_multiple),
        risk_per_trade=float(risk_per_trade),
        allow_fractional=bool(allow_fractional),
        slippage_bp=float(slippage_bp),
        fee_per_trade=float(fee_per_trade),
    )
    res = backtest_atr_breakout(
        symbol=symbol,
        start=start,
        end=end,
        starting_equity=float(starting_equity),
        params=params,
    )
    return res