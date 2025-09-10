# src/data/loader.py
from __future__ import annotations
import pandas as pd

def get_ohlcv(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Try Alpaca first, then fall back to yfinance.
    """
    try:
        from .alpaca_data import load_ohlcv as _alp
        return _alp(symbol, start, end)
    except Exception as e_alp:
        try:
            from .yf import load_ohlcv as _yf
            return _yf(symbol, start, end)
        except Exception as e_yf:
            raise RuntimeError(f"Failed to load data via Alpaca and yfinance.\nAlpaca: {e_alp}\nYF: {e_yf}")