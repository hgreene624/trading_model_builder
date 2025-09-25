# src/models/base_model_utils.py
from __future__ import annotations
import numpy as np
import pandas as pd

def atr_pct(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Convenience ATR% helper if you need it for filtering/analytics."""
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / n, adjust=False).mean()
    return (atr / df["close"]).replace([np.inf, -np.inf], np.nan)