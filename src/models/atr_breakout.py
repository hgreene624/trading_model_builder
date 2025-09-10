from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any

def _load_ohlcv(symbol: str, start: str, end: str):
    """
    Prefer Streamlit-cached data during app runs; fall back to direct Alpaca load
    when not in a Streamlit context (e.g., CLI tests).
    """
    try:
        from ..data.cache import get_ohlcv_cached
        return get_ohlcv_cached(symbol, start, end)
    except Exception:
        from ..data.alpaca_data import load_ohlcv as _alp
        return _alp(symbol, start, end)

def wilder_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df['high'], df['low'], df['close']
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    return atr

def backtest_single(symbol: str, start: str, end: str, breakout_n: int = 55, exit_n: int = 20, atr_n: int = 14, starting_equity: float = 10_000) -> Dict[str, Any]:
    df = _load_ohlcv(symbol, start, end)
    df = df.copy()
    df['atr'] = wilder_atr(df, n=atr_n)
    df['breakout_high'] = df['high'].rolling(window=breakout_n, min_periods=breakout_n).max().shift(1)
    df['exit_low'] = df['low'].rolling(window=exit_n, min_periods=exit_n).min().shift(1)

    in_pos = False
    pos = []
    for _, row in df.iterrows():
        if not in_pos:
            in_pos = bool(row['close'] > row['breakout_high']) if not np.isnan(row['breakout_high']) else False
        else:
            if not np.isnan(row['exit_low']) and row['close'] < row['exit_low']:
                in_pos = False
        pos.append(1 if in_pos else 0)
    df['pos'] = pd.Series(pos, index=df.index).astype(float)

    ret = df['close'].pct_change().fillna(0.0)
    strat_ret = ret * df['pos'].shift(1).fillna(0.0)
    equity = (1.0 + strat_ret).cumprod() * float(starting_equity)

    total_return = (equity.iloc[-1] / starting_equity) - 1.0
    daily = strat_ret
    sharpe = (daily.mean() / (daily.std() + 1e-12)) * np.sqrt(252) if daily.std() > 0 else 0.0
    running_max = equity.cummax()
    drawdown = (equity / running_max) - 1.0
    max_dd = drawdown.min()
    metrics = {
        "symbol": symbol,
        "total_return": float(total_return),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "start": str(equity.index[0].date()),
        "end": str(equity.index[-1].date()),
        "final_equity": float(equity.iloc[-1]),
    }
    return {"equity": equity, "metrics": metrics}
