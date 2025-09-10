# src/data/yf.py
from __future__ import annotations
import pandas as pd
import yfinance as yf

def load_ohlcv(symbol: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(f"No data for {symbol} in range {start} to {end}")
    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    })
    df.index.name = "date"
    return df[["open","high","low","close","volume"]].copy()