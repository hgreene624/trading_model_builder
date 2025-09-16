# src/data/yf.py
from __future__ import annotations

from datetime import datetime
import warnings
import pandas as pd
import yfinance as yf


def fetch_yf_data(symbol: str, start: datetime, end: datetime, *, auto_adjust: bool = True) -> pd.DataFrame:
    """
    Fetch daily OHLCV via yfinance. Returns DataFrame indexed by DatetimeIndex (UTC) with columns:
    open, high, low, close, volume (and adj_close if available)
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        df = yf.download(
            symbol,
            start=start,
            end=end,
            progress=False,
            auto_adjust=auto_adjust,
            interval="1d",
            threads=True,
        )

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )

    # tz-aware UTC index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    else:
        df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")

    df.sort_index(inplace=True)
    keep = [c for c in ("open", "high", "low", "close", "volume", "adj_close") if c in df.columns]
    return df[keep]


# --- Uniform alias so other modules can always call load_ohlcv on YF ---
def load_ohlcv(symbol: str, start: datetime, end: datetime, timeframe: str = "1d", *, auto_adjust: bool = True) -> pd.DataFrame:
    # timeframe is accepted for signature compatibility; YF leg uses daily bars.
    return fetch_yf_data(symbol, start, end, auto_adjust=auto_adjust)