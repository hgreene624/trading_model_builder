# src/data/alpaca_data.py
from __future__ import annotations
import os
import pandas as pd
from dotenv import load_dotenv

# Load .env so os.getenv works in CLI and Streamlit
load_dotenv()

# Prefer Streamlit secrets when running the app, else fall back to env vars
try:
    import streamlit as st
    _SECRETS = dict(st.secrets)
except Exception:
    _SECRETS = {}

def _get(key: str, default: str = "") -> str:
    if key in _SECRETS:
        return _SECRETS.get(key, default)
    return os.getenv(key, default)

def _client():
    # alpaca-py >= 0.20
    from alpaca.data import StockHistoricalDataClient
    api_key = _get("ALPACA_API_KEY")
    secret_key = _get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError(
            "Missing ALPACA_API_KEY / ALPACA_SECRET_KEY. Put them in .env or .streamlit/secrets.toml"
        )
    return StockHistoricalDataClient(api_key, secret_key)

def load_ohlcv(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Load DAILY OHLCV [start, end] inclusive from Alpaca.
    Returns a DataFrame indexed by date with columns: open, high, low, close, volume.
    Always uses IEX (free) unless ALPACA_FEED=sip is set.
    """
    from alpaca.data import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed

    # Resolve feed setting from env/secrets. Default: IEX
    feed_name = (_get("ALPACA_FEED", "iex") or "iex").lower()
    feed = DataFeed.IEX if feed_name == "iex" else DataFeed.SIP

    c = _client()
    start_ts = pd.Timestamp(start)
    # Alpaca 'end' is exclusive; add 1 day so our end is inclusive
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1)

    req = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Day,
        feed=feed,  # IEX by default; SIP only if ALPACA_FEED=sip
        start=start_ts.to_pydatetime(),
        end=end_ts.to_pydatetime(),
    )
    bars = c.get_stock_bars(req)

    # Newer alpaca-py returns an object with .df (MultiIndex on symbol/timestamp)
    if hasattr(bars, "df"):
        df = bars.df.copy()
        if not df.empty:
            if isinstance(df.index, pd.MultiIndex):
                try:
                    df = df.xs(symbol, level=0)
                except Exception:
                    if "symbol" in df.columns:
                        df = df[df["symbol"] == symbol]
            # make tz-naive and normalize to date
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df.rename(columns={
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            })
            out = df[["open","high","low","close","volume"]].copy()
            out.index = pd.to_datetime(out.index.date)
            out.index.name = "date"
            return out

    # Fallback: some versions return iterables per symbol
    rows = []
    try:
        sym_bars = bars[symbol]
    except Exception:
        sym_bars = getattr(bars, symbol, [])
    for b in sym_bars:
        ts = getattr(b, "timestamp", None)
        if ts is None:
            continue
        ts = pd.Timestamp(ts).tz_localize(None)
        rows.append({
            "date": ts.date(),
            "open": float(b.open),
            "high": float(b.high),
            "low": float(b.low),
            "close": float(b.close),
            "volume": int(b.volume),
        })
    if not rows:
        raise ValueError(f"No Alpaca data for {symbol} between {start} and {end}")
    out = pd.DataFrame(rows).set_index("date").sort_index()
    out.index = pd.to_datetime(out.index)
    out.index.name = "date"
    return out