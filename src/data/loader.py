# src/data/loader.py
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

# Local providers
from . import alpaca_data as A
from . import yf as Y


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure columns: open, high, low, close, volume; DatetimeIndex in UTC; sorted ascending."""
    if df is None or df.empty:
        return df
    df = df.copy()

    # Flatten multiindex columns if present (e.g., ("AAPL","open"))
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(x[-1]).lower() for x in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]

    # Rename common variants to standard names
    ren = {}
    alias_map = {
        "open": ("open", "o", "open_price", "adj_open", "open_adj"),
        "high": ("high", "h", "high_price", "adj_high", "high_adj"),
        "low": ("low", "l", "low_price", "adj_low", "low_adj"),
        "close": ("close", "c", "close_price", "adj_close", "close_adj"),
        "volume": ("volume", "v"),
    }
    for want, aliases in alias_map.items():
        if want not in df.columns:
            for cand in aliases:
                if cand in df.columns:
                    ren[cand] = want
                    break
    if ren:
        df.rename(columns=ren, inplace=True)

    # Ensure DatetimeIndex UTC
    if not isinstance(df.index, pd.DatetimeIndex):
        for ts_col in ("timestamp", "time", "date", "datetime"):
            if ts_col in df.columns:
                df.index = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
                if ts_col != "timestamp":
                    df.drop(columns=[ts_col], inplace=True, errors="ignore")
                break

    if isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")

    df.sort_index(inplace=True)
    return df


def _widen_daily_end(end: datetime, timeframe: str) -> datetime:
    """For daily bars, nudge end forward so we don't miss the latest completed session."""
    if timeframe.lower() in {"1d", "1day", "d", "day"}:
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        return (end + timedelta(days=1)).astimezone(timezone.utc)
    return end


def _print_one_line(provider: str, symbol: str, df: Optional[pd.DataFrame]):
    rows = 0 if df is None else len(df)
    cols = [] if (df is None or df.empty) else list(df.columns)
    print(f"[loader] provider={provider} symbol={symbol} rows={rows} cols={cols[:5]}{'...' if len(cols) > 5 else ''}")


def get_ohlcv(
    symbol: str,
    start: datetime,
    end: datetime,
    timeframe: str = "1d",
    *,
    force_provider: Optional[str] = None,
) -> pd.DataFrame:
    """
    Unified OHLCV loader with:
      - Provider selection: Alpaca first, then Yahoo fallback (default), or forced via env.
      - Wider end-bound for daily bars to avoid off-by-one misses.
      - One-line proof of provider used (to stdout).
    Env:
      DATA_PROVIDER = 'alpaca' | 'yf' | 'auto' (default: auto)
      ALPACA_FEED   = 'iex' | 'sip' (we pass through to Alpaca only)
    """
    provider_env = (force_provider or os.getenv("DATA_PROVIDER", "auto")).strip().lower()
    use_alpaca_first = provider_env in {"auto", "alpaca"}
    use_yf = provider_env in {"auto", "yf"}

    end_adj = _widen_daily_end(end, timeframe)

    last_err: Optional[Exception] = None

    # 1) Try Alpaca
    if use_alpaca_first:
        try:
            feed = os.getenv("ALPACA_FEED", "iex")
            df = A.load_ohlcv(symbol, start, end_adj, timeframe=timeframe, feed=feed)
            df = _normalize_ohlcv(df)
            # Ensure canonical columns are present
            if df is not None and not df.empty and {"open", "high", "low", "close"}.issubset(df.columns):
                _print_one_line("alpaca", symbol, df)
                return df
            # If we reach here, data is empty or missing columns; fall through
            _print_one_line("alpaca-empty", symbol, df)
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[loader] alpaca error symbol={symbol}: {repr(e)}")

    # 2) Fallback to Yahoo
    if use_yf:
        try:
            df = Y.load_ohlcv(symbol, start, end_adj, timeframe=timeframe)
            df = _normalize_ohlcv(df)
            if df is not None and not df.empty and {"open", "high", "low", "close"}.issubset(df.columns):
                _print_one_line("yahoo", symbol, df)
                return df
            _print_one_line("yahoo-empty", symbol, df)
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[loader] yahoo error symbol={symbol}: {repr(e)}")

    # If neither worked
    msg = f"No data returned for {symbol} ({provider_env})."
    if last_err is not None:
        raise RuntimeError(msg) from last_err
    raise RuntimeError(msg)