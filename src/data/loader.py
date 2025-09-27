# src/data/loader.py
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

# Providers
from . import alpaca_data as A
from . import yf as Y

from ._tz_utils import to_utc_index

# Storage roots
from src.storage import get_ohlcv_root

# Optional in-process RAM cache
try:
    from . import memory_cache as MEM
except Exception:
    class _NoMem:
        @staticmethod
        def get(*a, **k): return None
        @staticmethod
        def put(*a, **k): return None
        @staticmethod
        def clear(*a, **k): return None
    MEM = _NoMem()


# ---------------------------
# Normalization & formatting
# ---------------------------

def _normalize_ohlcv(df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Normalize to columns: open, high, low, close, volume
    UTC DateTimeIndex, ascending sort, numeric types, drop empty OHLC rows.
    """
    if df is None:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = df.copy()

    # Flatten multiindex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(x[-1]).lower() for x in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]

    alias_map = {
        "open": ("open", "o", "open_price", "adj_open", "open_adj"),
        "high": ("high", "h", "high_price", "adj_high", "high_adj"),
        "low": ("low", "l", "low_price", "adj_low", "low_adj"),
        "close": ("close", "c", "close_price", "adj_close", "close_adj"),
        "volume": ("volume", "v", "vol", "shares", "qty"),
    }
    ren: dict[str, str] = {}
    for std, aliases in alias_map.items():
        for a in aliases:
            if a in df.columns:
                ren[a] = std
    if ren:
        df = df.rename(columns=ren)

    for c in ("open", "high", "low", "close", "volume"):
        if c not in df.columns:
            df[c] = pd.NA

    # Index → datetime (UTC)
    idx_source = df.index
    if not isinstance(df.index, pd.DatetimeIndex):
        for c in ("datetime", "timestamp", "date", "time"):
            if c in df.columns:
                idx_source = df[c]
                df = df.drop(columns=[c])
                break

    try:
        df.index = to_utc_index(idx_source)
    except Exception:
        df.index = pd.DatetimeIndex([], tz="UTC")

    df = df.sort_index()
    df = df[["open", "high", "low", "close", "volume"]]
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    mask_valid = df[["open", "high", "low", "close"]].notna().any(axis=1)
    df = df.loc[mask_valid]
    return df


def _print_one_line(provider: str, symbol: str, df: pd.DataFrame | None) -> None:
    if df is None or df.empty:
        print(f"[loader] provider={provider} symbol={symbol} rows=0 cols=[]")
        return
    cols = list(df.columns)
    #print(f"[loader] provider={provider} symbol={symbol} rows={len(df)} cols={cols}")


def _widen_daily_end(end: datetime, timeframe: str) -> datetime:
    tf = (timeframe or "").lower()
    if tf in ("1d", "d", "day", "1day"):
        return (end + timedelta(days=1)).replace(tzinfo=timezone.utc)
    return (end + timedelta(minutes=5)).replace(tzinfo=timezone.utc)


def _as_utc_naive(dt: datetime) -> datetime:
    """
    Return a UTC-naive datetime equivalent to dt in UTC.
    Providers like Alpaca/Yahoo often call pd.to_datetime(..., tz="UTC") internally,
    which fails if a tz-aware datetime is passed in. We normalize here.
    """
    if dt.tzinfo is None:
        # Assume naive is already UTC-naive
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def _to_utc_timestamp(dt: datetime) -> pd.Timestamp:
    ts = pd.Timestamp(dt)
    if ts.tz is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


# ---------------
# Disk cache I/O (storage/data/ohlcv)
# ---------------

def _cache_root() -> Path:
    return get_ohlcv_root()


def _cache_path(provider: str, symbol: str, timeframe: str, start: datetime, end: datetime) -> Path:
    s = _to_utc_timestamp(start).date().isoformat()
    e = _to_utc_timestamp(end).date().isoformat()
    sym = symbol.upper().replace("/", "_")
    tf = (timeframe or "1D").upper()
    return _cache_root() / provider / sym / tf / f"{s}__{e}.parquet"


def _read_cache(path: Path) -> pd.DataFrame | None:
    try:
        if path.exists():
            df = pd.read_parquet(path)
            return _normalize_ohlcv(df)
    except Exception as e:
        print(f"[loader] cache read error {path}: {e!r}")
    return None


def _write_cache(path: Path, df: pd.DataFrame) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=True)
        tmp.replace(path)
    except Exception as e:
        print(f"[loader] cache write error {path}: {e!r}")


# ----------------
# Public function
# ----------------

def get_ohlcv(
    symbol: str,
    start: datetime,
    end: datetime,
    timeframe: str = "1D",
    *,
    force_provider: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV for [symbol, start, end, timeframe].
    Order: RAM -> disk (storage/data/ohlcv) -> Alpaca -> Yahoo.
    On success: normalize, print a one-liner, write disk cache, promote to RAM.
    """
    symbol = (symbol or "").strip().upper()
    if not symbol:
        raise ValueError("symbol is required")
    if not isinstance(start, datetime) or not isinstance(end, datetime):
        raise ValueError("start and end must be datetime")
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    provider_env = (force_provider or os.getenv("DATA_PROVIDER", "auto")).lower()
    use_alpaca_first = provider_env in {"alpaca", "auto", "alpaca_first"}
    use_yf = provider_env in {"yahoo", "auto", "alpaca_first", "yf_first"}

    end_adj = _widen_daily_end(end, timeframe)
    start_naive = _as_utc_naive(start)
    end_naive = _as_utc_naive(end_adj)

    # 0) RAM
    mem_df = MEM.get(symbol, start, end_adj, timeframe=timeframe)
    if mem_df is not None and not mem_df.empty:
        _print_one_line("memory", symbol, mem_df)
        return mem_df

    # 1) Disk cache
    for prov in ("alpaca", "yahoo"):
        cpath = _cache_path(prov, symbol, timeframe, start, end_adj)
        cdf = _read_cache(cpath)
        if cdf is not None and not cdf.empty:
            _print_one_line("cache", symbol, cdf)
            MEM.put(symbol, timeframe, cdf)  # promote
            s_ts = _to_utc_timestamp(start)
            e_ts = _to_utc_timestamp(end_adj)
            return cdf.loc[s_ts:e_ts]

    last_err: Optional[Exception] = None

    # 2) Alpaca
    if use_alpaca_first:
        try:
            feed = os.getenv("ALPACA_FEED", "iex")
            df = A.load_ohlcv(symbol, start_naive, end_naive, timeframe=timeframe, feed=feed)
            df = _normalize_ohlcv(df)
            if df is not None and not df.empty and {"open", "high", "low", "close"}.issubset(df.columns):
                _print_one_line("alpaca", symbol, df)
                _write_cache(_cache_path("alpaca", symbol, timeframe, start, end_adj), df)
                MEM.put(symbol, timeframe, df)
                return df
            _print_one_line("alpaca-empty", symbol, df)
        except Exception as e:
            last_err = e
            print(f"[loader] alpaca error symbol={symbol}: {repr(e)}")

    # 3) Yahoo fallback
    if use_yf:
        try:
            df = Y.load_ohlcv(symbol, start_naive, end_naive, timeframe=timeframe)
            df = _normalize_ohlcv(df)
            if df is not None and not df.empty and {"open", "high", "low", "close"}.issubset(df.columns):
                _print_one_line("yahoo", symbol, df)
                _write_cache(_cache_path("yahoo", symbol, timeframe, start, end_adj), df)
                MEM.put(symbol, timeframe, df)
                return df
            _print_one_line("yahoo-empty", symbol, df)
        except Exception as e:
            last_err = e
            print(f"[loader] yahoo error symbol={symbol}: {repr(e)}")

    msg = f"No data returned for {symbol} ({provider_env})."
    if last_err is not None:
        raise RuntimeError(msg) from last_err
    raise RuntimeError(msg)