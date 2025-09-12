# src/data/cache.py
from __future__ import annotations

from pathlib import Path
from typing import Optional
import pandas as pd


DATA_DIR = Path("storage/data/ohlcv")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(symbol: str) -> Path:
    return DATA_DIR / f"{symbol.upper()}.parquet"


def _load_local(symbol: str) -> Optional[pd.DataFrame]:
    p = _cache_path(symbol)
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    # Support both index and a 'date' column
    if "date" in df.columns:
        df = df.set_index(pd.to_datetime(df["date"])).drop(columns=["date"])
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _save_local(symbol: str, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out.to_parquet(_cache_path(symbol), engine="pyarrow")


def _slice(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    s = pd.to_datetime(start)
    e = pd.to_datetime(end)
    return df.loc[(df.index >= s) & (df.index <= e)].copy()


def get_ohlcv_cached(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Returns OHLCV with DatetimeIndex and columns: open, high, low, close, volume.
    Uses on-disk parquet cache under storage/data/ohlcv. If requested window is not
    fully covered locally, fetches the missing range from Alpaca and updates the cache.
    """
    # Local import to avoid circular deps
    from src.data.alpaca_data import load_ohlcv as _remote

    symbol = symbol.upper()
    local = _load_local(symbol)

    need_fetch = True
    if local is not None and not local.empty:
        s = pd.to_datetime(start)
        e = pd.to_datetime(end)
        if local.index.min() <= s and local.index.max() >= e:
            need_fetch = False

    if need_fetch:
        remote = _remote(symbol, start, end)
        if remote is None or remote.empty:
            # If we have some local data, return the slice we can; else raise
            if local is not None and not local.empty:
                return _slice(local, start, end)
            raise ValueError(f"No Alpaca data for {symbol} between {start} and {end}")

        if local is not None and not local.empty:
            merged = pd.concat([local, remote]).sort_index()
            merged = merged[~merged.index.duplicated(keep="last")]
            _save_local(symbol, merged)
            return _slice(merged, start, end)
        else:
            _save_local(symbol, remote)
            return _slice(remote, start, end)

    # Fully covered locally
    return _slice(local, start, end)