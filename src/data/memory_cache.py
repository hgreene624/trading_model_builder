# src/data/memory_cache.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd


@dataclass
class _Entry:
    df: pd.DataFrame
    start: pd.Timestamp
    end: pd.Timestamp


# Per-process store keyed by (SYMBOL, TF)
_STORE: Dict[Tuple[str, str], _Entry] = {}
_MAX_SYMBOLS = 512  # adjust for your RAM


def _key(symbol: str, timeframe: str) -> Tuple[str, str]:
    return symbol.upper(), (timeframe or "1D").upper()


def get(symbol, start, end, timeframe="1D") -> Optional[pd.DataFrame]:
    ent = _STORE.get(_key(symbol, timeframe))
    if ent is None:
        return None

    def _to_utc(ts):
        ts = pd.Timestamp(ts)
        # If tz-naive, localize to UTC; if tz-aware, convert to UTC
        return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")

    s = _to_utc(start)
    e = _to_utc(end)

    if ent.start <= s and e <= ent.end:
        return ent.df.loc[s:e]
    return None


def put(symbol: str, timeframe: str, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    k = _key(symbol, timeframe)
    s_new = pd.Timestamp(df.index.min()).tz_convert("UTC")
    e_new = pd.Timestamp(df.index.max()).tz_convert("UTC")
    if k in _STORE:
        old = _STORE[k]
        merged = pd.concat([old.df, df], axis=0).sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
        _STORE[k] = _Entry(df=merged, start=min(old.start, s_new), end=max(old.end, e_new))
    else:
        if len(_STORE) >= _MAX_SYMBOLS:
            _STORE.pop(next(iter(_STORE)))
        _STORE[k] = _Entry(df=df, start=s_new, end=e_new)


def clear(symbol: Optional[str] = None, timeframe: str = "1D") -> None:
    if symbol is None:
        _STORE.clear()
    else:
        _STORE.pop(_key(symbol, timeframe), None)