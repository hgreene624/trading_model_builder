# src/data/loader.py
from __future__ import annotations

from datetime import datetime
from typing import Optional
import importlib
import pandas as pd


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure columns: open, high, low, close, volume; DatetimeIndex in UTC; sorted ascending."""
    if df is None or df.empty:
        return df
    df = df.copy()

    # Flatten multiindex columns if present (e.g., ("AAPL","open"))
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(x[-1]).lower() for x in df.columns]

    # Lowercase lookup
    lower = {c.lower(): c for c in df.columns}

    def _alias(*names: str):
        for n in names:
            if n in lower:
                return lower[n]
        return None

    # Rename common variants to standard names
    ren = {}
    for want in ("open", "high", "low", "close", "volume"):
        if want not in lower:
            cand = _alias(want, want[0], f"{want}_price", f"adj_{want}", f"{want}_adj")
            if cand:
                ren[cand] = want
    if ren:
        df.rename(columns=ren, inplace=True)

    # If we still don’t have all OHLC, give it back (caller can choose to error/continue)
    if not {"open", "high", "low", "close"}.issubset({c.lower() for c in df.columns}):
        return df

    # Build a datetime index if needed
    if not isinstance(df.index, pd.DatetimeIndex):
        for t in ("timestamp", "time", "date", "datetime"):
            if t in lower:
                df.index = pd.to_datetime(df[lower[t]], utc=True, errors="coerce")
                # keep columns tidy
                if t != "timestamp":
                    df.drop(columns=[lower[t]], inplace=True, errors="ignore")
                break

    # Ensure UTC tz-aware index
    if isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")

    # Sort ascending by time
    df.sort_index(inplace=True)
    return df


def _call_provider(func, symbol: str, start: datetime, end: datetime, timeframe: Optional[str]):
    """Call provider function with flexible signature (with/without timeframe kw)."""
    try:
        return func(symbol, start, end, timeframe=timeframe)
    except TypeError:
        return func(symbol, start, end)


def get_ohlcv(symbol: str, start: datetime, end: datetime, timeframe: Optional[str] = "1Day") -> pd.DataFrame:
    """
    Unified OHLC loader.
    1) Try Alpaca (lazy import; tolerant if SDK/keys missing)
    2) Fallback to Yahoo Finance (lazy import)
    Returns a (possibly empty) DataFrame. Raises RuntimeError only if both fail.
    """
    alp_err = yf_err = None

    # --- Alpaca (lazy import so missing SDK doesn’t crash) ---
    try:
        alp_mod = importlib.import_module("src.data.alpaca_data")
        if hasattr(alp_mod, "load_ohlcv"):
            df = _call_provider(alp_mod.load_ohlcv, symbol, start, end, timeframe)
            df = _normalize_ohlcv(df)
            if df is not None and not df.empty:
                return df
    except Exception as e:
        alp_err = e

    # --- Yahoo Finance (lazy import + alias support) ---
    try:
        yf_mod = importlib.import_module("src.data.yf")
        if hasattr(yf_mod, "load_ohlcv"):
            df = _call_provider(yf_mod.load_ohlcv, symbol, start, end, timeframe)
        else:
            # fallbacks if older naming is present
            f = getattr(yf_mod, "fetch_yf_data", None) or getattr(yf_mod, "get_ohlcv", None)
            if f is None:
                raise ImportError("No Yahoo loader found (expected load_ohlcv/fetch_yf_data/get_ohlcv)")
            df = f(symbol, start, end)
        df = _normalize_ohlcv(df)
        if df is not None and not df.empty:
            return df
    except Exception as e:
        yf_err = e

    # If both paths failed:
    raise RuntimeError(
        "Failed to load data via Alpaca and yfinance.\n"
        f"Alpaca: {repr(alp_err)}\n"
        f"YF: {repr(yf_err)}"
    )