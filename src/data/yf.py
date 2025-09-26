# src/data/yf.py
from __future__ import annotations

import os
import time
import random
from datetime import datetime
from typing import Optional

import pandas as pd

from ._tz_utils import to_utc_index

# yfinance is only imported when called (so unit tests that don't need it won't fail on import)
def _lazy_import_yf():
    import yfinance as yf  # noqa: WPS433
    return yf


def _backoff_sleep(attempt: int):
    """
    Exponential backoff with jitter:
    attempt: 0,1,2,... -> sleep ~ 1, 2, 4, 8, 12 (+ small jitter)
    """
    base = [1, 2, 4, 8, 12]
    wait = base[min(attempt, len(base) - 1)]
    # jitter 0–300ms to de-sync parallel pulls
    time.sleep(wait + random.random() * 0.3)


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize to columns: open, high, low, close, volume; UTC DatetimeIndex; ascending."""
    if df is None or df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        # e.g. ('Open','AAPL') -> 'open'
        df.columns = [str(t[-1]).lower() for t in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]

    # Common aliases
    ren = {}
    for want in ("open", "high", "low", "close", "volume"):
        if want not in df.columns:
            for cand in (want, want[0], f"{want}_price", f"adj_{want}", f"{want}_adj"):
                if cand in df.columns:
                    ren[cand] = want
                    break
    if ren:
        df = df.rename(columns=ren)

    # Ensure datetime index in UTC
    idx_source = df.index
    if not isinstance(df.index, pd.DatetimeIndex):
        for ts_col in ("timestamp", "time", "date", "datetime"):
            if ts_col in df.columns:
                idx_source = df[ts_col]
                if ts_col != "timestamp":
                    df = df.drop(columns=[ts_col])
                break

    df.index = to_utc_index(idx_source)

    df = df.sort_index()
    # Return only the canonical columns if present
    cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    return df[cols] if cols else df


def load_ohlcv(
    symbol: str,
    start: datetime,
    end: datetime,
    timeframe: str = "1d",
    *,
    auto_adjust: Optional[bool] = None,
) -> pd.DataFrame:
    """
    Download OHLCV from Yahoo Finance with retries/backoff and normalize output.

    Env overrides:
      - YF_MAX_RETRIES   (int, default 5)
      - YF_HTTP_TIMEOUT  (int seconds, currently informational only; yfinance manages requests)
      - YF_VERBOSE       (1 to print attempts)
    """
    yf = _lazy_import_yf()

    # yfinance changed default auto_adjust=True; keep it explicit & overridable
    if auto_adjust is None:
        auto_adjust = True

    max_retries = int(os.getenv("YF_MAX_RETRIES", "5"))
    http_timeout = int(os.getenv("YF_HTTP_TIMEOUT", "25"))  # informational; yfinance manages internals
    verbose = os.getenv("YF_VERBOSE", "0") == "1"

    if timeframe.lower() not in {"1d", "1day", "d", "day"}:
        # This module is currently used for daily bars only in our pipeline.
        raise ValueError(f"Unsupported timeframe for Yahoo loader: {timeframe}")

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            if verbose:
                print(f"[yf] attempt {attempt+1}/{max_retries} symbol={symbol} "
                      f"start={start} end={end} auto_adjust={auto_adjust} timeout={http_timeout}s")

            # Keep interval='1d' and threads=False for reproducibility
            df = yf.download(
                symbol,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=auto_adjust,
                progress=False,
                threads=False,
            )

            # yfinance may return a single- or multi-indexed columns dataframe
            if df is None or df.empty:
                raise RuntimeError("Yahoo returned empty dataframe")

            out = _normalize_ohlcv(df)
            if out is None or out.empty or not {"open", "high", "low", "close"}.issubset(out.columns):
                raise RuntimeError(f"Yahoo returned dataframe missing OHLC columns. Got: {list(out.columns)}")

            return out
        except Exception as e:  # noqa: BLE001
            last_err = e
            if verbose:
                print(f"[yf] error attempt={attempt+1}: {repr(e)}")
            if attempt < max_retries - 1:
                _backoff_sleep(attempt)
            continue

    raise RuntimeError(f"Yahoo Finance download failed for {symbol} after {max_retries} attempts") from last_err