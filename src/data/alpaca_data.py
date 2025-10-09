# src/data/alpaca_data.py
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from ._tz_utils import to_utc_index


logger = logging.getLogger(__name__)

_RATE_LIMIT_STATUS = 429
_RATE_LIMIT_MESSAGE = "too many requests"
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BACKOFF_SECONDS = 1.0


def _as_text(value: Optional[object]) -> str:
    if value is None:
        return ""
    return str(value).lower()


def _is_rate_limit_error(exc: Exception) -> bool:
    """Return True if *exc* looks like an Alpaca rate-limit error."""

    status = getattr(exc, "status_code", None)
    code = getattr(exc, "code", None)
    text = _as_text(getattr(exc, "message", None)) or _as_text(exc)
    if status == _RATE_LIMIT_STATUS or code == _RATE_LIMIT_STATUS:
        return True
    return _RATE_LIMIT_MESSAGE in text


def _iso_utc(dt: datetime) -> datetime:
    """Return timezone-aware UTC datetime (alpaca-py accepts dt; no need for iso strings)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _get_data_client():
    """
    Lazy-import the modern alpaca-py Historical Data client.
    Reads ALPACA_API_KEY/ALPACA_SECRET_KEY if present; otherwise falls back to APCA_*.
    """
    from alpaca.data.historical import StockHistoricalDataClient  # lazy import
    api_key = os.getenv("ALPACA_API_KEY", os.getenv("APCA_API_KEY_ID"))
    secret_key = os.getenv("ALPACA_SECRET_KEY", os.getenv("APCA_API_SECRET_KEY"))
    if not api_key or not secret_key:
        raise RuntimeError("Missing ALPACA_API_KEY/ALPACA_SECRET_KEY (or APCA_*) in environment")
    return StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)


def _timeframe_to_ap(tf: str):
    """Map our string timeframe to alpaca-py TimeFrame."""
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    tf = (tf or "1Day").lower()
    if tf in ("1d", "1day", "day", "1 day"):
        return TimeFrame.Day
    if tf in ("1min", "1m"):
        return TimeFrame.Minute
    if tf in ("5min", "5m"):
        return TimeFrame(5, TimeFrameUnit.Minute)
    if tf in ("15min", "15m"):
        return TimeFrame(15, TimeFrameUnit.Minute)
    if tf in ("1h", "1hour", "60m"):
        return TimeFrame(1, TimeFrameUnit.Hour)
    return TimeFrame.Day


def load_ohlcv(symbol: str, start: datetime, end: datetime, timeframe: str = "1Day", **kwargs) -> pd.DataFrame:
    """
    Fetch OHLCV via alpaca-py Historical Data client.
    - Uses IEX feed (works on paper/basic plans; avoids 'recent SIP' errors).
    - Returns a DataFrame indexed by UTC DatetimeIndex with columns open, high, low, close, volume.
    - Accepts **kwargs to allow optional retry/backoff overrides.
    """
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed, Adjustment  # <-- fixed: Adjustment comes from enums

    client = _get_data_client()

    ap_tf = _timeframe_to_ap(timeframe)
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=ap_tf,
        start=_iso_utc(start),
        end=_iso_utc(end),
        adjustment=Adjustment.RAW,
        feed=DataFeed.IEX,  # avoid SIP
        limit=None,
    )

    max_retries = max(int(kwargs.get("max_retries", _DEFAULT_MAX_RETRIES)), 0)
    base_backoff = float(kwargs.get("backoff_seconds", _DEFAULT_BACKOFF_SECONDS))
    total_attempts = max_retries + 1

    last_exc = None
    for attempt in range(1, total_attempts + 1):
        try:
            bars = client.get_stock_bars(req)
            break
        except Exception as exc:  # pragma: no cover - network errors hard to simulate
            last_exc = exc
            if _is_rate_limit_error(exc) and attempt < total_attempts:
                delay = base_backoff * (2 ** (attempt - 1))
                if delay > 0:
                    logger.warning(
                        "Alpaca rate limit for %s (attempt %s/%s); retrying in %.1fs",
                        symbol,
                        attempt,
                        total_attempts,
                        delay,
                    )
                    time.sleep(delay)
                continue
            raise
    else:
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Unexpected rate-limit retry failure")

    df = bars.df

    if df is None or df.empty:
        return pd.DataFrame()

    # Drop symbol level if present
    if isinstance(df.index, pd.MultiIndex) and "symbol" in (df.index.names or []):
        try:
            df = df.xs(symbol, level="symbol")
        except Exception:
            pass

    # Normalize columns
    df.columns = [str(c).lower() for c in df.columns]
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]

    # Ensure UTC tz-aware index and sorted
    df.index = to_utc_index(df.index)
    df.sort_index(inplace=True)

    return df[keep]
