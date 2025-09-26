# src/data/_tz_utils.py
from __future__ import annotations

import pandas as pd


def to_utc_index(idx_like) -> pd.DatetimeIndex:
    """Return a UTC DatetimeIndex from any datetime-like input."""
    converted = pd.to_datetime(idx_like, errors="coerce", utc=False)

    if isinstance(converted, pd.DatetimeIndex):
        di = converted
    elif isinstance(converted, pd.Series):
        di = pd.DatetimeIndex(converted.array)
    elif isinstance(converted, pd.Index):
        di = pd.DatetimeIndex(converted)
    else:
        di = pd.DatetimeIndex(pd.Index(converted))

    if di.tz is None:
        return di.tz_localize("UTC")
    return di.tz_convert("UTC")
