from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Tuple

import pandas as pd

from src.data import loader


def _make_fake_df(rows: int) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, tz="UTC")
    data = {
        "timestamp": idx,
        "open": [100 + i for i in range(rows)],
        "high": [101 + i for i in range(rows)],
        "low": [99 + i for i in range(rows)],
        "close": [100.5 + i for i in range(rows)],
        "volume": [1000 + i for i in range(rows)],
    }
    return pd.DataFrame(data)


def setup_fake_provider(monkeypatch, tmp_path, rows: int = 3) -> Tuple[pd.DataFrame, Path]:
    """Install fake providers and cache root for loader.get_ohlcv tests."""

    loader.MEM.clear()

    cache_root = tmp_path / "ohlcv_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    base_df = _make_fake_df(rows)

    def _fake_alpaca(symbol: str, start: datetime, end: datetime, timeframe: str = "1D", **kwargs):
        # Providers should receive UTC-naive boundaries
        assert start.tzinfo is None
        assert end.tzinfo is None
        return base_df.copy(deep=True)

    def _fake_yahoo(*args, **kwargs):  # pragma: no cover - fallback not used in these tests
        return pd.DataFrame()

    monkeypatch.setattr(loader, "_cache_root", lambda: cache_root)
    monkeypatch.setattr(loader.A, "load_ohlcv", _fake_alpaca)
    monkeypatch.setattr(loader.Y, "load_ohlcv", _fake_yahoo)

    return base_df.copy(deep=True), cache_root
