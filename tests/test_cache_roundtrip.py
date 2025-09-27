from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.data import loader
from tests._loader_test_utils import setup_fake_provider


def test_get_ohlcv_writes_and_reads_cache(monkeypatch, tmp_path):
    _, cache_root = setup_fake_provider(monkeypatch, tmp_path)

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 3, tzinfo=timezone.utc)

    result = loader.get_ohlcv("AAPL", start, end, timeframe="1D", force_provider="alpaca")

    expected_path = cache_root / "alpaca" / "AAPL" / "1D" / "2024-01-01__2024-01-04.parquet"
    assert expected_path.exists()

    loader.MEM.clear()
    monkeypatch.setattr(loader.A, "load_ohlcv", lambda *a, **k: pd.DataFrame())

    cached = loader.get_ohlcv("AAPL", start, end, timeframe="1D", force_provider="alpaca")

    assert cached.index.equals(result.index)
    assert cached.equals(result)
