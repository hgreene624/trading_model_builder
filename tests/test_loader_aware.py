from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.data import loader
from tests._loader_test_utils import setup_fake_provider


def test_get_ohlcv_accepts_aware_bounds(monkeypatch, tmp_path):
    base_df, _ = setup_fake_provider(monkeypatch, tmp_path)

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 3, tzinfo=timezone.utc)

    result = loader.get_ohlcv("AAPL", start, end, timeframe="1D", force_provider="alpaca")

    expected_index = pd.DatetimeIndex(base_df["timestamp"])
    assert result.index.equals(expected_index)
    assert result.index.tz == expected_index.tz
