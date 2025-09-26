from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.data import loader
from tests._loader_test_utils import setup_fake_provider


def test_get_ohlcv_accepts_naive_bounds(monkeypatch, tmp_path):
    base_df, _ = setup_fake_provider(monkeypatch, tmp_path)

    start = datetime(2024, 1, 1)
    end = datetime(2024, 1, 3)

    result = loader.get_ohlcv("AAPL", start, end, timeframe="1D", force_provider="alpaca")

    assert list(result.columns) == ["open", "high", "low", "close", "volume"]
    assert len(result) == len(base_df)

    expected_index = pd.DatetimeIndex(base_df["timestamp"])
    assert result.index.equals(expected_index)
    assert result.index.tz == expected_index.tz
