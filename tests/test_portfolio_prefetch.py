from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.data import portfolio_prefetch as pf
from tests._loader_test_utils import setup_fake_provider


def test_prefetch_and_cached_window_roundtrip(monkeypatch, tmp_path):
    base_df, cache_root = setup_fake_provider(monkeypatch, tmp_path, rows=5)

    monkeypatch.setattr(pf, "get_ohlcv_root", lambda: cache_root)

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 5, tzinfo=timezone.utc)

    ranges = pf.prefetch_and_ranges(["AAPL"], start, end, timeframe="1D")

    assert "AAPL" in ranges
    assert ranges["AAPL"]["start"] is not None
    assert ranges["AAPL"]["end"] is not None

    cached = pf.load_cached_window("AAPL", start, end, timeframe="1D")

    assert not cached.empty
    assert list(cached.columns) == ["open", "high", "low", "close", "volume"]
    assert cached.index.tz == timezone.utc

    # ensure cached content matches fake provider after normalization
    expected = pd.DataFrame(
        {
            "open": base_df["open"].tolist(),
            "high": base_df["high"].tolist(),
            "low": base_df["low"].tolist(),
            "close": base_df["close"].tolist(),
            "volume": base_df["volume"].tolist(),
        },
        index=base_df["timestamp"].dt.tz_convert("UTC"),
    )

    pd.testing.assert_index_equal(cached.index, expected.index, check_names=False)
    pd.testing.assert_frame_equal(cached, expected, check_names=False)
