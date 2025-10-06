from __future__ import annotations

import pandas as pd

from src.backtest.engine import ATRParams, backtest_atr_breakout


def _mock_frame() -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=8, freq="D")
    close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0], index=dates)
    high = close + 0.5
    low = close - 0.5
    open_ = close
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=dates)


def test_persistence_requires_multiple_breakouts(monkeypatch) -> None:
    frame = _mock_frame()

    monkeypatch.setattr("src.backtest.engine.get_ohlcv", lambda *args, **kwargs: frame)
    monkeypatch.delenv("EXEC_DELAY_BARS", raising=False)
    monkeypatch.delenv("EXEC_FILL_WHERE", raising=False)

    base_params = ATRParams(breakout_n=2, exit_n=2, atr_n=2, atr_multiple=2.0, k_atr_buffer=0.0, persist_n=1)
    base_result = backtest_atr_breakout("TEST", frame.index[0], frame.index[-1], 100_000.0, base_params)
    assert base_result["trades"], "expected at least one trade in baseline run"
    base_entry = base_result["trades"][0]["entry_time"]

    persist_params = ATRParams(breakout_n=2, exit_n=2, atr_n=2, atr_multiple=2.0, k_atr_buffer=0.0, persist_n=3)
    persist_result = backtest_atr_breakout("TEST", frame.index[0], frame.index[-1], 100_000.0, persist_params)
    assert persist_result["trades"], "expected trade even with persistence"
    persist_entry = persist_result["trades"][0]["entry_time"]
    assert persist_entry > base_entry
    counters = persist_result.get("meta", {}).get("runtime_counters", {})
    assert counters.get("blocked_by_persistence", 0) >= 1


def test_atr_buffer_blocks_until_threshold(monkeypatch) -> None:
    frame = _mock_frame()

    def _loader(*_args, **_kwargs):
        return frame

    monkeypatch.setattr("src.backtest.engine.get_ohlcv", _loader)
    monkeypatch.delenv("EXEC_DELAY_BARS", raising=False)
    monkeypatch.delenv("EXEC_FILL_WHERE", raising=False)

    base_params = ATRParams(breakout_n=2, exit_n=2, atr_n=2, atr_multiple=2.0, k_atr_buffer=0.0, persist_n=1)
    base_result = backtest_atr_breakout("BUF", frame.index[0], frame.index[-1], 100_000.0, base_params)
    assert base_result["trades"], "baseline should trade"
    base_entry = base_result["trades"][0]["entry_time"]

    buffered_params = ATRParams(breakout_n=2, exit_n=2, atr_n=2, atr_multiple=2.0, k_atr_buffer=0.25, persist_n=1)
    buffered_result = backtest_atr_breakout("BUF", frame.index[0], frame.index[-1], 100_000.0, buffered_params)
    assert buffered_result["trades"], "buffered config should eventually trade"
    buffered_entry = buffered_result["trades"][0]["entry_time"]
    assert buffered_entry >= base_entry
    counters = buffered_result.get("meta", {}).get("runtime_counters", {})
    assert counters.get("blocked_by_buffer", 0) >= 1
