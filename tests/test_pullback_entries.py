import pandas as pd

from src.backtest.engine import ATRParams, backtest_atr_breakout, _build_pullback_signal, wilder_atr


def _frame_single_dip() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 100.0, 101.0, 102.0, 103.0, 104.0], index=dates)
    high = close + 1.0
    low = close - 1.0
    open_ = close.copy()
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=dates)


def _frame_two_dips() -> pd.DataFrame:
    dates = pd.date_range("2020-02-01", periods=12, freq="D")
    close = pd.Series(
        [100.0, 103.0, 106.0, 101.0, 103.0, 105.0, 100.5, 102.0, 104.5, 106.0, 107.0, 108.0],
        index=dates,
    )
    high = close + 1.0
    low = close - 1.0
    open_ = close.copy()
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=dates)


def test_pullback_entry_after_dip(monkeypatch) -> None:
    frame = _frame_single_dip()
    monkeypatch.setattr("src.backtest.engine.get_ohlcv", lambda *_, **__: frame)
    monkeypatch.delenv("EXEC_DELAY_BARS", raising=False)
    monkeypatch.delenv("EXEC_FILL_WHERE", raising=False)

    params = ATRParams(
        breakout_n=5,
        exit_n=3,
        atr_n=3,
        atr_multiple=2.0,
        entry_mode="pullback",
        trend_ma=3,
        dip_atr_from_high=0.5,
        dip_lookback_high=4,
        dip_rsi_max=80.0,
        dip_confirm="none",
        dip_cooldown_days=0,
    )

    result = backtest_atr_breakout("TEST", frame.index[0], frame.index[-1], 100_000.0, params)
    trades = result["trades"]
    assert trades, "Expected at least one pullback trade"
    atr = wilder_atr(frame["high"], frame["low"], frame["close"], n=params.atr_n)
    signal = _build_pullback_signal(frame, atr, params)
    expected_index = signal[signal].index[0]
    assert trades[0]["entry_time"] == expected_index
    counters = result.get("meta", {}).get("runtime_counters", {})
    assert counters.get("blocked_by_cooldown", 0) == 0


def test_pullback_cooldown_blocks_followup(monkeypatch) -> None:
    frame = _frame_two_dips()
    monkeypatch.setattr("src.backtest.engine.get_ohlcv", lambda *_, **__: frame)
    monkeypatch.delenv("EXEC_DELAY_BARS", raising=False)
    monkeypatch.delenv("EXEC_FILL_WHERE", raising=False)

    base_kwargs = dict(
        breakout_n=5,
        exit_n=3,
        atr_n=3,
        atr_multiple=2.0,
        entry_mode="pullback",
        trend_ma=3,
        dip_atr_from_high=0.4,
        dip_lookback_high=4,
        dip_rsi_max=90.0,
        dip_confirm="none",
        tp_multiple=0.0,
        holding_period_limit=1,
    )

    free_params = ATRParams(**base_kwargs, dip_cooldown_days=0)
    result_no_cooldown = backtest_atr_breakout("TEST", frame.index[0], frame.index[-1], 100_000.0, free_params)
    assert len(result_no_cooldown["trades"]) >= 2

    gated_params = ATRParams(**base_kwargs, dip_cooldown_days=4)
    result_cooldown = backtest_atr_breakout("TEST", frame.index[0], frame.index[-1], 100_000.0, gated_params)
    assert len(result_cooldown["trades"]) == 1
    counters = result_cooldown.get("meta", {}).get("runtime_counters", {})
    assert counters.get("blocked_by_cooldown", 0) >= 1
