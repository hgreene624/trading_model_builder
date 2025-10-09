import numpy as np
import pandas as pd

from src.backtest.engine import ATRParams, backtest_atr_breakout


def test_backtest_requests_warmup_history(monkeypatch):
    idx = pd.bdate_range("2023-01-02", periods=400, tz="UTC")
    base_price = pd.Series(100.0 + np.linspace(0, 40, len(idx)), index=idx)
    frame = pd.DataFrame(
        {
            "open": base_price,
            "high": base_price + 1.0,
            "low": base_price - 1.0,
            "close": base_price + 0.25,
            "volume": 1_000,
        }
    )

    captured: dict[str, pd.Timestamp] = {}

    def fake_loader(symbol, start, end):
        captured["start"] = pd.Timestamp(start)
        captured["end"] = pd.Timestamp(end)
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        else:
            start_ts = start_ts.tz_convert("UTC")
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")
        else:
            end_ts = end_ts.tz_convert("UTC")
        return frame.loc[start_ts:end_ts]

    monkeypatch.setattr("src.backtest.engine.get_ohlcv", fake_loader)

    start = pd.Timestamp("2023-11-01", tz="UTC")
    end = pd.Timestamp("2023-12-29", tz="UTC")
    params = ATRParams(breakout_n=30, exit_n=10, atr_n=20, atr_multiple=1.0)

    result = backtest_atr_breakout("TEST", start, end, 100_000.0, params)

    requested_start = captured["start"]
    if requested_start.tzinfo is None:
        requested_start = requested_start.tz_localize("UTC")
    else:
        requested_start = requested_start.tz_convert("UTC")
    assert requested_start < start

    business_days = len(
        pd.bdate_range(
            requested_start.tz_convert(None),
            start.tz_convert(None),
            inclusive="left",
        )
    )
    assert business_days >= params.breakout_n

    equity_index = result["equity"].index
    assert len(equity_index) > 0
    assert equity_index[0] == start

    meta = result["meta"]
    meta_start = pd.Timestamp(meta.get("data_start"))
    if meta_start.tzinfo is None:
        meta_start = meta_start.tz_localize("UTC")
    else:
        meta_start = meta_start.tz_convert("UTC")
    assert meta_start <= requested_start
    assert meta.get("warmup_bars") >= params.breakout_n
