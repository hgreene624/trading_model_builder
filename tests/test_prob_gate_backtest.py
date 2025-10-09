import numpy as np
import pandas as pd

from src.backtest.engine import ATRParams, backtest_atr_breakout
from src.backtest import prob_gate


def _make_trending_frame() -> pd.DataFrame:
    idx = pd.date_range("2023-09-01", periods=200, freq="B", tz="UTC")
    base = pd.Series(100.0 + np.arange(len(idx)) * 0.5, index=idx)
    df = pd.DataFrame(
        {
            "open": base,
            "high": base + 1.5,
            "low": base - 1.5,
            "close": base + 0.5,
            "volume": 1_000,
        }
    )
    return df


def test_prob_gate_blocks_entries(monkeypatch):
    frame = _make_trending_frame()

    def _fake_loader(symbol, start, end):
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

    monkeypatch.setattr("src.backtest.engine.get_ohlcv", _fake_loader)

    base_params = ATRParams(breakout_n=10, exit_n=8, atr_n=5, atr_multiple=1.0)
    start = frame.index[-60]
    end = frame.index[-1]

    base_result = backtest_atr_breakout("TEST", start, end, 100_000.0, base_params)
    base_trades = len(base_result["trades"])
    assert base_trades > 0

    monkeypatch.setattr(
        prob_gate,
        "score_probabilities",
        lambda df, params, model_id: pd.Series(0.05, index=df.index),
    )

    gated_params = ATRParams(breakout_n=10, exit_n=8, atr_n=5, atr_multiple=1.0)
    gated_params.prob_gate_enabled = True
    gated_params.prob_gate_threshold = 0.5
    gated_params.prob_model_id = "mock_model"

    gated_result = backtest_atr_breakout("TEST", start, end, 100_000.0, gated_params)
    assert len(gated_result["trades"]) == 0

    gated_params.prob_gate_enabled = False
    restored_result = backtest_atr_breakout("TEST", start, end, 100_000.0, gated_params)
    assert len(restored_result["trades"]) == base_trades
