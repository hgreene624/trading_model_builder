import pandas as pd

from src.backtest.engine import ATRParams, backtest_atr_breakout
from src.backtest import prob_gate


def _make_trending_frame() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=60, freq="D")
    base = pd.Series(100.0 + idx.dayofyear * 0.5, index=idx)
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
    monkeypatch.setattr(
        "src.backtest.engine.get_ohlcv",
        lambda symbol, start, end: frame.copy(),
    )

    base_params = ATRParams(breakout_n=10, exit_n=8, atr_n=5, atr_multiple=1.0)
    start = frame.index[0]
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
