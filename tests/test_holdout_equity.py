import types
from datetime import datetime

import pandas as pd

from src.utils import holdout_chart


def _make_fake_module(equity_values):
    mod = types.ModuleType("fake.strategy")

    def _run_strategy(symbol, start, end, starting_equity, params):
        idx = pd.date_range(start=datetime(2020, 1, 1), periods=len(equity_values), freq="D")
        series = pd.Series(equity_values, index=idx)
        return {
            "equity": series,
            "daily_returns": series.pct_change().fillna(0.0),
            "trades": [],
        }

    mod.run_strategy = _run_strategy
    return mod


def test_holdout_equity_returns_curve(monkeypatch):
    fake_mod = _make_fake_module([100_000.0, 101_000.0, 102_000.0])
    monkeypatch.setattr(holdout_chart, "import_module", lambda _: fake_mod)

    df = holdout_chart.holdout_equity(
        params={"foo": 1},
        start="2020-01-01",
        end="2020-01-03",
        tickers=["AAPL", "MSFT"],
        starting_equity=100_000.0,
        strategy="fake.strategy",
    )

    assert not df.empty
    assert list(df.columns) == ["date", "equity"]
    assert len(df) == 3
    assert float(df["equity"].iloc[0]) == 100_000.0
    assert float(df["equity"].iloc[-1]) > float(df["equity"].iloc[0])


def test_holdout_equity_handles_missing_strategy(monkeypatch):
    monkeypatch.setattr(holdout_chart, "import_module", lambda _: (_ for _ in ()).throw(ImportError))

    df = holdout_chart.holdout_equity(
        params={},
        start="2020-01-01",
        end="2020-01-02",
        tickers=["AAPL"],
        starting_equity=100_000.0,
        strategy="fake.other",
    )

    assert df.empty
    assert list(df.columns) == ["date", "equity"]
