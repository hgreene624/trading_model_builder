from __future__ import annotations

import math

import pandas as pd

from src.backtest.engine import CostModel, _apply_costs, _estimate_half_spread_bps


def test_estimate_half_spread_basic() -> None:
    bar = {"high": 102.0, "low": 98.0, "close": 100.0}
    cm = CostModel.from_inputs(enabled=True, use_range_impact=True, cap_range_impact_bps=500.0)
    bps = _estimate_half_spread_bps(bar, cm)
    assert math.isclose(bps, 200.0, rel_tol=1e-6)


def test_estimate_half_spread_missing_fields() -> None:
    bar = {"high": None, "low": None, "close": 0.0}
    cm = CostModel.from_inputs(enabled=True, use_range_impact=True, min_half_spread_bps=0.5)
    assert math.isclose(_estimate_half_spread_bps(bar, cm), cm.min_half_spread_bps, rel_tol=1e-6)


def test_apply_costs_buy_flow() -> None:
    cm = CostModel.from_inputs(
        enabled=True,
        fixed_bps=0.5,
        atr_k=0.25,
        min_half_spread_bps=1.0,
        use_range_impact=True,
        cap_range_impact_bps=500.0,
    )
    bar = pd.Series({"high": 101.0, "low": 99.0, "close": 100.0})
    price_after, slip_bps, fees_bps = _apply_costs(100.0, 10.0, bar, 0.01, cm)
    assert math.isclose(price_after, 101.0, rel_tol=1e-6)
    assert math.isclose(slip_bps, 100.0, rel_tol=1e-6)
    assert math.isclose(fees_bps, 0.5, rel_tol=1e-6)


def test_apply_costs_sell_flow() -> None:
    cm = CostModel.from_inputs(
        enabled=True,
        fixed_bps=0.0,
        atr_k=0.25,
        min_half_spread_bps=1.0,
        use_range_impact=True,
        cap_range_impact_bps=500.0,
    )
    bar = pd.Series({"high": 101.0, "low": 99.0, "close": 100.0})
    price_after, slip_bps, fees_bps = _apply_costs(100.0, -5.0, bar, 0.005, cm)
    assert math.isclose(price_after, 99.0, rel_tol=1e-6)
    assert math.isclose(slip_bps, 100.0, rel_tol=1e-6)
    assert math.isclose(fees_bps, 0.0, rel_tol=1e-6)


def test_apply_costs_disabled_returns_original() -> None:
    cm = CostModel.from_inputs(enabled=False)
    price_after, slip_bps, fees_bps = _apply_costs(50.0, 10.0, {}, None, cm)
    assert price_after == 50.0
    assert slip_bps == 0.0
    assert fees_bps == 0.0
