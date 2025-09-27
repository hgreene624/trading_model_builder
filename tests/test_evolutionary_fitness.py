"""Unit tests for evolutionary fitness calculation."""
from __future__ import annotations

from src.optimization.evolutionary import _clamped_fitness


_DEF_KWARGS = dict(
    min_trades=1,
    min_avg_holding_days_gate=0.0,
    require_hold_days=False,
    eps_mdd=1e-4,
    eps_sharpe=1e-4,
    alpha_cagr=1.0,
    beta_calmar=0.2,
    gamma_sharpe=0.0,
    delta_total_return=1.0,
    min_holding_days=0.0,
    max_holding_days=365.0,
    holding_penalty_weight=0.0,
    trade_rate_min=0.0,
    trade_rate_max=100.0,
    trade_rate_penalty_weight=0.0,
    num_symbols=1,
    years=1.0,
    calmar_cap=3.0,
)


def test_higher_total_return_increases_fitness() -> None:
    base_metrics = {
        "trades": 10,
        "avg_holding_days": 10.0,
        "total_return": 0.10,
        "cagr": 0.10,
        "calmar": 1.0,
        "max_drawdown": -0.10,
        "sharpe": 0.0,
    }
    better_return_metrics = {**base_metrics, "total_return": 0.25}

    base_score = _clamped_fitness(base_metrics, **_DEF_KWARGS)
    better_score = _clamped_fitness(better_return_metrics, **_DEF_KWARGS)

    assert better_score > base_score


def test_calmar_cap_prevents_dominance() -> None:
    balanced_metrics = {
        "trades": 10,
        "avg_holding_days": 10.0,
        "total_return": 0.30,
        "cagr": 0.10,
        "calmar": 2.0,
        "max_drawdown": -0.10,
        "sharpe": 0.0,
    }
    calmar_heavy_metrics = {
        "trades": 10,
        "avg_holding_days": 10.0,
        "total_return": 0.05,
        "cagr": 0.10,
        "calmar": 50.0,
        "max_drawdown": -0.10,
        "sharpe": 0.0,
    }

    balanced_score = _clamped_fitness(balanced_metrics, **_DEF_KWARGS)
    calmar_heavy_score = _clamped_fitness(calmar_heavy_metrics, **_DEF_KWARGS)

    assert balanced_score > calmar_heavy_score
