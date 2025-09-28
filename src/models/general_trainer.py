# src/models/general_trainer.py
"""
General trainer orchestrates:
- iterate tickers
- call strategy wrapper
- compute core metrics
- aggregate equity (equal-weight normalized)
- IMPORTANT: populate aggregate *activity* fields (trades, avg_holding_days, win_rate)
  so optimizers (EA) can apply min-trades gates and other selection logic.

This module has NO Streamlit code.
"""

from __future__ import annotations
from importlib import import_module
from typing import Any, Dict, List, Tuple

from datetime import date, datetime

import pandas as pd

from src.backtest.metrics import compute_core_metrics


def import_callable(dotted: str):
    """
    Import run_strategy from a dotted path, e.g., 'src.models.atr_breakout'.
    Strategy module MUST expose:
      run_strategy(symbol, start, end, starting_equity, params) -> BacktestResult
    """
    mod = import_module(dotted)
    if not hasattr(mod, "run_strategy"):
        raise ImportError(f"{dotted} missing run_strategy(...)")
    return getattr(mod, "run_strategy")


def _weighted_average(values: List[float], weights: List[float]) -> float:
    if not values or not weights or sum(weights) == 0:
        return 0.0
    s = 0.0
    w = 0.0
    for v, wt in zip(values, weights):
        if pd.notna(v) and pd.notna(wt):
            s += float(v) * float(wt)
            w += float(wt)
    return float(s / w) if w else 0.0


def _coerce_datetime(value: Any, label: str) -> datetime:
    """Best-effort conversion of common date inputs to ``datetime``."""

    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()

    if value is None:
        raise ValueError(f"{label} must be datetime")

    try:
        coerced = pd.to_datetime(value, errors="raise")
    except Exception as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"{label} must be datetime") from exc

    if isinstance(coerced, pd.Series):
        if coerced.empty:
            raise ValueError(f"{label} must be datetime")
        coerced = coerced.iloc[0]

    if pd.isna(coerced):
        raise ValueError(f"{label} must be datetime")

    if isinstance(coerced, pd.Timestamp):
        return coerced.to_pydatetime()
    if isinstance(coerced, datetime):
        return coerced

    # Fallback for numpy datetime64 or similar scalars
    try:
        return pd.Timestamp(coerced).to_pydatetime()
    except Exception as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"{label} must be datetime") from exc


def train_general_model(
    strategy_dotted: str,
    tickers: List[str],
    start,
    end,
    starting_equity: float,
    base_params: Dict[str, Any],
) -> Dict:
    """
    Returns:
      {
        "strategy": strategy_dotted,
        "params": base_params,
        "period": {"start": str, "end": str},
        "results": [
            {"symbol": str, "metrics": dict, "meta": dict, "trade_count": int},
            ...
        ],
        "aggregate": {
            "metrics": dict,  # includes curve metrics + populated activity fields
        },
      }
    """
    run = import_callable(strategy_dotted)

    start_dt = _coerce_datetime(start, "start")
    end_dt = _coerce_datetime(end, "end")

    per_symbol: List[Dict[str, Any]] = []
    # Collect normalized curves in a DataFrame for clean aggregation
    norm_curves: Dict[str, pd.Series] = {}

    # Also collect activity metrics for portfolio-level rollups
    symbol_trade_counts: List[int] = []
    symbol_avg_holds: List[float] = []
    symbol_win_rates: List[float] = []
    symbol_expectancies: List[float] = []

    # --- per-symbol loop ---
    for sym in tickers:
        result = run(sym, start_dt, end_dt, starting_equity, base_params)

        # Compute symbol-level metrics
        metrics = compute_core_metrics(result["equity"], result["daily_returns"], result["trades"])

        # Keep per-symbol record
        per_symbol.append({
            "symbol": sym,
            "metrics": metrics,
            "meta": result.get("meta", {}),
            "trade_count": metrics.get("trades", 0),
        })

        # Aggregate equity: equal-weight normalized curves
        eq = result.get("equity")
        if eq is not None and len(eq) > 0 and pd.notna(eq.iloc[0]) and eq.iloc[0] != 0:
            norm = (eq / float(eq.iloc[0])).astype(float)
            norm_curves[sym] = norm

        # Save activity for rollups
        symbol_trade_counts.append(int(metrics.get("trades", 0) or 0))
        symbol_avg_holds.append(float(metrics.get("avg_holding_days", 0.0) or 0.0))
        symbol_win_rates.append(float(metrics.get("win_rate", 0.0) or 0.0))
        symbol_expectancies.append(float(metrics.get("expectancy", 0.0) or 0.0))

    # --- aggregate equity & metrics ---
    agg_metrics: Dict[str, Any] = {}

    agg_curve_payload = None

    if norm_curves:
        # Align on common index; average across columns (equal weight)
        df_curves = pd.DataFrame(norm_curves).sort_index()
        # Option: require at least one non-NaN across symbols then forward-fill gaps per symbol if desired
        df_curves = df_curves.dropna(how="all")
        # Equal-weight average (ignore NaNs per row)
        aggregate_equity = df_curves.mean(axis=1, skipna=True)
        daily = aggregate_equity.pct_change().fillna(0.0)

        # Curve-quality metrics from aggregate equity
        agg_metrics = compute_core_metrics(aggregate_equity, daily, trades=[])  # trades list intentionally empty here

        # Provide explicit equity curve payloads for downstream visualizations (e.g., EA Inspector)
        agg_curve = aggregate_equity.dropna().astype(float)
        if not agg_curve.empty:
            try:
                idx = pd.to_datetime(agg_curve.index)
            except Exception:
                idx = agg_curve.index
            start_val = float(starting_equity) if starting_equity else 1.0
            curve_values = (agg_curve * start_val).tolist()
            curve_dates = [getattr(d, "isoformat", lambda: str(d))() for d in idx]
            agg_curve_payload = {"date": curve_dates, "equity": curve_values}
    else:
        # No curves → empty metrics
        aggregate_equity = pd.Series(dtype=float)
        agg_metrics = {}

    # --- populate aggregate *activity* fields so optimizers can gate on portfolio activity ---
    total_trades = int(sum(symbol_trade_counts))
    # trades-weighted averages (avoid divide-by-zero)
    w_avg_hold = _weighted_average(symbol_avg_holds, symbol_trade_counts)
    w_win_rate = _weighted_average(symbol_win_rates, symbol_trade_counts)
    w_expectancy = _weighted_average(symbol_expectancies, symbol_trade_counts)

    # Ensure the keys exist even if aggregate_equity was empty
    if not agg_metrics:
        agg_metrics = {"total_return": 0.0, "cagr": 0.0, "max_drawdown": 0.0, "calmar": 0.0, "sharpe": 0.0}

    # Inject portfolio activity lenses
    agg_metrics["trades"] = total_trades
    agg_metrics["avg_holding_days"] = w_avg_hold
    agg_metrics["win_rate"] = w_win_rate
    agg_metrics["expectancy"] = w_expectancy  # portfolio-level expectancy proxy

    aggregate_payload: Dict[str, Any] = {"metrics": agg_metrics}
    if agg_curve_payload:
        curve_pairs = list(zip(agg_curve_payload["date"], agg_curve_payload["equity"]))
        aggregate_payload["equity_curve"] = curve_pairs
        # Back-compat aliases for callers that expect dict/list payloads
        aggregate_payload["curve"] = dict(agg_curve_payload)
        aggregate_payload["equity"] = dict(agg_curve_payload)

    portfolio_payload = dict(agg_curve_payload) if agg_curve_payload else None

    result: Dict[str, Any] = {
        "strategy": strategy_dotted,
        "params": dict(base_params),
        "period": {"start": start_dt.isoformat(), "end": end_dt.isoformat()},
        "results": per_symbol,
        "aggregate": aggregate_payload,
    }
    if portfolio_payload:
        result["portfolio"] = portfolio_payload
    return result
