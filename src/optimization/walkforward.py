# src/optimization/walkforward.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, UTC
from typing import Any, Dict, List, Optional, Tuple, Callable
import importlib
import math

import pandas as pd

# Reuse existing building blocks in your tree
from src.data.loader import get_ohlcv as _load_ohlcv
from src.backtest.metrics import compute_core_metrics
from src.optimization.evolutionary import evolutionary_search
from src.utils.training_logger import TrainingLogger  # your existing logger


# Worker-safe progress type & default no-op callback
ProgressCb = Callable[[str, Dict[str, Any]], None]


def _noop_progress(event: str, payload: Dict[str, Any]) -> None:
    return


@dataclass
class SplitResult:
    idx: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    params_in: Dict[str, Any]            # params before EA (base params)
    params_used: Dict[str, Any]          # params actually used OOS (EA-best or base)
    in_sample: Dict[str, Any]            # aggregate IS metrics (equal-weight across tickers)
    out_sample: Dict[str, Any]           # aggregate OOS metrics (equal-weight across tickers)
    per_symbol: List[Dict[str, Any]]     # [{symbol, is_metrics, oos_metrics, is_trades, oos_trades}]
    oos_equity_curve: List[Tuple[str, float]]  # Normalised equity curve (date -> equity index)


def _import_callable(dotted: str):
    mod_name, _, attr = dotted.rpartition(".")
    if not mod_name or not attr:
        raise ValueError(f"Bad dotted path: {dotted}")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr, None)
    if fn is None:
        raise AttributeError(f"{dotted} not found")
    return fn


def _eq_to_daily_returns(eq: pd.Series) -> pd.Series:
    if eq is None or len(eq) == 0:
        return pd.Series(dtype=float)
    return eq.pct_change().fillna(0.0)


def _aggregate_equity(curves: Dict[str, pd.Series]) -> Tuple[pd.Series, pd.Series]:
    """Equal-weight aggregation of per-symbol equity curves → (equity_series, daily_returns)."""
    if not curves:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    df = pd.DataFrame({k: (v / float(v.iloc[0])) for k, v in curves.items() if v is not None and len(v) > 0})
    df = df.sort_index().dropna(how="all")
    if df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    eq = df.mean(axis=1, skipna=True)
    daily = eq.pct_change().fillna(0.0)
    return eq, daily


def _run_strategy_on_range(strategy_dotted: str, symbol: str, start: datetime, end: datetime,
                           starting_equity: float, params: Dict[str, Any]) -> Dict[str, Any]:
    run_strategy = _import_callable(strategy_dotted + ".run_strategy")
    # Ensure params only includes keys the strategy expects (defensive)
    # If your strategy already ignores extras, this is harmless.
    try:
        default_params = getattr(importlib.import_module(strategy_dotted), "DEFAULT_PARAMS", None)
        if isinstance(default_params, dict) and default_params:
            params = {k: params.get(k, v) for k, v in default_params.items()}
    except Exception:
        pass
    return run_strategy(symbol, start, end, starting_equity, dict(params))


def walk_forward(
    strategy_dotted: str,
    tickers: List[str],
    start: datetime,
    end: datetime,
    starting_equity: float,
    base_params: Dict[str, Any],
    *,
    splits: int = 3,
    train_days: int = 252,
    test_days: int = 63,
    step_days: Optional[int] = None,     # default = test_days if None
    use_ea: bool = False,
    ea_generations: int = 4,
    ea_pop: int = 12,
    min_trades: int = 3,
    n_jobs: int = 1,
    seed: Optional[int] = None,
    # logging
    log_file: str = "walkforward.jsonl",
    ea_kwargs: Optional[Dict[str, Any]] = None,
    progress_cb: Optional[ProgressCb] = None,
) -> Dict[str, Any]:
    """
    Rolling walk-forward validation.
    - Keeps function signature and return shape stable for your existing tests/UI.
    - Writes one JSONL log entry per split (checkpoint), including params_used and OOS metrics.
    Returns:
      {
        "config": {...},
        "splits": [SplitResult.__dict__, ...],
        "aggregate": {"oos_mean": {...}, "oos_median": {...}}
      }
    """
    if progress_cb is None:
        progress_cb = _noop_progress

    logger = TrainingLogger(log_file)

    if step_days is None:
        step_days = test_days

    # Build split boundaries
    # We advance by `step_days`. Each split uses [train_start:train_end) then OOS [test_start:test_end)
    total_days = (end - start).days
    if total_days <= train_days + test_days:
        raise ValueError("Not enough history for the requested train/test windows.")

    split_results: List[SplitResult] = []

    # We determine the first training window end so that the last test window ends at `end`
    # then slide backward computing the required number of splits.
    # Alternatively, we can slide forward from `start`; we’ll slide forward for readability.
    cur_train_end = start + timedelta(days=train_days)
    cur_test_end = cur_train_end + timedelta(days=test_days)

    # Ensure we have enough room to produce the requested number of splits
    # Compute how many forward steps fit until end
    max_forward_steps = max(0, math.floor((end - cur_test_end).days / step_days))
    max_splits_possible = 1 + max_forward_steps
    if splits > max_splits_possible:
        splits = max_splits_possible  # clamp rather than error; keeps test friendliness

    for idx in range(splits):
        train_start = cur_train_end - timedelta(days=train_days)
        train_end = cur_train_end
        test_start = train_end
        test_end = min(end, train_end + timedelta(days=test_days))

        # Make sure we don't have inverted windows
        if test_start >= test_end or train_start >= train_end:
            break

        # --- Choose params for this split ---
        params_in = dict(base_params)
        params_used = dict(base_params)

        if use_ea:
            # Lightweight EA: search only on this split’s train window
            # We’ll derive a tiny space around provided base_params if they exist, otherwise a default.
            space = {
                "breakout_n": (max(5, params_in.get("breakout_n", 14) - 3), params_in.get("breakout_n", 14) + 3),
                "exit_n": (max(3, params_in.get("exit_n", 6) - 2), params_in.get("exit_n", 6) + 2),
                "atr_n": (max(5, params_in.get("atr_n", 14) - 3), params_in.get("atr_n", 14) + 3),
                "atr_multiple": (max(0.5, params_in.get("atr_multiple", 2.0) * 0.7), params_in.get("atr_multiple", 2.0) * 1.3),
                "tp_multiple": (max(0.1, params_in.get("tp_multiple", 0.5) * 0.5), params_in.get("tp_multiple", 0.5) * 1.5),
                "holding_period_limit": (max(2, params_in.get("holding_period_limit", 5) - 2), params_in.get("holding_period_limit", 5) + 2),
            }
            # Evaluate EA on train window (equal-weight across symbols)
            resolved_ea_kwargs: Dict[str, Any] = {
                "generations": ea_generations,
                "pop_size": ea_pop,
                "min_trades": min_trades,
                "n_jobs": n_jobs,
                "seed": seed,
                "log_file": log_file,
            }
            if ea_kwargs:
                resolved_ea_kwargs.update({k: v for k, v in ea_kwargs.items() if v is not None})
            resolved_ea_kwargs.setdefault("progress_cb", progress_cb)
            top = evolutionary_search(
                strategy_dotted,
                tickers,
                train_start, train_end,
                starting_equity,
                space,
                **resolved_ea_kwargs,
            )
            if isinstance(top, list) and len(top) > 0 and isinstance(top[0], tuple):
                params_used = dict(top[0][0])  # best param set

        # --- Run strategy on train & OOS windows (per symbol) ---
        is_curves: Dict[str, pd.Series] = {}
        oos_curves: Dict[str, pd.Series] = {}
        per_symbol_rows: List[Dict[str, Any]] = []

        for sym in tickers:
            # In-sample
            is_res = _run_strategy_on_range(strategy_dotted, sym, train_start, train_end, starting_equity, params_used)
            is_eq: pd.Series = is_res.get("equity", pd.Series(dtype=float))
            # Guard: avoid truthiness of Series with "or"
            is_daily = is_res.get("daily_returns", None)
            if not isinstance(is_daily, pd.Series) or len(is_daily) == 0:
                is_daily = _eq_to_daily_returns(is_eq)
            is_trades: List[Dict[str, Any]] = is_res.get("trades", []) or []
            is_metrics = compute_core_metrics(is_eq, is_daily, is_trades)

            # OOS
            oos_res = _run_strategy_on_range(strategy_dotted, sym, test_start, test_end, starting_equity, params_used)
            oos_eq: pd.Series = oos_res.get("equity", pd.Series(dtype=float))
            # Guard: avoid truthiness of Series with "or"
            oos_daily = oos_res.get("daily_returns", None)
            if not isinstance(oos_daily, pd.Series) or len(oos_daily) == 0:
                oos_daily = _eq_to_daily_returns(oos_eq)
            oos_trades: List[Dict[str, Any]] = oos_res.get("trades", []) or []
            oos_metrics = compute_core_metrics(oos_eq, oos_daily, oos_trades)

            if is_eq is not None and len(is_eq) > 0 and pd.notna(is_eq.iloc[0]) and is_eq.iloc[0] != 0:
                is_curves[sym] = (is_eq / float(is_eq.iloc[0])).astype(float)
            if oos_eq is not None and len(oos_eq) > 0 and pd.notna(oos_eq.iloc[0]) and oos_eq.iloc[0] != 0:
                oos_curves[sym] = (oos_eq / float(oos_eq.iloc[0])).astype(float)

            per_symbol_rows.append({
                "symbol": sym,
                "is_metrics": is_metrics,
                "oos_metrics": oos_metrics,
                "is_trades": int(is_metrics.get("trades", 0) or 0),
                "oos_trades": int(oos_metrics.get("trades", 0) or 0),
            })

        # Aggregate IS/OOS curves (equal-weight)
        is_eq_agg, is_daily_agg = _aggregate_equity(is_curves)
        oos_eq_agg, oos_daily_agg = _aggregate_equity(oos_curves)
        is_metrics_agg = compute_core_metrics(is_eq_agg, is_daily_agg, trades=[])
        oos_metrics_agg = compute_core_metrics(oos_eq_agg, oos_daily_agg, trades=[])

        oos_curve_payload: List[Tuple[str, float]] = []
        if isinstance(oos_eq_agg, pd.Series) and len(oos_eq_agg) > 0:
            for ts, val in oos_eq_agg.items():
                if pd.isna(val):
                    continue
                try:
                    ts_str = ts.isoformat()
                except AttributeError:
                    ts_str = str(ts)
                oos_curve_payload.append((ts_str, float(val)))

        split = SplitResult(
            idx=len(split_results),
            train_start=train_start, train_end=train_end,
            test_start=test_start, test_end=test_end,
            params_in=dict(params_in),
            params_used=dict(params_used),
            in_sample=is_metrics_agg,
            out_sample=oos_metrics_agg,
            per_symbol=per_symbol_rows,
            oos_equity_curve=oos_curve_payload,
        )
        split_results.append(split)

        # -------- JSONL checkpoint (enhancement #3) --------
        logger.log("walkforward_split", {
            "idx": split.idx,
            "train": {"start": str(train_start), "end": str(train_end)},
            "test": {"start": str(test_start), "end": str(test_end)},
            "params_in": params_in,
            "params_used": params_used,
            "oos_metrics": oos_metrics_agg,
            "is_metrics": is_metrics_agg,
            "per_symbol": per_symbol_rows,
            "oos_equity_curve": oos_curve_payload,
        })

        # Slide the window forward
        cur_train_end = cur_train_end + timedelta(days=step_days)
        if cur_train_end >= end:
            break

    # -------- Aggregate across splits (enhancement #4) --------
    # Compute OOS mean & median over splits for stable keys present in all oos metrics
    oos_list = [sr.out_sample for sr in split_results if isinstance(sr.out_sample, dict)]
    agg_mean: Dict[str, float] = {}
    agg_median: Dict[str, float] = {}
    if oos_list:
        keys = sorted(set(k for d in oos_list for k in d.keys()))
        for k in keys:
            vals = [float(d.get(k)) for d in oos_list if isinstance(d.get(k, None), (int, float))]
            if not vals:
                continue
            s = pd.Series(vals, dtype=float)
            agg_mean[k] = float(s.mean())
            agg_median[k] = float(s.median())

    return {
        "config": {
            "strategy": strategy_dotted,
            "tickers": list(tickers),
            "period": {"start": str(start), "end": str(end)},
            "starting_equity": starting_equity,
            "wf": {
                "splits": len(split_results),
                "train_days": train_days,
                "test_days": test_days,
                "step_days": step_days,
            },
            "ea": {
                "use_ea": use_ea,
                "generations": ea_generations,
                "pop": ea_pop,
                "min_trades": min_trades,
                "n_jobs": n_jobs,
                "seed": seed,
            },
        },
        "splits": [sr.__dict__ for sr in split_results],
        "aggregate": {"oos_mean": agg_mean, "oos_median": agg_median},
    }