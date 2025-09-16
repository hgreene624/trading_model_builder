# src/validation/walkforward.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, UTC
from typing import Any, Dict, List, Optional, Tuple
import importlib
import math

import pandas as pd

# Reuse your existing bits
from src.data.loader import get_ohlcv as _load_ohlcv
from src.backtest.metrics import compute_core_metrics
from src.optimization.evolutionary import evolutionary_search
from src.utils.training_logger import TrainingLogger  # already in your tree

@dataclass
class SplitResult:
    split_idx: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    best_params: Dict[str, Any]
    train_score: float
    oos_metrics: Dict[str, Any]     # computed on OOS equity/returns/trades
    per_symbol: Dict[str, Dict[str, Any]]  # symbol -> metrics

def _import_callable(dotted: str):
    mod_name, func_name = dotted.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, func_name, None)
    if fn is None:
        raise AttributeError(f"{dotted} not found")
    return fn

def _import_strategy(dotted_mod: str):
    """dotted_mod like 'src.models.atr_breakout' exposing run_strategy()."""
    mod = importlib.import_module(dotted_mod)
    run = getattr(mod, "run_strategy", None)
    if run is None:
        raise AttributeError(f"{dotted_mod} missing run_strategy()")
    return run, mod

def _years(a: datetime, b: datetime) -> float:
    return max(1e-9, (b - a).days / 365.25)

def _slice(df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    return df.loc[(df.index >= start) & (df.index <= end)]

def walkforward_validate(
    strategy_dotted: str,
    tickers: List[str],
    start: datetime,
    end: datetime,
    starting_equity: float,
    param_space_or_params: Dict[str, Any | Tuple],
    *,
    n_splits: int = 5,
    train_days: int = 252,
    test_days: int = 63,
    step_days: Optional[int] = None,  # if None, step = test_days (standard WF)
    use_ea: bool = True,
    ea_generations: int = 10,
    ea_pop: int = 24,
    ea_min_trades: int = 5,
    ea_n_jobs: int = 4,
    # fitness shaping (forwarded to EA)
    min_avg_holding_days_gate: float = 1.0,
    require_hold_days: bool = False,
    eps_mdd: float = 1e-4,
    eps_sharpe: float = 1e-4,
    alpha_cagr: float = 1.0,
    beta_calmar: float = 1.0,
    gamma_sharpe: float = 0.25,
    min_holding_days: float = 3.0,
    max_holding_days: float = 30.0,
    holding_penalty_weight: float = 1.0,
    trade_rate_min: float = 5.0,
    trade_rate_max: float = 50.0,
    trade_rate_penalty_weight: float = 0.5,
    seed: Optional[int] = None,
    data_provider: Optional[str] = None,  # respects env DATA_PROVIDER if None
    log_file: str = "training.log",
    cache_data: bool = True,
) -> Dict[str, Any]:
    """
    Rolling walk-forward with (optional) EA on each train window, then OOS validation.

    Returns:
      {
        "strategy": str,
        "period": {"start": str, "end": str},
        "config": {...},
        "splits": [ SplitResult as dict ... ],
        "aggregate": { "oos_metrics": dict, "oos_mean": dict, "oos_median": dict }
      }
    """
    logger = TrainingLogger(log_file)
    run_strategy, strat_mod = _import_strategy(strategy_dotted)

    # 1) Preload data once per symbol for [start, end], then slice per split
    data_cache: Dict[str, pd.DataFrame] = {}
    def _get(symbol: str, s: datetime, e: datetime) -> pd.DataFrame:
        if cache_data:
            if symbol not in data_cache:
                df_full = _load_ohlcv(symbol, start, end)  # full span once
                data_cache[symbol] = df_full
            return _slice(data_cache[symbol], s, e)
        return _load_ohlcv(symbol, s, e)

    # 2) Build splits
    if step_days is None:
        step_days = test_days
    total_days = (end - start).days
    if total_days < (train_days + test_days):
        raise ValueError("Date range too short for requested train_days + test_days.")
    # Create split anchors from the tail forward so we end exactly at `end`
    anchors: List[Tuple[datetime, datetime, datetime, datetime]] = []
    t_end = end
    for i in range(n_splits):
        t_start = t_end - timedelta(days=test_days)
        tr_end = t_start - timedelta(days=1)
        tr_start = tr_end - timedelta(days=train_days)
        anchors.append((tr_start, tr_end, t_start, t_end))
        t_end = t_end - timedelta(days=step_days)
        if tr_start <= start:
            break
    anchors = list(reversed(anchors))
    if not anchors:
        raise ValueError("No walk-forward splits produced (adjust dates or split config).")

    # 3) Run splits
    split_results: List[SplitResult] = []
    for si, (tr_start, tr_end, te_start, te_end) in enumerate(anchors):
        logger.log("wf_split_start", {
            "split": si, "train_start": str(tr_start), "train_end": str(tr_end),
            "test_start": str(te_start), "test_end": str(te_end),
        })

        # ---- 3a) Find params on TRAIN (EA or use provided) ----
        if use_ea:
            # param_space_or_params should be a space: name -> (low, high) or (list,…)
            space = param_space_or_params
            top = evolutionary_search(
                strategy_dotted,
                tickers,
                tr_start, tr_end,
                starting_equity,
                space,
                generations=ea_generations,
                pop_size=ea_pop,
                n_jobs=ea_n_jobs,
                min_trades=ea_min_trades,
                min_avg_holding_days_gate=min_avg_holding_days_gate,
                require_hold_days=require_hold_days,
                eps_mdd=eps_mdd,
                eps_sharpe=eps_sharpe,
                alpha_cagr=alpha_cagr,
                beta_calmar=beta_calmar,
                gamma_sharpe=gamma_sharpe,
                min_holding_days=min_holding_days,
                max_holding_days=max_holding_days,
                holding_penalty_weight=holding_penalty_weight,
                trade_rate_min=trade_rate_min,
                trade_rate_max=trade_rate_max,
                trade_rate_penalty_weight=trade_rate_penalty_weight,
                progress_cb=lambda *_a, **_k: None,
                log_file=log_file,
                seed=seed,
            )
            best_params, train_score = top[0]
        else:
            # param_space_or_params is a concrete param dict
            best_params, train_score = dict(param_space_or_params), float("nan")

        # ---- 3b) Evaluate on TEST (per symbol → aggregate) ----
        per_symbol_metrics: Dict[str, Dict[str, Any]] = {}
        all_eq: Dict[str, pd.Series] = {}
        all_daily: Dict[str, pd.Series] = {}
        all_trades: int = 0

        for sym in tickers:
            # Use cached OOS window
            _ = _get(sym, te_start, te_end)  # keep for parity/debug; run_strategy loads internally
            res = run_strategy(sym, te_start, te_end, starting_equity, best_params)
            eq = res.get("equity")
            dr = res.get("daily_returns") or (eq.pct_change().fillna(0.0) if isinstance(eq, pd.Series) else None)
            trades = res.get("trades") or []

            # Compute symbol-level OOS metrics
            m = compute_core_metrics(eq, dr, trades)
            per_symbol_metrics[sym] = m
            all_trades += int(m.get("trades", 0) or 0)
            if isinstance(eq, pd.Series) and len(eq) > 0 and float(eq.iloc[0]) != 0.0:
                all_eq[sym] = (eq / float(eq.iloc[0])).astype(float)
            if isinstance(dr, pd.Series):
                all_daily[sym] = dr

        # Aggregate OOS across symbols (equal-weight normalized equity)
        if all_eq:
            df_curves = pd.DataFrame(all_eq).sort_index().dropna(how="all")
            agg_eq = df_curves.mean(axis=1, skipna=True)
            agg_daily = agg_eq.pct_change().fillna(0.0)
            oos_metrics = compute_core_metrics(agg_eq, agg_daily, trades=[])
        else:
            oos_metrics = {}

        # Inject activity lenses for OOS aggregate
        if "trades" not in oos_metrics:
            oos_metrics["trades"] = all_trades

        split_results.append(SplitResult(
            split_idx=si,
            train_start=tr_start, train_end=tr_end,
            test_start=te_start, test_end=te_end,
            best_params=best_params,
            train_score=train_score,
            oos_metrics=oos_metrics,
            per_symbol=per_symbol_metrics,
        ))
        logger.log("wf_split_done", {
            "split": si,
            "best_params": best_params,
            "train_score": train_score,
            "oos_metrics": {k: float(v) if isinstance(v, (int, float)) else v for k, v in oos_metrics.items()},
        })

    # 4) Aggregate OOS across splits (simple mean/median of numeric metrics)
    def _num_only(d: Dict[str, Any]) -> Dict[str, float]:
        out = {}
        for k, v in d.items():
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                out[k] = float(v)
        return out

    oos_all: List[Dict[str, float]] = [_num_only(sr.oos_metrics) for sr in split_results if sr.oos_metrics]
    agg_mean: Dict[str, float] = {}
    agg_median: Dict[str, float] = {}
    if oos_all:
        keys = sorted(set().union(*[d.keys() for d in oos_all]))
        for k in keys:
            vals = [d[k] for d in oos_all if k in d]
            if vals:
                s = pd.Series(vals, dtype=float)
                agg_mean[k] = float(s.mean())
                agg_median[k] = float(s.median())

    return {
        "strategy": strategy_dotted,
        "period": {"start": str(start), "end": str(end)},
        "config": {
            "tickers": tickers,
            "n_splits": n_splits,
            "train_days": train_days,
            "test_days": test_days,
            "step_days": step_days,
            "use_ea": use_ea,
            "ea": {
                "generations": ea_generations,
                "pop": ea_pop,
                "min_trades": ea_min_trades,
                "n_jobs": ea_n_jobs,
                "weights": {"alpha_cagr": alpha_cagr, "beta_calmar": beta_calmar, "gamma_sharpe": gamma_sharpe},
                "holding_prefs": {
                    "min_holding_days": min_holding_days, "max_holding_days": max_holding_days,
                    "holding_penalty_weight": holding_penalty_weight,
                },
                "trade_rate_prefs": {
                    "trade_rate_min": trade_rate_min, "trade_rate_max": trade_rate_max,
                    "trade_rate_penalty_weight": trade_rate_penalty_weight,
                },
            },
        },
        "splits": [sr.__dict__ for sr in split_results],
        "aggregate": {
            "oos_mean": agg_mean,
            "oos_median": agg_median,
        },
    }