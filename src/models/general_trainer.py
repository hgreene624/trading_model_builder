# src/models/general_trainer.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, List, Tuple, Callable, Optional, TYPE_CHECKING
import inspect
import numpy as np
import pandas as pd

# Safe at import time; if your cache module ever causes cycles, move this import
# into _safe_backtest().
from src.data.cache import get_ohlcv_cached

if TYPE_CHECKING:
    # Only for type hints; real import is done lazily inside train_general_model
    from src.models.strategy_adapter import StrategyAdapter

__all__ = ["TrainConfig", "train_general_model"]


# ================================ Config =====================================

@dataclass
class TrainConfig:
    """
    General model training knobs.

    K:                 samples (random param draws) per ticker
    n_folds:           CV folds inside the priors window
    min_trades_fold:   minimum trades required per fold to count as usable
    enforce_trades:    if True, penalize (or discard) params with too few trades
    seed:              RNG seed
    starting_equity:   passed through to engine for sizing
    """
    K: int = 256
    n_folds: int = 3
    min_trades_fold: int = 2
    enforce_trades: bool = False
    seed: int = 2025
    starting_equity: float = 10_000.0


# ============================== Metric Utils =================================

def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (list, dict)):
            return default
        return float(x)
    except Exception:
        return default


def _to_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        if isinstance(x, list):
            # Some engines return a list of trade records
            return len(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        return int(float(x))
    except Exception:
        return default


def _normalize_metrics(m: Dict[str, Any]) -> Dict[str, float]:
    """
    Normalize various possible engine return shapes into a stable metric dict.

    Expected flexible inputs:
      - {"metrics": {...}} style (adapter can pass through already-normalized)
      - flat keys: sharpe, total_return, max_drawdown, cagr, trades | trade_count
      - "trades" could be a number or a list of trades
    """
    if not isinstance(m, dict):
        m = {}

    # Some adapters wrap in {'metrics': {...}}
    if "metrics" in m and isinstance(m["metrics"], dict):
        m = m["metrics"]

    trades_raw = m.get("trades", m.get("num_trades", m.get("trade_count", 0)))

    return {
        "sharpe": _to_float(m.get("sharpe", m.get("sharpe_ratio", 0.0)), 0.0),
        "total_return": _to_float(m.get("total_return", m.get("tr", 0.0)), 0.0),
        "max_drawdown": _to_float(m.get("max_drawdown", m.get("dd", 0.0)), 0.0),
        "cagr": _to_float(m.get("cagr", 0.0), 0.0),
        "trades": _to_int(trades_raw, 0),
    }


# ============================== Backtest Runner ===============================

def _safe_backtest(
    adapter: "StrategyAdapter",
    symbol: str,
    start: str,
    end: str,
    params: Dict[str, Any],
    starting_equity: float,
) -> Dict[str, Any]:
    """
    Call adapter.backtest(...) with filtered kwargs.

    - Prechecks that data exists in [start, end]
    - Only passes kwargs accepted by the target callable
    - Returns a dict (possibly with '_error' on failure)
    """
    # 1) Quick data existence check
    try:
        df = get_ohlcv_cached(symbol, start, end)
        if df is None or len(df) < 50:
            return {"_error": f"no_data({symbol} {start}->{end})", "sharpe": 0.0, "trades": 0}
    except Exception as e:
        return {"_error": f"data_error({symbol}): {e}", "sharpe": 0.0, "trades": 0}

    # 2) Filter kwargs to what the backtest target accepts
    try:
        target = adapter.get_callable()
        sig = inspect.signature(target)
        allowed = set(sig.parameters.keys())
    except Exception:
        target = None
        allowed = set()

    kwargs = dict(symbol=symbol, start=start, end=end, starting_equity=starting_equity)
    for k, v in params.items():
        if not allowed or k in allowed:
            kwargs[k] = v

    # 3) Invoke
    try:
        res = adapter.backtest(**kwargs)
        if not isinstance(res, dict):
            return {"_error": "bad_return_shape", "sharpe": 0.0, "trades": 0}
        return res
    except Exception as e:
        return {"_error": f"bt_error({symbol}): {e}", "sharpe": 0.0, "trades": 0}


# ============================== CV / Folds ====================================

def _make_folds(priors_start: date, priors_end: date, n_folds: int) -> List[Tuple[str, str]]:
    """
    Split [priors_start, priors_end] into n_folds contiguous sub-windows.
    Guarantees at least 30 days per fold when possible.
    """
    if n_folds <= 1:
        return [(priors_start.isoformat(), priors_end.isoformat())]

    total_days = max(1, (priors_end - priors_start).days)
    approx = max(30, total_days // n_folds)
    folds: List[Tuple[str, str]] = []

    s = priors_start
    while s < priors_end and len(folds) < n_folds:
        e = min(priors_end, s + timedelta(days=approx))
        folds.append((s.isoformat(), e.isoformat()))
        s = e + timedelta(days=1)

    if not folds:
        folds = [(priors_start.isoformat(), priors_end.isoformat())]
    return folds


def _cv_eval(
    adapter: "StrategyAdapter",
    symbol: str,
    params: Dict[str, Any],
    folds: List[Tuple[str, str]],
    starting_equity: float,
    min_trades_fold: int,
    enforce_trades: bool,
) -> Dict[str, Any]:
    fold_sharpes: List[float] = []
    fold_trades: List[int] = []
    errors: List[str] = []

    for fs, fe in folds:
        raw = _safe_backtest(adapter, symbol, fs, fe, params, starting_equity)
        if "_error" in raw:
            errors.append(raw["_error"])
        m = _normalize_metrics(raw)
        fold_sharpes.append(m["sharpe"])
        fold_trades.append(m["trades"])

    if enforce_trades and any(t < int(min_trades_fold) for t in fold_trades):
        # Hard penalty when insufficient activity in any fold
        return {
            "cv_sharpe": -1e9,
            "cv_trades": int(np.sum(fold_trades)),
            "fold_sharpes": fold_sharpes,
            "fold_trades": fold_trades,
            "errors": errors,
        }

    return {
        "cv_sharpe": float(np.mean(fold_sharpes)) if fold_sharpes else -1e9,
        "cv_trades": int(np.sum(fold_trades)),
        "fold_sharpes": fold_sharpes,
        "fold_trades": fold_trades,
        "errors": errors,
    }


# ============================== Sampler =======================================

def _sample_params(
    priors: Dict[str, Dict[str, Any]],
    rng: np.random.Generator
) -> Dict[str, Any]:
    """
    Draw one parameter set from 'priors' where each param specifies:
      {"low": ..., "high": ..., "seed": {"dist": "uniform"|"gamma"|"bernoulli", ...}}
    """
    out: Dict[str, Any] = {}

    for k, spec in priors.items():
        lo = spec.get("low")
        hi = spec.get("high")
        seed = spec.get("seed", {})
        dist = str(seed.get("dist", "uniform")).lower()

        # Boolean / on-off param
        if k == "use_trend_filter":
            p = float(seed.get("p", 0.5))
            out[k] = bool(rng.random() < p)
            continue

        # Numeric ranges
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            if dist == "gamma":
                kshape = float(seed.get("k", 2.0))
                theta = float(seed.get("theta", 5.0))
                val = float(rng.gamma(kshape, theta))
                span = max(1e-9, (float(hi) - float(lo)))
                val = float(lo) + (val % span)  # wrap into [lo, hi)
            else:
                val = float(lo) + float(rng.random()) * (float(hi) - float(lo))

            # Keep integer-like params integral (except explicit float params)
            if all(isinstance(x, (int, np.integer,)) for x in (lo, hi)) and k not in {
                "atr_multiple", "risk_per_trade", "tp_multiple", "cost_bps"
            }:
                val = int(round(val))
            out[k] = val
        else:
            # If a prior is malformed, fall back to lo
            out[k] = lo

    return out


# =========================== Public Trainer API ===============================

def train_general_model(
    tickers: List[str],
    priors: Dict[str, Dict[str, Any]],
    priors_start: date,
    priors_end: date,
    strategy_dotted: str,
    cfg: TrainConfig,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, Any]:
    """
    Random-search baseline trainer.
    For each ticker:
      - sample K parameter sets from 'priors'
      - evaluate via CV across folds inside [priors_start, priors_end]
      - keep the best
    Returns:
      {"leaderboard": DataFrame, "debug": {...}}
    """
    # Lazy import to avoid heavy/circular imports at module import time
    try:
        from src.models.strategy_adapter import StrategyAdapter
    except Exception as e:
        raise ImportError(f"general_trainer: failed to import StrategyAdapter: {e}") from e

    # Build adapter
    adapter: "StrategyAdapter" = StrategyAdapter.from_name(strategy_dotted)

    # Folds
    folds = _make_folds(priors_start, priors_end, int(cfg.n_folds))

    rng = np.random.default_rng(int(cfg.seed))
    rows: List[Dict[str, Any]] = []
    debug: Dict[str, Any] = {"tickers": [], "errors": []}

    total = max(1, len(tickers) * int(cfg.K))
    done = 0

    for sym in tickers:
        debug["tickers"].append(sym)
        best: Optional[Dict[str, Any]] = None

        for _ in range(int(cfg.K)):
            p = _sample_params(priors, rng)
            cv = _cv_eval(
                adapter=adapter,
                symbol=sym,
                params=p,
                folds=folds,
                starting_equity=float(cfg.starting_equity),
                min_trades_fold=int(cfg.min_trades_fold),
                enforce_trades=bool(cfg.enforce_trades),
            )
            score = float(cv.get("cv_sharpe", -1e9))

            if (best is None) or (score > best["score"]):
                best = {"score": score, "params": p, "cv": cv}

            done += 1
            if progress_cb:
                progress_cb(done / total, f"{sym}: {done}/{total}")

        # Record best result for this ticker
        if best is None:
            debug["errors"].append(f"{sym}: no successful runs")
            rows.append({"ticker": sym, "cv_sharpe": np.nan, "cv_trades": 0})
        else:
            if best["cv"].get("errors"):
                debug["errors"].append({sym: best["cv"]["errors"]})
            rows.append(
                {
                    "ticker": sym,
                    "cv_sharpe": float(best["score"]),
                    "cv_trades": int(best["cv"].get("cv_trades", 0)),
                    **{f"p_{k}": v for k, v in best["params"].items()},
                }
            )

    # Build leaderboard (always well-formed)
    lb = pd.DataFrame(rows)
    if lb.empty:
        lb = pd.DataFrame(columns=["ticker", "cv_sharpe", "cv_trades"])
    else:
        if "cv_sharpe" not in lb.columns:
            lb["cv_sharpe"] = np.nan
        if "cv_trades" not in lb.columns:
            lb["cv_trades"] = 0
        lb = lb.sort_values(["cv_sharpe", "cv_trades"], ascending=[False, False]).reset_index(drop=True)

    if progress_cb:
        progress_cb(1.0, "done")

    return {"leaderboard": lb, "debug": debug}