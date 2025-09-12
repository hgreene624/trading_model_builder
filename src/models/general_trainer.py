# src/models/general_trainer.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable, Dict, List, Tuple, Any
import inspect
import math
import numpy as np
import pandas as pd
from src.data.cache import get_ohlcv_cached
import inspect
# Adapter that resolves the strategy module
from src.models.strategy_adapter import StrategyAdapter


# ------------------------------- Config ---------------------------------------
@dataclass
class TrainConfig:
    K: int = 256                   # samples per ticker
    n_folds: int = 3               # CV folds on the priors window
    min_trades_fold: int = 2       # min trades required per fold
    enforce_trades: bool = False   # if True, discard params that fail min trades
    seed: int = 2025               # RNG seed
    starting_equity: float = 10_000.0


# --------------------------- Metric Utilities ---------------------------------
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
        if isinstance(x, list):
            return len(x)
        if x is None:
            return default
        # Some libs return numpy scalar types
        if isinstance(x, (np.integer,)):
            return int(x)
        return int(float(x))
    except Exception:
        return default


def _normalize_metrics(m: Dict[str, Any]) -> Dict[str, float]:
    """
    Make sure we always have numeric sharpe/return/dd/cagr/trades, even if the
    engine returns a different shape (e.g., 'trades' as list).
    """
    if not isinstance(m, dict):
        m = {}

    # Try several common keys
    trades_raw = (
        m.get("trades")
        if "trades" in m
        else m.get("num_trades", m.get("trade_count", 0))
    )

    out = {
        "sharpe": _to_float(m.get("sharpe", m.get("sharpe_ratio", 0.0)), 0.0),
        "total_return": _to_float(m.get("total_return", m.get("tr", 0.0)), 0.0),
        "max_drawdown": _to_float(m.get("max_drawdown", m.get("dd", 0.0)), 0.0),
        "cagr": _to_float(m.get("cagr", 0.0), 0.0),
        "trades": _to_int(trades_raw, 0),
    }
    return out


# ----------------------------- Backtest runner --------------------------------
def _safe_backtest(
    adapter: StrategyAdapter,
    symbol: str,
    start: str,
    end: str,
    params: Dict[str, Any],
    starting_equity: float,
) -> Dict[str, Any]:
    """
    Call adapter.backtest(...) with filtered kwargs.
    - Prechecks that we actually have bars in [start, end]
    - Only passes kwargs accepted by the target
    - Normalizes tuple / {'metrics': {...}} returns
    - Returns {'_error': ...} on failure (handled later)
    """
    # 1) Pre-flight data check
    try:
        df = get_ohlcv_cached(symbol, start, end)  # your cache wrapper
        if df is None or len(df) < 50:  # too few bars to do anything meaningful
            return {"_error": f"no_data({symbol} {start}->{end})", "sharpe": 0.0, "trades": 0}
    except Exception as e:
        return {"_error": f"data_error({symbol}): {e}", "sharpe": 0.0, "trades": 0}

    # 2) Build kwargs limited to target signature
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

    # 3) Call and normalize
    try:
        res = adapter.backtest(**kwargs)
        # If adapter returned {'metrics': {...}}, it already normalizes to metrics dict
        if not isinstance(res, dict):
            return {"_error": "bad_return_shape", "sharpe": 0.0, "trades": 0}
        return res
    except Exception as e:
        return {"_error": f"bt_error({symbol}): {e}", "sharpe": 0.0, "trades": 0}

# ------------------------------ CV handling -----------------------------------
def _make_folds(priors_start: date, priors_end: date, n_folds: int) -> List[Tuple[str, str]]:
    """Split the priors window into n_folds contiguous sub-windows."""
    if n_folds <= 1:
        return [(priors_start.isoformat(), priors_end.isoformat())]
    days = (priors_end - priors_start).days
    step = max(30, days // n_folds)  # at least ~1 month per fold
    folds: List[Tuple[str, str]] = []
    s = priors_start
    for _ in range(n_folds):
        e = min(priors_end, s + timedelta(days=step))
        folds.append((s.isoformat(), e.isoformat()))
        s = e + timedelta(days=1)
        if s >= priors_end:
            break
    if not folds:
        folds = [(priors_start.isoformat(), priors_end.isoformat())]
    return folds


def _cv_eval(
    adapter: StrategyAdapter,
    symbol: str,
    params: Dict[str, Any],
    folds: List[Tuple[str, str]],
    starting_equity: float,
    min_trades_fold: int,
    enforce_trades: bool,
) -> Dict[str, Any]:
    shs, trs, errs = [], [], []
    for fs, fe in folds:
        raw = _safe_backtest(adapter, symbol, fs, fe, params, starting_equity)
        if "_error" in raw:
            errs.append(raw["_error"])
        m = _normalize_metrics(raw)
        shs.append(m["sharpe"])
        trs.append(m["trades"])

    if enforce_trades and any(t < int(min_trades_fold) for t in trs):
        return {
            "cv_sharpe": -1e9,
            "cv_trades": int(np.sum(trs)),
            "fold_sharpes": shs,
            "fold_trades": trs,
            "errors": errs,
        }

    return {
        "cv_sharpe": float(np.mean(shs)) if shs else -1e9,
        "cv_trades": int(np.sum(trs)),
        "fold_sharpes": shs,
        "fold_trades": trs,
        "errors": errs,
    }
# ----------------------------- Sampler (uniform/gamma/bernoulli) --------------
def _sample_params(priors: Dict[str, Dict[str, Any]], rng: np.random.Generator) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, spec in priors.items():
        lo = spec.get("low")
        hi = spec.get("high")
        seed = spec.get("seed", {})
        dist = seed.get("dist", "uniform")

        if k == "use_trend_filter":
            p = float(seed.get("p", 0.5))
            out[k] = bool(rng.random() < p)
            continue

        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            if dist == "gamma":
                kshape = float(seed.get("k", 2.0))
                theta = float(seed.get("theta", 5.0))
                val = float(rng.gamma(kshape, theta))
                # Wrap into [lo, hi)
                span = max(1e-9, (float(hi) - float(lo)))
                val = float(lo) + (val % span)
            else:
                val = float(lo) + float(rng.random()) * (float(hi) - float(lo))

            # Keep core integer params integral
            if all(isinstance(x, (int, np.integer)) for x in (lo, hi)) and k not in {
                "atr_multiple", "risk_per_trade", "tp_multiple", "cost_bps"
            }:
                val = int(round(val))
            out[k] = val
        else:
            out[k] = lo
    return out


# ------------------------------ Public Trainer API ----------------------------
def train_general_model(
    tickers: List[str],
    priors: Dict[str, Dict[str, Any]],
    priors_start: date,
    priors_end: date,
    strategy_dotted: str,
    cfg: TrainConfig,
    progress_cb: Callable[[float, str], None] | None = None,
) -> Dict[str, Any]:
    """
    Random-search CV over priors for each ticker.
    Returns {'leaderboard': DataFrame, 'debug': dict}.
    """
    rng = np.random.default_rng(int(cfg.seed))
    adapter = StrategyAdapter.from_name(strategy_dotted)

    folds = _make_folds(priors_start, priors_end, int(cfg.n_folds))
    rows = []
    debug = {"folds": folds, "tickers": [], "errors": []}

    total = max(1, len(tickers) * int(cfg.K))
    done = 0

    for sym in tickers:
        debug["tickers"].append(sym)
        best = None

        for _ in range(int(cfg.K)):
            # Sample params & evaluate
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
            score = cv["cv_sharpe"]

            # inside the sym-loop, after computing `best`
            if best is None:
                debug["errors"].append(f"{sym}: no successful runs")
            else:
                if best["cv"].get("errors"):
                    debug["errors"].append({sym: best["cv"]["errors"]})

            done += 1
            if progress_cb:
                progress_cb(done / total, f"{sym}: {done}/{total}")

        if best is None:
            debug["errors"].append(f"{sym}: no successful runs")
            continue

        rows.append(
            {
                "ticker": sym,
                "cv_sharpe": best["score"],
                "cv_trades": best["cv"]["cv_trades"],
                **{f"p_{k}": v for k, v in best["params"].items()},
            }
        )

    lb = pd.DataFrame(rows).sort_values(["cv_sharpe", "cv_trades"], ascending=[False, False]).reset_index(drop=True)
    return {"leaderboard": lb, "debug": debug}