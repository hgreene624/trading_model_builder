# src/models/general_trainer.py
from __future__ import annotations

import json
import math
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.storage import load_portfolio, save_training_log
from src.models.strategy_adapter import StrategyAdapter
from src.data.cache import get_ohlcv_cached

ProgressHook = Callable[[Dict[str, Any]], None]


# ────────────────────────────────────────────────────────────────────────────────
# Small logging helper so we can emit rich, structured logs per symbol and global
# ────────────────────────────────────────────────────────────────────────────────
class EventLogger:
    def __init__(self) -> None:
        self.global_events: List[Dict[str, Any]] = []
        self.symbol_events: Dict[str, List[Dict[str, Any]]] = {}

    def emit(self, *, phase: str, msg: str = "", symbol: str | None = None, **extra: Any) -> None:
        ev = {"ts": datetime.now().isoformat(timespec="seconds"), "phase": phase, "msg": msg}
        ev.update({k: v for k, v in extra.items() if v is not None})
        if symbol:
            self.symbol_events.setdefault(symbol, []).append(ev)
        else:
            self.global_events.append(ev)

    def to_dict(self) -> Dict[str, Any]:
        return {"global": self.global_events, "symbols": self.symbol_events}


@dataclass
class TrainConfig:
    portfolio: str
    strategy_dotted: str
    params: Dict[str, Any]
    folds: int
    starting_equity: float
    min_trades: int = 4
    workers: int = 1  # IGNORED in this single-thread version
    progress_hook: ProgressHook | None = None


# ────────────────────────────────────────────────────────────────────────────────
# Safe helpers
# ────────────────────────────────────────────────────────────────────────────────

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return default
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if isinstance(x, list):
            return int(len(x))
        if x is None:
            return default
        return int(x)
    except Exception:
        try:
            return int(round(float(x)))
        except Exception:
            return default


def _sanitize_params(p: Dict[str, Any]) -> Dict[str, Any]:
    """Clamp/clean obvious types & bounds so backtest won't explode."""
    q = dict(p)
    ints = [
        "breakout_n",
        "exit_n",
        "atr_n",
        "sma_fast",
        "sma_slow",
        "sma_long",
        "long_slope_len",
        "holding_period_limit",
    ]
    for k in ints:
        if k in q:
            q[k] = max(1, int(_safe_int(q[k], 1)))
    floats = ["atr_multiple", "tp_multiple", "risk_per_trade", "cost_bps", "atr_ratio_max", "chop_max"]
    for k in floats:
        if k in q:
            q[k] = float(_safe_float(q[k], 0.0))
    if "use_trend_filter" in q:
        q["use_trend_filter"] = bool(q["use_trend_filter"])
    if "execution" in q:
        q["execution"] = str(q["execution"])  # e.g., "close"
    # optional debug flag that engines/adapters may choose to honor
    q.setdefault("debug", True)
    return q


# ────────────────────────────────────────────────────────────────────────────────
# Fold construction & CV evaluation
# ────────────────────────────────────────────────────────────────────────────────

def _make_folds(df: pd.DataFrame, k: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Simple equal-span folds over the available index range."""
    if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return []
    idx = df.index.sort_values()
    start, end = idx[0], idx[-1]
    if k <= 1:
        return [(start, end)]
    spans = []
    total = (end - start).days
    if total < k:
        # fallback: split by index count
        n = len(idx)
        step = max(1, n // k)
        for i in range(k):
            s_i = idx[min(i * step, n - 1)]
            e_i = idx[min((i + 1) * step - 1, n - 1)]
            if s_i <= e_i:
                spans.append((s_i, e_i))
        return spans
    step_days = max(1, total // k)
    cursor = start
    for _ in range(k):
        s_i = cursor
        e_i = min(end, s_i + timedelta(days=step_days))
        spans.append((s_i, e_i))
        cursor = e_i
    # ensure last ends at 'end'
    spans[-1] = (spans[-1][0], end)
    return spans


def cv_eval(
    adapter: StrategyAdapter,
    symbol: str,
    params: Dict[str, Any],
    folds: Sequence[Tuple[pd.Timestamp, pd.Timestamp]],
    starting_equity: float,
    fold_hook: ProgressHook | None = None,
) -> Dict[str, Any]:
    """Run backtest over folds and aggregate metrics."""
    fold_metrics: List[Dict[str, Any]] = []
    for i, (s, e) in enumerate(folds, start=1):
        if fold_hook:
            fold_hook({
                "phase": "fold_start",
                "symbol": symbol,
                "i": i,
                "n": len(folds),
                "start": str(s),
                "end": str(e),
            })
        m = adapter.backtest(symbol, params, s, e, starting_equity) or {}
        # normalize
        trades_raw = m.get("trades", 0)
        trades = _safe_int(trades_raw, 0)
        sharpe = _safe_float(m.get("sharpe", 0.0), 0.0)
        cagr = _safe_float(m.get("cagr", 0.0), 0.0)
        maxdd = _safe_float(m.get("maxdd", 0.0), 0.0)
        # capture optional engine debug info if present
        dbg = m.get("debug") if isinstance(m.get("debug"), dict) else None
        fold_metrics.append({
            "sharpe": sharpe,
            "cagr": cagr,
            "trades": trades,
            "maxdd": maxdd,
            "debug": dbg,
        })
        if fold_hook:
            fold_hook({
                "phase": "fold_end",
                "symbol": symbol,
                "i": i,
                "n": len(folds),
                "sharpe": sharpe,
                "trades": trades,
            })

    # aggregate
    if not fold_metrics:
        return {"cv_sharpe": 0.0, "cv_cagr": 0.0, "cv_trades": 0, "cv_maxdd": 0.0, "folds": []}

    shs = [fm["sharpe"] for fm in fold_metrics]
    cgs = [fm["cagr"] for fm in fold_metrics]
    trs = [fm["trades"] for fm in fold_metrics]
    dds = [fm["maxdd"] for fm in fold_metrics]

    return {
        "cv_sharpe": float(np.nanmean(shs)) if len(shs) else 0.0,
        "cv_cagr": float(np.nanmean(cgs)) if len(cgs) else 0.0,
        "cv_trades": int(np.nansum(trs)) if len(trs) else 0,
        "cv_maxdd": float(np.nanmin(dds)) if len(dds) else 0.0,
        "folds": fold_metrics,
    }


# ────────────────────────────────────────────────────────────────────────────────
# Lightweight probe to understand *why* there are zero trades
# ────────────────────────────────────────────────────────────────────────────────

def _atr(series_high: pd.Series, series_low: pd.Series, series_close: pd.Series, n: int) -> pd.Series:
    prev_close = series_close.shift(1)
    tr = pd.concat([
        (series_high - series_low).abs(),
        (series_high - prev_close).abs(),
        (series_low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def _chop_index(h: pd.Series, l: pd.Series, n: int) -> pd.Series:
    # CHOP = 100 * log10( sum(TR(n)) / (max(high,n) - min(low,n)) ) / log10(n)
    tr = (h - l).abs()
    sum_tr = tr.rolling(n, min_periods=n).sum()
    hi_n = h.rolling(n, min_periods=n).max()
    lo_n = l.rolling(n, min_periods=n).min()
    rng = (hi_n - lo_n).replace(0, np.nan)
    with np.errstate(invalid='ignore', divide='ignore'):
        chop = 100.0 * (np.log10((sum_tr / rng))) / np.log10(n)
    return chop.replace([np.inf, -np.inf], np.nan)


def _probe_breakout_reasons(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    """Heuristic probe for why entries might be filtered out.
    Returns counters for candidate breakouts and gating filters.
    """
    need = {"high", "low", "close"}
    if df is None or df.empty or not need.issubset(set(map(str.lower, df.columns)) | set(df.columns)):
        return {"error": "df missing OHLC columns"}

    # be resilient to upper/lower case column names
    cols = {c.lower(): c for c in df.columns}
    H = df[cols.get("high")]
    L = df[cols.get("low")]
    C = df[cols.get("close")]

    bo_n = max(2, int(_safe_int(params.get("breakout_n", 55), 55)))
    ex_n = max(1, int(_safe_int(params.get("exit_n", 35), 35)))
    atr_n = max(2, int(_safe_int(params.get("atr_n", 14), 14)))
    sma_fast = max(1, int(_safe_int(params.get("sma_fast", 20), 20)))
    sma_slow = max(2, int(_safe_int(params.get("sma_slow", 60), 60)))
    sma_long = max(10, int(_safe_int(params.get("sma_long", 200), 200)))
    slope_len = max(2, int(_safe_int(params.get("long_slope_len", 20), 20)))

    use_tf = bool(params.get("use_trend_filter", True))
    chop_max = _safe_float(params.get("chop_max", 55.0), 55.0)
    atr_ratio_max = _safe_float(params.get("atr_ratio_max", 1.8), 1.8)

    atr = _atr(H, L, C, atr_n)
    atr_ma = atr.rolling(atr_n, min_periods=atr_n).mean()
    atr_ratio = (atr / atr_ma).replace([np.inf, -np.inf], np.nan)

    # donchian breakout level: prior N-day high
    prior_high = H.rolling(bo_n, min_periods=bo_n).max().shift(1)
    candidate = (C > prior_high)

    # trend filter
    fast = C.rolling(sma_fast, min_periods=sma_fast).mean()
    slow = C.rolling(sma_slow, min_periods=sma_slow).mean()
    long = C.rolling(sma_long, min_periods=sma_long).mean()
    long_slope = long - long.shift(slope_len)

    tf_pass = (~use_tf) | ((fast > slow) & (long_slope > 0))

    # choppiness proxy
    chop = _chop_index(H, L, max(atr_n, 14))
    chop_pass = (chop_max is None) | (chop.fillna(1000) <= chop_max)

    # atr ratio gate
    atr_pass = (atr_ratio_max is None) | (atr_ratio.fillna(0) <= atr_ratio_max)

    # aggregate reasons on candidate bars
    mask = candidate.fillna(False)
    n_cand = int(mask.sum())
    n_tf_reject = int((mask & (~tf_pass.fillna(False))).sum())
    n_chop_reject = int((mask & (~chop_pass.fillna(False))).sum())
    n_atr_reject = int((mask & (~atr_pass.fillna(False))).sum())
    n_survive = int((mask & tf_pass.fillna(False) & chop_pass.fillna(False) & atr_pass.fillna(False)).sum())

    return {
        "bars": int(len(df)),
        "bo_n": bo_n,
        "exit_n": ex_n,
        "atr_n": atr_n,
        "sma_fast": sma_fast,
        "sma_slow": sma_slow,
        "sma_long": sma_long,
        "candidate_breakouts": n_cand,
        "reject_trend": n_tf_reject,
        "reject_chop": n_chop_reject,
        "reject_atr_ratio": n_atr_reject,
        "candidates_after_filters": n_survive,
    }


# ────────────────────────────────────────────────────────────────────────────────
# Rescue sweep when CV shows 0 trades
# ────────────────────────────────────────────────────────────────────────────────

def _rescue_param_sweep(
    adapter: StrategyAdapter,
    symbol: str,
    base_params: Dict[str, Any],
    starting_equity: float,
) -> Dict[str, Any] | None:
    """Quick grid to find *any* trading activity as a fallback."""
    try:
        df_all = get_ohlcv_cached(symbol, "1900-01-01", "2100-01-01")
        if df_all is None or df_all.empty:
            return None
        idx = df_all.index if isinstance(df_all.index, pd.DatetimeIndex) else pd.to_datetime(df_all.index)
        if len(idx) < 150:
            return None

        # Use last ~250 trading days as a probe
        s = idx[max(0, len(idx) - 260)]
        e = idx[-1]
        best = None
        best_trades = -1

        for bo in [10, 20, 40, 60, 90]:
            for ex in [5, 10, 20, 30, 40, 60]:
                for am in [1.8, 2.2, 2.6, 3.2]:
                    for tp in [1.2, 1.6, 2.0, 2.6]:
                        q = dict(base_params)
                        q.update({
                            "breakout_n": bo,
                            "exit_n": ex,
                            "atr_multiple": am,
                            "tp_multiple": tp,
                        })
                        q = _sanitize_params(q)
                        m = adapter.backtest(symbol, q, s, e, starting_equity) or {}
                        trades = _safe_int(m.get("trades", 0), 0)
                        if trades > best_trades:
                            best_trades = trades
                            best = {"params": q, "metrics": m}
        return best
    except Exception:
        return None


# ────────────────────────────────────────────────────────────────────────────────
# Main entry
# ────────────────────────────────────────────────────────────────────────────────

def train_general_model(
    portfolio: str,
    strategy_dotted: str,
    params: Dict[str, Any],
    folds: int,
    starting_equity: float,
    min_trades: int = 4,
    workers: int = 1,  # ignored; single-threaded
    progress_hook: ProgressHook | None = None,
) -> Dict[str, Any]:
    """
    Single-threaded trainer. Emits progress via `progress_hook` and also returns
    a detailed JSON log with symbol-level probes when trades==0.
    """
    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    symbol_logs: List[Dict[str, Any]] = []
    elog = EventLogger()

    # Load portfolio
    pobj = load_portfolio(portfolio) or {}
    symbols: List[str] = list(pobj.get("tickers", []))
    if not symbols:
        return {"leaderboard": [], "errors": [{"error": f"Portfolio '{portfolio}' is empty."}], "log": {}}

    # Strategy adapter
    base_params = _sanitize_params(params)
    adapter = StrategyAdapter.from_name(strategy_dotted, base_params)

    # Prefetch (sequential)
    if progress_hook:
        progress_hook({"phase": "prefetch_start", "symbols_total": len(symbols), "workers": 1})
    elog.emit(phase="prefetch_start", msg=f"symbols={len(symbols)}")

    ok = 0
    for i, sym in enumerate(symbols, start=1):
        try:
            df = get_ohlcv_cached(sym, "1900-01-01", "2100-01-01")
            ok += 1 if (df is not None and not df.empty) else 0
            if progress_hook:
                progress_hook({"phase": "prefetch_progress", "i": i, "n": len(symbols), "symbol": sym, "ok": bool(df is not None and not df.empty)})
            elog.emit(phase="prefetch_progress", symbol=sym, msg="ok" if (df is not None and not df.empty) else "empty")
        except Exception as ex:
            errors.append({"symbol": sym, "error": f"Prefetch: {type(ex).__name__}: {ex}"})
            if progress_hook:
                progress_hook({"phase": "prefetch_progress", "i": i, "n": len(symbols), "symbol": sym, "ok": False})
            elog.emit(phase="prefetch_progress", symbol=sym, msg="error", error=f"{type(ex).__name__}: {ex}")

    if progress_hook:
        progress_hook({"phase": "prefetch_done", "ok": ok, "errors": len(errors)})
    elog.emit(phase="prefetch_done", msg=f"ok={ok} errors={len(errors)}")

    # Train (sequential)
    if progress_hook:
        progress_hook({"phase": "start", "symbols_total": len(symbols), "workers": 1})
    elog.emit(phase="start", msg=f"workers=1 symbols_total={len(symbols)}")

    for i, sym in enumerate(symbols, start=1):
        if progress_hook:
            progress_hook({"phase": "queued", "symbol": sym, "i": i, "n": len(symbols)})
            progress_hook({"phase": "symbol_start", "symbol": sym, "i": i, "n": len(symbols)})
        elog.emit(phase="symbol_start", symbol=sym, msg=f"{i}/{len(symbols)}")

        try:
            df_all = get_ohlcv_cached(sym, "1900-01-01", "2100-01-01")
            if df_all is None or df_all.empty:
                raise RuntimeError("No data")

            if not isinstance(df_all.index, pd.DatetimeIndex):
                df_all = df_all.copy()
                df_all.index = pd.to_datetime(df_all.index)

            fold_bounds = _make_folds(df_all, int(max(1, folds)))

            # per-fold callback (safe; we're single-threaded)
            def fold_hook(ev: Dict[str, Any]):
                if progress_hook:
                    progress_hook(ev)
                elog.emit(symbol=sym, phase=ev.get("phase", "fold"), msg="", **{k: v for k, v in ev.items() if k not in {"phase"}})

            adapter.params = dict(base_params)  # type: ignore[attr-defined]
            cv = cv_eval(adapter, sym, adapter.params, fold_bounds, starting_equity, fold_hook=fold_hook)

            trades = _safe_int(cv.get("cv_trades", 0), 0)
            sym_log: Dict[str, Any] = {
                "symbol": sym,
                "folds": [{"start": str(s), "end": str(e)} for (s, e) in fold_bounds],
                "cv": cv,
                "base_params": dict(base_params),
            }

            if trades < max(1, int(min_trades)):
                # add a quick probe to understand gating reasons
                probe = _probe_breakout_reasons(df_all, base_params)
                sym_log["probe"] = probe
                elog.emit(phase="probe", symbol=sym, msg="zero-trade probe", **(probe or {}))

                rescue = _rescue_param_sweep(adapter, sym, base_params, starting_equity)
                if rescue and rescue.get("params"):
                    p2 = _sanitize_params(rescue["params"])
                    adapter.params = p2  # type: ignore[attr-defined]
                    cv2 = cv_eval(adapter, sym, adapter.params, fold_bounds, starting_equity, fold_hook=fold_hook)
                    sym_log["rescue"] = {"params": p2, "single_window_metrics": rescue.get("metrics", {}), "cv": cv2}
                    row = {
                        "symbol": sym,
                        "cv_sharpe": _safe_float(cv2.get("cv_sharpe", 0.0)),
                        "cv_cagr": _safe_float(cv2.get("cv_cagr", 0.0)),
                        "cv_trades": _safe_int(cv2.get("cv_trades", 0)),
                        "cv_maxdd": _safe_float(cv2.get("cv_maxdd", 0.0)),
                        "params": p2,
                    }
                    elog.emit(phase="rescue_used", symbol=sym, msg=f"cv_trades={row['cv_trades']}")
                else:
                    row = {
                        "symbol": sym,
                        "cv_sharpe": _safe_float(cv.get("cv_sharpe", 0.0)),
                        "cv_cagr": _safe_float(cv.get("cv_cagr", 0.0)),
                        "cv_trades": _safe_int(cv.get("cv_trades", 0)),
                        "cv_maxdd": _safe_float(cv.get("cv_maxdd", 0.0)),
                        "params": dict(base_params),
                    }
                    elog.emit(phase="rescue_failed", symbol=sym, msg="kept base params")
            else:
                row = {
                    "symbol": sym,
                    "cv_sharpe": _safe_float(cv.get("cv_sharpe", 0.0)),
                    "cv_cagr": _safe_float(cv.get("cv_cagr", 0.0)),
                    "cv_trades": _safe_int(cv.get("cv_trades", 0)),
                    "cv_maxdd": _safe_float(cv.get("cv_maxdd", 0.0)),
                    "params": dict(base_params),
                }
                elog.emit(phase="symbol_ok", symbol=sym, msg=f"cv_trades={row['cv_trades']}")

            rows.append(row)
            symbol_logs.append(sym_log)

            if progress_hook:
                progress_hook({
                    "phase": "symbol_done",
                    "symbol": sym,
                    "i": i,
                    "n": len(symbols),
                    "cv_trades": _safe_int(row.get("cv_trades", 0)),
                    "cv_sharpe": _safe_float(row.get("cv_sharpe", 0.0)),
                })
            elog.emit(phase="symbol_done", symbol=sym, msg="done", cv_trades=row.get("cv_trades", 0))

        except Exception as ex:
            errors.append({"symbol": sym, "error": f"{type(ex).__name__}: {ex}", "traceback": traceback.format_exc()})
            if progress_hook:
                progress_hook({"phase": "symbol_done", "symbol": sym, "i": i, "n": len(symbols), "cv_trades": 0, "cv_sharpe": 0.0})
            elog.emit(phase="symbol_error", symbol=sym, msg=str(ex), error_type=type(ex).__name__)

    # build result
    log = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "portfolio": portfolio,
        "strategy": strategy_dotted,
        "symbols": symbols,
        "rows": len(rows),
        "errors": errors,
        "symbol_logs": symbol_logs,
        "events": elog.to_dict(),
    }

    try:
        save_training_log(portfolio, log)
    except Exception:
        pass

    return {"leaderboard": rows, "errors": errors, "log": log}