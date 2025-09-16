# tests/test_walkforward.py
from __future__ import annotations

import os
import sys
import importlib
from dataclasses import dataclass
from datetime import datetime, timedelta, UTC
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ======================
# Config via environment
# ======================
STRATEGY = os.getenv("WF_STRATEGY", "src.models.atr_breakout")
TICKERS = [t.strip() for t in os.getenv("WF_TICKERS", "AAPL,MSFT").split(",") if t.strip()]
YEARS_BACK = int(os.getenv("WF_YEARS_BACK", "3"))
DAYS_BACK = int(os.getenv("WF_DAYS_BACK", str(365 * YEARS_BACK)))
EQUITY = float(os.getenv("WF_EQUITY", "10000"))

N_SPLITS = int(os.getenv("WF_N_SPLITS", "3"))
TRAIN_DAYS = int(os.getenv("WF_TRAIN_DAYS", "252"))
TEST_DAYS = int(os.getenv("WF_TEST_DAYS", "63"))
STEP_DAYS = os.getenv("WF_STEP_DAYS")
STEP_DAYS = None if STEP_DAYS in (None, "", "None", "none") else int(STEP_DAYS)

# EA toggles
USE_EA = bool(int(os.getenv("WF_USE_EA", "0")))   # default OFF for speed
EA_GEN = int(os.getenv("WF_EA_GEN", "4"))
EA_POP = int(os.getenv("WF_EA_POP", "12"))
EA_MIN_TRADES = int(os.getenv("WF_EA_MIN_TRADES", "3"))
EA_NJOBS = int(os.getenv("WF_EA_NJOBS", "1"))
SEED = os.getenv("WF_SEED")
SEED = None if SEED in (None, "", "None", "none") else int(SEED)

# Default param space (only used if USE_EA=1)
EA_SPACE: Dict[str, Tuple] = {
    "breakout_n": (10, 30),
    "exit_n": (5, 15),
    "atr_n": (10, 30),
    "atr_multiple": (1.0, 3.0),
    "tp_multiple": (0.2, 1.0),
    "holding_period_limit": (2, 10),
}
# Default fixed params (used when USE_EA=0)
FIXED_PARAMS: Dict[str, Any] = {
    "breakout_n": 14,
    "exit_n": 6,
    "atr_n": 14,
    "atr_multiple": 2.0,
    "tp_multiple": 0.5,
    "holding_period_limit": 5,
}

# =============
# Test harness
# =============
def banner(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)

def leg(title: str, fn, *args, **kwargs):
    print("\n" + "-" * 72)
    print(title)
    print("-" * 72)
    try:
        out = fn(*args, **kwargs)
        print("RESULT: PASS")
        return True, out
    except Exception as e:
        print("RESULT: FAIL")
        print("ERROR :", repr(e))
        return False, e

def leg_env():
    banner("Environment")
    print("Python:", sys.version.split()[0], f"({sys.executable})")
    print("PWD   :", os.getcwd())
    print("ROOT  :", ROOT)
    print("Strategy :", STRATEGY)
    print("Tickers  :", TICKERS)
    print("DaysBack :", DAYS_BACK)
    print("Equity   :", EQUITY)
    print("WF splits:", N_SPLITS, "train_days:", TRAIN_DAYS, "test_days:", TEST_DAYS, "step_days:", STEP_DAYS or "<test_days>")
    print("EA use   :", USE_EA, "gen:", EA_GEN, "pop:", EA_POP, "min_trades:", EA_MIN_TRADES, "n_jobs:", EA_NJOBS, "seed:", SEED)

# ============================
# Local walk-forward utilities
# ============================
def _import_strategy_and_utils():
    L = importlib.import_module("src.data.loader")
    M = importlib.import_module("src.backtest.metrics")
    S = importlib.import_module(STRATEGY)
    E = importlib.import_module("src.optimization.evolutionary")
    run_strategy = getattr(S, "run_strategy")
    get_ohlcv = getattr(L, "get_ohlcv")
    compute_metrics = getattr(M, "compute_core_metrics")
    evo = getattr(E, "evolutionary_search")
    return get_ohlcv, run_strategy, compute_metrics, evo, L, S, M, E

@dataclass
class Split:
    i: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime

def _make_splits(end: datetime) -> List[Split]:
    # rolling splits ending at `end`, stepping by step_days or test_days
    step = STEP_DAYS or TEST_DAYS
    splits: List[Split] = []
    cursor_end = end
    for i in range(N_SPLITS - 1, -1, -1):
        test_end = cursor_end
        test_start = test_end - timedelta(days=TEST_DAYS)
        train_end = test_start
        train_start = train_end - timedelta(days=TRAIN_DAYS)
        splits.append(Split(i=i, train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end))
        cursor_end = test_end - timedelta(days=step)
    # chronological order
    splits.sort(key=lambda s: s.i)
    return splits

def _load_panel(get_ohlcv, tickers: List[str], start: datetime, end: datetime) -> Dict[str, pd.DataFrame]:
    data = {}
    for sym in tickers:
        df = get_ohlcv(sym, start, end)
        if df is None or df.empty:
            raise RuntimeError(f"No data for {sym} {start}→{end}")
        need = {"open","high","low","close"}
        if not need.issubset({c.lower() for c in df.columns}):
            raise RuntimeError(f"{sym}: missing OHLC columns")
        data[sym] = df
    return data

def _eval_params_on_period(run_strategy, compute_metrics, params: Dict[str, Any], tickers: List[str],
                           start: datetime, end: datetime) -> Dict[str, Any]:
    # equal-weight portfolio: average normalized equity across symbols
    curves = {}
    trades_agg: List[dict] = []

    for sym in tickers:
        res = run_strategy(sym, start, end, EQUITY, params)

        eq = res.get("equity")
        if eq is None or len(eq) == 0:
            # skip symbols with no curve
            continue

        # daily_returns: avoid boolean context on Series
        daily = res.get("daily_returns")
        if not isinstance(daily, pd.Series) or daily.empty:
            daily = eq.pct_change().fillna(0.0)

        # trades: normalize to a list of dicts
        tr = res.get("trades")
        if tr is None:
            tr = []
        elif isinstance(tr, pd.DataFrame):
            tr = tr.to_dict("records")
        elif not isinstance(tr, list):
            # fallback: wrap single trade-like object
            tr = [tr]

        trades_agg.extend(tr)

        # normalize equity to start=1 and collect
        first = float(eq.iloc[0])
        if first != 0:
            curves[sym] = (eq / first).astype(float)

    if not curves:
        return {"metrics": {"trades": 0}}

    port = pd.DataFrame(curves).mean(axis=1, skipna=True)
    port_daily = port.pct_change().fillna(0.0)

    m = compute_metrics(port, port_daily, trades_agg)
    return {"metrics": m}

def _ea_or_fixed(evo, use_ea: bool, param_space: Dict[str, Tuple], fixed: Dict[str, Any],
                 tickers: List[str], start: datetime, end: datetime) -> Dict[str, Any]:
    if not use_ea:
        return dict(fixed)
    top = evo(
        STRATEGY, tickers, start, end, EQUITY, param_space,
        generations=EA_GEN, pop_size=EA_POP, min_trades=EA_MIN_TRADES,
        n_jobs=EA_NJOBS, seed=SEED, progress_cb=lambda *_a, **_k: None, log_file="wf_ea.log"
    )
    if not top:
        return dict(fixed)
    return dict(top[0][0])  # best params

def walkforward_once(get_ohlcv, run_strategy, compute_metrics, evo):
    end = datetime.now(UTC)
    start_global = end - timedelta(days=DAYS_BACK)
    splits = _make_splits(end)

    results = {
        "strategy": STRATEGY,
        "period": {"start": str(start_global), "end": str(end)},
        "config": {
            "splits": N_SPLITS, "train_days": TRAIN_DAYS, "test_days": TEST_DAYS, "step_days": STEP_DAYS,
            "use_ea": USE_EA, "ea_gen": EA_GEN, "ea_pop": EA_POP, "ea_min_trades": EA_MIN_TRADES, "ea_n_jobs": EA_NJOBS,
        },
        "splits": [],
        "aggregate": {}
    }

    oos_rows = []
    for s in splits:
        # Load data once per split (data loader itself caches via your module-level cache)
        _ = _load_panel(get_ohlcv, TICKERS, s.train_start, s.test_end)

        # Choose params on TRAIN
        best_params = _ea_or_fixed(evo, USE_EA, EA_SPACE, FIXED_PARAMS, TICKERS, s.train_start, s.train_end)

        # Evaluate OOS on TEST
        oos = _eval_params_on_period(run_strategy, compute_metrics, best_params, TICKERS, s.test_start, s.test_end)
        met = oos["metrics"] or {}

        results["splits"].append({
            "split_idx": s.i,
            "train_start": str(s.train_start), "train_end": str(s.train_end),
            "test_start": str(s.test_start), "test_end": str(s.test_end),
            "best_params": best_params,
            "train_score": None,  # quick smoke: omitted for speed
            "oos_metrics": met,
            "per_symbol": {},     # quick smoke: omitted for speed
        })
        if met:
            oos_rows.append(met)

    # Aggregate simple mean/median over numeric keys we care about
    if oos_rows:
        df = pd.DataFrame(oos_rows)
        results["aggregate"]["oos_mean"] = {k: float(df[k].mean()) for k in df.columns if pd.api.types.is_numeric_dtype(df[k])}
        results["aggregate"]["oos_median"] = {k: float(df[k].median()) for k in df.columns if pd.api.types.is_numeric_dtype(df[k])}
    else:
        results["aggregate"]["oos_mean"] = {}
        results["aggregate"]["oos_median"] = {}

    return results

# ======================
# Test legs
# ======================
def leg_imports():
    banner("Import modules")
    get_ohlcv, run_strategy, compute_metrics, evo, L, S, M, E = _import_strategy_and_utils()
    print("\n" + "=" * 72)
    print("Module paths")
    print("=" * 72)
    print("loader file   :", getattr(L, "__file__", "<??>"))
    print("strategy file :", getattr(S, "__file__", "<??>"))
    print("metrics file  :", getattr(M, "__file__", "<??>"))
    print("evolutionary  :", getattr(E, "__file__", "<??>"))
    return get_ohlcv, run_strategy, compute_metrics, evo

def leg_walkforward(get_ohlcv, run_strategy, compute_metrics, evo):
    banner("Walk-Forward run (local implementation)")
    res = walkforward_once(get_ohlcv, run_strategy, compute_metrics, evo)

    # quick structural checks
    if not isinstance(res.get("splits"), list) or len(res["splits"]) == 0:
        raise AssertionError("walkforward returned empty splits")
    for i, sr in enumerate(res["splits"]):
        for k in ("split_idx","train_start","train_end","test_start","test_end","best_params","oos_metrics"):
            if k not in sr:
                raise AssertionError(f"split[{i}] missing key '{k}'")
        if "trades" not in (sr["oos_metrics"] or {}):
            # metrics always include 'trades' in your compute_core_metrics
            raise AssertionError(f"split[{i}].oos_metrics missing 'trades'")

    mean = res["aggregate"].get("oos_mean", {})
    print("\nSummary (OOS mean):")
    for k in ("total_return","cagr","calmar","sharpe","max_drawdown","trades"):
        v = mean.get(k)
        if v is not None:
            print(f"  {k:>14s}: {v:.4f}")

    s0 = res["splits"][0]
    print("\nFirst split preview:")
    print("  dates:", s0["train_start"], "→", s0["train_end"], "| OOS:", s0["test_start"], "→", s0["test_end"])
    pkeys = ("breakout_n","exit_n","atr_n","atr_multiple","tp_multiple","holding_period_limit")
    print("  params:", {k: s0["best_params"].get(k) for k in pkeys})
    print("  oos trades:", s0["oos_metrics"].get("trades"))
    return res

def main() -> int:
    leg_env()
    ok, mods = leg("Import modules", leg_imports)
    if not ok:
        return 1
    get_ohlcv, run_strategy, compute_metrics, evo = mods

    ok, _ = leg("Walk-Forward run", leg_walkforward, get_ohlcv, run_strategy, compute_metrics, evo)
    if not ok:
        return 1



    banner("Walk-Forward test completed.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())