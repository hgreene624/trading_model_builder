# tests/test_strategy_adapter_pipeline.py
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, UTC
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# What we will validate after we compute metrics locally
REQUIRED_METRIC_KEYS = {
    "trades",
    "total_return",
    "cagr",
    "max_drawdown",
    "calmar",
    "sharpe",
    "profit_factor",
    "expectancy",
}

STRATEGY_DOTTED = os.getenv("STRATEGY_DOTTED", "src.models.atr_breakout")
TICKERS = [t.strip() for t in os.getenv("STRAT_TEST_TICKERS", "AAPL,MSFT").split(",") if t.strip()]
DAYS_BACK = int(os.getenv("STRAT_TEST_DAYS", "365"))
STARTING_EQUITY = float(os.getenv("STRAT_TEST_EQUITY", "10000"))
PORTFOLIO_NAME = os.getenv("STRAT_TEST_PORTFOLIO", "")

BASE_PARAMS: Dict[str, Any] = {
    "breakout_n": 14,
    "exit_n": 6,
    "atr_n": 14,
    "atr_multiple": 2.0,
    "tp_multiple": 0.5,
    "holding_period_limit": 5,
    # extras present in other parts of the app — we’ll filter to match ATRParams:
    "use_trend_filter": False,
    "sma_fast": 20,
    "sma_slow": 50,
    "sma_long": 200,
    "long_slope_len": 20,
    "risk_per_trade": 0.005,
    "cost_bps": 1.0,
    "execution": "close",
}

EA_SPACE: Dict[str, Tuple] = {
    "breakout_n": (10, 20),
    "exit_n": (5, 10),
    "atr_n": (10, 20),
    "atr_multiple": (1.2, 2.5),
    "tp_multiple": (0.2, 0.8),
    "holding_period_limit": (2, 8),
}
EA_GENERATIONS = int(os.getenv("EA_TEST_GENERATIONS", "2"))
EA_POP = int(os.getenv("EA_TEST_POP", "6"))
EA_MIN_TRADES = int(os.getenv("EA_TEST_MIN_TRADES", "1"))
EA_NJOBS = int(os.getenv("EA_TEST_NJOBS", "1"))


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


def df_info(df: pd.DataFrame | pd.Series, name: str):
    if df is None or (hasattr(df, "empty") and df.empty):
        print(f"{name}: EMPTY")
        return
    if isinstance(df, pd.Series):
        print(f"{name}: rows={len(df)} index={type(df.index).__name__}")
    else:
        print(f"{name}: rows={len(df)} cols={list(df.columns)} index={type(df.index).__name__}")
    try:
        print("  head:", df.index[0], "tail:", df.index[-1])
    except Exception:
        pass


def assert_metrics_contract(metrics: Dict[str, Any]):
    missing = [k for k in REQUIRED_METRIC_KEYS if k not in metrics]
    if missing:
        raise AssertionError(f"metrics missing required keys: {missing}")
    for k in ["trades", "total_return", "cagr", "max_drawdown", "calmar", "sharpe", "profit_factor", "expectancy"]:
        v = metrics.get(k)
        if not isinstance(v, (int, float)):
            raise AssertionError(f"metrics['{k}'] should be numeric, got {type(v).__name__}")
    if metrics["trades"] < 0:
        raise AssertionError("metrics['trades'] must be >= 0")


def leg_env():
    banner("Environment")
    print("Python:", sys.version.split()[0], f"({sys.executable})")
    print("PWD   :", os.getcwd())
    print("ROOT  :", ROOT)
    print("Strategy dotted:", STRATEGY_DOTTED)
    print("Tickers        :", TICKERS)
    print("Days back      :", DAYS_BACK)
    print("Starting equity:", STARTING_EQUITY)
    if PORTFOLIO_NAME:
        print("Portfolio name :", PORTFOLIO_NAME)
    else:
        print("Portfolio name : <none> (trainer leg will be skipped)")


def leg_imports():
    import importlib
    banner("Import modules")
    L = importlib.import_module("src.data.loader")
    S = importlib.import_module(STRATEGY_DOTTED)
    GT = importlib.import_module("src.models.general_trainer")
    EV = importlib.import_module("src.optimization.evolutionary")
    M = importlib.import_module("src.backtest.metrics") # <- we’ll compute metrics here
    print("\n" + "=" * 72)
    print("Module paths")
    print("=" * 72)
    print("loader file   :", getattr(L, "__file__", "<??>"))
    print("strategy file :", getattr(S, "__file__", "<??>"))
    print("trainer file  :", getattr(GT, "__file__", "<??>"))
    print("evolutionary  :", getattr(EV, "__file__", "<??>"))
    print("metrics file  :", getattr(M, "__file__", "<??>"))
    return L, S, GT, EV, M


def _filter_params_for_strategy(S, base: Dict[str, Any]) -> Dict[str, Any]:
    params = dict(base)

    default_params = getattr(S, "DEFAULT_PARAMS", None)
    if isinstance(default_params, dict) and default_params:
        allowed = set(default_params.keys())
        filtered = {k: params.get(k, v) for k, v in default_params.items()}
        print(f"Using DEFAULT_PARAMS keys from strategy: {sorted(allowed)}")
        return filtered

    P = getattr(S, "ATRParams", None)
    allowed_keys = None
    if P is not None:
        try:
            from dataclasses import is_dataclass, fields
            if is_dataclass(P):
                allowed_keys = {f.name for f in fields(P)}
        except Exception:
            pass
        if allowed_keys is None:
            allowed_keys = set(getattr(P, "__annotations__", {}).keys()) or None
        if allowed_keys is None:
            mf = getattr(P, "model_fields", None)
            if isinstance(mf, dict) and mf:
                allowed_keys = set(mf.keys())
        if allowed_keys is None:
            fld = getattr(P, "__fields__", None)
            if isinstance(fld, dict) and fld:
                allowed_keys = set(fld.keys())

    if allowed_keys:
        print(f"Allowed ATRParams keys discovered: {sorted(allowed_keys)}")
        return {k: v for k, v in params.items() if k in allowed_keys}

    core = {"breakout_n", "exit_n", "atr_n", "atr_multiple", "tp_multiple", "holding_period_limit"}
    print("WARNING: Could not introspect ATRParams; using conservative param subset.")
    return {k: v for k, v in params.items() if k in core}


def leg_data_load(L):
    banner("Unified data load via loader.get_ohlcv")
    end = datetime.now(UTC)
    start = end - timedelta(days=DAYS_BACK)
    data: Dict[str, pd.DataFrame] = {}
    for sym in TICKERS:
        ok, out = leg(f"LOAD {sym}", getattr(L, "get_ohlcv"), sym, start, end)
        if not ok:
            raise out
        df = out
        df_info(df, sym)
        cols = {c.lower() for c in df.columns}
        for need in ("open", "high", "low", "close"):
            if need not in cols:
                raise AssertionError(f"{sym}: loader did not provide required column '{need}'")
        data[sym] = df
    return start, end, data


def _retry_strip_kwargs(callable_fn, sym, start, end, equity, params: Dict[str, Any], max_retries: int = 6):
    p = dict(params)
    for i in range(max_retries + 1):
        try:
            return callable_fn(sym, start, end, equity, p)
        except TypeError as e:
            msg = str(e)
            marker = "unexpected keyword argument"
            if marker in msg:
                bad = None
                if "'" in msg:
                    try:
                        bad = msg.split("'")[1]
                    except Exception:
                        bad = None
                if bad and bad in p:
                    print(f"[retry {i+1}] removing unsupported param: {bad}")
                    p.pop(bad, None)
                    continue
            raise
    raise RuntimeError("Exceeded max_retries while stripping unexpected kwargs.")


def leg_strategy_run(S, M, start, end):
    banner("Strategy.run_strategy() exact contract")
    run = getattr(S, "run_strategy", None)
    if run is None:
        raise AttributeError(f"{STRATEGY_DOTTED} is missing run_strategy")

    params = _filter_params_for_strategy(S, BASE_PARAMS)
    print("Params passed to run_strategy:", sorted(params.keys()))

    summaries = {}
    for sym in TICKERS:
        ok, out = leg(f"RUN run_strategy on {sym}", _retry_strip_kwargs, run, sym, start, end, STARTING_EQUITY, params)
        if not ok:
            raise out
        if not isinstance(out, dict):
            raise AssertionError(f"{sym}: run_strategy should return dict, got {type(out).__name__}")

        # Your strategy returns: equity (Series), daily_returns (Series), trades (list), meta (dict)
        equity = out.get("equity")
        trades = out.get("trades")
        daily = out.get("daily_returns")

        # Ensure daily_returns is available; derive if missing/empty
        if daily is None or (hasattr(daily, "empty") and daily.empty):
            daily = equity.pct_change().fillna(0.0)

        # Sanity printouts
        df_info(equity, f"{sym}.equity")
        print(f"{sym}.trades: {type(trades).__name__} len={len(trades) if isinstance(trades, list) else '??'}")
        df_info(daily, f"{sym}.daily_returns")

        # Compute metrics locally using your metrics module
        # Expectation: src.metrics exposes compute_core_metrics(equity: Series, trades: List[dict]|DataFrame) -> Dict[str, Any]
        compute = getattr(M, "compute_core_metrics", None)
        if compute is None:
            raise AttributeError("src.metrics missing compute_core_metrics()")

        metrics = compute(equity=equity, daily_returns=daily, trades=trades)
        assert_metrics_contract(metrics)
        print(f"{sym}: trades={metrics.get('trades')} total_return={metrics.get('total_return'):.4f} sharpe={metrics.get('sharpe'):.3f}")
        summaries[sym] = metrics
    return summaries


def leg_general_trainer(GT):
    banner("General Trainer (portfolio CV) — optional")
    if not PORTFOLIO_NAME:
        print("SKIP: No STRAT_TEST_PORTFOLIO provided.")
        return None
    train = getattr(GT, "train_general_model", None)
    if train is None:
        raise AttributeError("src.models.general_trainer missing train_general_model")
    ok, res = leg(
        f"train_general_model on portfolio='{PORTFOLIO_NAME}'",
        train,
        PORTFOLIO_NAME,
        STRATEGY_DOTTED,
        BASE_PARAMS,
        2,
        STARTING_EQUITY,
        1,
        1,
        None,
        None,
    )
    if not ok:
        raise res
    if not isinstance(res, dict):
        raise AssertionError(f"train_general_model should return dict, got {type(res).__name__}")
    for k in ("leaderboard", "logs", "errors"):
        if k not in res:
            raise AssertionError(f"train_general_model missing key '{k}'")
    print("trainer.leaderboard rows:", len(res.get("leaderboard", [])))
    return res


def leg_evolutionary(EV, start, end):
    banner("Evolutionary Search — smoke test")
    evo = getattr(EV, "evolutionary_search", None)
    if evo is None:
        raise AttributeError("src.optimization.evolutionary missing evolutionary_search")
    ok, top = leg(
        f"evolutionary_search on {TICKERS}",
        evo,
        STRATEGY_DOTTED,
        TICKERS,
        start,
        end,
        STARTING_EQUITY,
        EA_SPACE,
        generations=EA_GENERATIONS,
        pop_size=EA_POP,
        mutation_rate=0.4,
        elite_frac=0.5,
        random_inject_frac=0.2,
        n_jobs=EA_NJOBS,
        min_trades=EA_MIN_TRADES,
        # gates & clamps
        min_avg_holding_days_gate=1.0,
        require_hold_days=False,
        eps_mdd=1e-4,
        eps_sharpe=1e-4,
        # fitness weights
        alpha_cagr=1.0,
        beta_calmar=1.0,
        gamma_sharpe=0.25,
        # holding preferences (not day-trading, not buy/hold)
        min_holding_days=3.0,
        max_holding_days=30.0,
        holding_penalty_weight=1.0,
        # trade-rate preferences (per symbol per year)
        trade_rate_min=5.0,
        trade_rate_max=50.0,
        trade_rate_penalty_weight=0.5,
        # plumbing
        progress_cb=lambda *_a, **_k: None,
        log_file="training.log",
        seed=None,
    )
    if not ok:
        raise top
    if not isinstance(top, list) or len(top) == 0:
        raise AssertionError("evolutionary_search returned empty leaderboard or wrong type")
    p0, s0 = top[0]
    if not isinstance(p0, dict) or not isinstance(s0, (int, float)):
        raise AssertionError("leaderboard[0] should be (Dict[str, Any], float)")
    print("EA top[0]: score=", round(float(s0), 3), "params=", {k: p0.get(k) for k in list(EA_SPACE.keys())})
    return top


def main() -> int:
    banner("Environment")
    print("Python:", sys.version.split()[0], f"({sys.executable})")
    print("PWD   :", os.getcwd())
    print("ROOT  :", ROOT)
    print("Strategy dotted:", STRATEGY_DOTTED)
    print("Tickers        :", TICKERS)
    print("Days back      :", DAYS_BACK)
    print("Starting equity:", STARTING_EQUITY)
    print("Portfolio name :", PORTFOLIO_NAME or "<none>")

    ok, mods = leg("Import modules", leg_imports)
    if not ok:
        return 1
    L, S, GT, EV, M = mods

    ok, se = leg("Load data (all tickers)", leg_data_load, L)
    if not ok:
        return 1
    start, end, _ = se

    ok, _summ = leg("Strategy.run_strategy()", leg_strategy_run, S, M, start, end)
    if not ok:
        return 1

    ok, _tr = leg("General Trainer (optional)", leg_general_trainer, GT)
    if not ok:
        return 1

    ok, _ea = leg("Evolutionary Search (smoke)", leg_evolutionary, EV, start, end)
    if not ok:
        return 1

    banner("All strategy adapter pipeline legs completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())