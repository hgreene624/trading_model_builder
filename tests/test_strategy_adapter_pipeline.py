# tests/test_strategy_adapter_pipeline.py
from __future__ import annotations

import os
import sys
import inspect
from pathlib import Path
from datetime import datetime, timedelta, UTC
from typing import Callable, Dict, Any, List, Optional

import pandas as pd


# ───────────────────────── helpers ─────────────────────────

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def banner(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)

def leg(title: str, fn: Callable, *args, **kwargs):
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

def df_info(df: pd.DataFrame, name: str):
    if df is None or df.empty:
        print(f"{name}: EMPTY")
        return
    print(f"{name}: rows={len(df)} cols={list(df.columns)} index={type(df.index).__name__}")
    try:
        print("  head:", df.index[0], "tail:", df.index[-1])
    except Exception:
        pass

def now_utc():
    return datetime.now(UTC)


# ───────────────────────── config ─────────────────────────

# You can override these without editing the file:
#   STRAT_TEST_TICKERS="AAPL,MSFT" STRATEGY_DOTTED="src.models.atr_breakout" python -m tests.test_strategy_adapter_pipeline
TICKERS: List[str] = [t.strip() for t in os.getenv("STRAT_TEST_TICKERS", "AAPL,MSFT").split(",") if t.strip()]
STRATEGY_DOTTED: str = os.getenv("STRATEGY_DOTTED", "src.models.atr_breakout")
DAYS_BACK: int = int(os.getenv("STRAT_TEST_DAYS", "365"))  # date window
MIN_ROWS: int = int(os.getenv("STRAT_TEST_MIN_ROWS", "200"))  # sanity threshold for data sufficiency

# Parameters: we try to read DEFAULT_PARAMS off the strategy module; if missing we use this light fallback.
FALLBACK_PARAMS = {
    # common ATR breakout-like knobs (harmless if the strategy ignores them)
    "breakout_n": 14,
    "exit_n": 6,
    "atr_n": 14,
    "atr_multiple": 2.0,
    "tp_multiple": 0.5,
    "holding_period_limit": 5,
    "use_trend_filter": False,
    "sma_fast": 20,
    "sma_slow": 50,
    "sma_long": 200,
    "long_slope_len": 20,
    "risk_per_trade": 0.005,
    "cost_bps": 1.0,
    "execution": "close",
}

STRAT_FN_CANDIDATES = [
    # prefer explicit “generate_signals” style names:
    "generate_signals",
    "run_strategy",
    "run",
    "apply",
    "signals",
]

REQUIRED_Cols = {"open", "high", "low", "close"}  # loader output must include these


# ───────────────────────── legs ─────────────────────────

def leg_env():
    banner("Environment")
    print("Python:", sys.version.split()[0], "(", sys.executable, ")")
    print("PWD   :", os.getcwd())
    print("ROOT  :", ROOT)
    print("Tickers        :", TICKERS)
    print("Strategy dotted:", STRATEGY_DOTTED)
    print("Window (days)  :", DAYS_BACK)
    print("Min rows       :", MIN_ROWS)

def leg_import_modules():
    import importlib
    banner("Module paths")
    L = importlib.import_module("src.data.loader")
    print("loader file    :", getattr(L, "__file__", "<??>"))
    S = importlib.import_module(STRATEGY_DOTTED)
    print("strategy file  :", getattr(S, "__file__", "<??>"))
    return L, S

def leg_data_load(L):
    banner("Data load via unified loader")
    end = now_utc()
    start = end - timedelta(days=DAYS_BACK)
    results: Dict[str, pd.DataFrame] = {}
    for sym in TICKERS:
        ok, out = leg(f"LOAD {sym}", getattr(L, "get_ohlcv"), sym, start, end)
        if not ok:
            raise out
        df = out
        df_info(df, f"{sym}")
        # sanity
        if df is None or df.empty:
            raise RuntimeError(f"{sym}: data empty")
        cols_lower = {c.lower() for c in df.columns}
        if not REQUIRED_Cols.issubset(cols_lower):
            raise RuntimeError(f"{sym}: missing OHLC columns; got {list(df.columns)}")
        if len(df) < MIN_ROWS:
            print(f"WARNING: {sym} has only {len(df)} rows (< {MIN_ROWS}).")
        results[sym] = df
    return results

def _pick_strat_fn(mod) -> Callable[..., Any]:
    for name in STRAT_FN_CANDIDATES:
        if hasattr(mod, name) and callable(getattr(mod, name)):
            return getattr(mod, name)
    raise AttributeError(f"No strategy entry found on {mod}: tried {STRAT_FN_CANDIDATES}")

def _strategy_params(mod, df) -> Dict[str, Any]:
    # Try DEFAULT_PARAMS on the module; else fallback.
    params = getattr(mod, "DEFAULT_PARAMS", None)
    if not isinstance(params, dict):
        params = FALLBACK_PARAMS.copy()

    # Be nice: only pass parameters the function accepts to avoid TypeErrors.
    fn = _pick_strat_fn(mod)
    sig = inspect.signature(fn)
    allowed = set(sig.parameters.keys())
    trimmed = {k: v for k, v in params.items() if k in allowed}

    # If function requires a 'df' positional or keyword, we’ll pass it separately.
    return trimmed

def leg_strategy_exec(S, data_map: Dict[str, pd.DataFrame]):
    banner("Strategy execution sanity")
    fn = _pick_strat_fn(S)
    print("Using strategy function:", fn.__name__)
    p = _strategy_params(S, next(iter(data_map.values())))

    # Try each ticker independently; this mimics per-ticker specialization in the lab.
    out_summaries = {}
    for sym, df in data_map.items():
        def _call():
            try:
                # Common signatures we’ll attempt in order:
                # (df, params), (df), (prices_only)
                try:
                    return fn(df=df, params=p)
                except TypeError:
                    try:
                        return fn(df, p)
                    except TypeError:
                        return fn(df)
            except Exception:
                # As a last resort, try passing only close series if callable expects a Series.
                return fn(df["close"])

        ok, out = leg(f"RUN strategy on {sym}", _call)
        if not ok:
            raise out

        # Heuristics: accept df-like output. Count “signals” if present, else “orders”, else infer.
        if isinstance(out, pd.DataFrame):
            df_info(out, f"{sym} (strategy output)")
            cols = {c.lower() for c in out.columns}
            signal_cols = [c for c in out.columns if c.lower() in {"signal", "side", "order", "orders"}]
            nz = 0
            if "signal" in cols:
                s = out[[c for c in out.columns if c.lower() == "signal"][0]]
                try:
                    nz = int((s != 0).sum())
                except Exception:
                    nz = len(s.dropna())
            elif signal_cols:
                # count non-null entries in the first signal-ish column
                s = out[signal_cols[0]]
                nz = int((s != 0).sum()) if pd.api.types.is_numeric_dtype(s) else int(s.notna().sum())
            else:
                # fallback: look for boolean/±1 columns
                for c in out.columns:
                    s = out[c]
                    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
                        try:
                            nz = max(nz, int((s != 0).sum()))
                        except Exception:
                            pass
            print(f"{sym}: detected ~{nz} signal-like events")
            out_summaries[sym] = {"rows": len(out), "signals": nz, "columns": list(out.columns)}
        else:
            # non-DF outputs (dict, tuple, etc.) — just print type and consider pass if no exception
            print(f"{sym}: strategy output type={type(out).__name__}")
            out_summaries[sym] = {"type": type(out).__name__}
    return out_summaries


# ───────────────────────── main ─────────────────────────

def main():
    leg_env()
    ok, modpair = leg("Import modules", leg_import_modules)
    if not ok:
        return 1
    L, S = modpair

    ok, data_map = leg("Load data (all tickers)", leg_data_load, L)
    if not ok:
        return 1

    ok, summary = leg("Execute strategy (per ticker)", leg_strategy_exec, S, data_map)
    if not ok:
        return 1

    banner("Summary")
    print(pd.DataFrame(summary).T.to_string())

    print("\nAll strategy adapter pipeline legs completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())