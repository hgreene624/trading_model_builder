#!/usr/bin/env python3
"""Deterministic smoke harness for ATR breakout engine validation."""

from __future__ import annotations

import argparse
import json
import os
import random
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List

import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.metrics import TRADING_DAYS, max_drawdown, summarize_costs
from src.models.atr_breakout import backtest_single
from src.utils.logging_setup import setup_logging
from src.storage import get_ohlcv_root


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic ATR breakout smoke backtests.")
    parser.add_argument("--tickers", default=os.getenv("SMOKE_TICKERS", "SPY,QQQ,AAPL,MSFT,AMD"))
    parser.add_argument("--start", default=os.getenv("SMOKE_START", "2022-01-01"))
    parser.add_argument("--end", default=os.getenv("SMOKE_END", "2024-12-31"))
    parser.add_argument("--k-atr-buffer", type=float, default=float(os.getenv("K_ATR_BUFFER", 0.0)))
    parser.add_argument("--persist-n", type=int, default=int(os.getenv("PERSIST_N", 1)))
    parser.add_argument("--min-hold-days", type=int, default=int(os.getenv("MIN_HOLD_DAYS", 0)))
    parser.add_argument(
        "--reentry-cooldown-days", type=int, default=int(os.getenv("REENTRY_COOLDOWN_DAYS", 0))
    )
    parser.add_argument("--cost-enabled", type=int, choices=[0, 1], default=int(os.getenv("COST_ENABLED", 1)))
    parser.add_argument("--fixed-bps", type=float, default=float(os.getenv("COST_FIXED_BPS", 0.5)))
    parser.add_argument("--atr-k", type=float, default=float(os.getenv("COST_ATR_K", 0.25)))
    parser.add_argument("--min-hs-bps", type=float, default=float(os.getenv("COST_MIN_HS_BPS", 1.0)))
    parser.add_argument(
        "--use-range-impact",
        type=int,
        choices=[0, 1],
        default=int(os.getenv("COST_USE_RANGE_IMPACT", 0)),
        help="Enable range-based slippage proxy (Phase 1.2).",
    )  # PH1.2
    parser.add_argument(
        "--cap-range-impact-bps",
        type=float,
        default=float(os.getenv("CAP_RANGE_IMPACT_BPS", 10.0)),
        help="Cap for range-based slippage impact in bps.",
    )  # PH1.2
    parser.add_argument(
        "--per-trade-fee-usd",
        type=float,
        default=float(os.getenv("FEE_PER_TRADE_USD", os.getenv("ATR_PER_TRADE_FEE", 0.0))),
        help="Flat fee applied per fill in account currency.",
    )
    parser.add_argument("--exec-delay-bars", type=int, default=int(os.getenv("EXEC_DELAY_BARS", 0)))
    parser.add_argument(
        "--exec-fill-where", choices=["open", "close"], default=os.getenv("EXEC_FILL_WHERE", "close")
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--label", default="")
    parser.add_argument("--fast", action="store_true", help="Use deterministic fast-mode defaults.")
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Do not attempt network backfill; skip gracefully if data missing.",
    )
    parser.add_argument(
        "--max-bars",
        type=int,
        default=None,
        help="If provided, limit processing to approximately the most recent N bars.",
    )
    return parser.parse_args()


def _arg_was_provided(flag: str) -> bool:
    flag = flag.strip()
    if not flag.startswith("--"):
        flag = f"--{flag}"
    for token in sys.argv[1:]:
        if token == flag:
            return True
        if token.startswith(f"{flag}="):
            return True
    return False


_FAST_ENV_KEYS = {
    "tickers": "SMOKE_TICKERS",
    "start": "SMOKE_START",
    "end": "SMOKE_END",
    "k_atr_buffer": "K_ATR_BUFFER",
    "persist_n": "PERSIST_N",
    "cost_enabled": "COST_ENABLED",
    "fixed_bps": "COST_FIXED_BPS",
    "atr_k": "COST_ATR_K",
    "min_hs_bps": "COST_MIN_HS_BPS",
    "use_range_impact": "COST_USE_RANGE_IMPACT",
    "cap_range_impact_bps": "CAP_RANGE_IMPACT_BPS",
    "per_trade_fee_usd": "FEE_PER_TRADE_USD",
}

_FAST_SENSITIVE_ATTRS = {
    "cost_enabled",
    "fixed_bps",
    "atr_k",
    "min_hs_bps",
    "use_range_impact",
    "cap_range_impact_bps",
    "per_trade_fee_usd",
}


def _apply_fast_defaults(args: argparse.Namespace) -> None:
    if not args.fast:
        return

    overrides = {
        "tickers": ("--tickers", "SPY"),
        "start": ("--start", "2023-01-01"),
        "end": ("--end", "2023-06-30"),
        "k_atr_buffer": ("--k-atr-buffer", 0.0),
        "persist_n": ("--persist-n", 1),
        "cost_enabled": ("--cost-enabled", 1),
        "fixed_bps": ("--fixed-bps", 0.5),
        "atr_k": ("--atr-k", 0.25),
        "min_hs_bps": ("--min-hs-bps", 1.0),
        "use_range_impact": ("--use-range-impact", 0),  # PH1.2
        "cap_range_impact_bps": ("--cap-range-impact-bps", 10.0),  # PH1.2
        "per_trade_fee_usd": ("--per-trade-fee-usd", 0.0),
    }

    warnings: List[str] = []

    for attr, (flag, value) in overrides.items():
        cli_provided = _arg_was_provided(flag)
        env_key = _FAST_ENV_KEYS.get(attr)
        env_present = env_key is not None and env_key in os.environ
        if cli_provided or env_present:
            if attr in _FAST_SENSITIVE_ATTRS:
                source = "CLI flag" if cli_provided else f"ENV {env_key}"
                warnings.append(
                    f"WARNING: --fast default for {flag} ignored; using {source} value."
                )
            continue
        setattr(args, attr, value)

    if args.exec_delay_bars is None and not _arg_was_provided("--exec-delay-bars"):
        if "EXEC_DELAY_BARS" not in os.environ:
            args.exec_delay_bars = 0
    if not _arg_was_provided("--exec-fill-where") and "EXEC_FILL_WHERE" not in os.environ:
        args.exec_fill_where = "close"

    for message in warnings:
        print(message)


def _has_cached_data(symbol: str, timeframe: str = "1D") -> bool:
    root = get_ohlcv_root()
    sym = symbol.upper().replace("/", "_")
    tf = (timeframe or "1D").upper()
    for provider in ("alpaca", "yahoo"):
        provider_dir = root / provider / sym / tf
        if provider_dir.exists():
            for candidate in provider_dir.glob("*.parquet"):
                if candidate.is_file():
                    return True
    return False


def _coerce_bool(value) -> bool:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off", ""}:
            return False
        try:
            return bool(int(lowered))
        except ValueError:
            return True
    if isinstance(value, (int, float)):
        return bool(value)
    return bool(value)


def _serialize_bool(value) -> str:
    return "1" if _coerce_bool(value) else "0"


_ENV_OVERRIDE_META = [
    ("cost_enabled", "COST_ENABLED", "--cost-enabled", lambda v: _serialize_bool(v)),
    ("fixed_bps", "COST_FIXED_BPS", "--fixed-bps", lambda v: str(float(v))),
    ("atr_k", "COST_ATR_K", "--atr-k", lambda v: str(float(v))),
    ("min_hs_bps", "COST_MIN_HS_BPS", "--min-hs-bps", lambda v: str(float(v))),
    ("use_range_impact", "COST_USE_RANGE_IMPACT", "--use-range-impact", lambda v: _serialize_bool(v)),
    (
        "cap_range_impact_bps",
        "CAP_RANGE_IMPACT_BPS",
        "--cap-range-impact-bps",
        lambda v: str(float(v)),
    ),
    (
        "per_trade_fee_usd",
        "FEE_PER_TRADE_USD",
        "--per-trade-fee-usd",
        lambda v: str(float(v)),
    ),
    (
        "exec_delay_bars",
        "EXEC_DELAY_BARS",
        "--exec-delay-bars",
        lambda v: str(int(v) if v is not None else 0),
    ),
    (
        "exec_fill_where",
        "EXEC_FILL_WHERE",
        "--exec-fill-where",
        lambda v: str(v),
    ),
]


def _build_env_overrides(args: argparse.Namespace) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    for attr, env_key, flag, serializer in _ENV_OVERRIDE_META:
        cli_provided = _arg_was_provided(flag)
        env_present = env_key in os.environ
        if cli_provided or not env_present:
            value = getattr(args, attr)
            overrides[env_key] = serializer(value)
    # Maintain legacy compatibility for ATR_PER_TRADE_FEE without clobbering
    # explicit environment overrides. (PH1.2)
    if "FEE_PER_TRADE_USD" in overrides:
        if _arg_was_provided("--per-trade-fee-usd") or "ATR_PER_TRADE_FEE" not in os.environ:
            overrides.setdefault("ATR_PER_TRADE_FEE", overrides["FEE_PER_TRADE_USD"])
    return overrides


def _effective_cost_knobs(args: argparse.Namespace) -> Dict[str, float | bool]:
    def _resolve(attr: str, env_key: str, flag: str, cast, fallback_env: str | None = None):
        if _arg_was_provided(flag):
            source = getattr(args, attr)
        elif env_key in os.environ:
            source = os.environ[env_key]
        elif fallback_env and fallback_env in os.environ:
            source = os.environ[fallback_env]
        else:
            source = getattr(args, attr)
        return cast(source)

    return {
        "atr_k": float(_resolve("atr_k", "COST_ATR_K", "--atr-k", float)),
        "min_half_spread_bps": float(
            _resolve("min_hs_bps", "COST_MIN_HS_BPS", "--min-hs-bps", float)
        ),
        "use_range_impact": bool(
            _coerce_bool(
                _resolve("use_range_impact", "COST_USE_RANGE_IMPACT", "--use-range-impact", lambda v: v)
            )
        ),
        "cap_range_impact_bps": float(
            _resolve("cap_range_impact_bps", "CAP_RANGE_IMPACT_BPS", "--cap-range-impact-bps", float)
        ),
        "fee_bps": float(_resolve("fixed_bps", "COST_FIXED_BPS", "--fixed-bps", float)),
        "per_trade_fee_usd": float(
            _resolve(
                "per_trade_fee_usd",
                "FEE_PER_TRADE_USD",
                "--per-trade-fee-usd",
                float,
                fallback_env="ATR_PER_TRADE_FEE",
            )
        ),
    }


@contextmanager
def _temporary_env(overrides: Dict[str, str]) -> Iterator[None]:
    previous: Dict[str, str | None] = {}
    try:
        for key, value in overrides.items():
            previous[key] = os.environ.get(key)
            os.environ[key] = value
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _execute_smoke(args: argparse.Namespace, cost_knobs: Dict[str, float | bool]) -> None:
    setup_logging()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Effective cost knobs: {cost_knobs}")

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    tickers_requested = list(tickers)
    start_requested = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)

    if args.max_bars:
        approx_days = max(int(args.max_bars), 1)
        candidate_start = end - pd.Timedelta(days=approx_days)
        if candidate_start > start_requested:
            start_requested = candidate_start

    start = start_requested

    trades_frames: List[pd.DataFrame] = []
    equity_post_series: List[pd.Series] = []
    equity_pre_series: List[pd.Series] = []
    metadata: List[Dict] = []

    failures: List[Dict[str, str]] = []
    skip_run = False
    skip_reason = ""
    cache_missing: List[str] = []
    loop_tickers = list(tickers)
    if args.cache_only:
        cache_map = {sym: _has_cached_data(sym) for sym in tickers}
        loop_tickers = [sym for sym, has in cache_map.items() if has]
        cache_missing = [sym for sym, has in cache_map.items() if not has]
        if cache_missing:
            failures.extend({"symbol": sym, "error": "cache-missing"} for sym in cache_missing)
            missing_str = ", ".join(cache_missing)
            print(f"Cache-only mode: missing cached data for {missing_str}.")
        if not loop_tickers:
            skip_run = True
            skip_reason = (
                f"cache-only: missing cached data for {', '.join(cache_missing) or 'requested symbols'}"
            )

    tickers_processed: List[str] = []

    if not skip_run:
        for symbol in loop_tickers:
            try:
                result = backtest_single(
                    symbol=symbol,
                    start=start,
                    end=end,
                    breakout_n=20,
                    exit_n=10,
                    atr_n=14,
                    starting_equity=100_000.0,
                    atr_multiple=2.0,
                    k_atr_buffer=args.k_atr_buffer,
                    persist_n=max(1, args.persist_n),
                    per_trade_fee=float(args.per_trade_fee_usd),
                    enable_costs=bool(args.cost_enabled),
                    delay_bars=max(0, args.exec_delay_bars),
                    execution=args.exec_fill_where,
                )
            except Exception as exc:  # pragma: no cover - defensive path
                failures.append({"symbol": symbol, "error": str(exc)})
                if args.cache_only:
                    skip_run = True
                    skip_reason = f"cache-only: {exc}"
                    break
                continue
            tickers_processed.append(symbol)
            metadata.append(result.get("meta", {}))
            trades_df = result.get("trades_df")
            if isinstance(trades_df, pd.DataFrame) and not trades_df.empty:
                trades_frames.append(trades_df.copy())
            equity_post = result.get("equity")
            if isinstance(equity_post, pd.Series) and len(equity_post):
                equity_post_series.append(equity_post.astype(float))
            equity_pre = result.get("equity_pre_cost")
            if isinstance(equity_pre, pd.Series) and len(equity_pre):
                equity_pre_series.append(equity_pre.astype(float))

    combined_trades = pd.concat(trades_frames, ignore_index=True) if trades_frames else pd.DataFrame()
    required_cols = [
        "time",
        "symbol",
        "qty",
        "price_before",
        "price_after",
        "notional",
        "slip_bps",
        "fees_bps",
    ]
    for col in required_cols:
        if col not in combined_trades.columns:
            dtype = "datetime64[ns]" if col == "time" else ("object" if col == "symbol" else float)
            combined_trades[col] = pd.Series(dtype=dtype)

    equity_post_total = _aggregate_equity(equity_post_series)
    equity_pre_total = _aggregate_equity(equity_pre_series)

    cost_summary = summarize_costs(
        combined_trades if not combined_trades.empty else None,
        equity_pre_total,
        equity_post_total,
    )

    ir_post = None
    tracking_error_post = None
    maxdd_post = None
    if equity_post_total is not None and len(equity_post_total):
        returns_post = equity_post_total.pct_change().dropna()
        if len(returns_post):
            mean_ret = returns_post.mean()
            std_ret = returns_post.std(ddof=0)
            if std_ret != 0:
                ir_post = float(mean_ret / std_ret * np.sqrt(TRADING_DAYS))
            tracking_error_post = float(std_ret * np.sqrt(TRADING_DAYS))
        maxdd_post = float(max_drawdown(equity_post_total))

    counter_keys = [
        "entry_count",
        "exit_count",
        "blocked_by_buffer",
        "blocked_by_persistence",
        "blocked_by_min_hold",
        "blocked_by_cooldown",
    ]
    counters = {key: 0 for key in counter_keys}
    for meta in metadata:
        runtime = meta.get("runtime_counters", {}) if isinstance(meta, dict) else {}
        for key in counter_keys:
            counters[key] += int(runtime.get(key, 0) or 0)

    cost_inputs_effective = {
        "atr_k": float(cost_knobs["atr_k"]),
        "min_half_spread_bps": float(cost_knobs["min_half_spread_bps"]),
        "use_range_impact": bool(cost_knobs["use_range_impact"]),
        "cap_range_impact_bps": float(cost_knobs["cap_range_impact_bps"]),
        "fee_bps": float(cost_knobs.get("fee_bps", 0.0)),
        "per_trade_fee_usd": float(cost_knobs.get("per_trade_fee_usd", 0.0)),
    }
    for meta in metadata:  # PH1.2: reflect the actual knobs observed in engine metadata
        if not isinstance(meta, dict):
            continue
        meta_inputs = meta.get("cost_inputs")
        if not isinstance(meta_inputs, dict):
            continue
        for key in cost_inputs_effective:
            if key not in meta_inputs:
                continue
            value = meta_inputs[key]
            if key == "use_range_impact":
                cost_inputs_effective[key] = bool(value)
            else:
                try:
                    cost_inputs_effective[key] = float(value)
                except (TypeError, ValueError):
                    continue

    summary = {
        "inputs": {
            "tickers": tickers_requested,
            "start": str(start),
            "end": str(end),
            "k_atr_buffer": float(args.k_atr_buffer),
            "persist_n": int(max(1, args.persist_n)),
            "min_hold_days": int(max(0, args.min_hold_days)),
            "reentry_cooldown_days": int(max(0, args.reentry_cooldown_days)),
            "cost_enabled": int(args.cost_enabled),
            "fixed_bps": float(args.fixed_bps),
            "atr_k": cost_inputs_effective["atr_k"],  # PH1.2
            "min_half_spread_bps": cost_inputs_effective["min_half_spread_bps"],  # PH1.2
            "use_range_impact": cost_inputs_effective["use_range_impact"],  # PH1.2
            "cap_range_impact_bps": cost_inputs_effective["cap_range_impact_bps"],  # PH1.2
            "fee_bps": cost_inputs_effective["fee_bps"],
            "per_trade_fee_usd": cost_inputs_effective["per_trade_fee_usd"],
            "exec_delay_bars": int(max(0, args.exec_delay_bars)),
            "exec_fill_where": args.exec_fill_where,
            "seed": int(args.seed),
            "label": args.label,
            "fast_mode": bool(args.fast),
            "cache_only": bool(args.cache_only),
            "max_bars": int(args.max_bars) if args.max_bars else None,
            "tickers_processed": tickers_processed,
        },
        "metrics": {
            "pre_cost_cagr": _safe_float(cost_summary.get("pre_cost_cagr")),
            "post_cost_cagr": _safe_float(cost_summary.get("post_cost_cagr")),
            "annualized_drag_bps": _safe_float(cost_summary.get("annualized_drag_bps")),
            "IR_post": _safe_float(ir_post),
            "TE_post": _safe_float(tracking_error_post),
            "MaxDD_post": _safe_float(maxdd_post),
            "turnover_gross": _safe_float(cost_summary.get("turnover_gross")),
            "turnover_ratio": _safe_float(cost_summary.get("turnover_ratio")),
            "turnover_multiple": _safe_float(cost_summary.get("turnover_multiple")),
            "slippage_bps_weighted": _safe_float(cost_summary.get("slippage_bps_weighted")),
            "weighted_slippage_bps": _safe_float(cost_summary.get("weighted_slippage_bps")),
            "fees_bps_weighted": _safe_float(cost_summary.get("fees_bps_weighted")),
            "weighted_fees_bps": _safe_float(cost_summary.get("weighted_fees_bps")),
            "cost_per_turnover_bps": _safe_float(cost_summary.get("cost_per_turnover_bps")),
            "sharpe_gross": _safe_float(cost_summary.get("sharpe_gross")),
            "sharpe_net": _safe_float(cost_summary.get("sharpe_net")),
            "cagr_gross": _safe_float(cost_summary.get("cagr_gross")),
            "cagr_net": _safe_float(cost_summary.get("cagr_net")),
        },
        "counters": counters,
        "failures": failures,
    }
    summary["skipped"] = bool(skip_run)
    if skip_run:
        summary["skip_reason"] = skip_reason or "cache data missing"
    if cache_missing:
        summary["cache_missing"] = cache_missing

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    label = args.label.strip().replace(" ", "_") or f"seed{args.seed}"
    base_dir = Path("artifacts") / "smoke"
    base_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{timestamp}_{label}"
    run_dir = base_dir / run_name
    suffix = 1
    while run_dir.exists():
        run_dir = base_dir / f"{run_name}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_path = run_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    trades_path = run_dir / "trades.csv"
    combined_trades.to_csv(trades_path, index=False)

    log_source = Path("storage") / "logs" / "engine.log"
    log_target = run_dir / "engine.log"
    if log_source.exists():
        try:
            log_target.write_bytes(log_source.read_bytes())
        except Exception:
            log_target.write_text("", encoding="utf-8")
    else:
        log_target.write_text("", encoding="utf-8")

    if skip_run and args.cache_only:
        cache_note = run_dir / "cache_status.txt"
        cache_note.write_text("cache missing\n", encoding="utf-8")
        print(f"Smoke run skipped (cache-only). Artifacts saved to {run_dir} | reason={skip_reason}")
        return

    post_cagr = summary["metrics"].get("post_cost_cagr")
    drag_bps = summary["metrics"].get("annualized_drag_bps")
    print(
        f"Smoke run saved to {run_dir} | trades={len(combined_trades)} | post_cost_cagr={post_cagr} | drag_bps={drag_bps}"
    )


def _aggregate_equity(series_list: List[pd.Series]) -> pd.Series | None:
    if not series_list:
        return None
    eq_df = pd.concat(series_list, axis=1)
    eq_df = eq_df.ffill().dropna(how="all")  # PH1.2
    aggregated = eq_df.sum(axis=1)
    return aggregated.astype(float)


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def main() -> None:
    args = _parse_args()
    _apply_fast_defaults(args)
    if args.fast:
        os.environ.setdefault("SMOKE_FAST", "1")
        os.environ.setdefault("LOG_TRADES_SAMPLE", os.getenv("LOG_TRADES_SAMPLE", "10"))
        os.environ.setdefault("LOG_TRADES_HEAD", os.getenv("LOG_TRADES_HEAD", "50"))
    os.environ["SMOKE_CACHE_ONLY"] = "1" if args.cache_only else os.environ.get("SMOKE_CACHE_ONLY", "0")
    if args.max_bars is not None and args.max_bars < 0:
        args.max_bars = None
    os.environ.setdefault("LOG_LEVEL", os.getenv("LOG_LEVEL", "INFO"))

    env_overrides = _build_env_overrides(args)
    cost_knobs = _effective_cost_knobs(args)

    with _temporary_env(env_overrides):
        _execute_smoke(args, cost_knobs)



if __name__ == "__main__":
    main()

# Manual QA checklist:
#   1. Baseline: python scripts/run_smoke_backtest.py --k-atr-buffer 0.0 --persist-n 1 --cost-enabled 1
#   2. Tighter:  python scripts/run_smoke_backtest.py --k-atr-buffer 0.25 --persist-n 3 --cost-enabled 1
#      Expect turnover, slippage_bps_weighted, and annualized_drag_bps to decrease while IR_post remains stable or improves.
