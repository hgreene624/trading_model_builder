#!/usr/bin/env python3
"""Interactive cost troubleshooting harness for Phase 1.2 tuning."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # PH1.2
    sys.path.insert(0, str(ROOT))  # PH1.2

from src.backtest.engine import ATRParams, backtest_atr_breakout  # PH1.2

_COST_ENV_KEYS = (
    "COST_ENABLED",
    "COST_FIXED_BPS",
    "COST_ATR_K",
    "COST_MIN_HS_BPS",
    "COST_USE_RANGE_IMPACT",
    "CAP_RANGE_IMPACT_BPS",
)


@dataclass(slots=True)
class ScenarioSpec:
    """Parameter overrides for a troubleshooting scenario."""

    label: str
    k_atr_buffer: float
    persist_n: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run targeted ATR breakout backtests and report raw cost attribution metrics.",
    )
    parser.add_argument("--tickers", default="SPY", help="Comma-delimited list of symbols to evaluate.")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2023-06-30")
    parser.add_argument("--starting-equity", type=float, default=100_000.0)
    parser.add_argument("--breakout-n", type=int, default=20)
    parser.add_argument("--exit-n", type=int, default=10)
    parser.add_argument("--atr-n", type=int, default=14)
    parser.add_argument("--atr-multiple", type=float, default=2.0)
    parser.add_argument("--k-atr-buffer", type=float, default=0.0)
    parser.add_argument("--persist-n", type=int, default=1)
    parser.add_argument("--execution", choices=["close", "next_open"], default="close")
    parser.add_argument("--delay-bars", type=int, default=0)
    parser.add_argument("--cost-enabled", type=int, choices=[0, 1], default=1)
    parser.add_argument("--cost-fixed-bps", type=float, default=0.5)
    parser.add_argument("--cost-atr-k", type=float, default=0.05)
    parser.add_argument("--cost-min-hs-bps", type=float, default=0.5)
    parser.add_argument("--cost-use-range-impact", type=int, choices=[0, 1], default=0)
    parser.add_argument("--cap-range-impact-bps", type=float, default=10.0)
    parser.add_argument(
        "--include-filtA",
        action="store_true",
        help="Run an additional scenario with a modest breakout buffer.",
    )
    parser.add_argument(
        "--include-filtB",
        action="store_true",
        help="Run an additional scenario with increased persistence.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts") / "troubleshoot" / "troubleshoot_costs.json",
        help="Where to write the JSON report.",
    )
    return parser.parse_args()


def _scenario_variants(args: argparse.Namespace) -> List[ScenarioSpec]:
    base = [
        ScenarioSpec(
            label="base",
            k_atr_buffer=float(args.k_atr_buffer),
            persist_n=max(1, int(args.persist_n)),
        )
    ]
    if args.include_filtA:
        base.append(
            ScenarioSpec(
                label="filtA",
                k_atr_buffer=float(args.k_atr_buffer) + 0.1,
                persist_n=max(1, int(args.persist_n)),
            )
        )
    if args.include_filtB:
        base.append(
            ScenarioSpec(
                label="filtB",
                k_atr_buffer=float(args.k_atr_buffer),
                persist_n=max(1, int(args.persist_n) + 1),
            )
        )
    return base


def _set_cost_env(args: argparse.Namespace) -> Dict[str, Optional[str]]:
    """Apply Phase 1.2 env knobs, returning the previous environment snapshot."""

    previous = {key: os.environ.get(key) for key in _COST_ENV_KEYS}
    os.environ["COST_ENABLED"] = str(int(bool(args.cost_enabled)))
    os.environ["COST_FIXED_BPS"] = str(float(args.cost_fixed_bps))
    os.environ["COST_ATR_K"] = str(float(args.cost_atr_k))
    os.environ["COST_MIN_HS_BPS"] = str(float(args.cost_min_hs_bps))
    os.environ["COST_USE_RANGE_IMPACT"] = str(int(bool(args.cost_use_range_impact)))
    os.environ["CAP_RANGE_IMPACT_BPS"] = str(float(args.cap_range_impact_bps))
    return previous


def _restore_env(snapshot: Dict[str, Optional[str]]) -> None:
    for key, value in snapshot.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _env_snapshot() -> Dict[str, float | int]:
    snap: Dict[str, float | int] = {}
    for key in _COST_ENV_KEYS:
        val = os.environ.get(key)
        if val is None:
            continue
        if key in {"COST_ENABLED", "COST_USE_RANGE_IMPACT"}:
            snap[key] = int(val)
        else:
            try:
                snap[key] = float(val)
            except ValueError:
                snap[key] = val  # fall back to raw str for unexpected data
    return snap


def _cost_inputs_from_meta(meta: Dict) -> Dict[str, float | int | bool]:
    cost_inputs = meta.get("cost_inputs", {}) if isinstance(meta, dict) else {}
    normalized: Dict[str, float | int | bool] = {}
    for key in ("atr_k", "min_half_spread_bps", "use_range_impact", "cap_range_impact_bps"):
        if key in cost_inputs:
            value = cost_inputs[key]
            if key == "use_range_impact":
                normalized[key] = bool(value)
            else:
                try:
                    normalized[key] = float(value)
                except (TypeError, ValueError):
                    continue
    return normalized


def _summarize_result(meta: Dict, trades: Iterable[Dict]) -> Dict[str, object]:
    cost_summary = ((meta or {}).get("costs") or {}).get("summary") or {}
    runtime = (meta or {}).get("runtime_counters", {}) or {}
    slip_bps = cost_summary.get("weighted_slippage_bps")
    drag_bps = cost_summary.get("annualized_drag_bps", cost_summary.get("annualized_drag"))
    if drag_bps is None:
        drag_bps = 0.0
    else:
        try:
            drag_bps = float(drag_bps)
            if abs(drag_bps) < 1e-6 and cost_summary.get("annualized_drag") is not None:
                drag_bps = float(cost_summary["annualized_drag"]) * 10_000.0
        except (TypeError, ValueError):
            drag_bps = 0.0
    try:
        slip_bps = float(slip_bps)
    except (TypeError, ValueError):
        slip_bps = 0.0

    turnover_gross = cost_summary.get("turnover_gross", 0.0)
    try:
        turnover_gross = float(turnover_gross)
    except (TypeError, ValueError):
        turnover_gross = 0.0

    result_summary = {
        "slip_bps": slip_bps,
        "drag_bps": drag_bps,
        "turnover_gross": turnover_gross,
        "turnover_ratio": float(cost_summary.get("turnover_ratio", 0.0) or 0.0),
        "trades": len(list(trades)),
        "entry_count": int(runtime.get("entry_count", 0) or 0),
        "exit_count": int(runtime.get("exit_count", 0) or 0),
        "blocks": {
            "buffer": int(runtime.get("blocked_by_buffer", 0) or 0),
            "persistence": int(runtime.get("blocked_by_persistence", 0) or 0),
            "min_hold": int(runtime.get("blocked_by_min_hold", 0) or 0),
            "cooldown": int(runtime.get("blocked_by_cooldown", 0) or 0),
        },
        "raw_cost_summary": {k: cost_summary[k] for k in cost_summary},
        "cost_inputs": _cost_inputs_from_meta(meta),
    }
    return result_summary


def _format_blocks(blocks: Dict[str, int]) -> str:
    return "buf:{buffer} pers:{persistence} hold:{min_hold} cool:{cooldown}".format(**blocks)


def _render_table(rows: List[Dict[str, object]]) -> str:
    headers = ["symbol", "slip_bps", "drag_bps", "turnover_gross", "trades", "blocks"]
    lines = [" | ".join(headers)]
    lines.append("-+-".join("-" * len(h) for h in headers))
    for row in rows:
        line = " | ".join(
            [
                str(row.get("symbol", "")),
                f"{row.get('slip_bps', 0.0):6.2f}",
                f"{row.get('drag_bps', 0.0):7.2f}",
                f"{row.get('turnover_gross', 0.0):10.2f}",
                f"{row.get('trades', 0):5d}",
                row.get("blocks", ""),
            ]
        )
        lines.append(line)
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prev_env = _set_cost_env(args)
    scenarios = _scenario_variants(args)
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    start_ts = pd.Timestamp(args.start)
    end_ts = pd.Timestamp(args.end)

    report: Dict[str, object] = {
        "generated_at": datetime.utcnow().isoformat(),
        "scenarios": [],
    }

    try:
        for scenario in scenarios:
            env_used = _env_snapshot()
            scenario_rows: List[Dict[str, object]] = []
            scenario_payload: Dict[str, object] = {
                "label": scenario.label,
                "env": env_used,
                "runs": [],
            }
            for symbol in tickers:
                params = ATRParams(
                    breakout_n=int(args.breakout_n),
                    exit_n=int(args.exit_n),
                    atr_n=int(args.atr_n),
                    atr_multiple=float(args.atr_multiple),
                )
                params.k_atr_buffer = float(scenario.k_atr_buffer)
                params.persist_n = int(max(1, scenario.persist_n))
                params.enable_costs = bool(args.cost_enabled)
                params.delay_bars = int(max(0, args.delay_bars))
                try:
                    result = backtest_atr_breakout(
                        symbol=symbol,
                        start=start_ts,
                        end=end_ts,
                        starting_equity=float(args.starting_equity),
                        params=params,
                        execution=args.execution,
                        enable_costs=params.enable_costs,
                        delay_bars=params.delay_bars,
                        capture_trades_df=True,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    scenario_payload["runs"].append(
                        {
                            "symbol": symbol,
                            "error": str(exc),
                        }
                    )
                    scenario_rows.append(
                        {
                            "symbol": symbol,
                            "slip_bps": float("nan"),
                            "drag_bps": float("nan"),
                            "turnover_gross": float("nan"),
                            "trades": 0,
                            "blocks": "error",
                        }
                    )
                    continue

                meta = result.get("meta", {}) if isinstance(result, dict) else {}
                trades = result.get("trades", []) if isinstance(result, dict) else []
                summary = _summarize_result(meta, trades)
                row = {
                    "symbol": symbol,
                    "slip_bps": summary["slip_bps"],
                    "drag_bps": summary["drag_bps"],
                    "turnover_gross": summary["turnover_gross"],
                    "trades": summary["trades"],
                    "blocks": _format_blocks(summary["blocks"]),
                }
                scenario_rows.append(row)
                scenario_payload["runs"].append(
                    {
                        "symbol": symbol,
                        "summary": summary,
                        "runtime_counters": meta.get("runtime_counters", {}),
                        "cost_inputs": summary["cost_inputs"],
                        "phase0_cost_model": meta.get("phase0_cost_model"),
                    }
                )

            report["scenarios"].append(scenario_payload)
            print(f"\nScenario: {scenario.label}")
            print("ENV used per run:")
            for key in _COST_ENV_KEYS:
                if key in env_used:
                    print(f"  {key}={env_used[key]}")
            print()
            print(_render_table(scenario_rows))

    finally:
        _restore_env(prev_env)

    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nJSON report written to {output_path}")


if __name__ == "__main__":
    main()
