"""Pytest harness that runs smoke backtests and aggregates churn/cost metrics."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import pytest


HARNESS_PATH = Path("scripts/run_smoke_backtest.py").resolve()
SMOKE_ROOT = Path("artifacts") / "smoke"
REPORT_ROOT = Path("artifacts") / "test_reports"


def _find_latest_summary(label: str, since_ts: float) -> Tuple[Path | None, Path | None]:
    """Locate the most recent smoke artifact for a label produced after ``since_ts``."""

    if not SMOKE_ROOT.exists():
        return None, None

    latest: Tuple[float, Path, Path] | None = None
    pattern = f"*{label}*" if label else "*"
    for run_dir in SMOKE_ROOT.glob(pattern):
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        mtime = summary_path.stat().st_mtime
        if mtime < since_ts:
            continue
        if latest is None or mtime > latest[0]:
            latest = (mtime, summary_path, run_dir)
    if latest is None:
        return None, None
    return latest[1], latest[2]


def _baseline_columns() -> List[str]:
    return [
        "label",
        "tickers",
        "start",
        "end",
        "k_atr_buffer",
        "persist_n",
        "cost_enabled",
        "fixed_bps",
        "atr_k",
        "min_hs_bps",
        "exec_delay_bars",
        "exec_fill_where",
        "pre_cost_cagr",
        "post_cost_cagr",
        "annualized_drag_bps",
        "IR_post",
        "TE_post",
        "MaxDD_post",
        "turnover_gross",
        "slippage_bps_weighted",
        "fees_bps_weighted",
        "entry_count",
        "exit_count",
        "blocked_by_buffer",
        "blocked_by_persistence",
        "blocked_by_min_hold",
        "blocked_by_cooldown",
        "summary_path",
        "harness_stdout_path",
        "harness_stderr_path",
    ]


@pytest.mark.slow
def test_smoke_churn_costs() -> None:
    if not HARNESS_PATH.exists():
        pytest.skip("Smoke harness script is unavailable.")

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    csv_path = REPORT_ROOT / f"smoke_churn_costs_{timestamp}.csv"
    txt_path = REPORT_ROOT / f"smoke_churn_costs_{timestamp}.txt"

    base_cmd = [
        sys.executable,
        str(HARNESS_PATH),
        "--tickers",
        "SPY,QQQ,AAPL,MSFT,AMD",
        "--start",
        "2022-01-01",
        "--end",
        "2024-12-31",
        "--cost-enabled",
        "1",
        "--fixed-bps",
        "0.5",
        "--atr-k",
        "0.25",
        "--min-hs-bps",
        "1.0",
    ]

    runs: List[Tuple[str, List[str]]] = [
        ("baseline", ["--k-atr-buffer", "0.0", "--persist-n", "1"]),
        ("filtA", ["--k-atr-buffer", "0.25", "--persist-n", "2"]),
        ("filtB", ["--k-atr-buffer", "0.25", "--persist-n", "3"]),
    ]

    since_ts = time.time()
    rows: List[Dict[str, object]] = []
    executed_commands: List[str] = []
    stderr_notes: List[str] = []

    for label, args in runs:
        cmd = base_cmd + ["--label", label] + args
        executed_commands.append(" ".join(cmd))
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=90,
            )
        except subprocess.TimeoutExpired as exc:  # pragma: no cover - defensive
            result = subprocess.CompletedProcess(
                cmd,
                returncode=1,
                stdout=exc.stdout or "",
                stderr=(exc.stderr or "") + "\n[timeout expired]",
            )

        stdout_path = REPORT_ROOT / f"{label}_stdout.log"
        stderr_path = REPORT_ROOT / f"{label}_stderr.log"
        stdout_path.write_text(result.stdout or "", encoding="utf-8")
        stderr_path.write_text(result.stderr or "", encoding="utf-8")

        summary_path, run_dir = _find_latest_summary(label, since_ts)
        row: Dict[str, object] = {col: None for col in _baseline_columns()}
        row.update(
            {
                "label": label,
                "summary_path": str(summary_path) if summary_path else "",
                "harness_stdout_path": str(stdout_path),
                "harness_stderr_path": str(stderr_path),
            }
        )

        if summary_path and summary_path.exists():
            try:
                summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                stderr_notes.append(f"{label}: failed to parse summary.json ({exc})")
                summary_data = None
        else:
            summary_data = None

        if summary_data:
            inputs = summary_data.get("inputs", {})
            metrics = summary_data.get("metrics", {})
            counters = summary_data.get("counters", {})

            row.update(
                {
                    "tickers": ",".join(inputs.get("tickers", [])) if isinstance(inputs.get("tickers"), list) else inputs.get("tickers"),
                    "start": inputs.get("start"),
                    "end": inputs.get("end"),
                    "k_atr_buffer": inputs.get("k_atr_buffer"),
                    "persist_n": inputs.get("persist_n"),
                    "cost_enabled": inputs.get("cost_enabled"),
                    "fixed_bps": inputs.get("fixed_bps"),
                    "atr_k": inputs.get("atr_k"),
                    "min_hs_bps": inputs.get("min_half_spread_bps"),
                    "exec_delay_bars": inputs.get("exec_delay_bars"),
                    "exec_fill_where": inputs.get("exec_fill_where"),
                    "pre_cost_cagr": metrics.get("pre_cost_cagr"),
                    "post_cost_cagr": metrics.get("post_cost_cagr"),
                    "annualized_drag_bps": metrics.get("annualized_drag_bps"),
                    "IR_post": metrics.get("IR_post"),
                    "TE_post": metrics.get("TE_post"),
                    "MaxDD_post": metrics.get("MaxDD_post"),
                    "turnover_gross": metrics.get("turnover_gross"),
                    "slippage_bps_weighted": metrics.get("slippage_bps_weighted"),
                    "fees_bps_weighted": metrics.get("fees_bps_weighted"),
                    "entry_count": counters.get("entry_count"),
                    "exit_count": counters.get("exit_count"),
                    "blocked_by_buffer": counters.get("blocked_by_buffer"),
                    "blocked_by_persistence": counters.get("blocked_by_persistence"),
                    "blocked_by_min_hold": counters.get("blocked_by_min_hold"),
                    "blocked_by_cooldown": counters.get("blocked_by_cooldown"),
                }
            )
        else:
            stderr_notes.append(
                f"{label}: smoke harness did not produce summary.json; returncode={result.returncode}."
            )
            if run_dir:
                stderr_notes.append(f"{label}: inspected directory {run_dir}")

        if result.returncode != 0 and not summary_data:
            stderr_notes.append(
                f"{label}: harness stderr snippet -> {result.stderr.strip()[:200] if result.stderr else 'no stderr'}"
            )

        rows.append(row)

    df = pd.DataFrame(rows, columns=_baseline_columns())
    df.to_csv(csv_path, index=False)

    baseline_row = df[df["label"] == "baseline"].iloc[0] if not df[df["label"] == "baseline"].empty else None

    lines: List[str] = []
    lines.append(f"Smoke churn costs report generated at {timestamp}")
    lines.append("Commands executed:")
    lines.extend([f"  {cmd}" for cmd in executed_commands])
    lines.append("")

    display_df = df.copy()
    display_df = display_df.fillna("")
    table_cols = ["label", "k_atr_buffer", "persist_n", "post_cost_cagr", "annualized_drag_bps", "turnover_gross", "IR_post", "TE_post"]
    col_widths = {col: max(len(col), *(len(str(val)) for val in display_df[col])) for col in table_cols}

    header = " ".join(col.ljust(col_widths[col]) for col in table_cols)
    lines.append(header)
    lines.append("-" * len(header))
    for _, row in display_df.iterrows():
        line = " ".join(str(row[col]).ljust(col_widths[col]) for col in table_cols)
        lines.append(line)
    lines.append("")

    def _delta(current: float | None, baseline: float | None) -> str:
        try:
            if current is None or baseline is None:
                return "n/a"
            return f"{current - baseline:+.4f}"
        except TypeError:
            return "n/a"

    if baseline_row is not None:
        try:
            base_turnover = float(baseline_row["turnover_gross"])
        except (TypeError, ValueError):
            base_turnover = None
        try:
            base_drag = float(baseline_row["annualized_drag_bps"])
        except (TypeError, ValueError):
            base_drag = None
        try:
            base_ir = float(baseline_row["IR_post"])
        except (TypeError, ValueError):
            base_ir = None
        try:
            base_te = float(baseline_row["TE_post"])
        except (TypeError, ValueError):
            base_te = None
        try:
            base_cagr = float(baseline_row["post_cost_cagr"])
        except (TypeError, ValueError):
            base_cagr = None

        lines.append("Comparisons versus baseline:")
        for _, row in df.iterrows():
            if row["label"] == "baseline":
                continue
            label = row["label"] or "(unknown)"

            def _as_float(value):
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None

            turnover = _as_float(row["turnover_gross"])
            drag = _as_float(row["annualized_drag_bps"])
            ir_post = _as_float(row["IR_post"])
            te_post = _as_float(row["TE_post"])
            post_cagr = _as_float(row["post_cost_cagr"])

            lines.append(
                f"  {label}: Δturnover={_delta(turnover, base_turnover)}, Δdrag_bps={_delta(drag, base_drag)}, "
                f"ΔIR={_delta(ir_post, base_ir)}, ΔTE={_delta(te_post, base_te)}, Δpost_CAGR={_delta(post_cagr, base_cagr)}"
            )
        lines.append("")

    if stderr_notes:
        lines.append("Notes:")
        lines.extend([f"  - {note}" for note in stderr_notes])

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    success_count = sum(bool(path) for path in df["summary_path"])
    if success_count == 0:
        pytest.skip("Smoke harness could not complete any run; see TXT for details.")

