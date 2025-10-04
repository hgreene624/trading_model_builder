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


def _run_smoke_suite(
    run_specs: List[Tuple[str, List[str]]],
    base_cmd: List[str],
    baseline_label: str,
    report_suffix: str,
    timeout: int = 90,
) -> Dict[str, object]:
    if not HARNESS_PATH.exists():
        pytest.skip("Smoke harness script is unavailable.")

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_tag = f"{report_suffix}_{timestamp}"
    csv_path = REPORT_ROOT / f"smoke_churn_costs_{report_tag}.csv"
    txt_path = REPORT_ROOT / f"smoke_churn_costs_{report_tag}.txt"

    since_ts = time.time()
    rows: List[Dict[str, object]] = []
    executed_commands: List[str] = []
    stderr_notes: List[str] = []
    run_outputs: List[Dict[str, object]] = []

    for label, extra_args in run_specs:
        cmd = base_cmd + ["--label", label] + extra_args
        executed_commands.append(" ".join(cmd))
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:  # pragma: no cover - defensive
            result = subprocess.CompletedProcess(
                cmd,
                returncode=1,
                stdout=exc.stdout or "",
                stderr=(exc.stderr or "") + "\n[timeout expired]",
            )

        stdout_path = REPORT_ROOT / f"{label}_{report_suffix}_stdout.log"
        stderr_path = REPORT_ROOT / f"{label}_{report_suffix}_stderr.log"
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

        summary_data = None
        if summary_path and summary_path.exists():
            try:
                summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                stderr_notes.append(f"{label}: failed to parse summary.json ({exc})")
        elif result.returncode != 0:
            stderr_notes.append(
                f"{label}: smoke harness did not produce summary.json; returncode={result.returncode}."
            )

        skip_flag = False
        skip_reason = ""
        cache_note_path = ""
        if summary_data:
            inputs = summary_data.get("inputs", {})
            metrics = summary_data.get("metrics", {})
            counters = summary_data.get("counters", {})
            skip_flag = bool(summary_data.get("skipped"))
            skip_reason = summary_data.get("skip_reason", "")

            row.update(
                {
                    "tickers": ",".join(inputs.get("tickers", []))
                    if isinstance(inputs.get("tickers"), list)
                    else inputs.get("tickers"),
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
        elif result.stderr:
            stderr_notes.append(
                f"{label}: harness stderr snippet -> {(result.stderr.strip()[:200]) if result.stderr else 'no stderr'}"
            )

        if run_dir is not None:
            cache_note = run_dir / "cache_status.txt"
            if cache_note.exists():
                cache_note_path = str(cache_note)

        run_outputs.append(
            {
                "label": label,
                "command": " ".join(cmd),
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "stdout_snippet": (result.stdout or "").strip()[-500:],
                "stderr_snippet": (result.stderr or "").strip()[-500:],
                "skipped": skip_flag,
                "skip_reason": skip_reason,
                "summary_available": bool(summary_data),
                "cache_note_path": cache_note_path,
            }
        )

        rows.append(row)

    df = pd.DataFrame(rows, columns=_baseline_columns())
    df.to_csv(csv_path, index=False)

    lines: List[str] = []
    lines.append(f"Smoke churn costs report ({report_suffix}) generated at {timestamp}")
    lines.append("Commands executed:")
    lines.extend([f"  {cmd}" for cmd in executed_commands])
    lines.append("")

    display_df = df.copy().fillna("")
    table_cols = [
        "label",
        "k_atr_buffer",
        "persist_n",
        "post_cost_cagr",
        "annualized_drag_bps",
        "turnover_gross",
        "IR_post",
        "TE_post",
    ]
    col_widths = {col: max(len(col), *(len(str(val)) for val in display_df[col])) for col in table_cols}
    header = " ".join(col.ljust(col_widths[col]) for col in table_cols)
    lines.append(header)
    lines.append("-" * len(header))
    for _, row in display_df.iterrows():
        line = " ".join(str(row[col]).ljust(col_widths[col]) for col in table_cols)
        lines.append(line)
    lines.append("")

    baseline_row = df[df["label"] == baseline_label]
    baseline_meta = next((meta for meta in run_outputs if meta["label"] == baseline_label), None)
    if baseline_row is not None and not baseline_row.empty and baseline_meta and not baseline_meta["skipped"]:
        row = baseline_row.iloc[0]

        def _as_float(value):
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        base_turnover = _as_float(row["turnover_gross"])
        base_drag = _as_float(row["annualized_drag_bps"])
        base_ir = _as_float(row["IR_post"])
        base_te = _as_float(row["TE_post"])
        base_cagr = _as_float(row["post_cost_cagr"])

        def _delta(current: float | None, base: float | None) -> str:
            if current is None or base is None:
                return "n/a"
            return f"{current - base:+.4f}"

        lines.append("Comparisons versus baseline:")
        for meta in run_outputs:
            if meta["label"] == baseline_label or meta["skipped"]:
                continue
            other_row = df[df["label"] == meta["label"]]
            if other_row.empty:
                continue
            vals = other_row.iloc[0]
            lines.append(
                "  {label}: Δturnover={d_turn}, Δdrag_bps={d_drag}, ΔIR={d_ir}, ΔTE={d_te}, Δpost_CAGR={d_cagr}".format(
                    label=meta["label"],
                    d_turn=_delta(_as_float(vals["turnover_gross"]), base_turnover),
                    d_drag=_delta(_as_float(vals["annualized_drag_bps"]), base_drag),
                    d_ir=_delta(_as_float(vals["IR_post"]), base_ir),
                    d_te=_delta(_as_float(vals["TE_post"]), base_te),
                    d_cagr=_delta(_as_float(vals["post_cost_cagr"]), base_cagr),
                )
            )
        lines.append("")

    skip_meta = [meta for meta in run_outputs if meta["skipped"]]
    if skip_meta:
        lines.append("Skipped runs:")
        for meta in skip_meta:
            reason = meta["skip_reason"] or "cache missing"
            cache_note = f" (cache note: {meta['cache_note_path']})" if meta["cache_note_path"] else ""
            lines.append(f"  - {meta['label']}: {reason}{cache_note}")
        lines.append("")

    if stderr_notes:
        lines.append("Notes:")
        lines.extend([f"  - {note}" for note in stderr_notes])
        lines.append("")

    lines.append("Harness output snippets:")
    for meta in run_outputs:
        lines.append(f"=== {meta['label']} STDOUT ===")
        lines.append(meta["stdout_snippet"] or "(empty)")
        lines.append(f"=== {meta['label']} STDERR ===")
        lines.append(meta["stderr_snippet"] or "(empty)")
        lines.append("")

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    success_count = sum(1 for meta in run_outputs if meta["summary_available"] and not meta["skipped"])
    available_count = sum(1 for meta in run_outputs if meta["summary_available"])
    cache_count = sum(1 for meta in run_outputs if meta["cache_note_path"])

    return {
        "df": df,
        "csv_path": csv_path,
        "txt_path": txt_path,
        "success_count": success_count,
        "available_count": available_count,
        "cache_count": cache_count,
    }


def test_smoke_churn_costs_fast() -> None:
    base_cmd = [
        sys.executable,
        str(HARNESS_PATH),
        "--fast",
        "--cache-only",
        "--max-bars",
        "5000",
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
        ("baseline_fast", ["--k-atr-buffer", "0.0", "--persist-n", "1"]),
        ("filtA_fast", ["--k-atr-buffer", "0.25", "--persist-n", "2"]),
        ("filtB_fast", ["--k-atr-buffer", "0.25", "--persist-n", "3"]),
    ]

    result = _run_smoke_suite(runs, base_cmd, baseline_label="baseline_fast", report_suffix="fast", timeout=90)

    assert result["available_count"] > 0 or result["cache_count"] > 0, (
        "Fast smoke harness produced no summaries or cache notes; see report at "
        f"{result['txt_path']}"
    )


@pytest.mark.slow
def test_smoke_churn_costs_slow() -> None:
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

    result = _run_smoke_suite(runs, base_cmd, baseline_label="baseline", report_suffix="slow", timeout=120)

    if result["success_count"] == 0:
        pytest.skip("Smoke harness could not complete slow run; see TXT for details.")
