# pages/5_EA_Train_Test_Inspector.py
from __future__ import annotations

import json
import html
import math
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import warnings

from src.utils.tri_panel import render_tri_panel

# ---------- settings ----------
LOG_DIR = Path("storage/logs/ea")
DEFAULT_PAGE_TITLE = "EA Train/Test Inspector"
SHOW_COST_KPIS = True


# --- UI hardening: prevent button label wrapping globally ---
st.markdown(
    """
    <style>
    div.stButton > button { white-space: nowrap; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    .ea-cost-kpi-row { display: flex; gap: 0.75rem; flex-wrap: wrap; }
    .ea-cost-kpi-chip {
        background-color: var(--secondary-background-color, #f0f2f6);
        padding: 0.75rem 1rem;
        border-radius: 0.75rem;
        min-width: 160px;
        box-shadow: inset 0 0 0 1px rgba(0, 0, 0, 0.05);
    }
    .ea-cost-kpi-chip-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: rgba(49, 51, 63, 0.6);
        margin-bottom: 0.15rem;
    }
    .ea-cost-kpi-chip-value {
        font-size: 1.15rem;
        font-weight: 600;
        color: rgba(49, 51, 63, 0.95);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    .ea-trade-metric-row { display: flex; flex-wrap: wrap; gap: 0.75rem; margin-top: 0.5rem; }
    .ea-trade-metric-chip {
        background-color: var(--secondary-background-color, #f0f2f6);
        padding: 0.75rem 1rem;
        border-radius: 0.75rem;
        min-width: 160px;
        box-shadow: inset 0 0 0 1px rgba(0, 0, 0, 0.05);
    }
    .ea-trade-metric-chip-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: rgba(49, 51, 63, 0.6);
        margin-bottom: 0.15rem;
    }
    .ea-trade-metric-chip-value {
        font-size: 1.15rem;
        font-weight: 600;
        color: rgba(49, 51, 63, 0.95);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- small debug buffer ----------
def _dbg(msg: str):
    try:
        st.session_state.setdefault("ea_inspector_debug", []).append(str(msg))
    except Exception:
        pass

# ---------- helpers ----------

# --- display-only helper ---
def _ymd(x) -> str:
    """Return YYYY-MM-DD for display only; do not mutate the original value."""
    if x is None or x == "":
        return ""
    try:
        return pd.to_datetime(x, utc=True).strftime("%Y-%m-%d")
    except Exception:
        return str(x)

def _latest_log_file(dirpath: Path) -> Optional[Path]:
    if not dirpath.exists():
        return None
    files = sorted(dirpath.glob("*_ea.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None

def load_ea_log(log_path: str) -> pd.DataFrame:
    rows = []
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return pd.DataFrame(rows)

def _get_session_meta(df: pd.DataFrame) -> Dict[str, Any]:
    smeta_rows = (
        df[df["event"] == "session_meta"]["payload"].apply(pd.Series)
        if ("event" in df.columns and (df["event"] == "session_meta").any())
        else pd.DataFrame()
    )
    return smeta_rows.iloc[-1].to_dict() if not smeta_rows.empty else {}

def _get_holdout_meta(df: pd.DataFrame) -> Dict[str, Any]:
    hmeta_rows = (
        df[df["event"] == "holdout_meta"]["payload"].apply(pd.Series)
        if ("event" in df.columns and (df["event"] == "holdout_meta").any())
        else pd.DataFrame()
    )
    # last one wins
    return hmeta_rows.iloc[-1].to_dict() if not hmeta_rows.empty else {}

def _sanitize_obj_cols(df: pd.DataFrame) -> pd.DataFrame:
    """JSON-encode any dict/list columns (prevents hashing errors later)."""
    if df.empty:
        return df
    out = df.copy()
    for col in list(out.columns):
        if out[col].map(lambda x: isinstance(x, (dict, list))).any():
            out[col] = out[col].apply(lambda x: json.dumps(x, sort_keys=True) if isinstance(x, (dict, list)) else x)
    return out

def _eval_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return tidy table of individual evaluations with metrics.
    NOTE: JSON-encode 'params' and any dict/list columns to avoid caching/hashing issues.
    """
    if "event" not in df.columns:
        return pd.DataFrame()
    evals = df[df["event"] == "individual_evaluated"]["payload"].apply(pd.Series).reset_index(drop=True)
    if evals.empty:
        return pd.DataFrame()
    metrics = pd.json_normalize(evals["metrics"]).reset_index(drop=True)
    out = pd.concat([evals.drop(columns=["metrics"]), metrics], axis=1)
    # Avoid dicts; keep a JSON copy of params
    if "params" in out.columns:
        out["params_json"] = out["params"].apply(
            lambda d: json.dumps(d, sort_keys=True) if isinstance(d, dict) else (d if isinstance(d, str) else "{}")
        )
        out = out.drop(columns=["params"])
    out = _sanitize_obj_cols(out)
    # Make sure 'gen' is numeric
    if "gen" in out.columns:
        out["gen"] = pd.to_numeric(out["gen"], errors="coerce").fillna(0).astype(int)
    return out

def _gen_end_table(df: pd.DataFrame) -> pd.DataFrame:
    if "event" not in df.columns:
        return pd.DataFrame()
    g = df[df["event"] == "generation_end"]["payload"].apply(pd.Series)
    g = _sanitize_obj_cols(g)
    return g.reset_index(drop=True) if not g.empty else pd.DataFrame()

def _row_params(row: pd.Series) -> Dict[str, Any]:
    pj = row.get("params_json")
    try:
        return json.loads(pj) if isinstance(pj, str) else {}
    except Exception:
        return {}


def _strategy_module_name(strategy_dotted: str) -> str:
    if not strategy_dotted:
        return ""
    if ":" in strategy_dotted:
        return strategy_dotted.split(":", 1)[0]
    return strategy_dotted


def _import_strategy_runner(strategy_dotted: str):
    module_name = _strategy_module_name(strategy_dotted)
    if not module_name:
        return None
    try:
        module = import_module(module_name)
    except Exception as exc:
        _dbg(f"trades: failed to import {module_name}: {exc}")
        return None
    runner = getattr(module, "run_strategy", None)
    if runner is None:
        _dbg(f"trades: module {module_name} missing run_strategy")
    return runner


def _coerce_timestamp(value: Any) -> Optional[pd.Timestamp]:
    if value is None or value == "":
        return None
    try:
        ts = pd.to_datetime(value, utc=False)
    except Exception:
        return None
    if isinstance(ts, pd.Series):
        ts = ts.iloc[0] if not ts.empty else None
    if ts is None or pd.isna(ts):
        return None
    return pd.Timestamp(ts)


@st.cache_data(show_spinner=True, hash_funcs={dict: lambda d: json.dumps(d, sort_keys=True)})
def load_trades(
    strategy_dotted: str,
    tickers: List[str],
    start_iso: str,
    end_iso: str,
    starting_equity: float,
    params: Dict[str, Any],
) -> pd.DataFrame:
    runner = _import_strategy_runner(strategy_dotted)
    if runner is None:
        return pd.DataFrame()

    start_ts = _coerce_timestamp(start_iso)
    end_ts = _coerce_timestamp(end_iso)
    if start_ts is None or end_ts is None:
        return pd.DataFrame()

    params_payload = dict(params or {})
    params_payload.setdefault("model_key", strategy_dotted)

    if isinstance(tickers, str):
        tickers = [t.strip() for t in tickers.split(",") if t.strip()]
    symbols = [str(t).strip().upper() for t in tickers if str(t).strip()]
    if not symbols:
        return pd.DataFrame()

    records: List[Dict[str, Any]] = []

    for sym in symbols:
        try:
            result = runner(sym, start_ts, end_ts, starting_equity, params_payload)
        except Exception as exc:  # pragma: no cover - defensive UI helper
            _dbg(f"trades: {sym} run_strategy error {type(exc).__name__}: {exc}")
            continue
        trades = result.get("trades") if isinstance(result, dict) else None
        if isinstance(trades, list) and trades:
            for trade in trades:
                if isinstance(trade, dict):
                    rec = dict(trade)
                    rec["symbol"] = sym
                    records.append(rec)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    for col in ["entry_time", "exit_time", "signal_time", "exit_signal_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _best_row_for_gen(eval_df: pd.DataFrame, gen_idx: Optional[int]) -> Optional[pd.Series]:
    if gen_idx is None or eval_df.empty or "gen" not in eval_df.columns:
        return None
    try:
        gen_slice = eval_df[eval_df["gen"] == int(gen_idx)]
    except Exception:
        return None
    if gen_slice.empty or "total_return" not in gen_slice.columns:
        return None
    try:
        idx = gen_slice["total_return"].idxmax()
    except Exception:
        return None
    try:
        return gen_slice.loc[idx]
    except Exception:
        return None


def _first_float(row: pd.Series, keys: List[str]) -> Optional[float]:
    for key in keys:
        if key in row:
            val = row.get(key)
            if val is None:
                continue
            try:
                fval = float(val)
            except Exception:
                continue
            if math.isfinite(fval) and not pd.isna(fval):
                return fval
    return None


def _alpha_retention_ratio(row: pd.Series) -> Optional[float]:
    sharpe_pre = _first_float(
        row,
        [
            "sharpe_pre_cost",
            "sharpe_pre",
            "sharpe_gross",
            "sharpe_before_cost",
        ],
    )
    sharpe_post = _first_float(
        row,
        [
            "sharpe_post_cost",
            "sharpe_post",
            "sharpe_net",
        ],
    )
    if sharpe_pre is not None and abs(sharpe_pre) > 1e-12 and sharpe_post is not None:
        ratio = sharpe_post / sharpe_pre
        if math.isfinite(ratio):
            return ratio

    cagr_pre = _first_float(row, ["pre_cost_cagr", "cagr_pre", "cagr_gross", "cagr"])
    cagr_post = _first_float(row, ["post_cost_cagr", "cagr_post", "cagr_net"])
    if cagr_pre is not None and abs(cagr_pre) > 1e-12 and cagr_post is not None:
        ratio = cagr_post / cagr_pre
        if math.isfinite(ratio):
            return ratio
    return None


_PERFORMANCE_METRIC_TOOLTIPS = {
    "Sharpe (post-cost)": "Risk-adjusted return that already reflects slippage and fees. Higher is better.",
    "Sharpe (pre-cost)": "Risk-adjusted return before accounting for trading costs. Provides context for the raw edge.",
    "CAGR (post-cost)": "Annualized growth after trading costs. Compare against benchmarks on a net basis.",
    "CAGR (pre-cost)": "Annualized growth before costs. Highlights the gross opportunity before slippage/fees.",
    "Max Drawdown %": "Worst peak-to-trough equity decline. Smaller losses (closer to 0%) imply lower risk.",
    "Win Rate %": "Percent of trades that were profitable. Needs to be viewed alongside payoff ratios.",
    "Expectancy": "Average profit per trade. Positive expectancy is required for a sustainable edge.",
    "Edge Ratio": "Average win size versus loss size. Values above 1.0 indicate gains outweigh losses.",
    "Profit Factor": "Gross profits divided by gross losses. >1.0 indicates net profitability.",
    "Trades": "Number of trades evaluated for this individual. Larger samples lend greater confidence.",
}


_COST_KPI_TOOLTIPS = {
    "Alpha Retention %": (
        "How much of your edge survives costs. 100% means costs had no impact; "
        "70–95% is typical for liquid, low-churn setups. Higher is better."
    ),
    "Annualized Drag (bps/yr)": (
        "Per-year performance lost to slippage + fees. <50 bps/yr is good on very liquid symbols; "
        ">100 bps/yr is costly. Lower is better."
    ),
    "Weighted Slippage (bps)": (
        "Average execution penalty per trade (in basis points), weighted by fills. For SPY/QQQ, ~5–15 bps is typical. "
        "Lower is better."
    ),
    "Turnover (×/yr)": (
        "How many portfolio ‘turns’ per year. Higher turnover usually raises costs unless alpha improves proportionally. "
        "Context-dependent."
    ),
    "Cost per Turnover (bps per 1×)": (
        "Cost burden normalized by trading activity. <20 bps per 1× is healthy on liquid names. Lower is better."
    ),
}


def _format_percent(value: float, digits: int = 0) -> Optional[str]:
    try:
        fval = float(value)
    except Exception:
        return None
    if not math.isfinite(fval) or pd.isna(fval):
        return None
    scaled = fval * 100.0 if abs(fval) <= 1.0 else fval
    return f"{scaled:.{digits}f}%"


def _format_float(value: float, digits: int = 2) -> Optional[str]:
    try:
        fval = float(value)
    except Exception:
        return None
    if not math.isfinite(fval) or pd.isna(fval):
        return None
    return f"{fval:.{digits}f}"


def _format_int(value: float) -> Optional[str]:
    try:
        fval = float(value)
    except Exception:
        return None
    if not math.isfinite(fval) or pd.isna(fval):
        return None
    return f"{int(round(fval)):,}"


def _collect_performance_metric_rows(best_row: pd.Series) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    def _append(label: str, keys: List[str], formatter) -> None:
        val = _first_float(best_row, keys)
        if val is None:
            return
        formatted = formatter(val)
        if not formatted:
            return
        rows.append(
            {
                "Metric": label,
                "Value": formatted,
                "Guidance": _PERFORMANCE_METRIC_TOOLTIPS.get(label, ""),
            }
        )

    _append("Sharpe (post-cost)", ["sharpe_post_cost", "sharpe_post", "sharpe_net"], lambda v: _format_float(v, 2))
    _append("Sharpe (pre-cost)", ["sharpe_pre_cost", "sharpe_pre", "sharpe_gross", "sharpe_before_cost"], lambda v: _format_float(v, 2))
    _append("CAGR (post-cost)", ["post_cost_cagr", "cagr_post", "cagr_net"], lambda v: _format_percent(v, 1))
    _append("CAGR (pre-cost)", ["pre_cost_cagr", "cagr_pre", "cagr_gross", "cagr"], lambda v: _format_percent(v, 1))
    _append("Max Drawdown %", ["max_drawdown", "max_dd", "max_drawdown_pct"], lambda v: _format_percent(v, 1))
    _append("Win Rate %", ["win_rate", "winrate", "pct_winners"], lambda v: _format_percent(v, 0))
    _append("Expectancy", ["expectancy"], lambda v: _format_float(v, 2))
    _append("Edge Ratio", ["edge_ratio"], lambda v: _format_float(v, 2))
    _append("Profit Factor", ["profit_factor"], lambda v: _format_float(v, 2))
    _append("Trades", ["num_trades", "trades"], _format_int)

    return rows


def _collect_cost_metric_rows(best_row: pd.Series) -> List[Dict[str, str]]:
    alpha_ratio = _alpha_retention_ratio(best_row)
    drag_bps = _first_float(best_row, ["annualized_drag_bps", "annualized_drag"])
    if drag_bps is not None and "annualized_drag" in best_row and "annualized_drag_bps" not in best_row:
        drag_bps = drag_bps * 10_000.0
    slip_bps = _first_float(
        best_row,
        [
            "slippage_bps_weighted",
            "weighted_slippage_bps",
            "slip_bps_weighted",
        ],
    )
    turnover_ratio = _first_float(best_row, ["turnover_ratio", "turnover"])

    if (
        drag_bps is not None
        and turnover_ratio is not None
        and abs(turnover_ratio) > 1e-12
    ):
        cost_per_turnover = drag_bps / turnover_ratio
    else:
        cost_per_turnover = None

    kpis: List[Dict[str, str]] = []

    def _push(label: str, value: Optional[str]) -> None:
        if not value:
            return
        kpis.append(
            {
                "label": label,
                "value": value,
                "tooltip": _COST_KPI_TOOLTIPS.get(label, ""),
            }
        )

    if alpha_ratio is not None:
        _push("Alpha Retention %", _format_percent(alpha_ratio, 0))

    if drag_bps is not None:
        _push("Annualized Drag (bps/yr)", f"{drag_bps:.0f} bps/yr")

    if slip_bps is not None:
        slip_fmt = f"{slip_bps:.1f}" if abs(slip_bps) < 10 else f"{slip_bps:.0f}"
        _push("Weighted Slippage (bps)", f"{slip_fmt} bps")

    if turnover_ratio is not None:
        _push("Turnover (×/yr)", f"{turnover_ratio:.2f}×/yr")

    if cost_per_turnover is not None:
        _push("Cost per Turnover (bps per 1×)", f"{cost_per_turnover:.0f} bps/1×")

    return kpis


def _render_cost_kpis(kpis: List[Dict[str, str]]) -> None:
    if not kpis:
        return
    chips: List[str] = []
    for item in kpis:
        label_html = html.escape(item["label"], quote=True)
        value_html = html.escape(item["value"], quote=True)
        tooltip_html = html.escape(item.get("tooltip", ""), quote=True)
        chips.append(
            """
            <div class="ea-cost-kpi-chip" title="{tooltip}">
                <div class="ea-cost-kpi-chip-label">{label}</div>
                <div class="ea-cost-kpi-chip-value">{value}</div>
            </div>
            """.format(label=label_html, value=value_html, tooltip=tooltip_html)
        )
    st.markdown(f"<div class='ea-cost-kpi-row'>{''.join(chips)}</div>", unsafe_allow_html=True)


def _summarize_trades(trades: pd.DataFrame) -> List[Dict[str, str]]:
    if trades is None or trades.empty:
        return []

    total = len(trades)
    returns = pd.to_numeric(trades.get("return_pct", pd.Series(dtype=float)), errors="coerce")
    pnl = pd.to_numeric(trades.get("net_pnl", pd.Series(dtype=float)), errors="coerce")
    holding_days = pd.to_numeric(trades.get("holding_days", pd.Series(dtype=float)), errors="coerce")

    wins = returns[returns > 0]
    losses = returns[returns < 0]

    win_count = int((returns > 0).sum())
    loss_count = int((returns < 0).sum())
    breakeven_count = total - win_count - loss_count

    gross_profit = pnl[pnl > 0].sum()
    gross_loss = pnl[pnl < 0].sum()

    def _fmt_pct(value: Optional[float], digits: int = 1) -> Optional[str]:
        if value is None or not math.isfinite(value):
            return None
        return f"{value * 100.0:.{digits}f}%"

    def _fmt_float(value: Optional[float], digits: int = 2) -> Optional[str]:
        if value is None or not math.isfinite(value):
            return None
        return f"{value:.{digits}f}"

    expectancy = returns.mean() if not returns.empty else None
    avg_win = wins.mean() if not wins.empty else None
    avg_loss = losses.mean() if not losses.empty else None
    payoff = None
    if avg_win is not None and avg_loss is not None and avg_loss != 0:
        payoff = abs(avg_win / avg_loss)

    avg_hold = holding_days.mean() if not holding_days.empty else None
    total_pnl = pnl.sum() if not pnl.empty else None
    avg_trade_pnl = pnl.mean() if not pnl.empty else None
    profit_factor = None
    if gross_loss is not None and gross_loss != 0:
        profit_factor = abs(gross_profit / gross_loss)

    rows = [
        {
            "label": "Trades",
            "value": f"{total:,}",
            "tooltip": "Total executed trades in the selected window.",
        },
        {
            "label": "Net P&L",
            "value": _fmt_float(total_pnl, 2) or "–",
            "tooltip": "Sum of trade-level net P&L (strategy units).",
        },
        {
            "label": "Avg Trade P&L",
            "value": _fmt_float(avg_trade_pnl, 2) or "–",
            "tooltip": "Mean net P&L per trade (strategy units).",
        },
        {
            "label": "Win Rate",
            "value": _fmt_pct((win_count / total) if total else None, 1) or "–",
            "tooltip": "Winning trades divided by total trades.",
        },
        {
            "label": "Expectancy",
            "value": _fmt_pct(expectancy, 2) or "–",
            "tooltip": "Average return per trade (%%).",
        },
        {
            "label": "Avg Hold (days)",
            "value": _fmt_float(avg_hold, 2) or "–",
            "tooltip": "Mean holding period in days.",
        },
        {
            "label": "Wins",
            "value": f"{win_count:,}",
            "tooltip": "Count of trades with positive net return.",
        },
        {
            "label": "Losses",
            "value": f"{loss_count:,}",
            "tooltip": "Count of trades with negative net return.",
        },
        {
            "label": "Avg Win",
            "value": _fmt_pct(avg_win, 2) or "–",
            "tooltip": "Average return (%%) on winning trades.",
        },
        {
            "label": "Avg Loss",
            "value": _fmt_pct(avg_loss, 2) or "–",
            "tooltip": "Average return (%%) on losing trades.",
        },
        {
            "label": "Payoff Ratio",
            "value": _fmt_float(payoff, 2) or "–",
            "tooltip": "Absolute avg win divided by absolute avg loss.",
        },
        {
            "label": "Profit Factor",
            "value": _fmt_float(profit_factor, 2) or "–",
            "tooltip": "Gross profit divided by absolute gross loss.",
        },
    ]

    if breakeven_count:
        rows.append(
            {
                "label": "Breakeven",
                "value": f"{breakeven_count:,}",
                "tooltip": "Trades with near-zero return.",
            }
        )

    if pd.notna(gross_profit) and gross_profit:
        rows.append(
            {
                "label": "Gross Profit",
                "value": _fmt_float(gross_profit, 2) or "–",
                "tooltip": "Sum of P&L from profitable trades.",
            }
        )

    if pd.notna(gross_loss) and gross_loss:
        rows.append(
            {
                "label": "Gross Loss",
                "value": _fmt_float(gross_loss, 2) or "–",
                "tooltip": "Sum of P&L from losing trades.",
            }
        )

    return rows


def _render_trade_metrics(metrics: List[Dict[str, str]]) -> None:
    if not metrics:
        return
    chips: List[str] = []
    for item in metrics:
        label_html = html.escape(item.get("label", ""), quote=True)
        value_html = html.escape(item.get("value", ""), quote=True)
        tooltip_html = html.escape(item.get("tooltip", ""), quote=True)
        chips.append(
            (
                "<div class=\"ea-trade-metric-chip\" title=\"{tooltip}\">"
                "<div class=\"ea-trade-metric-chip-label\">{label}</div>"
                "<div class=\"ea-trade-metric-chip-value\">{value}</div>"
                "</div>"
            ).format(label=label_html, value=value_html, tooltip=tooltip_html)
        )
    st.markdown(f"<div class='ea-trade-metric-row'>{''.join(chips)}</div>", unsafe_allow_html=True)


def _prepare_trade_table(trades: pd.DataFrame) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame()

    df = trades.copy()
    df = df.sort_values(by=[col for col in ["entry_time", "exit_time"] if col in df.columns])

    def _col(name: str, default=None):
        return df[name] if name in df.columns else pd.Series([default] * len(df))

    table = pd.DataFrame(
        {
            "Window": _col("window"),
            "Symbol": _col("symbol"),
            "Side": _col("side"),
            "Entry": _col("entry_time"),
            "Exit": _col("exit_time"),
            "Holding Days": _col("holding_days").astype(float),
            "Entry Price": _col("entry_price").astype(float),
            "Exit Price": _col("exit_price").astype(float),
            "Net P&L": _col("net_pnl").astype(float),
            "Return %": _col("return_pct").astype(float) * 100.0,
        }
    )

    for col in ["Entry", "Exit"]:
        if col in table.columns:
            table[col] = pd.to_datetime(table[col], errors="coerce")

    return table


def _render_metric_dashboard(best_row: Optional[pd.Series]) -> None:
    if best_row is None:
        return

    perf_rows = _collect_performance_metric_rows(best_row)
    cost_rows: List[Dict[str, str]] = []
    if SHOW_COST_KPIS:
        cost_rows = _collect_cost_metric_rows(best_row)

    col_count = int(bool(perf_rows)) + int(bool(cost_rows))
    if col_count == 0:
        return

    st.markdown("#### Metrics Dashboard")
    cols = st.columns(col_count)
    idx = 0

    if perf_rows:
        perf_df = pd.DataFrame(perf_rows)
        with cols[idx]:
            st.markdown("**Performance Metrics**")
            st.dataframe(
                perf_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Metric": st.column_config.TextColumn(
                        "Metric",
                        help="Performance indicator for the selected individual.",
                    ),
                    "Value": st.column_config.TextColumn("Value"),
                    "Guidance": st.column_config.TextColumn(
                        "Guidance",
                        help="How to interpret this performance metric.",
                    ),
                },
            )
        idx += 1

    if cost_rows:
        with cols[idx]:
            st.markdown("**Costs Impact**")
            _render_cost_kpis(cost_rows)

# ---------- equity provider ----------

@st.cache_data(show_spinner=True, hash_funcs={dict: lambda d: json.dumps(d, sort_keys=True)})
def run_equity_curve(
    strategy_dotted: str,
    tickers: List[str],
    start_iso: str,
    end_iso: str,
    starting_equity: float,
    params: Dict[str, Any],
) -> pd.DataFrame:
    """
    Produce an equity curve [date,equity] for the window using the same path the EA uses.
    Tries general_trainer first. If no curve is found, tries multiple signatures against
    src.utils.holdout_chart (or fallback holdout_chart). Emits debug breadcrumbs.
    """
    # Normalize tickers
    if isinstance(tickers, str):
        tickers = [t.strip() for t in tickers.split(",") if t.strip()]
    tickers = list(tickers)

    # 1) general_trainer
    try:
        from src.models.general_trainer import train_general_model
        _dbg("trainer: calling train_general_model(...)")
        res = train_general_model(strategy_dotted, tickers, start_iso, end_iso, starting_equity, params)
        if isinstance(res, dict):
            agg = res.get("aggregate") or {}
            for key in ("equity_curve", "curve", "equity"):
                curve = agg.get(key)
                # list-of-pairs
                if isinstance(curve, list) and curve and isinstance(curve[0], (list, tuple)):
                    df = pd.DataFrame(curve, columns=["date", "equity"])
                    df["date"] = pd.to_datetime(df["date"])
                    _dbg(f"trainer: aggregate.{key} list-of-pairs")
                    return df
                # dict-of-arrays
                if isinstance(curve, dict) and {"date", "equity"}.issubset(curve.keys()):
                    df = pd.DataFrame({"date": pd.to_datetime(curve["date"]), "equity": curve["equity"]})
                    _dbg(f"trainer: aggregate.{key} dict-of-arrays")
                    return df
            p = res.get("portfolio")
            if isinstance(p, dict) and {"date", "equity"}.issubset(p.keys()):
                df = pd.DataFrame({"date": pd.to_datetime(p["date"]), "equity": p["equity"]})
                _dbg("trainer: portfolio{date,equity}")
                return df
        _dbg("trainer: no curve fields found")
    except Exception as e:
        _dbg(f"trainer: exception {type(e).__name__}: {e}")

    # 2) holdout runner (same module Strategy Adapter uses)
    hc = None
    try:
        try:
            from src.utils import holdout_chart as hc_mod
            hc = hc_mod
            _dbg("holdout: using src.utils.holdout_chart")
        except Exception:
            import holdout_chart as hc_mod  # root module fallback
            hc = hc_mod
            _dbg("holdout: using root holdout_chart")
    except Exception as e:
        _dbg(f"holdout: import failed {type(e).__name__}: {e}")

    if hc is not None:
        # Candidate function names and signature variants
        candidates = [
            ("holdout_equity", ("params", "start", "end", "tickers", "starting_equity", "strategy")),
            ("simulate_holdout", ("params", "start", "end", "tickers", "starting_equity", "strategy")),
            ("run_holdout", ("params", "start", "end", "tickers", "starting_equity", "strategy")),
        ]
        args = {
            "params": params,
            "start": start_iso,
            "end": end_iso,
            "tickers": tickers,
            "starting_equity": starting_equity,
            "strategy": strategy_dotted,
        }
        for fn_name, _sig in candidates:
            if hasattr(hc, fn_name):
                fn = getattr(hc, fn_name)
                try:
                    from inspect import signature
                    sig = signature(fn)
                    kwargs = {k: v for k, v in args.items() if k in sig.parameters}
                    _dbg(f"holdout: calling {fn_name} with {list(kwargs.keys())}")
                    ec = fn(**kwargs)
                    if isinstance(ec, tuple):
                        ec = ec[0]
                    if isinstance(ec, pd.DataFrame) and {"date", "equity"}.issubset(ec.columns):
                        df = ec[["date", "equity"]].copy()
                        df["date"] = pd.to_datetime(df["date"])
                        return df
                    if isinstance(ec, list) and ec and isinstance(ec[0], (list, tuple)):
                        df = pd.DataFrame(ec, columns=["date", "equity"])
                        df["date"] = pd.to_datetime(df["date"])
                        return df
                    _dbg(f"holdout: {fn_name} returned no usable curve")
                except Exception as e:
                    _dbg(f"holdout: {fn_name} raised {type(e).__name__}: {e}")

    # 3) last resort: flat line (UI warns)
    _dbg("fallback: flat 2-point line returned")
    return pd.DataFrame({"date": pd.to_datetime([start_iso, end_iso]), "equity": [starting_equity, starting_equity]})

# ---------- plotting ----------

def _plot_gen_topK(
    eval_df: pd.DataFrame,
    gen_idx: int,
    k: int,
    strategy: str,
    tickers: List[str],
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    starting_equity: float,
) -> go.Figure:
    G = eval_df[eval_df["gen"] == gen_idx].copy()
    if G.empty:
        return go.Figure()
    # rank by final return in that gen
    G = G.sort_values(by="total_return", ascending=False).head(min(k, len(G)))

    fig = go.Figure()
    flat_warn = False
    for _, row in G.iterrows():
        params = _row_params(row)
        ec_train = run_equity_curve(strategy, tickers, train_start, train_end, starting_equity, params)
        end_equity = ec_train["equity"].iloc[-1] if not ec_train.empty else starting_equity
        ec_test = run_equity_curve(strategy, tickers, test_start, test_end, end_equity, params)
        ec = pd.concat([ec_train, ec_test], ignore_index=True)
        if len(ec) <= 2 or ec["equity"].nunique() <= 1:
            flat_warn = True
        name = f"gen{gen_idx} idx{int(row['idx'])} (ret {row['total_return']:.3f})"
        fig.add_trace(
            go.Scatter(x=ec["date"], y=ec["equity"], mode="lines", name=name, line=dict(width=1))
        )

    if flat_warn:
        warnings.warn(
            "Equity curve(s) appear flat; the trainer didn't return a curve. Wire your holdout/trainer curve provider in run_equity_curve()."
        )

    # all-flat annotation
    if len(fig.data) > 0:
        all_flat = all((len(t.y) <= 2 or (np.max(t.y) - np.min(t.y) == 0)) for t in fig.data)
        if all_flat:
            fig.add_annotation(
                text="No equity curves returned by simulator. Check run_equity_curve wiring.",
                showarrow=False, xref="paper", yref="paper", x=0.01, y=0.95,
                bgcolor="#ffeeee", bordercolor="#cc0000",
            )

    fig.add_vline(x=pd.to_datetime(train_end), line_width=2, line_dash="dot", line_color="#888")
    fig.update_layout(
        title=f"Gen {gen_idx}: Top-{k} by final return (train + test)",
        xaxis_title="Date",
        yaxis_title="Equity",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=420,
    )
    return fig

def _plot_leaders_through_gen(
    eval_df: pd.DataFrame,
    upto_gen: int,
    strategy: str,
    tickers: List[str],
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    starting_equity: float,
) -> go.Figure:
    fig = go.Figure()
    all_gens = sorted(eval_df["gen"].unique())
    for g in [g for g in all_gens if g <= upto_gen]:
        G = eval_df[eval_df["gen"] == g]
        if G.empty:
            continue
        row = G.loc[G["total_return"].idxmax()]  # best by return in that gen
        params = _row_params(row)
        ec_train = run_equity_curve(strategy, tickers, train_start, train_end, starting_equity, params)
        end_equity = ec_train["equity"].iloc[-1] if not ec_train.empty else starting_equity
        ec_test = run_equity_curve(strategy, tickers, test_start, test_end, end_equity, params)
        ec = pd.concat([ec_train, ec_test], ignore_index=True)
        name = f"Gen {g} (ret {row['total_return']:.3f})"
        fig.add_trace(go.Scatter(x=ec["date"], y=ec["equity"], mode="lines", name=name, line=dict(width=1)))

    # all-flat annotation
    if len(fig.data) > 0:
        all_flat = all((len(t.y) <= 2 or (np.max(t.y) - np.min(t.y) == 0)) for t in fig.data)
        if all_flat:
            fig.add_annotation(
                text="No equity curves returned by simulator. Check run_equity_curve wiring.",
                showarrow=False, xref="paper", yref="paper", x=0.01, y=0.95,
                bgcolor="#ffeeee", bordercolor="#cc0000",
            )

    fig.add_vline(x=pd.to_datetime(train_end), line_width=2, line_dash="dot", line_color="#888")
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Equity",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=420,
    )
    return fig

# ---------- UI ----------

def _file_picker():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(LOG_DIR.glob("*_ea.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    labels = [f.name for f in files]
    if not files:
        st.error(f"No EA logs found in {LOG_DIR}")
        return None
    idx = st.selectbox("EA log", options=list(range(len(files))), format_func=lambda i: labels[i], index=0)
    return files[idx]

def main():
    st.set_page_config(page_title=DEFAULT_PAGE_TITLE, layout="wide")
    st.title(DEFAULT_PAGE_TITLE)

    # pick a log (default to latest)
    log_file = _file_picker() or _latest_log_file(LOG_DIR)
    if not log_file:
        st.stop()

    df = load_ea_log(str(log_file))
    eval_df = _eval_table(df)
    gen_df = _gen_end_table(df)

    # session + holdout meta
    smeta = _get_session_meta(df)
    hmeta = _get_holdout_meta(df)

    # Session meta is emitted by the EA run (preferred source).  When unavailable,
    # fall back to anything provided in the holdout metadata payload so the page
    # still has reasonable defaults even for older/partial logs.
    strategy = smeta.get("strategy") or hmeta.get("strategy") or "src.models.atr_breakout:Strategy"
    tickers = smeta.get("tickers") or hmeta.get("tickers") or []
    starting_equity = float(
        smeta.get("starting_equity")
        or hmeta.get("starting_equity")
        or 10000.0
    )
    train_start = smeta.get("train_start") or hmeta.get("train_start")
    train_end = smeta.get("train_end") or hmeta.get("train_end")

    # Controls row
    c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 2, 2])

    with c1:
        st.markdown("**Training window**")
        st.text_input(
            "Train start",
            value=_ymd(train_start) if train_start else "",
            key="ea_disp_train_start",
            disabled=True,
        )
        st.text_input(
            "Train end",
            value=_ymd(train_end) if train_end else "",
            key="ea_disp_train_end",
            disabled=True,
        )

    with c2:
        st.markdown("**Test window**")
        # default from holdout_meta, else next day after train_end
        default_test_start = hmeta.get("holdout_start")
        if not default_test_start and train_end:
            try:
                default_test_start = (pd.to_datetime(train_end) + pd.Timedelta(days=1)).date().isoformat()
            except Exception:
                default_test_start = ""
        # set underlying variables (full ISO) and show compact display fields
        test_start = str(default_test_start) if default_test_start else ""
        default_test_end = hmeta.get("holdout_end")
        test_end = str(default_test_end) if default_test_end else ""

        st.text_input(
            "Test start",
            value=_ymd(test_start) if test_start else "",
            key="ea_disp_test_start",
            disabled=True,
        )
        st.text_input(
            "Test end",
            value=_ymd(test_end) if test_end else "",
            key="ea_disp_test_end",
            disabled=True,
        )

        # Ensure empty user inputs fall back to holdout metadata when possible
        if not test_start:
            test_start = str(default_test_start) if default_test_start else ""
        if not test_end:
            test_end = str(default_test_end) if default_test_end else ""

    with c3:
        st.markdown("**Top-K (current gen)**")
        top_k = st.number_input("K", min_value=1, max_value=50, value=5, step=1)

    with c4:
        st.markdown("**Playback**")
        max_gen = int(eval_df["gen"].max()) if not eval_df.empty else 0
        if "ea_inspect_gen" not in st.session_state:
            st.session_state.ea_inspect_gen = 0

        # Wider buttons with a thin spacer; labels use non-breaking space to resist weird wraps
        prev_col, spacer, next_col = st.columns([1, 0.2, 1])

        with prev_col:
            if st.button("⟵ Prev", key="ea_prev", use_container_width=True):  # note the NBSP between words
                st.session_state.ea_inspect_gen = max(0, st.session_state.ea_inspect_gen - 1)

        with next_col:
            if st.button("Next ⟶", key="ea_next", use_container_width=True):
                st.session_state.ea_inspect_gen = min(max_gen, st.session_state.ea_inspect_gen + 1)

        st.caption(f"Max gen in log: {max_gen}")

    with c5:
        st.markdown("**Generation**")
        st.session_state.ea_inspect_gen = st.slider(
            "Gen",
            0,
            int(eval_df["gen"].max() if not eval_df.empty else 0),
            int(st.session_state.ea_inspect_gen),
        )

    try:
        current_gen = int(st.session_state.get("ea_inspect_gen", 0))
    except Exception:
        current_gen = None

    best_row = _best_row_for_gen(eval_df, current_gen)
    _render_metric_dashboard(best_row)

    # safety
    if not train_start or not train_end:
        st.warning("Training dates missing. Enter train_start/train_end (ISO) or run a new EA with session_meta logging.")
        st.stop()
    if not test_start or not test_end:
        st.info("Tip: set a test window to visualize out-of-sample performance (e.g., next 60–90 days).")

    st.markdown("### Individual Trade Review")
    gen_slice = eval_df[eval_df["gen"] == current_gen].copy() if current_gen is not None else pd.DataFrame()
    if gen_slice.empty:
        st.info("No individuals recorded for this generation in the EA log.")
    else:
        gen_slice = gen_slice.sort_values(by=["score", "total_return"], ascending=[False, False])
        option_records: List[Dict[str, Any]] = []
        for df_idx, row in gen_slice.iterrows():
            trades_cnt = int(row.get("trades", 0) or 0)
            score_val = float(row.get("score", 0.0) or 0.0)
            if not math.isfinite(score_val):
                score_val = 0.0
            ret_val = float(row.get("total_return", 0.0) or 0.0)
            if not math.isfinite(ret_val):
                ret_val = 0.0
            individual_id = int(row.get("idx", df_idx) or 0)
            label = (
                f"Idx {individual_id} • score {score_val:.3f} • ret {ret_val:.3f} • trades {trades_cnt}"
            )
            option_records.append(
                {
                    "label": label,
                    "df_index": df_idx,
                    "individual_id": individual_id,
                }
            )

        search_query = st.text_input(
            "Search individual by ID",
            value="",
            key="ea_trade_individual_search",
            placeholder="e.g., 12 for Idx 12",
        ).strip()

        filtered_records = option_records
        if search_query:
            lowered = search_query.lower()
            filtered_records = [rec for rec in option_records if lowered in rec["label"].lower()]
            if lowered.isdigit():
                target_id = int(lowered)
                id_matches = [rec for rec in option_records if rec["individual_id"] == target_id]
                if id_matches:
                    filtered_records = id_matches

        if not filtered_records:
            st.info("No individuals matched that search. Showing all entries.")
            filtered_records = option_records

        option_labels = [rec["label"] for rec in filtered_records]
        default_index = 0
        selected_label = st.selectbox(
            "Individual (current generation)",
            options=option_labels,
            index=min(default_index, max(len(option_labels) - 1, 0)),
            key="ea_trade_individual",
        )
        selected_idx = next(
            (rec["df_index"] for rec in filtered_records if rec["label"] == selected_label),
            None,
        )
        selected_row = gen_slice.loc[selected_idx] if selected_idx is not None else None

        if selected_row is None:
            st.warning("Select an individual to review their trades.")
        else:
            params = _row_params(selected_row)
            if not params:
                st.warning("Parameters missing for this individual; cannot regenerate trades.")
            else:
                train_trades = load_trades(
                    strategy,
                    tickers,
                    train_start,
                    train_end,
                    starting_equity,
                    params,
                )

                train_curve = run_equity_curve(
                    strategy,
                    tickers,
                    train_start,
                    train_end,
                    starting_equity,
                    params,
                )
                if not train_curve.empty and "equity" in train_curve:
                    end_equity = float(train_curve["equity"].iloc[-1])
                else:
                    end_equity = float(starting_equity)

                test_trades = pd.DataFrame()
                if test_start and test_end:
                    test_trades = load_trades(
                        strategy,
                        tickers,
                        test_start,
                        test_end,
                        end_equity,
                        params,
                    )

                window_frames: Dict[str, pd.DataFrame] = {}
                if not train_trades.empty:
                    window_frames["Train"] = train_trades.assign(window="Train")
                if not test_trades.empty:
                    window_frames["Test"] = test_trades.assign(window="Test")
                if len(window_frames) > 1:
                    combined = pd.concat(window_frames.values(), ignore_index=True)
                    sort_cols = [col for col in ["entry_time", "exit_time"] if col in combined.columns]
                    if sort_cols:
                        combined = combined.sort_values(by=sort_cols)
                    window_frames["Combined"] = combined

                if not window_frames:
                    st.info("No trades were generated for the selected individual and windows.")
                else:
                    view_label = st.radio(
                        "Data window",
                        list(window_frames.keys()),
                        index=0,
                        horizontal=True,
                        key="ea_trade_window",
                    )
                    trades_df = window_frames.get(view_label, pd.DataFrame())
                    metrics = _summarize_trades(trades_df)
                    _render_trade_metrics(metrics)

                    table = _prepare_trade_table(trades_df)
                    if table.empty:
                        st.info("Trades could not be formatted for display.")
                    else:
                        st.dataframe(
                            table,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Holding Days": st.column_config.NumberColumn("Holding Days", format="%.2f"),
                                "Entry Price": st.column_config.NumberColumn("Entry Price", format="%.2f"),
                                "Exit Price": st.column_config.NumberColumn("Exit Price", format="%.2f"),
                                "Net P&L": st.column_config.NumberColumn("Net P&L", format="%.2f"),
                                "Return %": st.column_config.NumberColumn("Return %", format="%.2f%%"),
                            },
                        )

    # ---- Chart 1: top-K of current generation (by return) ----
    st.subheader("Chart 1 — Top-K of current generation (by final return)")
    fig1 = _plot_gen_topK(
        eval_df=eval_df,
        gen_idx=int(st.session_state.ea_inspect_gen),
        k=int(top_k),
        strategy=strategy,
        tickers=tickers,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        starting_equity=starting_equity,
    )
    st.plotly_chart(fig1, use_container_width=True)

    # ---- Chart 2: leaders up to the current generation ----
    st.subheader("Chart 2 — Best-by-return per generation (up to current gen)")
    fig2 = _plot_leaders_through_gen(
        eval_df=eval_df,
        upto_gen=int(st.session_state.ea_inspect_gen),
        strategy=strategy,
        tickers=tickers,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        starting_equity=starting_equity,
    )
    st.plotly_chart(fig2, use_container_width=True)

    tri_curve = pd.Series(dtype=float)
    if best_row is not None and current_gen is not None:
        try:
            params = _row_params(best_row)
            ec_train = run_equity_curve(
                strategy,
                tickers,
                train_start,
                train_end,
                starting_equity,
                params,
            )
            end_equity = ec_train["equity"].iloc[-1] if not ec_train.empty else starting_equity
            ec_test = run_equity_curve(
                strategy,
                tickers,
                test_start,
                test_end,
                end_equity,
                params,
            )
            ec = pd.concat([ec_train, ec_test], ignore_index=True)
            if {"date", "equity"}.issubset(ec.columns):
                ec = ec.dropna(subset=["date", "equity"])
                if not ec.empty:
                    ec["date"] = pd.to_datetime(ec["date"])
                    ec = ec.sort_values("date").drop_duplicates(subset=["date"])
                    tri_curve = ec.set_index("date")["equity"]
        except Exception as tri_err:  # pragma: no cover - defensive UI helper
            _dbg(f"tri_panel: {type(tri_err).__name__}: {tri_err}")

    render_tri_panel(
        tri_curve,
        test_start=test_start,
        test_end=test_end,
        portfolio_tickers=tickers,
    )

    # Debug trace from the equity provider
    with st.expander("Debug: equity provider trace", expanded=False):
        msgs = st.session_state.get("ea_inspector_debug", [])
        if msgs:
            st.code("\n".join(msgs))
        else:
            st.caption("No debug messages yet.")
        if hmeta:
            st.caption("Holdout metadata snapshot from EA log:")
            st.json(hmeta)

    st.caption(
        "A vertical dotted line marks the end of the training window. "
        "Curves continue into the test window on the same scale."
    )

if __name__ == "__main__":
    main()