from importlib import import_module
from typing import Any, Callable, Dict, List

from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.storage import get_benchmark_total_return


# ---- Internal state keys ------------------------------------------------------

_SS_BASE = "hc"  # namespace prefix in st.session_state to avoid collisions

def _ss_key(name: str) -> str:
    return f"{_SS_BASE}:{name}"


# ---- Rendering ----------------------------------------------------------------

def _render(curves: List[Dict[str, Any]], placeholder: st.delta_generator.DeltaGenerator) -> None:
    """Render holdout curves with benchmark + relative performance shading when available."""

    fig = go.Figure()

    if not curves:
        fig.update_layout(
            title="Holdout equity (awaiting Gen 0)",
            margin=dict(l=10, r=10, t=35, b=10),
            showlegend=False,
            xaxis_title="Date",
            yaxis_title="Equity ($)",
        )
        fig.add_trace(go.Scatter(x=[], y=[], mode="lines", showlegend=False))
        _plot(fig, placeholder)
        return

    # Plot historical best curves in grey for context (strategy + optional benchmark).
    if len(curves) > 1:
        for run in curves[:-1]:
            x_vals = _to_datetime_list(run.get("x"))
            strategy_vals = _to_float_array(run.get("equity") or run.get("y"))
            if not x_vals or len(x_vals) != len(strategy_vals):
                continue

            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=[None if np.isnan(v) else float(v) for v in strategy_vals],
                    mode="lines",
                    line=dict(width=1.0, color="rgba(150,150,150,0.55)"),
                    name="",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            bench_vals = _to_float_array(run.get("benchmark"))
            if len(bench_vals) == len(strategy_vals) and np.isfinite(bench_vals).any():
                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=[None if np.isnan(v) else float(v) for v in bench_vals],
                        mode="lines",
                        line=dict(width=1.0, color="rgba(120,120,120,0.35)", dash="dot"),
                        name="",
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

    best = curves[-1]
    best_gen = best.get("gen")
    best_label = best.get("label") or (f"Gen {best_gen}" if best_gen is not None else "Current best")
    best_score = best.get("score")
    legend_name = best_label if best_score is None else f"{best_label} (score={best_score:.3f})"

    x_vals = _to_datetime_list(best.get("x"))
    strategy_vals = _to_float_array(best.get("equity") or best.get("y"))
    bench_vals = _to_float_array(best.get("benchmark"))
    bench_label = best.get("benchmark_label", "Benchmark")

    has_strategy = bool(x_vals) and len(x_vals) == len(strategy_vals)
    has_benchmark = has_strategy and len(bench_vals) == len(strategy_vals) and np.isfinite(bench_vals).any()

    if has_benchmark:
        valid_points = np.array([x is not None for x in x_vals], dtype=bool)
        strategy_clean = np.where(np.isfinite(strategy_vals), strategy_vals, np.nan)
        bench_clean = np.where(np.isfinite(bench_vals), bench_vals, np.nan)
        diff = np.where(np.isfinite(strategy_clean) & np.isfinite(bench_clean), strategy_clean - bench_clean, np.nan)
        outperform_mask = np.where(np.isfinite(diff) & (diff >= 0) & valid_points, True, False)
        underperform_mask = np.where(np.isfinite(diff) & (diff < 0) & valid_points, True, False)

        def _add_fill(mask: np.ndarray, name: str, color: str) -> None:
            if not mask.any():
                return
            upper: List[float | None] = []
            lower: List[float | None] = []
            custom: List[float | None] = []
            for is_active, strat_val, bench_val, delta in zip(mask, strategy_clean, bench_clean, diff):
                if not is_active or not np.isfinite(strat_val) or not np.isfinite(bench_val):
                    upper.append(None)
                    lower.append(None)
                    custom.append(None)
                    continue
                upper.append(float(max(strat_val, bench_val)))
                lower.append(float(min(strat_val, bench_val)))
                custom.append(float(delta))

            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=upper,
                    mode="lines",
                    line=dict(width=0),
                    hoverinfo="skip",
                    showlegend=False,
                    connectgaps=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=lower,
                    mode="lines",
                    fill="tonexty",
                    fillcolor=color,
                    line=dict(width=0),
                    name=name,
                    showlegend=True,
                    connectgaps=False,
                    customdata=custom,
                    hovertemplate="Date=%{x|%Y-%m-%d}<br>Diff=$%{customdata:.2f}<extra></extra>",
                )
            )

        _add_fill(outperform_mask, "Outperformance", "rgba(40, 167, 69, 0.25)")
        _add_fill(underperform_mask, "Underperformance", "rgba(220, 53, 69, 0.25)")

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=[None if np.isnan(v) else float(v) for v in bench_clean],
                mode="lines",
                name=bench_label,
                line=dict(width=1.8, color="#6c757d", dash="dash"),
            )
        )

    if has_strategy:
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=[None if np.isnan(v) else float(v) for v in strategy_vals],
                mode="lines",
                line=dict(width=2.4, color="#1f78b4"),
                name=legend_name,
                showlegend=True,
            )
        )

    title = best_label if best_gen is not None else "Holdout equity (current best)"
    fig.update_layout(
        title=title,
        margin=dict(l=10, r=10, t=35, b=10),
        showlegend=True,
        xaxis_title="Date",
        yaxis_title="Equity ($)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
        ),
    )

    _plot(fig, placeholder)


def _plot(fig: "go.Figure", placeholder: st.delta_generator.DeltaGenerator) -> None:
    """Plot with a unique key per render to avoid duplicate element ID collisions."""
    seq_key = _ss_key("render_seq")
    seq = st.session_state.get(seq_key, 0) + 1
    st.session_state[seq_key] = seq
    render_key = f"holdout_equity_chart_{seq}"
    placeholder.plotly_chart(fig, use_container_width=True, key=render_key)


# ---- Helpers ------------------------------------------------------------------

def _blank_curve() -> Dict[str, Any]:
    return {"x": [], "equity": [], "benchmark": None, "benchmark_label": "Benchmark"}


def _to_float_array(values: Any) -> np.ndarray:
    arr: List[float] = []
    if values is None:
        return np.array(arr, dtype=float)
    for val in values:
        try:
            num = float(val)
        except (TypeError, ValueError):
            arr.append(np.nan)
        else:
            arr.append(num if np.isfinite(num) else np.nan)
    return np.array(arr, dtype=float)


def _to_datetime_list(values: Any) -> List[Any]:
    if values is None:
        return []
    try:
        idx = pd.to_datetime(list(values), errors="coerce")
    except Exception:
        return []
    out: List[Any] = []
    for ts in idx:
        if pd.isna(ts):
            out.append(None)
        elif isinstance(ts, pd.Timestamp):
            out.append(ts.to_pydatetime())
        elif isinstance(ts, datetime):
            out.append(ts)
        else:
            out.append(None)
    return out


def _coerce_equity_series(obj: Any) -> pd.Series | None:
    series: pd.Series | None
    if isinstance(obj, pd.Series):
        series = obj.copy()
    elif isinstance(obj, pd.DataFrame):
        if "equity" in obj.columns:
            series = obj["equity"].copy()
        elif not obj.columns.empty:
            series = obj.iloc[:, 0].copy()
        else:
            return None
    else:
        try:
            series = pd.Series(obj)
        except Exception:
            return None

    series = series.dropna()
    if series.empty:
        return None
    series = series[~series.index.duplicated(keep="last")]
    if not isinstance(series.index, pd.DatetimeIndex):
        try:
            series.index = pd.to_datetime(series.index, errors="coerce")
        except Exception:
            return None
    series = series[~series.index.isna()].sort_index()
    if series.empty:
        return None
    try:
        series = series.astype(float)
    except Exception:
        return None
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return None
    if isinstance(series.index, pd.DatetimeIndex) and series.index.tz is not None:
        try:
            series.index = series.index.tz_convert(None)
        except Exception:
            series.index = series.index.tz_localize(None)
    return series.sort_index()


def _align_benchmark_series(
    raw_series: Any,
    equity_series: pd.Series,
    starting_equity: float,
) -> pd.Series | None:
    if raw_series is None:
        return None
    series = _coerce_equity_series(raw_series)
    if series is None or series.empty:
        return None

    eq_index = equity_series.index
    if not isinstance(eq_index, pd.DatetimeIndex):
        return None

    series = series.reindex(eq_index.union(series.index)).ffill().bfill()
    series = series.reindex(eq_index).ffill().bfill()
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return None

    try:
        strategy_start = float(equity_series.iloc[0])
    except Exception:
        strategy_start = float(starting_equity)
    if not np.isfinite(strategy_start) or abs(strategy_start) < 1e-9:
        strategy_start = float(starting_equity)

    bench_start = float(series.iloc[0])
    if not np.isfinite(bench_start) or abs(bench_start) < 1e-12:
        return None

    scaled = (series / bench_start) * strategy_start
    return scaled.reindex(eq_index)


def _load_external_benchmark(
    equity_series: pd.Series,
    holdout_start: Any,
    holdout_end: Any,
    starting_equity: float,
) -> pd.Series | None:
    start_dt = _coerce_datetime(holdout_start) or equity_series.index.min()
    end_dt = _coerce_datetime(holdout_end) or equity_series.index.max()
    try:
        series = get_benchmark_total_return(start=start_dt, end=end_dt)
    except Exception:
        series = None
    if series is None or getattr(series, "empty", True):
        return None
    return _align_benchmark_series(series, equity_series, starting_equity)


# ---- Holdout simulation -------------------------------------------------------

def _simulate_holdout_equity(
    params: Dict[str, Any],
    loader_fn: Callable[..., Dict[str, pd.DataFrame]],
    engine_fn: Callable[..., pd.DataFrame],
    symbols: List[str],
    holdout_start: Any,
    holdout_end: Any,
    starting_equity: float,
) -> Dict[str, Any]:
    """Runs an out-of-sample (holdout) backtest and returns curve metadata for plotting."""

    result = _blank_curve()
    if not symbols:
        return result

    data = loader_fn(symbols=symbols, start=holdout_start, end=holdout_end)
    equity_df = engine_fn(params=params, data=data, starting_equity=starting_equity)

    equity_series = None
    benchmark_series: pd.Series | None = None
    if isinstance(equity_df, pd.DataFrame):
        equity_series = _coerce_equity_series(equity_df.get("equity", equity_df))
        bench_candidates = [
            col
            for col in equity_df.columns
            if str(col).lower() in {"benchmark", "benchmark_equity", "benchmark_ratio", "benchmark_total_return"}
        ]
        if bench_candidates:
            benchmark_series = _align_benchmark_series(
                equity_df[bench_candidates[0]],
                equity_series if equity_series is not None else pd.Series(dtype=float),
                starting_equity,
            )
    elif isinstance(equity_df, pd.Series):
        equity_series = _coerce_equity_series(equity_df)
    else:
        return result

    if equity_series is None or equity_series.empty:
        return result

    if benchmark_series is None:
        benchmark_series = _load_external_benchmark(equity_series, holdout_start, holdout_end, starting_equity)

    idx = equity_series.index
    if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
        idx = idx.tz_convert(None)
    x_values = idx.to_list()
    equity_values = equity_series.astype(float).tolist()
    benchmark_values = None
    if benchmark_series is not None and not benchmark_series.empty:
        bench_aligned = benchmark_series.reindex(equity_series.index).ffill().bfill()
        if not bench_aligned.dropna().empty:
            benchmark_values = bench_aligned.astype(float).tolist()

    return {
        "x": x_values,
        "equity": equity_values,
        "benchmark": benchmark_values,
        "benchmark_label": "Benchmark",
    }


# ---- Public API ---------------------------------------------------------------

def init_chart(
    placeholder: st.delta_generator.DeltaGenerator,
    starting_equity: float,
    holdout_start: Any,
    holdout_end: Any,
    loader_fn: Callable[..., Dict[str, pd.DataFrame]],
    engine_fn: Callable[..., pd.DataFrame],
    symbols: List[str],
    max_curves: int = 8,
) -> None:
    """Initialize/Reset chart and state; renders a blank chart immediately."""
    st.session_state[_ss_key("placeholder")] = placeholder
    st.session_state[_ss_key("render_seq")] = 0
    st.session_state[_ss_key("history")] = []
    st.session_state[_ss_key("best_score")] = None
    st.session_state[_ss_key("cfg")] = {
        "starting_equity": float(starting_equity),
        "holdout_start": holdout_start,
        "holdout_end": holdout_end,
        "loader_fn": loader_fn,
        "engine_fn": engine_fn,
        "symbols": list(symbols),
        "max_curves": int(max_curves),
    }
    _render([], placeholder)


def set_config(
    starting_equity: float | None = None,
    holdout_start: Any | None = None,
    holdout_end: Any | None = None,
    loader_fn: Callable[..., Dict[str, pd.DataFrame]] | None = None,
    engine_fn: Callable[..., pd.DataFrame] | None = None,
    symbols: List[str] | None = None,
    max_curves: int | None = None,
) -> None:
    """Update chart config during a run if UI changes occur (e.g., symbols or window)."""
    cfg = st.session_state.get(_ss_key("cfg"), {})
    if starting_equity is not None:
        cfg["starting_equity"] = float(starting_equity)
    if holdout_start is not None:
        cfg["holdout_start"] = holdout_start
    if holdout_end is not None:
        cfg["holdout_end"] = holdout_end
    if loader_fn is not None:
        cfg["loader_fn"] = loader_fn
    if engine_fn is not None:
        cfg["engine_fn"] = engine_fn
    if symbols is not None:
        cfg["symbols"] = list(symbols)
    if max_curves is not None:
        cfg["max_curves"] = int(max_curves)
    st.session_state[_ss_key("cfg")] = cfg


def on_generation_end(gen_idx: int, best_score: float, best_params: Dict[str, Any]) -> None:
    """EA callback: append and render a new curve ONLY when the score improves."""
    prev_best = st.session_state.get(_ss_key("best_score"), None)
    if (prev_best is not None) and not (best_score > prev_best):
        return  # no improvement; do nothing

    st.session_state[_ss_key("best_score")] = float(best_score)

    cfg = st.session_state.get(_ss_key("cfg"), {})
    placeholder = st.session_state.get(_ss_key("placeholder"))
    if not cfg or placeholder is None:
        return

    curve = _simulate_holdout_equity(
        params=best_params,
        loader_fn=cfg.get("loader_fn"),
        engine_fn=cfg.get("engine_fn"),
        symbols=cfg.get("symbols", []),
        holdout_start=cfg.get("holdout_start"),
        holdout_end=cfg.get("holdout_end"),
        starting_equity=cfg.get("starting_equity", 100_000.0),
    )
    x = curve.get("x", [])
    y = curve.get("equity", [])
    if not (x and y):
        return

    history = st.session_state.get(_ss_key("history"), [])
    history.append(
        {
            "gen": gen_idx,
            "x": x,
            "equity": y,
            "benchmark": curve.get("benchmark"),
            "benchmark_label": curve.get("benchmark_label"),
            "label": f"Gen {gen_idx}",
            "score": best_score,
        }
    )
    max_curves = int(cfg.get("max_curves", 8))
    if len(history) > max_curves:
        history = history[-max_curves:]
    st.session_state[_ss_key("history")] = history

    _render(history, placeholder)


# ---- Standalone equity helper -------------------------------------------------

def _coerce_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if value is None:
        return None

    try:
        coerced = pd.to_datetime(value, errors="raise")
    except Exception:
        return None

    if isinstance(coerced, pd.Series):
        if coerced.empty:
            return None
        coerced = coerced.iloc[0]

    if pd.isna(coerced):
        return None

    if isinstance(coerced, pd.Timestamp):
        return coerced.to_pydatetime()
    if isinstance(coerced, datetime):
        return coerced

    try:
        return pd.Timestamp(coerced).to_pydatetime()
    except Exception:
        return None


def holdout_equity(
    params: Dict[str, Any] | None = None,
    start: Any | None = None,
    end: Any | None = None,
    tickers: List[str] | str | None = None,
    starting_equity: float | None = None,
    strategy: str | None = None,
) -> pd.DataFrame:
    """Simulate the aggregated equity curve for the requested window.

    This mirrors the Strategy Adapter's portfolio simulation logic so that
    external tools (e.g. EA Train/Test Inspector) can render the same holdout
    curve without importing Streamlit UI helpers.
    """

    if not strategy:
        return pd.DataFrame(columns=["date", "equity"])

    params = params or {}
    tickers = tickers or []
    if isinstance(tickers, str):
        tickers = [t.strip() for t in tickers.split(",") if t.strip()]

    if not tickers:
        return pd.DataFrame(columns=["date", "equity"])

    try:
        mod = import_module(strategy)
        run_strategy = getattr(mod, "run_strategy")
    except Exception:
        return pd.DataFrame(columns=["date", "equity"])

    start_dt = _coerce_datetime(start)
    end_dt = _coerce_datetime(end)
    if start_dt is None or end_dt is None:
        return pd.DataFrame(columns=["date", "equity"])

    base_equity = float(starting_equity or 100_000.0)
    curves: Dict[str, pd.Series] = {}
    for sym in tickers:
        try:
            result = run_strategy(sym, start_dt, end_dt, starting_equity, params)
        except Exception:
            continue

        eq = result.get("equity") if isinstance(result, dict) else None
        if eq is None or len(eq) == 0:
            continue

        if isinstance(eq, pd.DataFrame):
            if "equity" in eq.columns:
                eq = eq["equity"]
            else:
                eq = eq.iloc[:, 0]
        elif not isinstance(eq, pd.Series):
            try:
                eq = pd.Series(eq)
            except Exception:
                continue

        eq = eq.dropna()
        if eq.empty:
            continue

        if not isinstance(eq.index, pd.DatetimeIndex):
            try:
                eq.index = pd.to_datetime(eq.index, errors="coerce")
            except Exception:
                continue

        eq = eq[~eq.index.isna()].sort_index()
        if eq.empty:
            continue

        try:
            eq = eq.astype(float)
        except Exception:
            continue

        first_val = float(eq.iloc[0]) if len(eq) else 0.0
        if abs(first_val) < 1e-12:
            anchored = eq + base_equity
        else:
            anchored = eq

        anchored = anchored.astype(float)
        if anchored.empty:
            continue

        start_value = float(anchored.iloc[0]) if len(anchored) else 0.0
        if abs(start_value) < 1e-12:
            continue

        curves[sym] = (anchored / start_value).astype(float)

    if not curves:
        return pd.DataFrame(columns=["date", "equity"])

    df_curves = pd.DataFrame(curves).sort_index().ffill().dropna(how="all")
    if df_curves.empty:
        return pd.DataFrame(columns=["date", "equity"])

    base_equity = float(starting_equity or 100_000.0)
    portfolio = df_curves.mean(axis=1, skipna=True) * base_equity
    portfolio = portfolio.dropna()
    if portfolio.empty:
        return pd.DataFrame(columns=["date", "equity"])

    date_index = portfolio.index
    if isinstance(date_index, pd.DatetimeIndex):
        date_index = date_index.tz_localize(None)
    return pd.DataFrame({"date": date_index, "equity": portfolio.values})
