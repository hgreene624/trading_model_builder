"""Render a Streamlit panel comparing strategy equity to an S&P 500 TRI proxy."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:  # pragma: no cover - defensive import per spec
    from src.storage import load_price_history  # type: ignore
except Exception:  # pragma: no cover - handled dynamically in render
    load_price_history = None  # type: ignore


def _normalize_curve(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    if not isinstance(series, pd.Series):
        try:
            series = pd.Series(series)
        except Exception:
            return pd.Series(dtype=float)
    if not isinstance(series.index, pd.DatetimeIndex):
        try:
            series.index = pd.to_datetime(series.index)
        except Exception:
            return pd.Series(dtype=float)
    series = series.sort_index()
    series = series.astype(float)
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    series = series[series > 0]
    if series.empty:
        return pd.Series(dtype=float)
    first = series.iloc[0]
    if first == 0 or not np.isfinite(first):
        return pd.Series(dtype=float)
    return series / first


def _compute_cagr(series: pd.Series) -> Optional[float]:
    if series.empty:
        return None
    start, end = series.iloc[0], series.iloc[-1]
    if start <= 0 or end <= 0:
        return None
    delta_days = (series.index[-1] - series.index[0]).days
    if delta_days <= 0:
        return None
    years = delta_days / 365.25
    if years <= 0:
        return None
    try:
        return (end / start) ** (1.0 / years) - 1.0
    except Exception:
        return None


def _compute_max_drawdown(series: pd.Series) -> Optional[float]:
    if series.empty:
        return None
    running_max = series.cummax()
    dd = series / running_max - 1.0
    if dd.empty:
        return None
    return float(dd.min())


def _compute_tracking_error(strategy: pd.Series, benchmark: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    if strategy.empty or benchmark.empty:
        return None, None
    daily = pd.concat([strategy, benchmark], axis=1, join="inner").pct_change().dropna()
    if daily.empty or daily.shape[1] < 2:
        return None, None
    strategy_returns = daily.iloc[:, 0]
    benchmark_returns = daily.iloc[:, 1]
    active = strategy_returns - benchmark_returns
    if active.empty or len(active) < 2:
        return None, None
    tracking_error = float(active.std(ddof=1) * math.sqrt(252))
    if not np.isfinite(tracking_error) or tracking_error == 0.0:
        return None, None
    info_ratio = float((active.mean() * 252) / tracking_error)
    if not np.isfinite(info_ratio):
        info_ratio = None
    return tracking_error, info_ratio


def _format_metric(name: str, value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or not np.isfinite(value))):
        return "—"
    if "Ratio" in name:
        return f"{value:.2f}"
    return f"{value * 100:.2f}%"


def render_tri_panel(strategy_curve: pd.Series, title: str = "S&P 500 Total-Return (TRI proxy) comparison") -> None:
    expander = st.expander("▸ " + title, expanded=False)
    with expander:
        if load_price_history is None:
            st.warning("Price history loader unavailable; cannot display TRI comparison.")
            return
        strategy_norm = _normalize_curve(strategy_curve)
        if strategy_norm.empty:
            st.info("Strategy curve unavailable or insufficient for TRI comparison.")
            return
        try:
            spy_df = load_price_history("SPY", timeframe="1D")  # type: ignore[misc]
        except Exception as err:  # pragma: no cover - defensive feedback
            st.warning(f"Unable to load SPY data: {err}")
            return
        if not isinstance(spy_df, pd.DataFrame) or spy_df.empty:
            st.info("No SPY data returned for TRI comparison.")
            return
        price_col = None
        for candidate in ("adj_close", "adjusted_close", "close"):
            if candidate in spy_df.columns:
                price_col = candidate
                break
        if price_col is None:
            st.info("SPY data lacks an adjusted/close column for TRI comparison.")
            return
        spy_series = spy_df[price_col]
        if not isinstance(spy_series, pd.Series):
            spy_series = pd.Series(spy_series)
        if not isinstance(spy_series.index, pd.DatetimeIndex):
            if "date" in spy_df.columns:
                try:
                    spy_series.index = pd.to_datetime(spy_df["date"], errors="coerce")
                except Exception:
                    st.info("SPY price series lacks a valid DatetimeIndex.")
                    return
            else:
                try:
                    spy_series.index = pd.to_datetime(spy_series.index)
                except Exception:
                    st.info("SPY price series lacks a valid DatetimeIndex.")
                    return
        spy_series = spy_series.sort_index()
        spy_series = spy_series.astype(float)
        spy_series = spy_series.replace([np.inf, -np.inf], np.nan).dropna()
        spy_series = spy_series[spy_series > 0]
        if spy_series.empty:
            st.info("SPY series empty after cleaning; cannot compute TRI comparison.")
            return
        spy_norm = _normalize_curve(spy_series)
        if spy_norm.empty:
            st.info("SPY normalization failed; cannot compute TRI comparison.")
            return
        combined = pd.concat(
            [strategy_norm.rename("Strategy"), spy_norm.rename("SPY (TRI)")], axis=1, join="inner"
        ).dropna()
        if combined.empty or len(combined) < 10:
            st.info("Not enough overlapping data between strategy and SPY for TRI comparison.")
            return
        strategy_aligned = combined["Strategy"]
        spy_aligned = combined["SPY (TRI)"]
        strategy_cagr = _compute_cagr(strategy_aligned)
        strategy_maxdd = _compute_max_drawdown(strategy_aligned)
        spy_cagr = _compute_cagr(spy_aligned)
        spy_maxdd = _compute_max_drawdown(spy_aligned)
        tracking_error, info_ratio = _compute_tracking_error(strategy_aligned, spy_aligned)
        metrics_df = pd.DataFrame(
            [(name, _format_metric(name, value)) for name, value in [
                ("Strategy CAGR", strategy_cagr),
                ("Strategy Max Drawdown", strategy_maxdd),
                ("SPY (TRI) CAGR", spy_cagr),
                ("SPY (TRI) Max Drawdown", spy_maxdd),
                ("Tracking Error (ann.)", tracking_error),
                ("Information Ratio", info_ratio),
            ]],
            columns=["Metric", "Value"],
        )
        st.line_chart(combined)
        st.dataframe(metrics_df, hide_index=True, use_container_width=True)
        st.caption(
            "Strategy and SPY curves normalized to 1.0 on first overlapping date. SPY adjusted close used as a total return proxy when available."
        )
