from __future__ import annotations

import math
from typing import Optional

import pandas as pd
import streamlit as st

try:
    from src.storage import load_price_history  # type: ignore
except Exception:  # pragma: no cover - loader optional in some envs
    load_price_history = None  # type: ignore


def _normalize_curve(series: pd.Series) -> Optional[pd.Series]:
    if series is None:
        return None
    try:
        s = series.dropna().astype(float)
    except Exception:
        return None
    if s.empty:
        return None
    if not isinstance(s.index, pd.DatetimeIndex):
        try:
            s.index = pd.to_datetime(s.index)
        except Exception:
            return None
    s = s.sort_index()
    s = s[s > 0]
    if s.empty:
        return None
    first = s.iloc[0]
    if first == 0:
        s = s[s > 0]
        if s.empty:
            return None
        first = s.iloc[0]
    if first <= 0:
        return None
    return s / float(first)


def _extract_spy_tri(start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> Optional[pd.Series]:
    if load_price_history is None:
        return None
    try:
        kwargs = {"timeframe": "1D"}
        if start is not None:
            kwargs.setdefault("start", start)
        if end is not None:
            kwargs.setdefault("end", end)
        try:
            df = load_price_history("SPY", **kwargs)
        except TypeError:
            df = load_price_history("SPY", timeframe="1D")
    except Exception:
        return None
    if df is None:
        return None
    try:
        spy_df = df.copy()
    except Exception:
        return None
    if spy_df is None or len(spy_df) == 0:
        return None
    cols = {str(c).lower(): c for c in spy_df.columns}
    tri_col = None
    for cand in ("adj_close", "adjusted_close", "adjclose", "close"):
        if cand in cols:
            tri_col = cols[cand]
            break
    if tri_col is None:
        return None
    try:
        series = spy_df[tri_col].dropna().astype(float)
    except Exception:
        return None
    if series.empty:
        return None
    if not isinstance(series.index, pd.DatetimeIndex):
        for ts_col in ("date", "datetime", "timestamp"):
            if ts_col in spy_df.columns:
                try:
                    idx = pd.to_datetime(spy_df[ts_col])
                    series.index = idx
                    break
                except Exception:
                    continue
    if not isinstance(series.index, pd.DatetimeIndex):
        try:
            series.index = pd.to_datetime(series.index)
        except Exception:
            return None
    series = series.sort_index()
    series = series[series > 0]
    if series.empty:
        return None
    return series / float(series.iloc[0])


def _compute_cagr(series: pd.Series) -> Optional[float]:
    if series is None or series.empty:
        return None
    start_value = series.iloc[0]
    end_value = series.iloc[-1]
    if start_value <= 0 or end_value <= 0:
        return None
    delta = series.index[-1] - series.index[0]
    if not isinstance(delta, pd.Timedelta):
        return None
    days = delta.days
    if days <= 0:
        return None
    years = days / 365.25
    if years <= 0:
        return None
    try:
        return (end_value / start_value) ** (1.0 / years) - 1.0
    except Exception:
        return None


def _compute_max_dd(series: pd.Series) -> Optional[float]:
    if series is None or series.empty:
        return None
    rolling_max = series.cummax()
    if rolling_max is None or rolling_max.empty:
        return None
    drawdowns = series / rolling_max - 1.0
    if drawdowns is None or drawdowns.empty:
        return None
    try:
        return float(drawdowns.min())
    except Exception:
        return None


def _compute_tracking_stats(strategy: pd.Series, benchmark: pd.Series) -> tuple[Optional[float], Optional[float]]:
    if strategy is None or benchmark is None:
        return None, None
    if strategy.empty or benchmark.empty:
        return None, None
    returns = pd.concat([strategy.pct_change(), benchmark.pct_change()], axis=1, join="inner")
    returns = returns.dropna()
    if returns.empty:
        return None, None
    returns.columns = ["strategy", "benchmark"]
    active = returns["strategy"] - returns["benchmark"]
    active = active.dropna()
    if active.empty:
        return None, None
    std_active = float(active.std(ddof=0))
    if std_active == 0 or math.isnan(std_active):
        return None, None
    mean_active = float(active.mean())
    tracking_error = std_active * math.sqrt(252)
    information_ratio = (mean_active * 252.0) / (std_active * math.sqrt(252))
    return tracking_error, information_ratio


def _format_pct(value: Optional[float]) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value:.2%}"


def _format_ratio(value: Optional[float]) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value:.2f}"


def render_tri_panel(strategy_curve: pd.Series, title: str = "S&P 500 Total-Return (TRI proxy) comparison") -> None:
    with st.expander(f"\u25b8 {title}"):
        strategy_norm = _normalize_curve(strategy_curve)
        if strategy_norm is None or strategy_norm.empty:
            st.info("Strategy equity curve unavailable; cannot compute TRI comparison.")
            return
        spy_norm = _extract_spy_tri(
            start=strategy_norm.index.min() if not strategy_norm.empty else None,
            end=strategy_norm.index.max() if not strategy_norm.empty else None,
        )
        if spy_norm is None or spy_norm.empty:
            st.warning("SPY TRI data unavailable (loader missing or returned no data).")
            return
        combined = pd.concat(
            [strategy_norm.rename("Strategy"), spy_norm.rename("SPY (TRI)")],
            axis=1,
            join="inner",
        ).dropna()
        if combined.empty or len(combined) < 10:
            st.info("Insufficient overlapping history between strategy and SPY (requires at least 10 points).")
            return
        strategy_aligned = combined["Strategy"]
        spy_aligned = combined["SPY (TRI)"]
        tracking_error, information_ratio = _compute_tracking_stats(strategy_aligned, spy_aligned)
        metrics = pd.DataFrame(
            {
                "Metric": [
                    "Strategy CAGR",
                    "Strategy MaxDD",
                    "SPY (TRI) CAGR",
                    "SPY (TRI) MaxDD",
                    "Tracking Error (annualized)",
                    "Information Ratio",
                ],
                "Value": [
                    _format_pct(_compute_cagr(strategy_aligned)),
                    _format_pct(_compute_max_dd(strategy_aligned)),
                    _format_pct(_compute_cagr(spy_aligned)),
                    _format_pct(_compute_max_dd(spy_aligned)),
                    _format_pct(tracking_error),
                    _format_ratio(information_ratio),
                ],
            }
        )
        st.line_chart(combined)
        st.dataframe(metrics, use_container_width=True, hide_index=True)
        st.caption(
            "Strategy and SPY curves normalized to 1.0 on first overlapping date. "
            "SPY adjusted close used as TRI proxy when available."
        )
