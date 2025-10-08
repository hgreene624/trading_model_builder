from __future__ import annotations

import logging
import math
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.cache import get_ohlcv_cached

logger = logging.getLogger(__name__)

try:
    from src.storage import get_benchmark_total_return  # type: ignore
except Exception:  # pragma: no cover - optional helper
    get_benchmark_total_return = None  # type: ignore

try:
    from src.storage import load_price_history  # type: ignore
except Exception:  # pragma: no cover - loader optional in some envs
    load_price_history = None  # type: ignore


def _widget_key(title: str, suffix: str) -> str:
    base = f"{abs(hash(title))}" if title else "tri_panel"
    return f"tri_panel_{suffix}_{base}"


def _normalize_curve(series: pd.Series) -> Optional[pd.Series]:
    if series is None:
        logger.debug("normalize_curve received None series")
        return None
    try:
        s = series.dropna().astype(float)
    except Exception:
        logger.exception("normalize_curve failed to coerce series to float")
        return None
    if s.empty:
        return None
    if not isinstance(s.index, pd.DatetimeIndex):
        try:
            s.index = pd.to_datetime(s.index)
        except Exception:
            logger.exception("normalize_curve failed to convert index to datetime")
            return None
    s = s.sort_index()
    if isinstance(s.index, pd.DatetimeIndex) and getattr(s.index, "tz", None) is not None:
        try:
            s.index = s.index.tz_convert(None)
        except Exception:
            try:
                s.index = s.index.tz_localize(None)
            except Exception:
                logger.debug("normalize_curve failed to drop timezone", extra={"tz": str(s.index.tz)})
                return None
    s = s[s > 0]
    if s.empty:
        return None
    first = s.iloc[0]
    if first == 0:
        s = s[s > 0]
        if s.empty:
            logger.debug("normalize_curve filtered out non-positive values")
            return None
        first = s.iloc[0]
    if first <= 0:
        logger.debug("normalize_curve first value non-positive", extra={"first": first})
        return None
    return s / float(first)


def _normalize_ticker_list(tickers: Optional[Iterable[str]]) -> List[str]:
    if tickers is None:
        return []
    if isinstance(tickers, str):
        tickers = [tickers]
    normalized: List[str] = []
    for ticker in tickers:
        if ticker is None:
            continue
        value = str(ticker).strip()
        if not value:
            continue
        normalized.append(value.upper())
    # Deduplicate while preserving order
    seen = set()
    unique: List[str] = []
    for ticker in normalized:
        if ticker in seen:
            continue
        seen.add(ticker)
        unique.append(ticker)
    return unique


def _timestamp_to_iso(value: pd.Timestamp) -> str:
    ts = pd.Timestamp(value)
    if getattr(ts, "tzinfo", None) is not None:
        try:
            ts = ts.tz_convert(None)
        except Exception:
            ts = ts.tz_localize(None)
    return ts.isoformat()


@st.cache_data(show_spinner=False)
def _load_price_series_cached(ticker: str, start: str, end: str) -> Optional[pd.Series]:
    try:
        df = get_ohlcv_cached(ticker, start, end)
    except Exception:
        logger.exception("tri_panel failed to load price history", extra={"ticker": ticker})
        return None
    if df is None or len(df) == 0:
        return None
    cols = {str(c).lower(): c for c in df.columns}
    close_col = None
    for candidate in ("adj_close", "adjusted_close", "adjclose", "close"):
        if candidate in cols:
            close_col = cols[candidate]
            break
    if close_col is None:
        close_col = "close" if "close" in df.columns else None
    if close_col is None:
        logger.debug("tri_panel no close column", extra={"columns": list(df.columns), "ticker": ticker})
        return None
    series = df[close_col].dropna()
    if series.empty:
        return None
    try:
        series = series.astype(float)
    except Exception:
        logger.debug("tri_panel failed to coerce price series", extra={"ticker": ticker})
        return None
    if not isinstance(series.index, pd.DatetimeIndex):
        try:
            series.index = pd.to_datetime(series.index)
        except Exception:
            logger.debug("tri_panel failed to convert price index", extra={"ticker": ticker})
            return None
    series = series.sort_index()
    if getattr(series.index, "tz", None) is not None:
        try:
            series.index = series.index.tz_convert(None)
        except Exception:
            try:
                series.index = series.index.tz_localize(None)
            except Exception:
                logger.debug("tri_panel failed to drop timezone from price series", extra={"ticker": ticker})
                return None
    return series


def _load_portfolio_price_curves(
    tickers: Iterable[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Tuple[Dict[str, pd.Series], List[str]]:
    tickers_list = list(tickers)
    if start is None or end is None:
        return {}, tickers_list
    start_iso = _timestamp_to_iso(start)
    end_iso = _timestamp_to_iso(end)
    curves: Dict[str, pd.Series] = {}
    missing: List[str] = []
    for ticker in tickers_list:
        series = _load_price_series_cached(ticker, start_iso, end_iso)
        if series is None or series.empty:
            missing.append(ticker)
            continue
        curves[ticker] = series
    return curves, missing


def _extract_spy_tri(start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> Optional[pd.Series]:
    mode = "TRI"
    if hasattr(st, "secrets"):
        try:
            mode = str(st.secrets.get("BENCHMARK_MODE", "TRI") or "TRI").upper()
        except Exception:
            mode = "TRI"

    series: Optional[pd.Series]
    series = None
    if get_benchmark_total_return is not None:
        try:
            series = get_benchmark_total_return(mode=mode, start=start, end=end)
        except Exception:
            logger.exception("tri_panel benchmark helper failed", extra={"mode": mode})
            series = None

    if (series is None or series.empty) and load_price_history is not None:
        try:
            kwargs = {"timeframe": "1D"}
            if start is not None:
                kwargs.setdefault("start", start)
            if end is not None:
                kwargs.setdefault("end", end)
            try:
                df = load_price_history("SPY", **kwargs)
            except TypeError:
                logger.debug("tri_panel fallback load_price_history signature", extra={"kwargs": kwargs})
                df = load_price_history("SPY", timeframe="1D")
        except Exception:
            logger.exception("tri_panel failed while loading SPY history")
            return None
        if df is None:
            logger.warning("tri_panel load_price_history returned None")
            return None
        if df is None or len(df) == 0:
            logger.info("tri_panel SPY dataframe empty")
            return None
        cols = {str(c).lower(): c for c in df.columns}
        if mode == "PRI":
            preferred_cols = ("close", "adj_close", "adjusted_close", "adjclose")
        else:
            preferred_cols = ("adj_close", "adjusted_close", "adjclose", "close")
        tri_col = None
        for cand in preferred_cols:
            if cand in cols:
                tri_col = cols[cand]
                break
        if tri_col is None:
            logger.warning("tri_panel could not locate benchmark column", extra={"columns": list(df.columns)})
            return None
        try:
            series = df[tri_col].dropna().astype(float)
        except Exception:
            logger.exception("tri_panel failed to process benchmark column", extra={"tri_col": tri_col})
            return None
        if series.empty:
            logger.info("tri_panel benchmark series empty after dropna")
            return None
        if not isinstance(series.index, pd.DatetimeIndex):
            for ts_col in ("date", "datetime", "timestamp"):
                if ts_col in df.columns:
                    try:
                        series.index = pd.to_datetime(df[ts_col])
                        break
                    except Exception:
                        logger.exception("tri_panel failed parsing timestamp column", extra={"column": ts_col})
                        continue
        if not isinstance(series.index, pd.DatetimeIndex):
            try:
                series.index = pd.to_datetime(series.index)
            except Exception:
                logger.exception("tri_panel failed to coerce series index to datetime")
                return None
        if isinstance(series.index, pd.DatetimeIndex) and getattr(series.index, "tz", None) is not None:
            try:
                series.index = series.index.tz_convert(None)
            except Exception:
                try:
                    series.index = series.index.tz_localize(None)
                except Exception:
                    logger.debug(
                        "tri_panel failed to drop timezone from benchmark series",
                        extra={"tz": str(series.index.tz)},
                    )
                    return None
    if series is None or series.empty:
        logger.warning("tri_panel benchmark series unavailable")
        return None

    series = series.sort_index().dropna()
    if isinstance(series.index, pd.DatetimeIndex) and getattr(series.index, "tz", None) is not None:
        try:
            series.index = series.index.tz_convert(None)
        except Exception:
            try:
                series.index = series.index.tz_localize(None)
            except Exception:
                logger.debug("tri_panel failed to drop timezone", extra={"tz": str(series.index.tz)})
                return None
    series = series[series > 0]
    if series.empty:
        logger.info("tri_panel benchmark series empty after filtering positives")
        return None
    first = series.iloc[0]
    if first <= 0:
        logger.info("tri_panel benchmark first value non-positive", extra={"value": float(first)})
        return None
    return series / float(first)


def _compute_cagr(series: pd.Series) -> Optional[float]:
    if series is None or series.empty:
        logger.debug("tri_panel CAGR requested on empty series")
        return None
    start_value = series.iloc[0]
    end_value = series.iloc[-1]
    if start_value <= 0 or end_value <= 0:
        logger.debug(
            "tri_panel CAGR invalid start/end",
            extra={"start": float(start_value), "end": float(end_value)},
        )
        return None
    delta = series.index[-1] - series.index[0]
    if not isinstance(delta, pd.Timedelta):
        logger.debug("tri_panel CAGR delta not timedelta", extra={"type": type(delta).__name__})
        return None
    days = delta.days
    if days <= 0:
        logger.debug("tri_panel CAGR non-positive days", extra={"days": days})
        return None
    years = days / 365.25
    if years <= 0:
        logger.debug("tri_panel CAGR non-positive years", extra={"years": years})
        return None
    try:
        return (end_value / start_value) ** (1.0 / years) - 1.0
    except Exception:
        logger.exception("tri_panel CAGR computation failed")
        return None


def _compute_max_dd(series: pd.Series) -> Optional[float]:
    if series is None or series.empty:
        logger.debug("tri_panel max drawdown on empty series")
        return None
    rolling_max = series.cummax()
    if rolling_max is None or rolling_max.empty:
        logger.debug("tri_panel rolling max empty")
        return None
    drawdowns = series / rolling_max - 1.0
    if drawdowns is None or drawdowns.empty:
        logger.debug("tri_panel drawdowns empty")
        return None
    try:
        return float(drawdowns.min())
    except Exception:
        logger.exception("tri_panel max drawdown computation failed")
        return None


def _compute_tracking_stats(strategy: pd.Series, benchmark: pd.Series) -> tuple[Optional[float], Optional[float]]:
    if strategy is None or benchmark is None:
        logger.debug("tri_panel tracking stats missing input")
        return None, None
    if strategy.empty or benchmark.empty:
        logger.debug("tri_panel tracking stats empty series")
        return None, None
    returns = pd.concat([strategy.pct_change(), benchmark.pct_change()], axis=1, join="inner")
    returns = returns.dropna()
    if returns.empty:
        logger.debug("tri_panel tracking stats no overlapping returns")
        return None, None
    returns.columns = ["strategy", "benchmark"]
    active = returns["strategy"] - returns["benchmark"]
    active = active.dropna()
    if active.empty:
        logger.debug("tri_panel tracking stats active empty")
        return None, None
    std_active = float(active.std(ddof=0))
    if std_active == 0 or math.isnan(std_active):
        logger.debug(
            "tri_panel tracking stats zero std",
            extra={"std": std_active},
        )
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


def _coerce_timestamp(value: "pd.Timestamp | str | None") -> Optional[pd.Timestamp]:
    if value is None or value == "":
        return None
    try:
        ts = pd.Timestamp(value)
    except Exception:
        logger.debug("tri_panel failed to parse timestamp", extra={"value": value})
        return None
    if getattr(ts, "tzinfo", None) is not None:
        try:
            ts = ts.tz_convert(None)
        except Exception:
            try:
                ts = ts.tz_localize(None)
            except Exception:
                logger.debug("tri_panel failed to drop timezone", extra={"value": value})
                return None
    return ts


def render_tri_panel(
    strategy_curve: pd.Series,
    title: str = "S&P 500 Total-Return (TRI proxy) comparison",
    test_start: "pd.Timestamp | str | None" = None,
    test_end: "pd.Timestamp | str | None" = None,
    show_test_toggle: bool = True,
    portfolio_tickers: Optional[Iterable[str]] = None,
) -> None:
    st.markdown(f"#### {title}")
    strategy_norm = _normalize_curve(strategy_curve)
    if strategy_norm is None or strategy_norm.empty:
        logger.info("tri_panel no strategy curve available")
        st.info("Strategy equity curve unavailable; cannot compute TRI comparison.")
        return
    spy_norm = _extract_spy_tri(
        start=strategy_norm.index.min() if not strategy_norm.empty else None,
        end=strategy_norm.index.max() if not strategy_norm.empty else None,
    )
    if spy_norm is None or spy_norm.empty:
        logger.warning("tri_panel SPY normalization failed")
        st.warning("SPY TRI data unavailable (loader missing or returned no data).")
        return
    combined = pd.concat(
        [strategy_norm.rename("Strategy"), spy_norm.rename("SPY (TRI)")],
        axis=1,
        join="inner",
    ).dropna()
    if isinstance(combined.index, pd.DatetimeIndex) and getattr(combined.index, "tz", None) is not None:
        try:
            combined.index = combined.index.tz_convert(None)
        except Exception:
            try:
                combined.index = combined.index.tz_localize(None)
            except Exception:
                logger.debug("tri_panel failed to drop timezone from combined index", extra={"tz": str(combined.index.tz)})
                st.info("Unable to align timezones for TRI comparison.")
                return
    if combined.empty or len(combined) < 10:
        logger.info(
            "tri_panel insufficient overlap",
            extra={"rows": int(len(combined))},
        )
        st.info("Insufficient overlapping history between strategy and SPY (requires at least 10 points).")
        return
    combined_view = combined
    test_start_ts = _coerce_timestamp(test_start)
    test_end_ts = _coerce_timestamp(test_end)
    test_toggle_available = False
    restrict_to_test = False
    if show_test_toggle:
        if test_start_ts is None or test_end_ts is None:
            st.info("Test window not provided; displaying full history.")
        elif test_start_ts > test_end_ts:
            st.info("Test window start is after end; displaying full history.")
        else:
            test_toggle_available = True
            restrict_to_test = st.checkbox(
                "Test-only view",
                value=False,
                key=f"tri_panel_test_toggle_{title}",
            )
    if test_toggle_available and restrict_to_test:
        test_view = combined.loc[(combined.index >= test_start_ts) & (combined.index <= test_end_ts)]
        if test_view.empty or len(test_view) < 10:
            logger.info(
                "tri_panel insufficient test overlap",
                extra={"rows": int(len(test_view))},
            )
            st.info(
                "Insufficient overlapping history in the test window (requires at least 10 points). Displaying full history.",
            )
        else:
            first_row = test_view.iloc[0]
            if (first_row <= 0).any():
                logger.info(
                    "tri_panel non-positive starting values in test window",
                    extra={"values": first_row.to_dict()},
                )
                st.info("Test window start values must be positive; displaying full history.")
            else:
                combined_view = test_view.divide(first_row)
    strategy_aligned = combined_view["Strategy"]
    spy_aligned = combined_view["SPY (TRI)"]
    excess_index = strategy_aligned / spy_aligned
    if not excess_index.empty:
        first_excess = excess_index.iloc[0]
        if pd.notna(first_excess) and first_excess > 0:
            excess_index = excess_index / float(first_excess)
    combined_chart = combined_view.copy()
    combined_chart["Excess Index"] = excess_index
    portfolio_trace_names: List[str] = []
    normalized_tickers = _normalize_ticker_list(portfolio_tickers)
    if normalized_tickers and isinstance(combined_chart.index, pd.DatetimeIndex):
        show_prices = st.checkbox(
            "Add portfolio ticker prices",
            value=False,
            key=_widget_key(title, "show_prices"),
            help="Overlay normalized closing prices for tickers traded in the portfolio.",
        )
        if show_prices:
            combined_start = combined_chart.index.min()
            combined_end = combined_chart.index.max()
            curves, missing = _load_portfolio_price_curves(normalized_tickers, combined_start, combined_end)
            for ticker in normalized_tickers:
                series = curves.get(ticker)
                if series is None:
                    continue
                normalized = _normalize_curve(series)
                if normalized is None or normalized.empty:
                    continue
                overlap = normalized.loc[(normalized.index >= combined_start) & (normalized.index <= combined_end)]
                if overlap.empty:
                    continue
                first_value = overlap.iloc[0]
                try:
                    first_value_float = float(first_value)
                except Exception:
                    continue
                if first_value_float <= 0 or math.isnan(first_value_float):
                    continue
                rebased = overlap / first_value_float
                combined_chart = combined_chart.join(rebased.rename(ticker), how="left")
                portfolio_trace_names.append(ticker)
            if portfolio_trace_names:
                missing_filtered = [m for m in missing if m not in portfolio_trace_names]
                if missing_filtered:
                    st.info(
                        "Price data unavailable for: " + ", ".join(missing_filtered),
                        icon="ℹ️",
                    )
            else:
                if missing:
                    st.warning("Unable to load price data for the selected portfolio tickers.")
                else:
                    st.info("No overlapping price history found for the selected tickers.")

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

    fig = go.Figure()
    for column, color in (
        ("Strategy", "#1f77b4"),
        ("SPY (TRI)", "#ff7f0e"),
        ("Excess Index", "#2ca02c"),
    ):
        series = combined_chart[column].dropna()
        if series.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=column,
                line=dict(width=2, color=color),
            )
        )

    if portfolio_trace_names:
        ticker_colors = [
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
            "#d62728",
            "#ff9896",
            "#98df8a",
        ]
        for idx, ticker in enumerate(portfolio_trace_names):
            if ticker not in combined_chart.columns:
                continue
            series = combined_chart[ticker].dropna()
            if series.empty:
                continue
            color = ticker_colors[idx % len(ticker_colors)]
            fig.add_trace(
                go.Scatter(
                    x=series.index,
                    y=series.values,
                    mode="lines",
                    name=f"{ticker} price",
                    line=dict(width=1.5, dash="dot", color=color),
                )
            )

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=0.0,
        ),
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Normalized return",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
    )
    st.dataframe(metrics, use_container_width=True, hide_index=True)
    st.caption(
        "Strategy and SPY curves normalized to 1.0 on first overlapping date. "
        "SPY adjusted close used as TRI proxy when available."
    )
