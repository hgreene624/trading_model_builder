# src/tuning/auto_bounds.py
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from datetime import date, timedelta

from src.data.cache import get_ohlcv_cached

def _wilder_atr(df: pd.DataFrame, n: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()

def _adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = (-low.diff())
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr.replace(0, np.nan))
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0.0)
    adx = dx.ewm(alpha=1 / n, adjust=False).mean()
    return adx

def _classify(df: pd.DataFrame) -> Tuple[str, str, Dict[str, float]]:
    atr14 = _wilder_atr(df, 14)
    atr_pct = (atr14 / df["close"]).replace(0, np.nan).dropna()
    med_atr_pct = float(np.median(atr_pct.values)) * 100.0 if len(atr_pct) else 0.0

    adx14 = _adx(df, 14)
    med_adx = float(np.nanmedian(adx14.values)) if len(adx14) else 0.0

    if med_atr_pct < 1.5: vol_class = "low"
    elif med_atr_pct <= 3.0: vol_class = "medium"
    else: vol_class = "high"

    if   med_adx < 20: trend_class = "choppy"
    elif med_adx <= 25: trend_class = "mixed"
    else: trend_class = "trending"

    return vol_class, trend_class, {"median_atr_pct": med_atr_pct, "median_adx": med_adx}

def suggest_bounds_for(symbol: str, start: date | str | None, end: date | str | None) -> Dict:
    """Return a dict of suggested *bounds* based on recent OHLCV."""
    if isinstance(start, str): start = date.fromisoformat(start)
    if isinstance(end, str):   end = date.fromisoformat(end)
    if not end:   end = date.today()
    if not start or start >= end: start = end - timedelta(days=365*2)

    # --- Resilient data load: try Alpaca with shrinking windows, then yfinance as a last resort ---
    tried_notes = []
    df = None

    # Try multiple lookback windows with Alpaca
    for days in (365 * 2, 365, 180, 120, 90):
        s = end - timedelta(days=days)
        if s >= end:
            continue
        try:
            dft = get_ohlcv_cached(symbol, s.isoformat(), end.isoformat())
            if dft is not None and not dft.empty:
                df = dft
                start = s  # use the actual start we succeeded with
                tried_notes.append(f"Alpaca OK {s}→{end}")
                break
            else:
                tried_notes.append(f"Alpaca empty {s}→{end}")
        except Exception as e:
            tried_notes.append(f"Alpaca error {s}→{end}: {type(e).__name__}")
            continue

    # Fallback: try yfinance loader if available
    if (df is None or df.empty):
        try:
            from src.data.yf import load_ohlcv as yf_load_ohlcv
            for days in (365 * 2, 365, 180, 120, 90):
                s = end - timedelta(days=days)
                if s >= end:
                    continue
                try:
                    dft = yf_load_ohlcv(symbol, s.isoformat(), end.isoformat())
                    if dft is not None and not dft.empty:
                        df = dft
                        start = s
                        tried_notes.append(f"yfinance OK {s}→{end}")
                        break
                    else:
                        tried_notes.append(f"yfinance empty {s}→{end}")
                except Exception as e:
                    tried_notes.append(f"yfinance error {s}→{end}: {type(e).__name__}")
                    continue
        except Exception as e:
            tried_notes.append("yfinance loader unavailable")

    if df is None or df.empty:
        # Sensible generic defaults when no data available anywhere
        return {
            "breakout_min": 20, "breakout_max": 120,
            "exit_min": 10, "exit_max": 60,
            "atr_min": 7, "atr_max": 30,
            "atr_multiple_min": 1.5, "atr_multiple_max": 5.0,
            "risk_per_trade_min": 0.2, "risk_per_trade_max": 1.0,
            "tp_multiple_min": 0.0, "tp_multiple_max": 3.0,
            "holding_period_min": 0, "holding_period_max": 120,
            "cost_bps_min": 1.0, "cost_bps_max": 5.0,
            "allow_trend_filter": True,
            "sma_fast_min": 10, "sma_fast_max": 40,
            "sma_slow_min": 40, "sma_slow_max": 100,
            "sma_long_min": 150, "sma_long_max": 300,
            "long_slope_len_min": 10, "long_slope_len_max": 30,
            "notes": f"fallback defaults (no data). Attempts: {'; '.join(tried_notes)}",
        }

    vol_class, trend_class, stats = _classify(df)

    # Breakout/exit
    if trend_class == "trending":   breakout_min, breakout_max = 50, 120
    elif trend_class == "mixed":    breakout_min, breakout_max = 35, 100
    else:                           breakout_min, breakout_max = 20, 60
    exit_min = max(10, int(breakout_min * 0.3))
    exit_max = max(exit_min + 1, int(breakout_max * 0.7))

    # ATR window
    atr_min, atr_max = (12, 28) if trend_class == "trending" else (10, 20)

    # ATR multiple & risk by vol
    if vol_class == "low":
        atr_multiple_min, atr_multiple_max = 1.5, 3.0
        risk_min, risk_max = 0.3, 1.0
    elif vol_class == "medium":
        atr_multiple_min, atr_multiple_max = 2.0, 3.5
        risk_min, risk_max = 0.2, 0.8
    else:
        atr_multiple_min, atr_multiple_max = 2.5, 4.0
        risk_min, risk_max = 0.1, 0.6

    # TP multiple
    tp_min, tp_max = (2.0, 6.0) if trend_class == "trending" else (0.0, 2.5)

    # Holding period & costs
    hold_min, hold_max = 0, max(120, int(breakout_max * 2))
    cost_min, cost_max = 1.0, 5.0

    allow_trend = trend_class != "choppy"
    sma_fast_min, sma_fast_max = 10, 40
    sma_slow_min, sma_slow_max = max(40, sma_fast_min + 10), 100
    sma_long_min, sma_long_max = 150, 300
    long_slope_len_min, long_slope_len_max = 10, 30

    notes = (f"{symbol}: vol={vol_class} (median ATR%={stats['median_atr_pct']:.2f}), "
             f"trend={trend_class} (median ADX={stats['median_adx']:.1f})")

    return {
        "breakout_min": breakout_min, "breakout_max": breakout_max,
        "exit_min": exit_min, "exit_max": exit_max,
        "atr_min": atr_min, "atr_max": atr_max,
        "atr_multiple_min": atr_multiple_min, "atr_multiple_max": atr_multiple_max,
        "risk_per_trade_min": risk_min, "risk_per_trade_max": risk_max,
        "tp_multiple_min": tp_min, "tp_multiple_max": tp_max,
        "holding_period_min": hold_min, "holding_period_max": hold_max,
        "cost_bps_min": cost_min, "cost_bps_max": cost_max,
        "allow_trend_filter": allow_trend,
        "sma_fast_min": sma_fast_min, "sma_fast_max": sma_fast_max,
        "sma_slow_min": sma_slow_min, "sma_slow_max": sma_slow_max,
        "sma_long_min": sma_long_min, "sma_long_max": sma_long_max,
        "long_slope_len_min": long_slope_len_min, "long_slope_len_max": long_slope_len_max,
        "notes": notes,
    }

def apply_bounds_to_streamlit_state(rec: Dict) -> None:
    """Best-effort: write suggested bounds into st.session_state using your current widget keys.
    Applies mins first, then maxes to avoid Streamlit constraint warnings.
    """
    try:
        import streamlit as st
    except Exception:
        return

    ss = st.session_state

    # Map incoming keys -> your widget keys
    # Note: risk_per_trade_* are in **percent** for the UI; rec already provides percent values.
    key_map_min = {
        "breakout_min": "b_breakout_min",
        "exit_min": "b_exit_min",
        "atr_min": "b_atr_min",
        "atr_multiple_min": "b_atrmult_min",
        "risk_per_trade_min": "b_rpt_min",
        "tp_multiple_min": "b_tp_min",
        "cost_bps_min": "b_cost_min",
        "holding_period_min": "b_hold_min",
        "sma_fast_min": "b_sma_fast_min",
        "sma_slow_min": "b_sma_slow_min",
        "sma_long_min": "b_sma_long_min",
        "long_slope_len_min": "b_slope_min",
    }
    key_map_max = {
        "breakout_max": "b_breakout_max",
        "exit_max": "b_exit_max",
        "atr_max": "b_atr_max",
        "atr_multiple_max": "b_atrmult_max",
        "risk_per_trade_max": "b_rpt_max",
        "tp_multiple_max": "b_tp_max",
        "cost_bps_max": "b_cost_max",
        "holding_period_max": "b_hold_max",
        "sma_fast_max": "b_sma_fast_max",
        "sma_slow_max": "b_sma_slow_max",
        "sma_long_max": "b_sma_long_max",
        "long_slope_len_max": "b_slope_max",
    }

    # Notes (for the caption)
    notes = rec.get("notes")
    if notes:
        ss["auto_bounds_notes"] = notes

    # First pass: mins
    for src_key, dst_key in key_map_min.items():
        if src_key in rec:
            try:
                ss[dst_key] = rec[src_key]
            except Exception:
                pass

    # Second pass: maxes — ensure > min where widgets enforce it
    for src_key, dst_key in key_map_max.items():
        if src_key in rec:
            try:
                if src_key.endswith("_max"):
                    min_src = src_key.replace("_max", "_min")
                    if min_src in rec:
                        # If equal or inverted, bump max slightly
                        try:
                            if float(rec[src_key]) <= float(rec[min_src]):
                                bump = 0.1 if any(k in src_key for k in ("multiple", "cost")) else 1
                                ss[dst_key] = float(rec[min_src]) + bump
                                continue
                        except Exception:
                            pass
                ss[dst_key] = rec[src_key]
            except Exception:
                pass

    # Boolean toggle
    if "allow_trend_filter" in rec:
        try:
            ss["b_allow_trend"] = bool(rec["allow_trend_filter"])  # your checkbox key
        except Exception:
            pass