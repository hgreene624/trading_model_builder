# src/backtest/engine.py
"""
Backtest engine for ATR breakout (long-only by default).

This file defines:
- ATRParams: dataclass of strategy parameters
- wilder_atr(): ATR indicator
- backtest_atr_breakout(): run the backtest and produce BacktestResult
- CONTRACT docstrings for trades/meta so other modules remain decoupled

IMPORTANT: This engine **does not** do any UI. It relies on `src.data.loader.get_ohlcv`
to fetch bars. Adjust that import to your real data source.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---- Data loader (swap to your real import) ----
try:
    from src.data.loader import get_ohlcv
except Exception:
    # Minimal placeholder so the file is importable; replace in your repo.
    def get_ohlcv(symbol: str, start, end) -> pd.DataFrame:
        raise RuntimeError("Implement src.data.loader.get_ohlcv(symbol, start, end)")

# ---- Metrics helper (only type reference; do NOT compute here) ----
# The trainer will call metrics.compute_core_metrics on the result.
# from src.backtest.metrics import compute_core_metrics

# ---------- CONTRACTS ----------

"""
BacktestResult = {
    "equity": pd.Series,               # cumulative equity
    "daily_returns": pd.Series,        # daily pct returns
    "trades": list[dict],              # see Trade dict below
    "meta": dict,                      # symbol, params, costs, date range, exec_mode, notes
}

Trade = {
    "entry_time": pd.Timestamp,
    "exit_time": pd.Timestamp,
    "entry_price": float,
    "exit_price": float,
    "side": "long"|"short",
    "return_pct": float,
    "holding_days": int,
    "mfe": float, "mae": float,               # return terms (e.g., +0.03, -0.02)
    "day_low": float, "day_high": float,      # entry day range
    "day_low_exit": float, "day_high_exit": float,  # exit day range
    "decision_price": float, "fill_price": float    # optional execution diag (if you support live)
}
"""

# ---------- Parameters ----------

@dataclass
class ATRParams:
    breakout_n: int = 20             # recent high lookback to trigger long entry
    exit_n: int = 10                  # recent low lookback for exit
    atr_n: int = 14
    atr_multiple: float = 2.0         # volatility filter: price > rolling_high - k * ATR
    tp_multiple: float = 0.0          # optional profit target (in ATRs). 0 disables.
    holding_period_limit: int = 0     # optional max holding days. 0 disables.
    allow_short: bool = False         # optional shorting; default off for simplicity

# ---------- Indicators ----------

def wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's ATR (EMA of True Range)."""
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False).mean()

# ---------- Core backtest ----------

def backtest_atr_breakout(
    symbol: str,
    start,
    end,
    starting_equity: float,
    params: ATRParams | Dict,
    execution: str = "close",   # "close" or "next_open" (if you have opens)
    commission_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> Dict:
    """
    Simple daily ATR breakout, long-only by default.

    Entry:
      - Long when close > rolling_high(breakout_n) - atr_multiple * ATR
    Exit:
      - Exit when close < rolling_low(exit_n)
      - Or when tp_multiple ATR target hit (if tp_multiple>0)
      - Or when holding_period_limit reached (if >0)

    This is intentionally simple and readable. All timing diagnostics (MFE/MAE, efficiencies)
    are computed from available bars within each holding window.
    """
    if isinstance(params, dict):
        params = ATRParams(**params)

    df = get_ohlcv(symbol, start, end).copy()
    if df.empty:
        return {
            "equity": pd.Series(dtype=float),
            "daily_returns": pd.Series(dtype=float),
            "trades": [],
            "meta": {"symbol": symbol, "start": start, "end": end, "params": params.__dict__},
        }

    # Ensure basic columns
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"Data for {symbol} missing required column '{col}'")

    df = df.sort_index()
    atr = wilder_atr(df["high"], df["low"], df["close"], n=params.atr_n)
    roll_high = df["close"].rolling(params.breakout_n).max()
    roll_low = df["close"].rolling(params.exit_n).min()

    # Entry trigger (volatility-adjusted)
    entry_th = roll_high - params.atr_multiple * atr
    long_signal = df["close"] > entry_th

    # State machine
    in_pos = False
    entry_px = np.nan
    entry_idx = None
    trades: List[Dict] = []

    equity = [starting_equity]
    # (we’ll mark-to-market as either flat or 1x fully invested; position sizing deliberately omitted here)
    position_qty = 0.0

    # Iterate daily
    for i, (ts, row) in enumerate(df.iterrows()):
        price = float(row["close"])
        day_lo = float(row["low"])
        day_hi = float(row["high"])

        # Entry
        if not in_pos and bool(long_signal.iloc[i]):
            in_pos = True
            entry_px = price if execution == "close" else float(row.get("open", price))
            # apply immediate costs on entry fill
            fill_mult = 1.0 + (commission_bps + slippage_bps) / 10_000.0
            entry_fill = entry_px * fill_mult
            entry_idx = i
            entry_time = ts
            # buy 1 unit notionally (equity accounting here is synthetic; sizing logic is left to portfolio layer)
            position_qty = equity[-1] / entry_fill  # all-in for didactic simplicity
            decision_px = entry_px  # same as entry for backtest; separate in live

        # Exit checks
        exit_reason = None
        do_exit = False
        exit_px = None

        if in_pos:
            # Exit on rolling low break
            if price < roll_low.iloc[i]:
                do_exit = True
                exit_reason = "roll_low"

            # Exit on TP in ATRs (check intraday range)
            if not do_exit and params.tp_multiple > 0:
                target = entry_px * (1.0 + params.tp_multiple * atr.iloc[entry_idx] / entry_px)
                # if day's high reached target intraday
                if day_hi >= target:
                    do_exit = True
                    exit_reason = "tp_hit_intraday"
                    # assume exit at target (conservative could be near close)
                    exit_px = target

            # Exit on holding period
            if not do_exit and params.holding_period_limit > 0:
                held = i - entry_idx
                if held >= params.holding_period_limit:
                    do_exit = True
                    exit_reason = "holding_limit"

        # Commit exit
        if in_pos and do_exit:
            # Exit price
            if exit_px is None:
                exit_px = price if execution == "close" else float(row.get("open", price))
            # apply costs on exit
            exit_fill = exit_px * (1.0 - (commission_bps + slippage_bps) / 10_000.0)

            # Compute MFE/MAE over holding window
            win_slice = df.iloc[entry_idx:i+1]
            # returns relative to entry
            rel = (win_slice["high"] / entry_px) - 1.0
            mfe = float(rel.max()) if len(rel) else 0.0
            rel2 = (win_slice["low"] / entry_px) - 1.0
            mae = float(rel2.min()) if len(rel2) else 0.0

            trade_ret = (exit_fill - entry_fill) / entry_fill
            holding_days = (ts - entry_time).days or (i - entry_idx)

            trades.append({
                "entry_time": entry_time,
                "exit_time": ts,
                "entry_price": float(entry_fill),
                "exit_price": float(exit_fill),
                "side": "long",
                "return_pct": float(trade_ret),
                "holding_days": int(holding_days),
                "mfe": float(mfe),
                "mae": float(mae),
                "day_low": float(df["low"].iloc[entry_idx]),
                "day_high": float(df["high"].iloc[entry_idx]),
                "day_low_exit": day_lo,
                "day_high_exit": day_hi,
                "decision_price": float(decision_px),
                "fill_price": float(entry_fill),
                "exit_reason": exit_reason,
            })

            # Flat after exit; mark equity to cash
            equity.append(position_qty * exit_fill)
            position_qty = 0.0
            in_pos = False
            entry_px = np.nan
            entry_idx = None
            entry_time = None

        else:
            # Mark-to-market
            if in_pos and position_qty > 0:
                equity.append(position_qty * price)
            else:
                equity.append(equity[-1])

    eq = pd.Series(equity[1:], index=df.index)  # drop initial seed
    daily_returns = eq.pct_change().fillna(0.0)

    return {
        "equity": eq,
        "daily_returns": daily_returns,
        "trades": trades,
        "meta": {
            "symbol": symbol,
            "start": pd.Timestamp(start),
            "end": pd.Timestamp(end),
            "params": params.__dict__,
            "execution": execution,
            "commission_bps": commission_bps,
            "slippage_bps": slippage_bps,
            "notes": "ATR breakout long-only reference engine",
        },
    }