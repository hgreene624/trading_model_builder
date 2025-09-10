# src/backtest/engine.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from src.data.cache import get_ohlcv_cached
from .metrics import summarize_equity, summarize_trades


@dataclass
class ATRParams:
    breakout_n: int = 55
    exit_n: int = 20
    atr_n: int = 14
    atr_multiple: float = 3.0        # stop distance in ATRs
    risk_per_trade: float = 0.01     # fraction of equity risked per position
    allow_fractional: bool = True
    slippage_bp: float = 5.0         # round-trip modeled per trade leg in bps (0.0001)
    fee_per_trade: float = 0.0       # flat fee per trade leg


def wilder_atr(df: pd.DataFrame, n: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def _entry_exit_levels(df: pd.DataFrame, breakout_n: int, exit_n: int) -> pd.DataFrame:
    out = df.copy()
    out["breakout_high"] = out["high"].rolling(window=breakout_n, min_periods=breakout_n).max().shift(1)
    out["exit_low"] = out["low"].rolling(window=exit_n, min_periods=exit_n).min().shift(1)
    return out


def backtest_atr_breakout(
    symbol: str,
    start: str,
    end: str,
    starting_equity: float,
    params: ATRParams,
) -> Dict[str, object]:
    """
    Donchian-style breakout with ATR stop & sizing.
    Execution model:
      - Signals evaluated on day t using previous-window levels.
      - Entries/exits on the next day's OPEN (t+1), with slippage/fees.
      - Hard stop is checked intraday on each bar: if low <= stop, exit on that bar
        at stop price (or at OPEN if gap below stop).
    Returns:
      {
        "equity": pd.Series (daily equity),
        "trades": List[Dict],  # trade log
        "metrics": Dict[str, float]  # summary metrics
      }
    """
    df = get_ohlcv_cached(symbol, start, end).copy()
    if df.empty:
        raise ValueError(f"No data for {symbol} between {start} and {end}")

    df["atr"] = wilder_atr(df, params.atr_n)
    df = _entry_exit_levels(df, params.breakout_n, params.exit_n)

    # We need the next day OPEN to execute scheduled orders
    df["next_open"] = df["open"].shift(-1)

    # State
    cash = float(starting_equity)
    shares = 0.0
    stop_price: Optional[float] = None
    entry_price: Optional[float] = None
    entry_date: Optional[pd.Timestamp] = None

    pending_entry = False
    pending_exit = False

    eq_dates: List[pd.Timestamp] = []
    eq_vals: List[float] = []
    trades: List[Dict] = []

    slippage = params.slippage_bp / 10000.0  # convert bps to fraction

    idx = df.index

    for i in range(len(df)):
        dt = idx[i]
        row = df.iloc[i]

        # --- Intraday stop check (if in position) ---
        if shares > 0:
            # If gap down below stop at the open, we exit at OPEN (worse than stop).
            if row["open"] <= (stop_price or -np.inf):
                exit_px = float(row["open"] * (1 - slippage))
                cash += shares * exit_px - params.fee_per_trade
                pnl = (exit_px - (entry_price or exit_px)) * shares
                trades.append({
                    "symbol": symbol,
                    "entry_date": entry_date.date() if entry_date is not None else None,
                    "entry_price": float(entry_price),
                    "exit_date": pd.Timestamp(dt).date(),
                    "exit_price": float(exit_px),
                    "shares": float(shares),
                    "pnl": float(pnl),
                    "return_pct": float((exit_px / (entry_price or exit_px)) - 1.0),
                    "holding_days": int((pd.Timestamp(dt) - (entry_date or pd.Timestamp(dt))).days),
                    "reason": "stop_gap",
                })
                shares = 0.0
                stop_price = None
                entry_price = None
                entry_date = None
                pending_exit = False  # clear any scheduled exit since we already exited

            # If not gapped below at open, but low pierced stop intra-day, exit at stop (with slippage)
            elif row["low"] <= (stop_price or -np.inf):
                exit_px_raw = float(stop_price or 0.0)
                exit_px = float(exit_px_raw * (1 - slippage))
                cash += shares * exit_px - params.fee_per_trade
                pnl = (exit_px - (entry_price or exit_px)) * shares
                trades.append({
                    "symbol": symbol,
                    "entry_date": entry_date.date() if entry_date is not None else None,
                    "entry_price": float(entry_price),
                    "exit_date": pd.Timestamp(dt).date(),
                    "exit_price": float(exit_px),
                    "shares": float(shares),
                    "pnl": float(pnl),
                    "return_pct": float((exit_px / (entry_price or exit_px)) - 1.0),
                    "holding_days": int((pd.Timestamp(dt) - (entry_date or pd.Timestamp(dt))).days),
                    "reason": "stop_intraday",
                })
                shares = 0.0
                stop_price = None
                entry_price = None
                entry_date = None
                pending_exit = False

        # --- Execute scheduled orders at today's OPEN (after handling stop) ---
        if i > 0:
            prev = df.iloc[i - 1]
            # Execute entry scheduled yesterday
            if pending_entry and shares == 0 and not np.isnan(row["open"]):
                px = float(row["open"] * (1 + slippage))
                # Sizing by ATR risk
                atr_val = float(prev["atr"]) if not np.isnan(prev["atr"]) else None
                if atr_val and atr_val > 0:
                    risk_per_share = params.atr_multiple * atr_val
                    target_risk_dollars = params.risk_per_trade * (cash)  # equity == cash if flat
                    raw_shares = target_risk_dollars / max(risk_per_share, 1e-12)
                    # Cash constraint
                    affordable = cash / max(px, 1e-12)
                    if params.allow_fractional:
                        use_shares = min(raw_shares, affordable)
                    else:
                        use_shares = float(np.floor(min(raw_shares, affordable)))
                    if use_shares > (0.0 if params.allow_fractional else 0.5):
                        cost = use_shares * px + params.fee_per_trade
                        if cost <= cash + 1e-9:
                            cash -= cost
                            shares = use_shares
                            entry_price = px
                            entry_date = pd.Timestamp(dt)
                            stop_price = px - params.atr_multiple * atr_val
                pending_entry = False

            # Execute exit scheduled yesterday (if still in position)
            if pending_exit and shares > 0 and not np.isnan(row["open"]):
                px = float(row["open"] * (1 - slippage))
                cash += shares * px - params.fee_per_trade
                pnl = (px - (entry_price or px)) * shares
                trades.append({
                    "symbol": symbol,
                    "entry_date": entry_date.date() if entry_date is not None else None,
                    "entry_price": float(entry_price),
                    "exit_date": pd.Timestamp(dt).date(),
                    "exit_price": float(px),
                    "shares": float(shares),
                    "pnl": float(pnl),
                    "return_pct": float((px / (entry_price or px)) - 1.0),
                    "holding_days": int((pd.Timestamp(dt) - (entry_date or pd.Timestamp(dt))).days),
                    "reason": "rule_exit",
                })
                shares = 0.0
                stop_price = None
                entry_price = None
                entry_date = None
                pending_exit = False

        # --- Generate today's signals to schedule for next bar (if not already exited by stop) ---
        if i < len(df) - 1:  # only schedule if we have a next bar
            if shares == 0:
                # Entry: close > prior breakout_high
                if not np.isnan(row["breakout_high"]) and row["close"] > row["breakout_high"]:
                    pending_entry = True
                    pending_exit = False
            else:
                # Exit: close < prior exit_low
                if not np.isnan(row["exit_low"]) and row["close"] < row["exit_low"]:
                    pending_exit = True

        # --- Mark-to-market equity at today's close ---
        equity_val = cash + shares * float(row["close"])
        eq_dates.append(pd.Timestamp(dt))
        eq_vals.append(float(equity_val))

    equity = pd.Series(eq_vals, index=pd.to_datetime(eq_dates), name="equity").astype(float)

    # Summaries
    eq_summary = summarize_equity(equity, starting_equity)
    tr_summary = summarize_trades(trades)

    metrics = {
        **eq_summary,
        **tr_summary,
        "symbol": symbol,
    }
    return {
        "equity": equity,
        "trades": trades,
        "metrics": metrics,
    }