# src/backtest/engine.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.data.cache import get_ohlcv_cached
from .metrics import summarize_equity, summarize_trades


@dataclass
class ATRParams:
    # Core Donchian/ATR model
    breakout_n: int = 55
    exit_n: int = 20
    atr_n: int = 14

    # Risk model
    atr_multiple: float = 3.0        # stop distance (ATR multiples)
    risk_per_trade: float = 0.01     # fraction of equity risked per position
    allow_fractional: bool = True

    # Costs model (per leg)
    slippage_bp: float = 5.0         # slippage basis points per leg
    cost_bps: float = 1.0            # commissions/fees (bps) per leg
    fee_per_trade: float = 0.0       # flat fee per leg (absolute $)

    # Optional extensions
    tp_multiple: Optional[float] = None   # take-profit target in ATRs (e.g., 2.0). None = off
    use_trend_filter: bool = False
    sma_fast: int = 30
    sma_slow: int = 50
    sma_long: int = 150
    long_slope_len: int = 15
    holding_period_limit: Optional[int] = None  # bars; None = off


def wilder_atr(df: pd.DataFrame, n: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def _entry_exit_levels(df: pd.DataFrame, breakout_n: int, exit_n: int) -> pd.DataFrame:
    out = df.copy()
    # Use previous N-day windows (shifted) so signals on day t execute at t+1 open
    out["breakout_high"] = (
        out["high"].rolling(window=breakout_n, min_periods=breakout_n).max().shift(1)
    )
    out["exit_low"] = (
        out["low"].rolling(window=exit_n, min_periods=exit_n).min().shift(1)
    )
    return out


def _compute_trend_cols(df: pd.DataFrame, p: ATRParams) -> pd.DataFrame:
    out = df.copy()
    out["sma_fast"] = out["close"].rolling(p.sma_fast, min_periods=p.sma_fast).mean()
    out["sma_slow"] = out["close"].rolling(p.sma_slow, min_periods=p.sma_slow).mean()
    out["sma_long"] = out["close"].rolling(p.sma_long, min_periods=p.sma_long).mean()
    out["sma_long_prev"] = out["sma_long"].shift(p.long_slope_len)
    out["long_slope_up"] = (out["sma_long"] - out["sma_long_prev"]) > 0
    out["fast_above_slow"] = out["sma_fast"] > out["sma_slow"]
    out["trend_ok"] = out["fast_above_slow"] & out["long_slope_up"]
    return out


def backtest_atr_breakout(
    symbol: str,
    start: str,
    end: str,
    starting_equity: float,
    params: ATRParams,
) -> Dict[str, object]:
    """
    Donchian-style breakout with ATR sizing, stop-loss, optional take-profit,
    optional trend filter, optional holding period limit, and a simple per-leg cost model.

    Execution model:
      • Signals are computed on bar t (using prior windows) and scheduled for execution at t+1 OPEN.
      • If in a position, we check for:
          - Gap below stop at OPEN → exit at OPEN (worse than stop).
          - Intraday stop: if LOW <= stop, exit at stop.
          - Gap above TP at OPEN → exit at OPEN (better than TP).
          - Intraday TP: if HIGH >= TP, exit at TP.
        (Stop checks run first to be conservative in ambiguous OHLC bars.)
      • Costs: per-leg bps = slippage_bp + cost_bps; buys use (1 + bps), sells use (1 - bps).

    Returns dict with:
      "equity": pd.Series of equity over time
      "trades": List[Dict] (trade log)
      "metrics": Dict[str, float]
    """
    df = get_ohlcv_cached(symbol, start, end).copy()
    if df.empty:
        raise ValueError(f"No data for {symbol} between {start} and {end}")

    # Indicators & levels
    df["atr"] = wilder_atr(df, params.atr_n)
    df = _entry_exit_levels(df, params.breakout_n, params.exit_n)
    if params.use_trend_filter:
        df = _compute_trend_cols(df, params)
    else:
        df["trend_ok"] = True

    # For next-day executions
    df["next_open"] = df["open"].shift(-1)

    # State
    cash = float(starting_equity)
    shares = 0.0
    stop_price: Optional[float] = None
    tp_price: Optional[float] = None
    entry_price: Optional[float] = None
    entry_date: Optional[pd.Timestamp] = None
    bars_in_trade: int = 0

    pending_entry = False
    pending_exit = False

    eq_dates: List[pd.Timestamp] = []
    eq_vals: List[float] = []
    trades: List[Dict] = []

    # unified per-leg cost
    bps = (params.slippage_bp + params.cost_bps) / 10000.0
    buy_mult = 1.0 + bps
    sell_mult = 1.0 - bps

    idx = df.index

    for i in range(len(df)):
        dt = idx[i]
        row = df.iloc[i]

        # --------- Intraday risk management (if in position) ----------
        if shares > 0:
            # 1) Gap below stop at open
            if stop_price is not None and row["open"] <= stop_price:
                exit_px = float(row["open"] * sell_mult)
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
                tp_price = None
                entry_price = None
                entry_date = None
                bars_in_trade = 0
                pending_exit = False  # cleared since we exited

            # 2) Intraday stop (conservative: prefer stop if both TP and stop touched)
            elif stop_price is not None and row["low"] <= stop_price:
                exit_px_raw = float(stop_price)
                exit_px = float(exit_px_raw * sell_mult)
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
                tp_price = None
                entry_price = None
                entry_date = None
                bars_in_trade = 0
                pending_exit = False

            # 3) Gap above TP at open
            elif tp_price is not None and row["open"] >= tp_price:
                exit_px = float(row["open"] * sell_mult)
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
                    "reason": "tp_gap",
                })
                shares = 0.0
                stop_price = None
                tp_price = None
                entry_price = None
                entry_date = None
                bars_in_trade = 0
                pending_exit = False

            # 4) Intraday TP
            elif tp_price is not None and row["high"] >= tp_price:
                exit_px_raw = float(tp_price)
                exit_px = float(exit_px_raw * sell_mult)
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
                    "reason": "tp_intraday",
                })
                shares = 0.0
                stop_price = None
                tp_price = None
                entry_price = None
                entry_date = None
                bars_in_trade = 0
                pending_exit = False

        # --------- Execute scheduled orders (from previous day) at today's OPEN ----------
        if i > 0:
            # Execute entry
            if pending_entry and shares == 0 and not np.isnan(row["open"]):
                px = float(row["open"] * buy_mult)
                prev = df.iloc[i - 1]
                atr_val = float(prev["atr"]) if not np.isnan(prev["atr"]) else None
                if atr_val and atr_val > 0:
                    risk_per_share = max(params.atr_multiple * atr_val, 1e-12)
                    target_risk_dollars = params.risk_per_trade * cash  # equity≈cash if flat
                    raw_shares = target_risk_dollars / risk_per_share
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
                            bars_in_trade = 0
                            stop_price = px - params.atr_multiple * atr_val
                            tp_price = (
                                px + (params.tp_multiple * atr_val)
                                if params.tp_multiple and params.tp_multiple > 0
                                else None
                            )
                pending_entry = False

            # Execute exit
            if pending_exit and shares > 0 and not np.isnan(row["open"]):
                px = float(row["open"] * sell_mult)
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
                tp_price = None
                entry_price = None
                entry_date = None
                bars_in_trade = 0
                pending_exit = False

        # --------- Generate today's signals (schedule for next bar) ----------
        if i < len(df) - 1:
            if shares == 0:
                # Entry signal: breakout + (optional) trend filter
                if (
                    not np.isnan(row["breakout_high"])
                    and row["close"] > row["breakout_high"]
                    and bool(row.get("trend_ok", True))
                ):
                    pending_entry = True
                    pending_exit = False
            else:
                # Exit signal: price below exit channel
                if not np.isnan(row["exit_low"]) and row["close"] < row["exit_low"]:
                    pending_exit = True

                # Holding period limit schedules exit for next bar
                if (
                    params.holding_period_limit is not None
                    and params.holding_period_limit > 0
                    and bars_in_trade + 1 >= params.holding_period_limit
                ):
                    pending_exit = True

        # --------- Mark-to-market equity, update holding counter ----------
        equity_val = cash + shares * float(row["close"])
        eq_dates.append(pd.Timestamp(dt))
        eq_vals.append(float(equity_val))

        if shares > 0:
            bars_in_trade += 1

    # Build outputs
    equity = (
        pd.Series(eq_vals, index=pd.to_datetime(eq_dates), name="equity").astype(float)
        if eq_vals else pd.Series(dtype=float, name="equity")
    )

    eq_summary = summarize_equity(equity, starting_equity)
    tr_summary = summarize_trades(trades)
    metrics = {**eq_summary, **tr_summary, "symbol": symbol}

    return {
        "equity": equity,
        "trades": trades,
        "metrics": metrics,
    }