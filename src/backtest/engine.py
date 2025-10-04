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
import os
import math
from typing import Any, Dict, List, Optional, Tuple

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
    "equity": pd.Series,               # cumulative equity (post-cost)
    "daily_returns": pd.Series,        # daily pct returns (post-cost)
    "trades": list[dict],              # see Trade dict below
    "meta": dict,                      # symbol, params, costs, date range, exec_mode, notes
    "equity_pre_cost": pd.Series,      # optional (gross equity curve)
    "daily_returns_pre_cost": pd.Series,
    "trades_df": pd.DataFrame,         # enriched trade log with cost attribution
}

Trade = {
    "entry_time": pd.Timestamp,
    "exit_time": pd.Timestamp,
    "entry_price": float,              # cost-adjusted fill price (per share)
    "exit_price": float,
    "side": "long"|"short",
    "return_pct": float,               # net of costs
    "holding_days": int,
    "mfe": float, "mae": float,               # return terms (e.g., +0.03, -0.02)
    "day_low": float, "day_high": float,      # entry day range
    "day_low_exit": float, "day_high_exit": float,  # exit day range
    "decision_price": float, "fill_price": float,   # execution diagnostics
    # --- Optional cost diagnostics ---
    "gross_entry_price": float,
    "gross_exit_price": float,
    "entry_slippage_bps": float,
    "entry_fee_bps": float,
    "exit_slippage_bps": float,
    "exit_fee_bps": float,
    "signal_time": pd.Timestamp | None,
    "signal_price": float | None,
    "exit_signal_time": pd.Timestamp | None,
    "exit_signal_price": float | None,
    "quantity": float,
    "notional_entry": float,
    "notional_exit": float,
    "entry_slippage_cost": float,
    "entry_fee_cost": float,
    "entry_fixed_cost": float,
    "exit_slippage_cost": float,
    "exit_fee_cost": float,
    "exit_fixed_cost": float,
    "total_cost": float,
    "gross_return_pct": float,
    "gross_pnl": float,
    "net_pnl": float,
}
"""

# ---------- Cost model helpers ----------


def _coerce_float(value, default: float = 0.0) -> float:
    try:
        f = float(value)
    except Exception:
        return default
    if not math.isfinite(f):
        return default
    return f


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float = 0.0) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    return _coerce_float(raw, default=default)


@dataclass
class CostModel:
    commission_bps: float = 0.0
    slippage_bps: float = 0.0
    per_trade_fee: float = 0.0
    enabled: bool = False

    @classmethod
    def from_inputs(
        cls,
        commission_bps: float = 0.0,
        slippage_bps: float = 0.0,
        per_trade_fee: float = 0.0,
        enabled: bool = False,
    ) -> "CostModel":
        commission_bps = max(0.0, _coerce_float(commission_bps, 0.0))
        slippage_bps = max(0.0, _coerce_float(slippage_bps, 0.0))
        per_trade_fee = max(0.0, _coerce_float(per_trade_fee, 0.0))
        active = enabled or (commission_bps > 0.0 or slippage_bps > 0.0 or per_trade_fee > 0.0)
        return cls(
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            per_trade_fee=per_trade_fee,
            enabled=bool(active),
        )

    def total_bps(self) -> float:
        return float(self.commission_bps + self.slippage_bps)

    def as_dict(self) -> dict:
        return {
            "commission_bps": float(self.commission_bps),
            "slippage_bps": float(self.slippage_bps),
            "per_trade_fee": float(self.per_trade_fee),
            "enabled": bool(self.enabled),
        }

    def compute_fill(
        self,
        side: str,
        action: str,
        price: float,
        qty: float,
    ) -> dict:
        price = _coerce_float(price, 0.0)
        qty = _coerce_float(qty, 0.0)
        if not self.enabled or price <= 0.0 or qty <= 0.0:
            return {
                "fill_price": price,
                "slippage_cost": 0.0,
                "commission_cost": 0.0,
                "fee_cost": 0.0,
                "total_cost": 0.0,
                "slippage_bps": 0.0,
                "commission_bps": 0.0,
            }

        total_bps = self.total_bps() / 10_000.0
        notional = price * qty
        slippage_cost = notional * (self.slippage_bps / 10_000.0)
        commission_cost = notional * (self.commission_bps / 10_000.0)
        fee_cost = self.per_trade_fee

        # Adjust fill directionally based on side/action
        direction = 1.0
        if side == "long":
            direction = 1.0 if action == "entry" else -1.0
        elif side == "short":
            direction = -1.0 if action == "entry" else 1.0

        fill_adjust = direction * total_bps
        fill_price = price * (1.0 + fill_adjust)

        return {
            "fill_price": float(fill_price),
            "slippage_cost": float(slippage_cost),
            "commission_cost": float(commission_cost),
            "fee_cost": float(fee_cost),
            "total_cost": float(slippage_cost + commission_cost + fee_cost),
            "slippage_bps": float(self.slippage_bps),
            "commission_bps": float(self.commission_bps),
        }

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
    cost_bps: float = 0.0             # legacy single cost slider (mapped into slippage)
    commission_bps: float = 0.0       # explicit commission component
    slippage_bps: float = 0.0         # explicit slippage component
    per_trade_fee: float = 0.0        # flat fee per fill (in account currency)
    enable_costs: bool = False        # toggle attribution / post-cost accounting
    delay_bars: int = 0               # optional execution delay (signal -> fill after N bars)

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
    enable_costs: Optional[bool] = None,
    cost_model: Optional[CostModel] = None,
    delay_bars: Optional[int] = None,
    capture_trades_df: Optional[bool] = None,
) -> Dict:
    """
    Simple daily ATR breakout, long-only by default.

    Adds optional execution delay (signal on close → execute at future open)
    and cost attribution via a CostModel that applies slippage/fees at fills.
    Defaults preserve the original behaviour (costs off, delay=0).
    """

    extra_params: Dict[str, Any] = {}
    if isinstance(params, dict):
        params_dict = dict(params)
        known = set(ATRParams.__dataclass_fields__.keys())
        params_core = {k: params_dict[k] for k in list(params_dict.keys()) if k in known}
        extra_params = {k: params_dict[k] for k in params_dict if k not in known}
        params = ATRParams(**params_core)

    delay_candidate = _coerce_float(getattr(params, "delay_bars", 0), 0.0)
    if delay_bars is not None:
        delay_candidate = _coerce_float(delay_bars, delay_candidate)
    env_delay = _env_float("ATR_EXECUTION_DELAY_BARS", delay_candidate)
    delay_bars = int(max(0, _coerce_float(env_delay, delay_candidate)))

    if capture_trades_df is None:
        capture_trades_df = _env_flag("ATR_CAPTURE_TRADES_DF", True)
    capture_trades_df = bool(capture_trades_df)

    param_enable_costs = bool(getattr(params, "enable_costs", False))
    env_enable_costs = _env_flag("ATR_ENABLE_COSTS", False)
    if enable_costs is None:
        enable_costs = param_enable_costs or env_enable_costs
    else:
        enable_costs = bool(enable_costs)

    commission_cfg = _coerce_float(getattr(params, "commission_bps", commission_bps), 0.0)
    slippage_cfg = _coerce_float(getattr(params, "slippage_bps", slippage_bps), 0.0)
    if commission_bps:
        commission_cfg = _coerce_float(commission_bps, commission_cfg)
    if slippage_bps:
        slippage_cfg = _coerce_float(slippage_bps, slippage_cfg)
    if commission_cfg == 0.0 and slippage_cfg == 0.0:
        slippage_cfg = max(slippage_cfg, _coerce_float(getattr(params, "cost_bps", 0.0), 0.0))
    fee_cfg = _coerce_float(getattr(params, "per_trade_fee", 0.0), 0.0)

    commission_cfg = _env_float("ATR_COMMISSION_BPS", commission_cfg)
    slippage_cfg = _env_float("ATR_SLIPPAGE_BPS", slippage_cfg)
    fee_cfg = _env_float("ATR_PER_TRADE_FEE", fee_cfg)

    if cost_model is None:
        cost_model = CostModel.from_inputs(
            commission_bps=commission_cfg,
            slippage_bps=slippage_cfg,
            per_trade_fee=fee_cfg,
            enabled=enable_costs,
        )
    else:
        cost_model = CostModel.from_inputs(
            commission_bps=getattr(cost_model, "commission_bps", commission_cfg),
            slippage_bps=getattr(cost_model, "slippage_bps", slippage_cfg),
            per_trade_fee=getattr(cost_model, "per_trade_fee", fee_cfg),
            enabled=enable_costs or getattr(cost_model, "enabled", False),
        )
    enable_costs = bool(cost_model.enabled)

    df = get_ohlcv(symbol, start, end).copy()
    base_meta = {
        "symbol": symbol,
        "start": pd.Timestamp(start),
        "end": pd.Timestamp(end),
        "params": dict(params.__dict__),
        "extra_params": extra_params,
        "execution": execution,
        "delay_bars": delay_bars,
        "cost_model": cost_model.as_dict(),
        "notes": "ATR breakout long-only reference engine",
    }

    if df.empty:
        empty_series = pd.Series(dtype=float)
        meta = dict(base_meta)
        meta["costs"] = {
            "enabled": enable_costs,
            "summary": {
                "turnover": 0.0,
                "weighted_slippage_bps": 0.0,
                "weighted_fees_bps": 0.0,
                "pre_cost_cagr": 0.0,
                "post_cost_cagr": 0.0,
                "annualized_drag": 0.0,
                "total_cost": 0.0,
            },
        }
        return {
            "equity": empty_series,
            "daily_returns": empty_series,
            "trades": [],
            "equity_pre_cost": empty_series,
            "daily_returns_pre_cost": empty_series,
            "trades_df": pd.DataFrame(),
            "meta": meta,
        }

    # Ensure basic columns
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"Data for {symbol} missing required column '{col}'")

    df = df.sort_index()
    atr = wilder_atr(df["high"], df["low"], df["close"], n=params.atr_n)
    roll_high = df["close"].rolling(params.breakout_n).max()
    roll_low = df["close"].rolling(params.exit_n).min()

    entry_th = roll_high - params.atr_multiple * atr
    long_signal = df["close"] > entry_th

    in_pos = False
    entry_idx: Optional[int] = None
    entry_time: Optional[pd.Timestamp] = None
    entry_decision_price = 0.0
    entry_signal_time: Optional[pd.Timestamp] = None
    entry_signal_price: Optional[float] = None
    entry_breakdown: Dict[str, float] = {"total_cost": 0.0, "slippage_cost": 0.0, "commission_cost": 0.0, "fee_cost": 0.0,
                                         "slippage_bps": 0.0, "commission_bps": 0.0, "fill_price": 0.0}
    entry_notional = 0.0

    cash_gross = float(starting_equity)
    cash_net = float(starting_equity)
    position_qty = 0.0

    equity_gross = [float(starting_equity)]
    equity_net = [float(starting_equity)]

    trades: List[Dict] = []
    trade_rows: List[Dict] = []

    pending_entry: Optional[Dict[str, Any]] = None
    pending_exit: Optional[Dict[str, Any]] = None

    index_list = list(df.index)
    n_bars = len(index_list)

    def _safe_price(value: Any, fallback: float) -> float:
        out = _coerce_float(value, fallback)
        if out <= 0.0:
            return fallback
        return out

    last_close_price: Optional[float] = None

    for i, (ts, row) in enumerate(df.iterrows()):
        price_close = _safe_price(row.get("close"), last_close_price if last_close_price is not None else 0.0)
        if price_close <= 0.0 and last_close_price is not None:
            price_close = last_close_price
        if price_close > 0.0:
            last_close_price = price_close
        day_lo = _safe_price(row.get("low"), price_close)
        day_hi = _safe_price(row.get("high"), price_close)
        price_open = _safe_price(row.get("open"), price_close)

        exit_trigger: Optional[Dict[str, Any]] = None
        entry_trigger: Optional[Dict[str, Any]] = None

        if pending_exit and in_pos:
            if i - pending_exit.get("signal_idx", i) >= delay_bars:
                exit_trigger = dict(pending_exit)
                exit_trigger["from_pending"] = True
                pending_exit = None

        if pending_entry and not in_pos:
            if i - pending_entry.get("signal_idx", i) >= delay_bars:
                entry_trigger = dict(pending_entry)
                entry_trigger["from_pending"] = True
                pending_entry = None

        if in_pos and exit_trigger is None:
            exit_reason = None
            exit_px = None
            if price_close < roll_low.iloc[i]:
                exit_reason = "roll_low"
            if exit_reason is None and params.tp_multiple > 0 and entry_idx is not None:
                atr_at_entry = atr.iloc[entry_idx]
                if np.isfinite(atr_at_entry) and entry_decision_price > 0:
                    target = entry_decision_price * (1.0 + params.tp_multiple * atr_at_entry / entry_decision_price)
                    if day_hi >= target:
                        exit_reason = "tp_hit_intraday"
                        exit_px = target
            if exit_reason is None and params.holding_period_limit > 0 and entry_idx is not None:
                if (i - entry_idx) >= params.holding_period_limit:
                    exit_reason = "holding_limit"

            if exit_reason is not None:
                trigger = {
                    "signal_idx": i,
                    "signal_time": ts,
                    "signal_price": price_close,
                    "reason": exit_reason,
                }
                if exit_px is not None:
                    trigger["override_price"] = exit_px
                if delay_bars > 0 and exit_reason != "tp_hit_intraday":
                    pending_exit = trigger
                else:
                    exit_trigger = trigger

        if in_pos and exit_trigger is None and i == n_bars - 1:
            exit_trigger = {
                "signal_idx": i,
                "signal_time": ts,
                "signal_price": price_close,
                "reason": "final_bar_flatten",
            }

        if not in_pos and entry_trigger is None and bool(long_signal.iloc[i]):
            trigger = {
                "signal_idx": i,
                "signal_time": ts,
                "signal_price": price_close,
            }
            if delay_bars > 0:
                pending_entry = trigger
            else:
                entry_trigger = trigger

        if exit_trigger is not None and in_pos and position_qty > 0:
            base_price = exit_trigger.get("override_price")
            if base_price is None:
                if exit_trigger.get("from_pending") and delay_bars > 0:
                    base_price = price_open
                else:
                    base_price = price_close if execution == "close" else price_open
            exit_price = _safe_price(base_price, price_close)

            qty = position_qty
            proceeds = exit_price * qty
            cash_gross += proceeds
            cash_net += proceeds
            exit_breakdown = cost_model.compute_fill("long", "exit", exit_price, qty)
            cash_net -= exit_breakdown["total_cost"]
            exit_fill = exit_breakdown["fill_price"]

            mfe = mae = 0.0
            if entry_idx is not None:
                win_slice = df.iloc[entry_idx:i+1]
                if not win_slice.empty and entry_decision_price > 0:
                    rel = (win_slice["high"] / entry_decision_price) - 1.0
                    rel2 = (win_slice["low"] / entry_decision_price) - 1.0
                    mfe = float(rel.max()) if len(rel) else 0.0
                    mae = float(rel2.min()) if len(rel2) else 0.0

            gross_pnl = (exit_price - entry_decision_price) * qty
            entry_cost_total = entry_breakdown.get("total_cost", 0.0)
            net_pnl = gross_pnl - entry_cost_total - exit_breakdown["total_cost"]
            notional_exit = exit_price * qty
            trade_ret = net_pnl / entry_notional if entry_notional else 0.0
            holding_days = 0
            if entry_time is not None:
                holding_days = (ts - entry_time).days or (i - (entry_idx or i))

            trade_record = {
                "entry_time": entry_time,
                "exit_time": ts,
                "entry_price": float(entry_breakdown.get("fill_price", entry_decision_price)),
                "exit_price": float(exit_fill),
                "side": "long",
                "return_pct": float(trade_ret),
                "holding_days": int(holding_days),
                "mfe": float(mfe),
                "mae": float(mae),
                "day_low": float(df["low"].iloc[entry_idx]) if entry_idx is not None else float("nan"),
                "day_high": float(df["high"].iloc[entry_idx]) if entry_idx is not None else float("nan"),
                "day_low_exit": float(day_lo),
                "day_high_exit": float(day_hi),
                "decision_price": float(entry_decision_price),
                "fill_price": float(entry_breakdown.get("fill_price", entry_decision_price)),
                "exit_reason": exit_trigger.get("reason"),
                "gross_entry_price": float(entry_decision_price),
                "gross_exit_price": float(exit_price),
                "entry_slippage_bps": float(entry_breakdown.get("slippage_bps", 0.0)),
                "entry_fee_bps": float(entry_breakdown.get("commission_bps", 0.0)),
                "exit_slippage_bps": float(exit_breakdown.get("slippage_bps", 0.0)),
                "exit_fee_bps": float(exit_breakdown.get("commission_bps", 0.0)),
                "signal_time": entry_signal_time,
                "signal_price": float(entry_signal_price) if entry_signal_price is not None else None,
                "exit_signal_time": exit_trigger.get("signal_time"),
                "exit_signal_price": float(exit_trigger.get("signal_price")) if exit_trigger.get("signal_price") is not None else None,
                "quantity": float(qty),
                "notional_entry": float(entry_notional),
                "notional_exit": float(notional_exit),
                "entry_slippage_cost": float(entry_breakdown.get("slippage_cost", 0.0)),
                "entry_fee_cost": float(entry_breakdown.get("commission_cost", 0.0)),
                "entry_fixed_cost": float(entry_breakdown.get("fee_cost", 0.0)),
                "exit_slippage_cost": float(exit_breakdown.get("slippage_cost", 0.0)),
                "exit_fee_cost": float(exit_breakdown.get("commission_cost", 0.0)),
                "exit_fixed_cost": float(exit_breakdown.get("fee_cost", 0.0)),
                "total_cost": float(entry_cost_total + exit_breakdown["total_cost"]),
                "gross_return_pct": float((exit_price / entry_decision_price) - 1.0) if entry_decision_price > 0 else 0.0,
                "gross_pnl": float(gross_pnl),
                "net_pnl": float(net_pnl),
            }

            trades.append(trade_record)
            trade_rows.append(trade_record)

            position_qty = 0.0
            in_pos = False
            entry_idx = None
            entry_time = None
            entry_decision_price = 0.0
            entry_notional = 0.0
            entry_signal_time = None
            entry_signal_price = None
            entry_breakdown = {"total_cost": 0.0, "slippage_cost": 0.0, "commission_cost": 0.0, "fee_cost": 0.0,
                               "slippage_bps": 0.0, "commission_bps": 0.0, "fill_price": 0.0}

        if entry_trigger is not None and not in_pos:
            base_price = entry_trigger.get("override_price")
            if base_price is None:
                if entry_trigger.get("from_pending") and delay_bars > 0:
                    base_price = price_open
                else:
                    base_price = price_close if execution == "close" else price_open
            entry_price = _safe_price(base_price, price_close)
            if entry_price > 0 and cash_gross > 0:
                qty = cash_gross / entry_price
                position_qty = float(qty)
                entry_idx = i
                entry_time = ts
                entry_decision_price = float(entry_price)
                entry_notional = entry_decision_price * position_qty
                cash_gross -= entry_notional
                cash_net -= entry_notional
                entry_breakdown = cost_model.compute_fill("long", "entry", entry_decision_price, position_qty)
                cash_net -= entry_breakdown["total_cost"]
                entry_signal_time = entry_trigger.get("signal_time")
                entry_signal_price = entry_trigger.get("signal_price")
                in_pos = True

        current_gross = cash_gross + position_qty * price_close
        current_net = cash_net + position_qty * price_close
        equity_gross.append(float(current_gross))
        equity_net.append(float(current_net))

    eq_gross = pd.Series(equity_gross[1:], index=df.index)
    eq_net = pd.Series(equity_net[1:], index=df.index)

    daily_returns_net = eq_net.pct_change().fillna(0.0)
    daily_returns_gross = eq_gross.pct_change().fillna(0.0)

    trades_df = pd.DataFrame(trade_rows) if capture_trades_df and trade_rows else pd.DataFrame(trade_rows)

    total_notional = float(trades_df["notional_entry"].sum() + trades_df.get("notional_exit", pd.Series(dtype=float)).sum()) if not trades_df.empty else 0.0
    start_equity = eq_gross.iloc[0] if len(eq_gross) else float(starting_equity)
    turnover = (total_notional / start_equity) if start_equity else 0.0

    def _weighted(col: str) -> float:
        if trades_df.empty or total_notional == 0.0 or col not in trades_df.columns:
            return 0.0
        if col.endswith("_bps"):
            weight = trades_df["notional_entry"]
            if col.startswith("exit_"):
                weight = trades_df["notional_exit"]
            weighted = (trades_df[col] * weight).sum()
            return float(weighted / weight.sum()) if weight.sum() else 0.0
        return float(trades_df[col].mean())

    total_slip_cost = 0.0
    total_fee_cost = 0.0
    if not trades_df.empty:
        total_slip_cost = float(trades_df["entry_slippage_cost"].sum() + trades_df["exit_slippage_cost"].sum())
        total_fee_cost = float(trades_df["entry_fee_cost"].sum() + trades_df["exit_fee_cost"].sum() + trades_df["entry_fixed_cost"].sum() + trades_df["exit_fixed_cost"].sum())

    def _cagr(series: pd.Series) -> float:
        if series is None or len(series) < 2:
            return 0.0
        start_val = _coerce_float(series.iloc[0], 0.0)
        end_val = _coerce_float(series.iloc[-1], 0.0)
        if not (start_val > 0 and end_val > 0):
            return 0.0
        days = (series.index[-1] - series.index[0]).days
        if days <= 0:
            return 0.0
        years = days / 365.25
        if years <= 0:
            return 0.0
        return float((end_val / start_val) ** (1 / years) - 1.0)

    pre_cagr = _cagr(eq_gross)
    post_cagr = _cagr(eq_net)
    annualized_drag = pre_cagr - post_cagr

    cost_summary = {
        "turnover": float(turnover),
        "weighted_slippage_bps": float(_weighted("entry_slippage_bps") + _weighted("exit_slippage_bps")),
        "weighted_fees_bps": float(_weighted("entry_fee_bps") + _weighted("exit_fee_bps")),
        "pre_cost_cagr": float(pre_cagr),
        "post_cost_cagr": float(post_cagr),
        "annualized_drag": float(annualized_drag),
        "total_slippage_cost": float(total_slip_cost),
        "total_fee_cost": float(total_fee_cost),
        "total_cost": float(total_slip_cost + total_fee_cost),
    }

    meta = dict(base_meta)
    meta["costs"] = {"enabled": enable_costs, "summary": cost_summary}

    return {
        "equity": eq_net,
        "daily_returns": daily_returns_net,
        "trades": trades,
        "equity_pre_cost": eq_gross,
        "daily_returns_pre_cost": daily_returns_gross,
        "trades_df": trades_df if capture_trades_df else pd.DataFrame(trade_rows),
        "meta": meta,
    }