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
import logging
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.backtest import prob_gate

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
try:
    from src.backtest.metrics import summarize_costs as _summarize_costs  # type: ignore
except Exception:
    _summarize_costs = None

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


# ---------- Logging ----------

logger = logging.getLogger("backtest.engine")
_ENGINE_LOGGER_CONFIGURED = False


def _ensure_engine_logger() -> None:
    """Configure rotating file logging for the engine (idempotent)."""

    global _ENGINE_LOGGER_CONFIGURED
    log_level_name = str(os.getenv("LOG_LEVEL", "INFO")).upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    logger.setLevel(log_level)
    if _ENGINE_LOGGER_CONFIGURED:
        return
    try:
        log_dir = os.path.join("storage", "logs")
        os.makedirs(log_dir, exist_ok=True)
        has_rotating = any(isinstance(h, RotatingFileHandler) for h in logger.handlers)
        if not has_rotating:
            root_logger = logging.getLogger()
            has_rotating = any(isinstance(h, RotatingFileHandler) for h in root_logger.handlers)
        if not has_rotating:
            handler = RotatingFileHandler(
                os.path.join(log_dir, "engine.log"), maxBytes=5 * 1024 * 1024, backupCount=3
            )
            handler.setLevel(log_level)
            handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
            logger.addHandler(handler)
    except Exception:
        # Logging is best-effort; failures should not break the engine.
        pass
    _ENGINE_LOGGER_CONFIGURED = True


@dataclass
class CostModel:
    commission_bps: float = 0.0
    slippage_bps: float = 0.0
    per_trade_fee: float = 0.0
    enabled: bool = False
    fixed_bps: float = 0.0
    atr_k: float = 0.05  # PH1.2: ATR-linked slippage scale
    min_half_spread_bps: float = 0.5  # PH1.2: floor on half-spread proxy
    use_range_impact: bool = False  # PH1.2: toggle for range-based impact
    cap_range_impact_bps: float = 10.0  # PH1.2: clamp range impact if enabled
    mode: str = "simple_spread"

    @classmethod
    def from_inputs(
        cls,
        commission_bps: float = 0.0,
        slippage_bps: float = 0.0,
        per_trade_fee: float = 0.0,
        enabled: bool = False,
        fixed_bps: float | None = None,
        atr_k: float | None = None,
        min_half_spread_bps: float | None = None,
        use_range_impact: bool | None = None,
        cap_range_impact_bps: float | None = None,
        mode: str | None = None,
    ) -> "CostModel":
        commission_bps = max(0.0, _coerce_float(commission_bps, 0.0))
        slippage_bps = max(0.0, _coerce_float(slippage_bps, 0.0))
        per_trade_fee = max(0.0, _coerce_float(per_trade_fee, 0.0))
        active = enabled or (commission_bps > 0.0 or slippage_bps > 0.0 or per_trade_fee > 0.0)
        fixed = max(0.0, _coerce_float(fixed_bps, 0.0)) if fixed_bps is not None else 0.0
        atr_scale = (
            max(0.0, _coerce_float(atr_k, 0.05)) if atr_k is not None else 0.05
        )  # PH1.2
        min_spread = (
            max(0.0, _coerce_float(min_half_spread_bps, 0.5))
            if min_half_spread_bps is not None
            else 0.5
        )  # PH1.2
        range_impact = bool(use_range_impact) if use_range_impact is not None else False  # PH1.2
        cap_range = (
            max(0.0, _coerce_float(cap_range_impact_bps, 10.0))
            if cap_range_impact_bps is not None
            else 10.0
        )  # PH1.2
        mode_clean = str(mode).strip().lower() if mode is not None else "simple_spread"
        if mode_clean not in {"simple_spread"}:
            mode_clean = "simple_spread"
        return cls(
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            per_trade_fee=per_trade_fee,
            enabled=bool(active),
            fixed_bps=fixed,
            atr_k=atr_scale,
            min_half_spread_bps=min_spread,
            use_range_impact=range_impact,
            cap_range_impact_bps=cap_range,
            mode=mode_clean,
        )

    def total_bps(self) -> float:
        return float(self.commission_bps + self.slippage_bps)

    def as_dict(self) -> dict:
        return {
            "commission_bps": float(self.commission_bps),
            "slippage_bps": float(self.slippage_bps),
            "per_trade_fee": float(self.per_trade_fee),
            "enabled": bool(self.enabled),
            "fixed_bps": float(self.fixed_bps),
            "atr_k": float(self.atr_k),
            "min_half_spread_bps": float(self.min_half_spread_bps),
            "use_range_impact": bool(self.use_range_impact),
            "cap_range_impact_bps": float(self.cap_range_impact_bps),
            "mode": str(self.mode),
        }

    @classmethod
    def from_env(cls) -> "CostModel":
        enabled = _env_flag("COST_ENABLED", True)
        fixed_bps = max(0.0, _env_float("COST_FIXED_BPS", 0.5))
        fee_bps_override = os.getenv("FEE_BPS")
        if fee_bps_override is not None:
            fixed_bps = max(0.0, _coerce_float(fee_bps_override, fixed_bps))
        atr_k = max(0.0, _env_float("COST_ATR_K", 0.05))  # PH1.2
        min_half_spread = max(0.0, _env_float("COST_MIN_HS_BPS", 0.5))  # PH1.2
        use_range_impact = _env_flag("COST_USE_RANGE_IMPACT", False)  # PH1.2
        cap_range_impact_bps = max(0.0, _env_float("CAP_RANGE_IMPACT_BPS", 10.0))  # PH1.2
        per_trade_fee_env = os.getenv("FEE_PER_TRADE_USD")
        if per_trade_fee_env is None:
            per_trade_fee_env = os.getenv("ATR_PER_TRADE_FEE")
        per_trade_fee = max(0.0, _coerce_float(per_trade_fee_env, 0.0))
        mode = os.getenv("COST_MODE", "simple_spread")
        mode = str(mode).strip().lower()
        if mode not in {"simple_spread"}:
            mode = "simple_spread"
        return cls(
            commission_bps=float(fixed_bps),
            slippage_bps=0.0,
            per_trade_fee=float(per_trade_fee),
            enabled=bool(enabled),
            fixed_bps=float(fixed_bps),
            atr_k=float(atr_k),  # PH1.2
            min_half_spread_bps=float(min_half_spread),  # PH1.2
            use_range_impact=bool(use_range_impact),  # PH1.2
            cap_range_impact_bps=float(cap_range_impact_bps),  # PH1.2
            mode=mode,
        )

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
                "per_trade_fee_bps": 0.0,
            }

        total_bps = self.total_bps() / 10_000.0
        notional = abs(price * qty)
        slippage_cost = notional * (self.slippage_bps / 10_000.0)
        commission_cost = notional * (self.commission_bps / 10_000.0)
        fee_cost = abs(self.per_trade_fee)
        per_trade_fee_bps = 0.0
        if notional > 0 and fee_cost > 0:
            per_trade_fee_bps = fee_cost / notional * 10_000.0

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
            "commission_bps": float(self.commission_bps + per_trade_fee_bps),
            "per_trade_fee_bps": float(per_trade_fee_bps),
            "commission_bps_raw": float(self.commission_bps),
        }


def _estimate_half_spread_bps(bar: Dict[str, Any] | pd.Series, cm: CostModel | None) -> float:
    if cm is None:
        return 0.0
    if not getattr(cm, "use_range_impact", False):
        return float(getattr(cm, "min_half_spread_bps", 0.0))  # PH1.2
    try:
        high = _coerce_float(bar.get("high", 0.0), 0.0)  # type: ignore[arg-type]
        low = _coerce_float(bar.get("low", 0.0), 0.0)  # type: ignore[arg-type]
        close = _coerce_float(bar.get("close", 0.0), 0.0)  # type: ignore[arg-type]
    except AttributeError:
        return float(cm.min_half_spread_bps)
    if high <= 0.0 or low <= 0.0:
        return float(cm.min_half_spread_bps)
    spread = max(0.0, high - low)
    if spread <= 0.0:
        return float(cm.min_half_spread_bps)
    rng_bps = 0.5 * spread / max(1e-12, close) * 10_000.0  # PH1.2
    cap = max(0.0, _coerce_float(getattr(cm, "cap_range_impact_bps", 10.0), 10.0))  # PH1.2
    if cap <= 0.0:
        return float(rng_bps)
    return float(min(rng_bps, cap))  # PH1.2


def _apply_costs(
    fill_price: float,
    qty: float,
    bar: Dict[str, Any] | pd.Series,
    atr_pct: float | None,
    cm: CostModel,
) -> tuple[float, float, float]:
    price = _coerce_float(fill_price, 0.0)
    quantity = _coerce_float(qty, 0.0)
    if not cm or not getattr(cm, "enabled", False) or quantity == 0.0 or price <= 0.0:
        return float(price), 0.0, 0.0
    atr_value = 0.0
    if atr_pct is not None:
        atr_value = max(0.0, _coerce_float(atr_pct, 0.0))
    base_bps = atr_value * cm.atr_k * 10_000.0  # PH1.2
    half_spread = _estimate_half_spread_bps(bar, cm)  # PH1.2
    slip_bps = max(base_bps, half_spread, cm.min_half_spread_bps)  # PH1.2
    fees_bps = max(0.0, _coerce_float(cm.fixed_bps, 0.0))
    per_trade_fee = max(0.0, _coerce_float(getattr(cm, "per_trade_fee", 0.0), 0.0))
    notional = abs(price * quantity)
    if per_trade_fee > 0.0 and notional > 0.0:
        fees_bps += per_trade_fee / notional * 10_000.0
    direction = 1.0 if quantity > 0 else -1.0
    price_after = price * (1.0 + direction * slip_bps / 10_000.0)
    return float(price_after), float(slip_bps), float(fees_bps)


# ---------- Parameters ----------

@dataclass
class ATRParams:
    breakout_n: int = 20             # recent high lookback to trigger long entry
    exit_n: int = 10                  # recent low lookback for exit
    atr_n: int = 14
    atr_multiple: float = 2.0         # volatility filter: price > rolling_high - k * ATR
    k_atr_buffer: float = 0.0         # Phase 1.1: breakout buffer (in ATRs) beyond prior high/low
    persist_n: int = 1                # Phase 1.1: require N consecutive breakout bars
    tp_multiple: float = 0.0          # optional profit target (in ATRs). 0 disables.
    holding_period_limit: int = 0     # optional max holding days. 0 disables.
    allow_short: bool = False         # optional shorting; default off for simplicity
    cost_bps: float = 0.0             # legacy single cost slider (mapped into slippage)
    commission_bps: float = 0.0       # explicit commission component
    slippage_bps: float = 0.0         # explicit slippage component
    per_trade_fee: float = 0.0        # flat fee per fill (in account currency)
    enable_costs: bool = False        # toggle attribution / post-cost accounting
    delay_bars: int = 0               # optional execution delay (signal -> fill after N bars)
    prob_gate_enabled: bool = False   # probability gate toggle (calibrated LR)
    prob_gate_threshold: float = 0.0  # minimum probability to allow entry
    prob_model_id: str = ""           # identifier for stored calibrated model

# Defaults for optional dip overlay when strategies request it via extra_params.
DIP_OVERLAY_DEFAULTS: Dict[str, Any] = {
    "trend_ma": 200,
    "dip_atr_from_high": 2.0,
    "dip_lookback_high": 60,
    "dip_rsi_max": 55.0,
    "dip_confirm": False,
    "dip_cooldown_days": 5,
}
DIP_RSI_PERIOD = 14

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


def _relative_strength_index(close: pd.Series, period: int = DIP_RSI_PERIOD) -> pd.Series:
    """Compute RSI using Wilder's smoothing with defensive guards."""

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = avg_loss.where(avg_loss != 0.0, np.nan)
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.clip(lower=0.0, upper=100.0)
    return rsi.fillna(50.0)

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

    _ensure_engine_logger()

    extra_params: Dict[str, Any] = {}
    if isinstance(params, dict):
        params_dict = dict(params)
        known = set(ATRParams.__dataclass_fields__.keys())
        params_core = {k: params_dict[k] for k in list(params_dict.keys()) if k in known}
        extra_params = {k: params_dict[k] for k in params_dict if k not in known}
        params = ATRParams(**params_core)

    entry_mode_extra = str(extra_params.get("entry_mode", "") or "").strip().lower()
    dip_overlay_raw: Dict[str, Any] = {}
    if isinstance(extra_params.get("dip_overlay"), dict):
        dip_overlay_raw = dict(extra_params.get("dip_overlay", {}))
    else:
        direct_overlay: Dict[str, Any] = {}
        for key in DIP_OVERLAY_DEFAULTS:
            if key in extra_params:
                direct_overlay[key] = extra_params[key]
        if direct_overlay:
            dip_overlay_raw = direct_overlay
    dip_overlay_enabled = entry_mode_extra == "dip" or bool(dip_overlay_raw)
    dip_overlay_config = dict(DIP_OVERLAY_DEFAULTS)
    if dip_overlay_enabled:
        for key in DIP_OVERLAY_DEFAULTS:
            if key in dip_overlay_raw and dip_overlay_raw[key] is not None:
                dip_overlay_config[key] = dip_overlay_raw[key]
        dip_overlay_config["trend_ma"] = max(
            1, int(_coerce_float(dip_overlay_config.get("trend_ma"), DIP_OVERLAY_DEFAULTS["trend_ma"]))
        )
        dip_overlay_config["dip_atr_from_high"] = max(
            0.0,
            _coerce_float(
                dip_overlay_config.get("dip_atr_from_high"), DIP_OVERLAY_DEFAULTS["dip_atr_from_high"]
            ),
        )
        dip_overlay_config["dip_lookback_high"] = max(
            1,
            int(
                _coerce_float(
                    dip_overlay_config.get("dip_lookback_high"), DIP_OVERLAY_DEFAULTS["dip_lookback_high"]
                )
            ),
        )
        dip_overlay_config["dip_rsi_max"] = float(
            max(
                0.0,
                min(
                    100.0,
                    _coerce_float(
                        dip_overlay_config.get("dip_rsi_max"), DIP_OVERLAY_DEFAULTS["dip_rsi_max"]
                    ),
                ),
            )
        )
        dip_overlay_config["dip_confirm"] = bool(
            dip_overlay_config.get("dip_confirm", DIP_OVERLAY_DEFAULTS["dip_confirm"])
        )
        dip_overlay_config["dip_cooldown_days"] = max(
            0,
            int(
                _coerce_float(
                    dip_overlay_config.get("dip_cooldown_days"),
                    DIP_OVERLAY_DEFAULTS["dip_cooldown_days"],
                )
            ),
        )
        extra_params["dip_overlay"] = dict(dip_overlay_config)
    else:
        dip_overlay_config = dict(DIP_OVERLAY_DEFAULTS)

    delay_candidate = _coerce_float(getattr(params, "delay_bars", 0), 0.0)
    if delay_bars is not None:
        delay_candidate = _coerce_float(delay_bars, delay_candidate)
    env_delay_override = os.getenv("EXEC_DELAY_BARS")
    if env_delay_override is not None:
        delay_candidate = _coerce_float(env_delay_override, delay_candidate)
    env_delay = _env_float("ATR_EXECUTION_DELAY_BARS", delay_candidate)
    delay_bars = int(max(0, _coerce_float(env_delay, delay_candidate)))

    execution_mode = str(execution).strip().lower()
    if execution_mode in {"next_open", "open"}:
        exec_fill_where = "open"
    else:
        exec_fill_where = "close"
    env_exec_where = os.getenv("EXEC_FILL_WHERE")
    if env_exec_where:
        env_exec_where = env_exec_where.strip().lower()
        if env_exec_where in {"open", "close"}:
            exec_fill_where = env_exec_where

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
    commission_cfg = _env_float("FEE_BPS", commission_cfg)
    fee_cfg = _env_float("ATR_PER_TRADE_FEE", fee_cfg)
    fee_cfg = _env_float("FEE_PER_TRADE_USD", fee_cfg)

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
    phase0_cost_model = CostModel.from_env()
    if phase0_cost_model.enabled:
        cost_model.enabled = False
    enable_costs = bool(cost_model.enabled or phase0_cost_model.enabled)

    df = get_ohlcv(symbol, start, end).copy()
    base_meta = {
        "symbol": symbol,
        "start": pd.Timestamp(start),
        "end": pd.Timestamp(end),
        "params": dict(params.__dict__),
        "extra_params": extra_params,
        "execution": execution,
        "exec_fill_where": exec_fill_where,
        "delay_bars": delay_bars,
        "cost_model": cost_model.as_dict(),
        "phase0_cost_model": phase0_cost_model.as_dict(),
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
    prob_gate_enabled = bool(getattr(params, "prob_gate_enabled", False))
    prob_gate_threshold = float(getattr(params, "prob_gate_threshold", 0.0) or 0.0)
    prob_model_id = str(getattr(params, "prob_model_id", "") or extra_params.get("prob_model_id", ""))
    gate_probabilities: Optional[pd.Series] = None
    if prob_gate_enabled and prob_gate_threshold > 0.0 and prob_model_id:
        try:
            gate_probabilities = prob_gate.score_probabilities(df, params, prob_model_id)
        except FileNotFoundError:
            logger.warning("prob_gate model %s not found; disabling gate", prob_model_id)
            prob_gate_enabled = False
            gate_probabilities = None
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.exception("prob_gate scoring failed", exc_info=exc)
            prob_gate_enabled = False
            gate_probabilities = None
    elif prob_gate_enabled:
        prob_gate_enabled = False
    atr = wilder_atr(df["high"], df["low"], df["close"], n=params.atr_n)
    roll_high = df["close"].rolling(params.breakout_n).max()
    roll_low = df["close"].rolling(params.exit_n).min()
    prior_high = roll_high.shift(1)

    k_atr_buffer = _coerce_float(getattr(params, "k_atr_buffer", 0.0), 0.0)
    if k_atr_buffer < 0.0:
        k_atr_buffer = 0.0
    persist_candidate = _coerce_float(getattr(params, "persist_n", 1), 1.0)
    if persist_candidate < 1:
        persist_candidate = 1.0
    persist_n = int(persist_candidate) if persist_candidate else 1
    if persist_n < 1:
        persist_n = 1

    entry_th = roll_high - params.atr_multiple * atr
    long_base = (df["close"] > entry_th).fillna(False)
    long_pre_buffer = long_base.copy()
    buffer_gate = pd.Series(True, index=df.index)
    if k_atr_buffer > 0.0:
        buffer_threshold = (prior_high + k_atr_buffer * atr)
        buffer_gate = (df["close"] > buffer_threshold).fillna(False)
        long_base = long_base & buffer_gate
    long_base = long_base.fillna(False)
    buffer_gate = buffer_gate.fillna(False)

    if persist_n > 1:
        long_signal_series = (
            long_base.astype(int)
            .rolling(window=persist_n, min_periods=persist_n)
            .sum()
            .eq(persist_n)
        )
        long_signal = long_signal_series.fillna(False).astype(bool)
    else:
        long_signal = long_base.astype(bool)

    raw_long_breakout = long_base.astype(bool)

    dip_overlay_active = bool(dip_overlay_enabled)
    dip_trend_ok = pd.Series(True, index=df.index, dtype=bool)
    dip_depth_ok = pd.Series(True, index=df.index, dtype=bool)
    dip_rsi_ok = pd.Series(True, index=df.index, dtype=bool)
    dip_confirm_ok = pd.Series(True, index=df.index, dtype=bool)
    dip_conditions_ok = pd.Series(True, index=df.index, dtype=bool)
    dip_cooldown_days = 0
    dip_cooldown_until_idx = -1
    dip_block_conditions = 0
    dip_block_trend = 0
    dip_block_depth = 0
    dip_block_rsi = 0
    dip_block_confirm = 0
    dip_block_cooldown = 0
    dip_entries_allowed = 0
    dip_atr_from_high = 0.0
    dip_rsi_cap = DIP_OVERLAY_DEFAULTS["dip_rsi_max"]
    dip_require_confirm = bool(dip_overlay_config.get("dip_confirm", DIP_OVERLAY_DEFAULTS["dip_confirm"]))
    if dip_overlay_active and not df.empty:
        trend_ma = int(dip_overlay_config.get("trend_ma", DIP_OVERLAY_DEFAULTS["trend_ma"]))
        if trend_ma > 1:
            trend_ma_series = df["close"].rolling(window=trend_ma).mean()
            dip_trend_ok = (df["close"] >= trend_ma_series).fillna(False)
        dip_atr_from_high = float(dip_overlay_config.get("dip_atr_from_high", 0.0) or 0.0)
        lookback_high = int(
            dip_overlay_config.get("dip_lookback_high", DIP_OVERLAY_DEFAULTS["dip_lookback_high"])
        )
        if dip_atr_from_high > 0.0 and lookback_high > 0:
            rolling_high = df["close"].rolling(window=lookback_high).max()
            dip_depth_ok = ((rolling_high - df["close"]) >= (dip_atr_from_high * atr)).fillna(False)
        dip_rsi_cap = float(
            dip_overlay_config.get("dip_rsi_max", DIP_OVERLAY_DEFAULTS["dip_rsi_max"])
        )
        if dip_rsi_cap < 100.0:
            rsi_series = _relative_strength_index(df["close"], DIP_RSI_PERIOD)
            dip_rsi_ok = (rsi_series <= dip_rsi_cap).fillna(False)
        dip_require_confirm = bool(
            dip_overlay_config.get("dip_confirm", DIP_OVERLAY_DEFAULTS["dip_confirm"])
        )
        if dip_require_confirm:
            dip_confirm_ok = (df["close"] >= df["close"].shift(1)).fillna(False)
        dip_conditions_ok = dip_trend_ok & dip_depth_ok & dip_rsi_ok & dip_confirm_ok
        dip_cooldown_days = int(
            dip_overlay_config.get("dip_cooldown_days", DIP_OVERLAY_DEFAULTS["dip_cooldown_days"])
        )

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
    entry_gate_probability: Optional[float] = None

    index_list = list(df.index)
    n_bars = len(index_list)

    long_streak_counter = 0
    state_tracking: Dict[str, int] = {
        "long_streak": 0,
        "short_streak": 0,
        "long_streak_max": 0,
        "entry_count": 0,
        "exit_count": 0,
        "blocked_by_buffer": 0,
        "blocked_by_persistence": 0,
        "blocked_by_prob_gate": 0,
        "blocked_by_min_hold": 0,
        "blocked_by_cooldown": 0,
    }
    entry_count = 0
    exit_count = 0
    blocked_by_buffer = 0
    blocked_by_persistence = 0
    blocked_by_prob_gate = 0
    blocked_by_min_hold = 0
    blocked_by_cooldown = 0

    log_per_trade = logger.isEnabledFor(logging.DEBUG) and _env_flag("LOG_TRADES", False)
    try:
        log_sample_every = max(1, int(os.getenv("LOG_TRADES_SAMPLE", "1")))
    except ValueError:
        log_sample_every = 1
    try:
        log_sample_head = max(0, int(os.getenv("LOG_TRADES_HEAD", "0")))
    except ValueError:
        log_sample_head = 0
    trade_log_counter = 0

    def _should_log_trade(counter: int) -> bool:
        if log_sample_head > 0 and counter <= log_sample_head:
            return True
        if log_sample_every <= 1:
            return True
        return counter % log_sample_every == 0

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
        atr_val = _coerce_float(atr.iloc[i], 0.0) if i < len(atr) else 0.0
        atr_pct_current = (atr_val / price_close) if (price_close > 0 and atr_val > 0) else None

        exit_trigger: Optional[Dict[str, Any]] = None
        entry_trigger: Optional[Dict[str, Any]] = None

        raw_break_today = bool(raw_long_breakout.iloc[i]) if i < len(raw_long_breakout) else False
        if raw_break_today:
            long_streak_counter += 1
        else:
            long_streak_counter = 0
        state_tracking["long_streak"] = long_streak_counter
        if long_streak_counter > state_tracking.get("long_streak_max", 0):
            state_tracking["long_streak_max"] = long_streak_counter
        state_tracking["short_streak"] = 0

        base_without_buffer = bool(long_pre_buffer.iloc[i]) if i < len(long_pre_buffer) else False
        buffer_pass_today = bool(buffer_gate.iloc[i]) if i < len(buffer_gate) else True
        long_after_buffer = bool(long_base.iloc[i]) if i < len(long_base) else False
        signal_today = bool(long_signal.iloc[i]) if i < len(long_signal) else False
        if (
            not in_pos
            and pending_entry is None
            and k_atr_buffer > 0.0
            and base_without_buffer
            and not buffer_pass_today
        ):
            blocked_by_buffer += 1
            state_tracking["blocked_by_buffer"] = blocked_by_buffer
        if (
            not in_pos
            and pending_entry is None
            and persist_n > 1
            and long_after_buffer
            and not signal_today
        ):
            blocked_by_persistence += 1
            state_tracking["blocked_by_persistence"] = blocked_by_persistence

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

        gate_probability: Optional[float] = None
        if prob_gate_enabled and gate_probabilities is not None and i < len(gate_probabilities):
            try:
                val = gate_probabilities.iloc[i]
            except Exception:  # pragma: no cover - defensive alignment guard
                val = np.nan
            if pd.notna(val):
                gate_probability = float(val)

        if not in_pos and entry_trigger is None and bool(long_signal.iloc[i]):
            if dip_overlay_active:
                cond_pass = True
                if i < len(dip_conditions_ok):
                    cond_pass = bool(dip_conditions_ok.iloc[i])
                if not cond_pass:
                    if raw_break_today:
                        dip_block_conditions += 1
                        if i < len(dip_trend_ok) and not bool(dip_trend_ok.iloc[i]):
                            dip_block_trend += 1
                        if dip_atr_from_high > 0.0 and i < len(dip_depth_ok) and not bool(dip_depth_ok.iloc[i]):
                            dip_block_depth += 1
                        if dip_rsi_cap < 100.0 and i < len(dip_rsi_ok) and not bool(dip_rsi_ok.iloc[i]):
                            dip_block_rsi += 1
                        if dip_require_confirm and i < len(dip_confirm_ok) and not bool(dip_confirm_ok.iloc[i]):
                            dip_block_confirm += 1
                    continue
                if dip_cooldown_days > 0 and i <= dip_cooldown_until_idx:
                    blocked_by_cooldown += 1
                    dip_block_cooldown += 1
                    state_tracking["blocked_by_cooldown"] = blocked_by_cooldown
                    continue
                dip_entries_allowed += 1
            if prob_gate_enabled and gate_probability is not None and gate_probability < prob_gate_threshold:
                blocked_by_prob_gate += 1
                state_tracking["blocked_by_prob_gate"] = blocked_by_prob_gate
            else:
                trigger = {
                    "signal_idx": i,
                    "signal_time": ts,
                    "signal_price": price_close,
                }
                if gate_probability is not None:
                    trigger["prob_gate_probability"] = gate_probability
                if delay_bars > 0:
                    pending_entry = trigger
                else:
                    entry_trigger = trigger

        if exit_trigger is not None and in_pos and position_qty > 0:
            base_price = exit_trigger.get("override_price")
            if base_price is None:
                if exit_trigger.get("from_pending") and delay_bars > 0:
                    base_price = price_open if exec_fill_where == "open" else price_close
                else:
                    base_price = price_close if exec_fill_where == "close" else price_open
            exit_price = _safe_price(base_price, price_close)

            qty = position_qty
            proceeds = exit_price * qty
            cash_gross += proceeds
            cash_net += proceeds
            exit_breakdown = cost_model.compute_fill("long", "exit", exit_price, qty)
            if phase0_cost_model.enabled:
                exit_fill_price, exit_slip_bps, exit_fee_bps = _apply_costs(
                    exit_price, -qty, row, atr_pct_current, phase0_cost_model
                )
                slip_cost_exit = abs(exit_fill_price - exit_price) * abs(qty)
                fee_cost_exit = abs(exit_fee_bps) / 10_000.0 * exit_price * abs(qty)
                exit_breakdown.update(
                    {
                        "fill_price": float(exit_fill_price),
                        "slippage_bps": float(exit_slip_bps),
                        "commission_bps": float(exit_fee_bps),
                        "slippage_cost": float(slip_cost_exit),
                        "commission_cost": float(fee_cost_exit),
                        "fee_cost": 0.0,
                        "total_cost": float(slip_cost_exit + fee_cost_exit),
                        "price_before": float(exit_price),
                        "price_after": float(exit_fill_price),
                        "per_trade_fee_bps": 0.0,
                    }
                )
            else:
                exit_breakdown.setdefault("price_before", float(exit_price))
                exit_breakdown.setdefault(
                    "price_after", float(exit_breakdown.get("fill_price", exit_price))
                )
            cash_net -= exit_breakdown["total_cost"]
            exit_fill = float(exit_breakdown.get("fill_price", exit_price))

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
            notional_exit = exit_fill * qty
            trade_ret = net_pnl / entry_notional if entry_notional else 0.0
            holding_days = 0
            if entry_time is not None:
                holding_days = (ts - entry_time).days or (i - (entry_idx or i))

            entry_price_before = float(entry_breakdown.get("price_before", entry_decision_price))
            entry_price_after = float(entry_breakdown.get("price_after", entry_breakdown.get("fill_price", entry_decision_price)))
            exit_price_before = float(exit_breakdown.get("price_before", exit_price))
            exit_price_after = float(exit_breakdown.get("price_after", exit_fill))
            combined_notional = abs(entry_notional) + abs(notional_exit)
            slip_bps_weighted = 0.0
            fee_bps_weighted = 0.0
            if combined_notional > 0:
                slip_bps_weighted = (
                    abs(entry_breakdown.get("slippage_bps", 0.0)) * abs(entry_notional)
                    + abs(exit_breakdown.get("slippage_bps", 0.0)) * abs(notional_exit)
                ) / combined_notional
                fee_bps_weighted = (
                    abs(entry_breakdown.get("commission_bps", 0.0)) * abs(entry_notional)
                    + abs(exit_breakdown.get("commission_bps", 0.0)) * abs(notional_exit)
                ) / combined_notional

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
                "entry_per_trade_fee_bps": float(entry_breakdown.get("per_trade_fee_bps", 0.0)),
                "exit_slippage_bps": float(exit_breakdown.get("slippage_bps", 0.0)),
                "exit_fee_bps": float(exit_breakdown.get("commission_bps", 0.0)),
                "exit_per_trade_fee_bps": float(exit_breakdown.get("per_trade_fee_bps", 0.0)),
                "signal_time": entry_signal_time,
                "signal_price": float(entry_signal_price) if entry_signal_price is not None else None,
                "exit_signal_time": exit_trigger.get("signal_time"),
                "exit_signal_price": float(exit_trigger.get("signal_price")) if exit_trigger.get("signal_price") is not None else None,
                "quantity": float(qty),
                "qty": float(qty),
                "symbol": symbol,
                "time": ts,
                "notional_entry": float(entry_notional),
                "notional_exit": float(notional_exit),
                "notional": float(abs(entry_notional)),
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
                "entry_price_before": entry_price_before,
                "entry_price_after": entry_price_after,
                "exit_price_before": exit_price_before,
                "exit_price_after": exit_price_after,
                "price_before": entry_price_before,
                "price_after": entry_price_after,
                "slip_bps": float(slip_bps_weighted),
                "fees_bps": float(fee_bps_weighted),
                "prob_gate_probability": float(entry_gate_probability)
                if entry_gate_probability is not None
                else None,
            }

            trades.append(trade_record)
            trade_rows.append(trade_record)

            exit_count += 1
            state_tracking["exit_count"] = exit_count

            if log_per_trade:
                trade_log_counter += 1
                if _should_log_trade(trade_log_counter):
                    logger.debug(
                    "%s,%s,%s,qty=%.6f,%.4f->%.4f,slip_bps=%.2f,fees_bps=%.2f,reason=exit,detail=%s,streaks=%s,blocks=%s",
                    ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                    symbol,
                    "long",
                    float(qty),
                    exit_price_before,
                    exit_price_after,
                    float(exit_breakdown.get("slippage_bps", 0.0)),
                    float(exit_breakdown.get("commission_bps", 0.0)),
                    exit_trigger.get("reason"),
                    {"long": state_tracking.get("long_streak", 0), "max": state_tracking.get("long_streak_max", 0)},
                    {
                        "buffer": blocked_by_buffer,
                        "persistence": blocked_by_persistence,
                        "min_hold": blocked_by_min_hold,
                        "cooldown": blocked_by_cooldown,
                    },
                )

            position_qty = 0.0
            in_pos = False
            entry_idx = None
            entry_time = None
            entry_decision_price = 0.0
            entry_notional = 0.0
            entry_signal_time = None
            entry_signal_price = None
            entry_gate_probability = None
            if dip_overlay_active and dip_cooldown_days > 0:
                dip_cooldown_until_idx = max(dip_cooldown_until_idx, i + dip_cooldown_days)
            entry_breakdown = {"total_cost": 0.0, "slippage_cost": 0.0, "commission_cost": 0.0, "fee_cost": 0.0,
                               "slippage_bps": 0.0, "commission_bps": 0.0, "fill_price": 0.0}

        if entry_trigger is not None and not in_pos:
            base_price = entry_trigger.get("override_price")
            if base_price is None:
                if entry_trigger.get("from_pending") and delay_bars > 0:
                    base_price = price_open if exec_fill_where == "open" else price_close
                else:
                    base_price = price_close if exec_fill_where == "close" else price_open
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
                if phase0_cost_model.enabled:
                    entry_fill_price, entry_slip_bps, entry_fee_bps = _apply_costs(
                        entry_decision_price, position_qty, row, atr_pct_current, phase0_cost_model
                    )
                    slip_cost_entry = abs(entry_fill_price - entry_decision_price) * position_qty
                    fee_cost_entry = abs(entry_fee_bps) / 10_000.0 * entry_decision_price * position_qty
                    entry_breakdown.update(
                        {
                            "fill_price": float(entry_fill_price),
                            "slippage_bps": float(entry_slip_bps),
                            "commission_bps": float(entry_fee_bps),
                            "slippage_cost": float(slip_cost_entry),
                            "commission_cost": float(fee_cost_entry),
                            "fee_cost": 0.0,
                            "total_cost": float(slip_cost_entry + fee_cost_entry),
                            "price_before": float(entry_decision_price),
                            "price_after": float(entry_fill_price),
                            "per_trade_fee_bps": 0.0,
                        }
                    )
                else:
                    entry_breakdown.setdefault("price_before", float(entry_decision_price))
                    entry_breakdown.setdefault(
                        "price_after", float(entry_breakdown.get("fill_price", entry_decision_price))
                    )
                cash_net -= entry_breakdown["total_cost"]
                entry_notional = entry_breakdown.get("price_after", entry_decision_price) * position_qty
                entry_signal_time = entry_trigger.get("signal_time")
                entry_signal_price = entry_trigger.get("signal_price")
                entry_gate_probability = entry_trigger.get("prob_gate_probability")
                in_pos = True
                entry_count += 1
                state_tracking["entry_count"] = entry_count

                if log_per_trade:
                    trade_log_counter += 1
                    if _should_log_trade(trade_log_counter):
                        entry_price_before = float(entry_breakdown.get("price_before", entry_decision_price))
                        entry_price_after = float(
                            entry_breakdown.get(
                                "price_after", entry_breakdown.get("fill_price", entry_decision_price)
                            )
                        )
                        logger.debug(
                            "%s,%s,%s,qty=%.6f,%.4f->%.4f,slip_bps=%.2f,fees_bps=%.2f,reason=enter,streaks=%s,blocks=%s",
                            ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                            symbol,
                            "long",
                            float(position_qty),
                        entry_price_before,
                        entry_price_after,
                        float(entry_breakdown.get("slippage_bps", 0.0)),
                        float(entry_breakdown.get("commission_bps", 0.0)),
                        {"long": state_tracking.get("long_streak", 0), "max": state_tracking.get("long_streak_max", 0)},
                        {
                            "buffer": blocked_by_buffer,
                            "persistence": blocked_by_persistence,
                            "min_hold": blocked_by_min_hold,
                            "cooldown": blocked_by_cooldown,
                        },
                    )

        current_gross = cash_gross + position_qty * price_close
        current_net = cash_net + position_qty * price_close
        equity_gross.append(float(current_gross))
        equity_net.append(float(current_net))

    eq_gross = pd.Series(equity_gross[1:], index=df.index)
    eq_net = pd.Series(equity_net[1:], index=df.index)

    daily_returns_net = eq_net.pct_change().fillna(0.0)
    daily_returns_gross = eq_gross.pct_change().fillna(0.0)

    trades_df = pd.DataFrame(trade_rows)
    if capture_trades_df and trade_rows:
        trades_df = pd.DataFrame(trade_rows)

    required_float_cols = [
        "qty",
        "price_before",
        "price_after",
        "notional",
        "slip_bps",
        "fees_bps",
        "notional_entry",
        "notional_exit",
        "entry_per_trade_fee_bps",
        "exit_per_trade_fee_bps",
    ]
    for col in required_float_cols:
        if col not in trades_df.columns:
            trades_df[col] = pd.Series(dtype=float)
    if "symbol" not in trades_df.columns:
        trades_df["symbol"] = pd.Series(dtype="object")
    if "time" not in trades_df.columns:
        trades_df["time"] = pd.Series(dtype="datetime64[ns]")

    total_notional = 0.0
    if not trades_df.empty:
        total_notional = float(
            trades_df.get("notional_entry", pd.Series(dtype=float)).abs().sum()
            + trades_df.get("notional_exit", pd.Series(dtype=float)).abs().sum()
        )
    start_equity = eq_gross.iloc[0] if len(eq_gross) else float(starting_equity)
    turnover_multiple = (total_notional / start_equity) if start_equity else 0.0

    period_days = 0.0

    def _span_days(series: pd.Series) -> float:
        if series is None or len(series) < 2:
            return 0.0
        try:
            delta = series.index[-1] - series.index[0]
        except Exception:
            return 0.0
        try:
            return float(delta.days)
        except Exception:
            return 0.0

    period_days = _span_days(eq_gross)
    if period_days <= 0.0:
        period_days = _span_days(eq_net)
    years = period_days / 365.25 if period_days > 0 else 0.0
    turnover_ratio = turnover_multiple / years if years > 0 else 0.0

    total_slip_cost = 0.0
    total_fee_cost = 0.0
    if not trades_df.empty:
        total_slip_cost = float(
            trades_df.get("entry_slippage_cost", pd.Series(dtype=float)).sum()
            + trades_df.get("exit_slippage_cost", pd.Series(dtype=float)).sum()
        )
        fee_cols = [
            trades_df.get("entry_fee_cost", pd.Series(dtype=float)).sum(),
            trades_df.get("exit_fee_cost", pd.Series(dtype=float)).sum(),
            trades_df.get("entry_fixed_cost", pd.Series(dtype=float)).sum(),
            trades_df.get("exit_fixed_cost", pd.Series(dtype=float)).sum(),
        ]
        total_fee_cost = float(sum(fee_cols))

    cost_summary: dict[str, float | None] = {}
    base_summary: dict | None = None
    if _summarize_costs is not None:
        try:
            base_summary = _summarize_costs(
                trades_df if not trades_df.empty else None,
                eq_gross if len(eq_gross) else None,
                eq_net if len(eq_net) else None,
            )
        except Exception:
            base_summary = None
    if base_summary:
        for key, value in base_summary.items():
            if value is None:
                cost_summary[key] = None
            else:
                try:
                    cost_summary[key] = float(value)
                except (TypeError, ValueError):
                    cost_summary[key] = None
    else:
        cost_summary.update(
            {
                "turnover_gross": float(total_notional),
                "turnover_avg_daily": 0.0,
                "slippage_bps_weighted": 0.0,
                "fees_bps_weighted": 0.0,
                "pre_cost_cagr": 0.0,
                "post_cost_cagr": 0.0,
                "annualized_drag_bps": 0.0,
                "sharpe_gross": 0.0,
                "sharpe_net": 0.0,
                "sharpe_pre_cost": 0.0,
                "sharpe_post_cost": 0.0,
                "cagr_gross": 0.0,
                "cagr_net": 0.0,
            }
        )

    if "pre_cost_cagr" in cost_summary and "post_cost_cagr" in cost_summary:
        try:
            cost_summary["annualized_drag"] = float(
                float(cost_summary.get("pre_cost_cagr", 0.0) or 0.0)
                - float(cost_summary.get("post_cost_cagr", 0.0) or 0.0)
            )
        except (TypeError, ValueError):
            cost_summary["annualized_drag"] = 0.0

    drag_bps = cost_summary.get("annualized_drag_bps")
    if drag_bps is None and cost_summary.get("annualized_drag") is not None:
        drag_bps = float(cost_summary.get("annualized_drag", 0.0)) * 10_000.0
    try:
        cost_summary["annualized_drag_bps"] = float(drag_bps) if drag_bps is not None else 0.0
    except (TypeError, ValueError):
        cost_summary["annualized_drag_bps"] = 0.0

    for alias in ("slippage_bps_weighted", "fees_bps_weighted"):
        if alias not in cost_summary or cost_summary[alias] is None:
            cost_summary[alias] = 0.0

    for key in (
        "pre_cost_cagr",
        "post_cost_cagr",
        "sharpe_gross",
        "sharpe_net",
        "sharpe_pre_cost",
        "sharpe_post_cost",
        "cagr_gross",
        "cagr_net",
    ):
        if key in cost_summary and cost_summary[key] is None:
            cost_summary[key] = 0.0

    cost_summary["weighted_slippage_bps"] = float(cost_summary.get("slippage_bps_weighted", 0.0) or 0.0)
    cost_summary["weighted_fees_bps"] = float(cost_summary.get("fees_bps_weighted", 0.0) or 0.0)
    cost_summary["turnover_gross"] = float(cost_summary.get("turnover_gross", total_notional) or 0.0)
    cost_summary["turnover_avg_daily"] = float(cost_summary.get("turnover_avg_daily", 0.0) or 0.0)
    cost_summary["turnover_multiple"] = float(cost_summary.get("turnover_multiple", turnover_multiple) or 0.0)
    cost_summary["turnover_ratio"] = float(cost_summary.get("turnover_ratio", turnover_ratio) or 0.0)
    cost_summary["turnover"] = float(cost_summary.get("turnover", cost_summary["turnover_ratio"]))
    cost_summary["total_slippage_cost"] = float(total_slip_cost)
    cost_summary["total_fee_cost"] = float(total_fee_cost)
    cost_summary["total_cost"] = float(total_slip_cost + total_fee_cost)
    if cost_summary["turnover_ratio"] > 0 and cost_summary["annualized_drag_bps"]:
        cost_summary["cost_per_turnover_bps"] = (
            float(cost_summary["annualized_drag_bps"]) / float(cost_summary["turnover_ratio"])
        )

    meta = dict(base_meta)
    meta["costs"] = {"enabled": enable_costs, "summary": cost_summary}
    effective_cost_model = (
        phase0_cost_model if getattr(phase0_cost_model, "enabled", False) else cost_model
    )  # PH1.2
    effective_cost_dict = effective_cost_model.as_dict() if effective_cost_model else {}
    meta["cost_inputs"] = {  # PH1.2: surface effective cost knobs for downstream logging
        "enabled": bool(getattr(effective_cost_model, "enabled", False)),
        "commission_bps": float(effective_cost_dict.get("commission_bps", 0.0) or 0.0),
        "slippage_bps": float(effective_cost_dict.get("slippage_bps", 0.0) or 0.0),
        "fixed_bps": float(effective_cost_dict.get("fixed_bps", 0.0) or 0.0),
        "fee_bps": float(effective_cost_dict.get("fixed_bps", 0.0) or 0.0),
        "per_trade_fee_usd": float(effective_cost_dict.get("per_trade_fee", 0.0) or 0.0),
        "atr_k": float(effective_cost_dict.get("atr_k", getattr(effective_cost_model, "atr_k", 0.0))),
        "min_half_spread_bps": float(
            effective_cost_dict.get("min_half_spread_bps", getattr(effective_cost_model, "min_half_spread_bps", 0.0))
        ),
        "use_range_impact": bool(
            effective_cost_dict.get("use_range_impact", getattr(effective_cost_model, "use_range_impact", False))
        ),
        "cap_range_impact_bps": float(
            effective_cost_dict.get("cap_range_impact_bps", getattr(effective_cost_model, "cap_range_impact_bps", 10.0))
            or 0.0
        ),
    }
    meta.setdefault("runtime_counters", {})
    meta["runtime_counters"].update(
        {
            "long_streak": int(state_tracking.get("long_streak", 0)),
            "long_streak_max": int(state_tracking.get("long_streak_max", 0)),
            "short_streak": int(state_tracking.get("short_streak", 0)),
            "entry_count": int(entry_count),
            "exit_count": int(exit_count),
            "persist_n": int(persist_n),
            "k_atr_buffer": float(k_atr_buffer),
            "blocked_by_buffer": int(blocked_by_buffer),
            "blocked_by_persistence": int(blocked_by_persistence),
            "blocked_by_prob_gate": int(blocked_by_prob_gate),
            "blocked_by_min_hold": int(blocked_by_min_hold),
            "blocked_by_cooldown": int(blocked_by_cooldown),
        }
    )
    if dip_overlay_active:
        meta["runtime_counters"].update(
            {
                "dip_entries_allowed": int(dip_entries_allowed),
                "dip_blocked_conditions": int(dip_block_conditions),
                "dip_blocked_trend": int(dip_block_trend),
                "dip_blocked_depth": int(dip_block_depth),
                "dip_blocked_rsi": int(dip_block_rsi),
                "dip_blocked_confirm": int(dip_block_confirm),
                "dip_blocked_cooldown": int(dip_block_cooldown),
            }
        )

    if prob_gate_enabled or gate_probabilities is not None:
        meta["prob_gate"] = {
            "enabled": bool(prob_gate_enabled),
            "threshold": float(prob_gate_threshold),
            "model_id": str(prob_model_id),
            "blocked": int(blocked_by_prob_gate),
            "scores_available": gate_probabilities is not None,
        }

    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "run_complete symbol=%s trades=%d entries=%d exits=%d buffer_blocks=%d persistence_blocks=%d delay_bars=%d costs_enabled=%s",
            symbol,
            len(trades),
            int(entry_count),
            int(exit_count),
            int(blocked_by_buffer),
            int(blocked_by_persistence),
            int(delay_bars),
            bool(enable_costs),
        )

    if dip_overlay_active:
        dip_meta_runtime = {
            "entries_allowed": int(dip_entries_allowed),
            "blocked_conditions": int(dip_block_conditions),
            "blocked_trend": int(dip_block_trend),
            "blocked_depth": int(dip_block_depth),
            "blocked_rsi": int(dip_block_rsi),
            "blocked_confirm": int(dip_block_confirm),
            "cooldown_hits": int(dip_block_cooldown),
            "cooldown_days": int(dip_cooldown_days),
        }
        if dip_cooldown_until_idx >= 0:
            dip_meta_runtime["cooldown_active_until_index"] = int(dip_cooldown_until_idx)
        meta["dip_overlay"] = {
            "config": dict(dip_overlay_config),
            "runtime": dip_meta_runtime,
        }

    return {
        "equity": eq_net,
        "daily_returns": daily_returns_net,
        "trades": trades,
        "equity_pre_cost": eq_gross,
        "daily_returns_pre_cost": daily_returns_gross,
        "trades_df": trades_df if capture_trades_df else pd.DataFrame(trade_rows),
        "meta": meta,
    }


# QA Checklist:
#   1. Enable Phase-0 costs via `COST_ENABLED=1 COST_FIXED_BPS=0.5 COST_ATR_K=0.25 COST_MIN_HS_BPS=1 streamlit run Home.py`.
#   2. Test delayed execution with `EXEC_DELAY_BARS=1 EXEC_FILL_WHERE=open streamlit run Home.py`.
#   3. In the Streamlit app, open Simulate Portfolio and expand "Cost Attribution (post-cost)" to review metrics.
#   4. Verify the TRI panel test toggle remains available in the EA Train/Test Inspector page.
#   5. Adjust `k_atr_buffer` and `persist_n` from defaults and confirm turnover/trade counts decline while default settings match legacy results.
