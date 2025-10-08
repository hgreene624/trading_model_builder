"""ATR breakout strategy with integrated buy-the-dip overlay."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from src.backtest.engine import ATRParams, backtest_atr_breakout, CostModel

MODEL_KEY = "atr_dip_overlay"

DIP_KEYS = (
    "trend_ma",
    "dip_atr_from_high",
    "dip_lookback_high",
    "dip_rsi_max",
    "dip_confirm",
    "dip_cooldown_days",
)

DEFAULT_DIP_OVERLAY: Dict[str, Any] = {
    "trend_ma": 200,
    "dip_atr_from_high": 2.0,
    "dip_lookback_high": 60,
    "dip_rsi_max": 55.0,
    "dip_confirm": False,
    "dip_cooldown_days": 5,
}

PARAMS_ALLOWED = tuple(
    list(ATRParams.__dataclass_fields__.keys())
    + list(DIP_KEYS)
    + ["execution", "entry_mode", "model_key"]
)


def _split_execution(p: Dict[str, Any] | Any) -> Tuple[Dict[str, Any] | Any, str]:
    if isinstance(p, dict):
        payload = dict(p)
        exec_mode = str(payload.pop("execution", "close"))
        return payload, exec_mode
    return p, "close"


def _prepare_overlay(raw: Dict[str, Any]) -> Dict[str, Any]:
    overlay = dict(DEFAULT_DIP_OVERLAY)
    for key in DIP_KEYS:
        if key in raw:
            overlay[key] = raw[key]

    overlay["trend_ma"] = max(1, int(float(overlay.get("trend_ma", DEFAULT_DIP_OVERLAY["trend_ma"]))))
    overlay["dip_atr_from_high"] = float(max(0.0, overlay.get("dip_atr_from_high", DEFAULT_DIP_OVERLAY["dip_atr_from_high"])))
    overlay["dip_lookback_high"] = max(1, int(float(overlay.get("dip_lookback_high", DEFAULT_DIP_OVERLAY["dip_lookback_high"]))))
    overlay["dip_rsi_max"] = float(max(0.0, min(100.0, overlay.get("dip_rsi_max", DEFAULT_DIP_OVERLAY["dip_rsi_max"]))))
    overlay["dip_confirm"] = bool(overlay.get("dip_confirm", DEFAULT_DIP_OVERLAY["dip_confirm"]))
    overlay["dip_cooldown_days"] = max(0, int(float(overlay.get("dip_cooldown_days", DEFAULT_DIP_OVERLAY["dip_cooldown_days"]))))
    return overlay


def run_strategy(
    symbol: str,
    start,
    end,
    starting_equity: float,
    params: Dict[str, Any] | ATRParams,
) -> Dict:
    payload, exec_mode = _split_execution(params)
    model_key = MODEL_KEY
    overlay_cfg: Dict[str, Any] = dict(DEFAULT_DIP_OVERLAY)

    if isinstance(payload, dict):
        params_dict = dict(payload)
        model_key = str(params_dict.pop("model_key", MODEL_KEY) or MODEL_KEY)
        params_dict.pop("entry_mode", None)
        overlay_cfg = _prepare_overlay(params_dict)
        for key in DIP_KEYS:
            params_dict.pop(key, None)
        p = ATRParams(**params_dict)
    elif isinstance(payload, ATRParams):
        p = payload
    else:
        p = ATRParams()

    enable_costs = bool(getattr(p, "enable_costs", False))
    delay_bars = int(getattr(p, "delay_bars", 0) or 0)
    commission_override = getattr(p, "commission_bps", None)
    slippage_override = getattr(p, "slippage_bps", None)
    per_trade_fee = getattr(p, "per_trade_fee", 0.0)
    total_cost_bps = getattr(p, "cost_bps", 0.0)
    if commission_override is None:
        commission_override = 0.0
    if slippage_override is None or (slippage_override == 0.0 and commission_override == 0.0 and total_cost_bps):
        slippage_override = float(total_cost_bps)

    cost_model = CostModel.from_inputs(
        commission_bps=commission_override,
        slippage_bps=slippage_override,
        per_trade_fee=per_trade_fee,
        enabled=enable_costs,
    )

    params_payload = dict(p.__dict__)
    params_payload["model_key"] = model_key
    params_payload["entry_mode"] = "dip"
    params_payload["dip_overlay"] = dict(overlay_cfg)

    return backtest_atr_breakout(
        symbol=symbol,
        start=start,
        end=end,
        starting_equity=starting_equity,
        params=params_payload,
        execution=exec_mode,
        commission_bps=float(commission_override or 0.0),
        slippage_bps=float(slippage_override or 0.0),
        enable_costs=enable_costs,
        delay_bars=delay_bars,
        cost_model=cost_model,
    )
