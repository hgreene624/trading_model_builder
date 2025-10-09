# src/models/atr_breakout.py
"""
Pluggable strategy wrapper for ATR breakout.

This keeps the UI/trainer decoupled: they call run_strategy(...) without caring
about how ATRParams works internally.
"""

from __future__ import annotations
from typing import Any, Dict, Tuple

from src.backtest.engine import ATRParams, backtest_atr_breakout, CostModel
from src.models._warmup import DISABLE_WARMUP_FLAG
from src.backtest.metrics import compute_core_metrics

MODEL_KEY = "atr_breakout"

def _split_execution(p: Dict[str, Any] | Any) -> Tuple[Dict[str, Any] | Any, str]:
    if isinstance(p, dict):
        q = dict(p)
        exec_mode = str(q.pop("execution", "close"))
        return q, exec_mode
    return p, "close"

def run_strategy(
    symbol: str,
    start,
    end,
    starting_equity: float,
    params: Dict[str, Any] | ATRParams,
) -> Dict:
    p, exec_mode = _split_execution(params)
    model_key = MODEL_KEY
    disable_warmup = False
    if isinstance(p, dict):
        payload = dict(p)
        disable_warmup = bool(payload.pop(DISABLE_WARMUP_FLAG, False))
        model_key = str(payload.pop("model_key", MODEL_KEY) or MODEL_KEY)
        p = ATRParams(**payload)
    else:
        model_key = getattr(p, "model_key", MODEL_KEY) if hasattr(p, "model_key") else MODEL_KEY
        disable_warmup = bool(getattr(p, DISABLE_WARMUP_FLAG, False))
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
        use_warmup=not disable_warmup,
    )


def backtest_single(
    symbol: str,
    start,
    end,
    breakout_n: int,
    exit_n: int,
    atr_n: int,
    starting_equity: float,
    atr_multiple: float = 2.0,
    k_atr_buffer: float = 0.0,
    persist_n: int = 1,
    tp_multiple: float = 0.0,
    holding_period_limit: int = 0,
    cost_bps: float = 0.0,
    commission_bps: float = 0.0,
    slippage_bps: float = 0.0,
    per_trade_fee: float = 0.0,
    enable_costs: bool = False,
    delay_bars: int = 0,
    execution: str = "close",
    **kwargs,
) -> Dict:
    params = ATRParams(
        breakout_n=int(breakout_n),
        exit_n=int(exit_n),
        atr_n=int(atr_n),
        atr_multiple=float(atr_multiple),
    )
    params.k_atr_buffer = float(k_atr_buffer or 0.0)
    if params.k_atr_buffer < 0.0:
        params.k_atr_buffer = 0.0
    persist_clean = int(persist_n or 1)
    if persist_clean < 1:
        persist_clean = 1
    params.persist_n = persist_clean
    params.tp_multiple = float(tp_multiple or 0.0)
    params.holding_period_limit = int(holding_period_limit or 0)
    params.cost_bps = float(cost_bps or 0.0)
    params.commission_bps = float(commission_bps or 0.0)
    params.slippage_bps = float(slippage_bps or 0.0)
    params.per_trade_fee = float(per_trade_fee or 0.0)
    params.enable_costs = bool(enable_costs or params.cost_bps > 0.0 or params.commission_bps > 0.0 or params.slippage_bps > 0.0 or params.per_trade_fee > 0.0)
    params.delay_bars = int(delay_bars or 0)

    cost_model = CostModel.from_inputs(
        commission_bps=params.commission_bps,
        slippage_bps=params.slippage_bps if params.slippage_bps > 0.0 else params.cost_bps,
        per_trade_fee=params.per_trade_fee,
        enabled=params.enable_costs,
    )

    result = backtest_atr_breakout(
        symbol=symbol,
        start=start,
        end=end,
        starting_equity=starting_equity,
        params=params,
        execution=execution,
        commission_bps=params.commission_bps,
        slippage_bps=params.slippage_bps if params.slippage_bps > 0.0 else params.cost_bps,
        enable_costs=params.enable_costs,
        delay_bars=params.delay_bars,
        cost_model=cost_model,
    )

    metrics = compute_core_metrics(result["equity"], result["daily_returns"], result["trades"])
    eq = result["equity"]
    final_equity = float(eq.iloc[-1]) if len(eq) else float(starting_equity)
    start_equity = float(eq.iloc[0]) if len(eq) else float(starting_equity)
    metrics.update({
        "final_equity": final_equity,
        "start_equity": start_equity,
        "start": start,
        "end": end,
    })
    result["metrics"] = metrics
    return result
