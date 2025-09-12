# src/models/strategy_adapter.py
from __future__ import annotations
import importlib, inspect
from typing import Callable, Dict, Any

class StrategyAdapter:
    """
    Safe adapter around a strategy backtest function like:
      backtest_single(symbol, start, end, *, starting_equity=..., **params) -> dict
    """

    def __init__(self, fn: Callable):
        self.fn = fn
        try:
            self.sig = inspect.signature(fn)
        except Exception:
            self.sig = None

    @classmethod
    def from_name(cls, dotted: str, attr: str = "backtest_single") -> "StrategyAdapter":
        """
        Example: StrategyAdapter.from_name("src.models.atr_breakout")
        """
        mod = importlib.import_module(dotted)
        fn = getattr(mod, attr)
        return cls(fn)

    def _filter_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.sig:
            return params
        allowed = set(self.sig.parameters.keys())
        # Required args are provided positionally by run(); here we keep only **kwargs
        forbidden = {"symbol", "start", "end", "starting_equity"}
        keep = {k: v for k, v in params.items() if k in allowed and k not in forbidden}
        return keep

    def run(self, symbol: str, start: str, end: str, params: Dict[str, Any], starting_equity: float = 10_000.0) -> Dict[str, Any]:
        try:
            filtered = self._filter_params(params or {})
            return self.fn(symbol, start, end, starting_equity=starting_equity, **filtered)
        except Exception as e:
            # Always return a dict so callers never crash
            return {"error": str(e), "sharpe": 0.0, "trades": 0}

    # Quick “pipeline is alive” check with very easy entries
    def smoke_test(self, symbol: str, start: str, end: str) -> Dict[str, Any]:
        easy = {
            "breakout_n": 10,          # small lookback → should trigger breaks
            "exit_n": 10,
            "atr_n": 14,
            "atr_multiple": 2.0,
            "tp_multiple": 1.5,
            "risk_per_trade": 0.008,
            "use_trend_filter": False, # keep it off to allow trades
            "cost_bps": 2.0,
        }
        return self.run(symbol, start, end, easy, starting_equity=10_000.0)