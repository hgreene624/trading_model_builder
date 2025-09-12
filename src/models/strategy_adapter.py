# src/models/strategy_adapter.py
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


@dataclass
class StrategyAdapter:
    dotted: str
    _fn: Optional[Callable] = None

    @classmethod
    def from_name(cls, dotted: str) -> "StrategyAdapter":
        mod = importlib.import_module(dotted)
        # Try common names in priority order
        for name in ("backtest", "backtest_single", "backtest_atr_breakout"):
            fn = getattr(mod, name, None)
            if callable(fn):
                return cls(dotted=dotted, _fn=fn)
        raise ImportError(
            f"No backtest fn found in {dotted}. Tried backtest(), backtest_single(), backtest_atr_breakout()."
        )

    def get_callable(self) -> Callable:
        if self._fn is None:
            self._fn = StrategyAdapter.from_name(self.dotted)._fn
        return self._fn

    def backtest(self, **kwargs) -> Dict[str, Any]:
        """Call the underlying backtest and normalize the return to a metrics dict."""
        fn = self.get_callable()
        res = fn(**kwargs)

        # Normalize shapes:
        # - tuple: (metrics, trades?, equity?)
        if isinstance(res, tuple):
            # Prefer the first dict-ish item
            for x in res:
                if isinstance(x, dict):
                    res = x
                    break
            else:
                return {}

        # - dict with nested 'metrics'
        if isinstance(res, dict) and "metrics" in res and isinstance(res["metrics"], dict):
            return res["metrics"]

        return res if isinstance(res, dict) else {}