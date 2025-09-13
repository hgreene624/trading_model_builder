# src/models/strategy_adapter.py
from __future__ import annotations

import importlib
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, Tuple, Optional

from src.data.cache import get_ohlcv_cached  # returns a DataFrame (start, end inclusive)

Window = Tuple[date, date]


@dataclass
class StrategyAdapter:
    """Unifies how general_trainer talks to a concrete strategy module.

    Standard call we support from the trainer:
        adapter.backtest(symbol, (start, end), starting_equity) -> metrics dict
    """
    mod: Any
    params: Dict[str, Any]

    # ---------- Construction ----------
    @classmethod
    def from_name(cls, dotted: str, params: Dict[str, Any]) -> "StrategyAdapter":
        mod = importlib.import_module(dotted)
        return cls(mod, dict(params or {}))

    # ---------- Utilities ----------
    def _fetch_df(self, symbol: str, start: date, end: date):
        # ISO strings so cache key is stable
        return get_ohlcv_cached(symbol, start.isoformat(), end.isoformat())

    # ---------- Backtest entry point ----------
    def backtest(self, symbol: str, window: Window, starting_equity: float) -> Dict[str, Any]:
        """Call the underlying strategy in a forgiving way.

        We support any of these function signatures in the strategy module:
            1) backtest(symbol, start, end, params, starting_equity)
            2) backtest_atr_breakout(symbol, start, end, params, starting_equity)
            3) backtest_on_df(df, params, starting_equity)
        """
        start, end = window

        # 1) Fully unified 'backtest'
        if hasattr(self.mod, "backtest"):
            return self.mod.backtest(
                symbol=symbol,
                start=start,
                end=end,
                params=self.params,
                starting_equity=starting_equity,
            )

        # 2) Legacy ATR function name
        if hasattr(self.mod, "backtest_atr_breakout"):
            return self.mod.backtest_atr_breakout(
                symbol,
                start,
                end,
                self.params,
                starting_equity,
            )

        # 3) DF-based backtest
        if hasattr(self.mod, "backtest_on_df"):
            df = self._fetch_df(symbol, start, end)
            return self.mod.backtest_on_df(
                df=df,
                params=self.params,
                starting_equity=starting_equity,
            )

        # If none of the above, raise a clear error so trainer logs it
        raise TypeError(
            f"{self.mod.__name__} has no compatible backtest function. "
            "Expected one of: backtest(...), backtest_atr_breakout(...), backtest_on_df(...)."
        )