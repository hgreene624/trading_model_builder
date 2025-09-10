# bt_check.py
from src.models.atr_breakout import backtest_single

res = backtest_single(
    "AAPL", "2024-01-01", "2025-09-10",
    breakout_n=55, exit_n=20, atr_n=14,
    starting_equity=10000, atr_multiple=3.0, risk_per_trade=0.01,
)
print(res["metrics"])
print("Trades:", len(res["trades"]))
print(res["equity"].tail())