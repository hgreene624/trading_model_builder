# tests/test_ea_logger.py
from pathlib import Path
import sys

# make project root (that contains /src and /tests) importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datetime import date
from src.optimization.evolutionary import evolutionary_search
from src.utils.progress import console_progress


if __name__ == "__main__":
    # Narrow param space for a quick smoke run
    param_space = {
        "breakout_n": (10, 15),
        "exit_n": (5, 8),
        "atr_n": (10, 14),
        "atr_multiple": (1.5, 2.5),
        "tp_multiple": (0.0, 1.0),
        "holding_period_limit": (0, 5),
    }

    results = evolutionary_search(
        strategy_dotted="src.models.atr_breakout",
        tickers=["AAPL"],  # must exist in your data loader
        start=date(2020, 1, 1),
        end=date(2021, 1, 1),
        starting_equity=10000.0,
        param_space=param_space,
        generations=2,
        pop_size=6,
        # EA diversity
        mutation_rate=0.5,
        elite_frac=0.5,
        random_inject_frac=0.2,
        # Fitness gates
        min_trades=3,
        require_hold_days=False,
        eps_mdd=1e-4,
        eps_sharpe=1e-4,
        # Fitness weights (growth vs risk)
        alpha_cagr=1.0,
        beta_calmar=1.0,
        gamma_sharpe=0.25,
        # Holding window preference (avoid day trades; avoid buy/hold)
        min_holding_days=3.0,
        max_holding_days=30.0,
        holding_penalty_weight=0.1,
        # Progress & logs
        progress_cb=console_progress,
        log_file="ea_test.log",
    )

    print("\nTop results:")
    for params, score in results:
        print(f"Score={score:.3f} Params={params}")