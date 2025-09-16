# src/utils/progress.py
"""
Generic progress reporting callback for training runs.
Expanded to surface richer telemetry from the EA loop.
"""

from typing import Callable, Dict, Any


def console_progress(event: str, payload: Dict[str, Any]):
    if event == "generation_start":
        print(f"[Gen {payload['gen']}] Starting population of {payload['pop_size']}")
    elif event == "individual_evaluated":
        m = payload.get("metrics", {}) or {}
        trades = m.get("trades", 0)
        score = payload.get("score", 0.0)
        print(f"[Gen {payload['gen']} | {payload['idx']}] Score={score:.3f} Trades={trades}")
    elif event == "generation_end":
        print(
            f"[Gen {payload['gen']}] "
            f"Best={payload.get('best_score', 0.0):.3f} "
            f"Avg={payload.get('avg_score', 0.0):.3f} "
            f"AvgTrades={payload.get('avg_trades', 0.0):.2f} "
            f"NoTrades%={100.0*payload.get('pct_no_trades', 0.0):.1f}% "
            f"[elite={payload.get('elite_n')}, breed={payload.get('breed_n')}, inject={payload.get('inject_n')}]"
        )
    elif event == "done":
        print(f"Training complete in {payload['elapsed_sec']:.1f}s. Best params: {payload['best']}")
    else:
        print(f"[{event}] {payload}")


ProgressCallback = Callable[[str, Dict[str, Any]], None]