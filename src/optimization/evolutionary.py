# src/optimization/evolutionary.py
"""
Evolutionary parameter search with progress reporting + logging.

Enhancements:
- Exposed EA knobs (mutation_rate, elite_frac, random_inject_frac)
- Fitness safety gates (min_trades, avg_holding_days > 0)
- Ratio clamping to avoid degenerate scores (eps for max_drawdown & sharpe)
- Survivor selection enforces min_trades gate
- Gen telemetry: best/avg score, avg trades, % no-trades, elite/breed/inject counts
- Anomaly logging: under_min_trades, degenerate_fitness
"""

from __future__ import annotations
import random
import time
from typing import Any, Dict, List, Tuple

from src.models.general_trainer import train_general_model
from src.utils.training_logger import TrainingLogger
from src.utils.progress import ProgressCallback, console_progress


def random_param(param_space: Dict[str, Tuple]) -> Dict[str, Any]:
    """Sample a random param set from (low, high) ranges. Int vs float inferred from bounds."""
    out: Dict[str, Any] = {}
    for k, v in param_space.items():
        low, high = v
        if isinstance(low, float) or isinstance(high, float):
            out[k] = float(random.uniform(float(low), float(high)))
        else:
            out[k] = int(random.randint(int(low), int(high)))
    return out


def mutate(params: Dict[str, Any], param_space: Dict[str, Tuple], rate: float = 0.3) -> Dict[str, Any]:
    """Randomly tweak a subset of params according to `rate`."""
    new = dict(params)
    for k, v in param_space.items():
        if random.random() < rate:
            low, high = v
            if isinstance(low, float) or isinstance(high, float):
                new[k] = float(random.uniform(float(low), float(high)))
            else:
                new[k] = int(random.randint(int(low), int(high)))
    return new


def crossover(p1: Dict[str, Any], p2: Dict[str, Any]) -> Dict[str, Any]:
    """Uniform crossover."""
    return {k: (p1[k] if random.random() < 0.5 else p2[k]) for k in p1}


def _clamped_fitness(
    metrics: Dict[str, Any],
    *,
    min_trades: int,
    require_hold_days: bool,
    eps_mdd: float,
    eps_sharpe: float,
) -> float:
    """
    Robust fitness to avoid degenerate explosions when there are no trades
    or near-zero denominators.

    Score = calmar + 0.5 * sharpe, with gates:
      - if trades < min_trades: score = 0
      - if require_hold_days and avg_holding_days <= 0: score = 0
      - if |max_drawdown| < eps_mdd: calmar = 0
      - if |sharpe| < eps_sharpe:   sharpe = 0
    """
    trades = int(metrics.get("trades", 0) or 0)
    if trades < min_trades:
        return 0.0
    if require_hold_days and float(metrics.get("avg_holding_days", 0.0) or 0.0) <= 0.0:
        return 0.0

    calmar = float(metrics.get("calmar", 0.0) or 0.0)
    sharpe = float(metrics.get("sharpe", 0.0) or 0.0)
    mdd = float(abs(metrics.get("max_drawdown", 0.0) or 0.0))

    if mdd < eps_mdd:
        calmar = 0.0
    if abs(sharpe) < eps_sharpe:
        sharpe = 0.0

    return calmar + 0.5 * sharpe


def evolutionary_search(
    strategy_dotted: str,
    tickers: List[str],
    start,
    end,
    starting_equity: float,
    param_space: Dict[str, Tuple],
    *,
    generations: int = 10,
    pop_size: int = 20,
    # Exposed EA knobs
    mutation_rate: float = 0.3,
    elite_frac: float = 0.5,            # keep top 50%
    random_inject_frac: float = 0.2,    # inject 20% fresh randoms each gen
    # Fitness gates & clamps
    min_trades: int = 5,
    require_hold_days: bool = False,
    eps_mdd: float = 1e-4,
    eps_sharpe: float = 1e-4,
    # Progress & logging
    progress_cb: ProgressCallback = console_progress,
    log_file: str = "training.log",
) -> List[Tuple[Dict[str, Any], float]]:
    """
    Run EA and return top parameter sets [(params, score)].

    Notes:
    - Survivor selection enforces `min_trades` (via the fitness gate).
    - Diversity is maintained via `random_inject_frac`.
    - Mutation uses `mutation_rate`.
    """
    logger = TrainingLogger(log_file)
    population = [random_param(param_space) for _ in range(pop_size)]
    scored: List[Tuple[Dict[str, Any], float]] = []
    t0 = time.time()

    for gen in range(generations):
        progress_cb("generation_start", {"gen": gen, "pop_size": len(population)})
        logger.log("generation_start", {"gen": gen, "pop_size": len(population)})

        gen_scores: List[Tuple[Dict[str, Any], float]] = []
        gen_trades: List[int] = []
        no_trade_count = 0

        for i, params in enumerate(population):
            t1 = time.time()
            try:
                res = train_general_model(strategy_dotted, tickers, start, end, starting_equity, params)
                metrics = res.get("aggregate", {}).get("metrics", {}) or {}

                # robust fitness with gates/clamps
                score = _clamped_fitness(
                    metrics,
                    min_trades=min_trades,
                    require_hold_days=require_hold_days,
                    eps_mdd=eps_mdd,
                    eps_sharpe=eps_sharpe,
                )
                elapsed = time.time() - t1

                trades = int(metrics.get("trades", 0) or 0)
                gen_trades.append(trades)

                payload = {
                    "gen": gen,
                    "idx": i,
                    "params": params,
                    "score": score,
                    "metrics": metrics,
                    "elapsed_sec": elapsed,
                }
                progress_cb("individual_evaluated", payload)
                logger.log("individual_evaluated", payload)

                if trades < min_trades:
                    logger.log("under_min_trades", payload)
                    if trades == 0 and (metrics.get("calmar", 0) or 0) != 0:
                        logger.log("degenerate_fitness", payload)
                    no_trade_count += (1 if trades == 0 else 0)

                gen_scores.append((params, score))

            except Exception as e:
                # hard error path
                ctx = {"gen": gen, "idx": i, "params": params}
                logger.log_error(ctx, e)

        # sort by score desc
        gen_scores.sort(key=lambda x: x[1], reverse=True)
        scored.extend(gen_scores)

        # determine counts
        elite_n = max(1, int(pop_size * elite_frac))
        inject_n = max(0, int(pop_size * random_inject_frac))
        breed_n = pop_size - elite_n - inject_n
        if breed_n < 0:
            # if fractions over-allocate, reduce random inject
            inject_n = max(0, pop_size - elite_n)
            breed_n = pop_size - elite_n - inject_n

        # survivors must pass min_trades gate (fitness gate already forces s>0)
        survivors: List[Dict[str, Any]] = []
        for p, s in gen_scores:
            if s > 0:
                survivors.append(p)
            if len(survivors) == elite_n:
                break

        # if too few survivors due to gate, pad with top-scoring anyway (keep population viable)
        idx = 0
        while len(survivors) < elite_n and idx < len(gen_scores):
            survivors.append(gen_scores[idx][0])
            idx += 1

        # breed children from survivors (if at least 2 exist; else use random parents)
        children: List[Dict[str, Any]] = []
        parents = survivors if len(survivors) >= 2 else [random_param(param_space) for _ in range(2)]
        for _ in range(max(0, breed_n)):
            p1 = random.choice(parents)
            p2 = random.choice(parents)
            child = mutate(crossover(p1, p2), param_space, rate=mutation_rate)
            children.append(child)

        # random injections
        injections = [random_param(param_space) for _ in range(max(0, inject_n))]

        # new population
        population = survivors + children + injections
        # if population drifted, re-pad randomly
        while len(population) < pop_size:
            population.append(random_param(param_space))
        if len(population) > pop_size:
            population = population[:pop_size]

        avg_score = (sum(s for _, s in gen_scores) / len(gen_scores)) if gen_scores else 0.0
        best_score = gen_scores[0][1] if gen_scores else 0.0
        avg_trades = (sum(gen_trades) / len(gen_trades)) if gen_trades else 0.0
        pct_no_trades = (no_trade_count / len(gen_trades)) if gen_trades else 0.0

        end_payload = {
            "gen": gen,
            "best_score": best_score,
            "avg_score": avg_score,
            "avg_trades": avg_trades,
            "pct_no_trades": pct_no_trades,
            "top_params": gen_scores[0][0] if gen_scores else {},
            "elite_n": elite_n,
            "breed_n": breed_n,
            "inject_n": inject_n,
        }
        progress_cb("generation_end", end_payload)
        logger.log("generation_end", end_payload)

    elapsed_total = time.time() - t0
    scored.sort(key=lambda x: x[1], reverse=True)
    best = scored[0] if scored else ({}, 0.0)
    done_payload = {"elapsed_sec": elapsed_total, "best": best[0], "score": best[1]}
    progress_cb("done", done_payload)
    logger.log("session_end", done_payload)
    return scored[:5]