# src/tuning/evolve.py
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

from src.backtest.engine import ATRParams, backtest_atr_breakout


@dataclass
class Bounds:
    # Integer windows
    breakout_min: int = 20
    breakout_max: int = 120
    exit_min: int = 10
    exit_max: int = 60
    atr_min: int = 7
    atr_max: int = 30
    # Risk params (floats)
    atr_multiple_min: float = 1.5
    atr_multiple_max: float = 5.0
    risk_per_trade_min: float = 0.002   # 0.2%
    risk_per_trade_max: float = 0.02    # 2%


def _clip_int(x: int, lo: int, hi: int) -> int:
    return int(min(max(x, lo), hi))


def _clip_float(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def _fix_constraints(indiv: Dict, b: Bounds) -> Dict:
    # Ensure breakout > exit
    if indiv["breakout_n"] <= indiv["exit_n"]:
        indiv["breakout_n"] = indiv["exit_n"] + 1
    # Clip to bounds
    indiv["breakout_n"] = _clip_int(indiv["breakout_n"], b.breakout_min, b.breakout_max)
    indiv["exit_n"] = _clip_int(indiv["exit_n"], b.exit_min, b.exit_max)
    indiv["atr_n"] = _clip_int(indiv["atr_n"], b.atr_min, b.atr_max)
    indiv["atr_multiple"] = _clip_float(indiv["atr_multiple"], b.atr_multiple_min, b.atr_multiple_max)
    indiv["risk_per_trade"] = _clip_float(indiv["risk_per_trade"], b.risk_per_trade_min, b.risk_per_trade_max)
    return indiv


def _random_individual(b: Bounds, rng: random.Random) -> Dict:
    indiv = {
        "breakout_n": rng.randint(b.breakout_min, b.breakout_max),
        "exit_n": rng.randint(b.exit_min, b.exit_max),
        "atr_n": rng.randint(b.atr_min, b.atr_max),
        "atr_multiple": rng.uniform(b.atr_multiple_min, b.atr_multiple_max),
        "risk_per_trade": rng.uniform(b.risk_per_trade_min, b.risk_per_trade_max),
    }
    return _fix_constraints(indiv, b)


def _mutate(indiv: Dict, b: Bounds, rng: random.Random) -> Dict:
    out = dict(indiv)
    # Integer params: small integer steps
    if rng.random() < 0.7:
        out["breakout_n"] += rng.randint(-5, 5)
    if rng.random() < 0.7:
        out["exit_n"] += rng.randint(-3, 3)
    if rng.random() < 0.6:
        out["atr_n"] += rng.randint(-2, 2)
    # Floats: multiplicative jitter (log-normal-ish)
    if rng.random() < 0.6:
        out["atr_multiple"] *= (1.0 + rng.uniform(-0.2, 0.2))
    if rng.random() < 0.6:
        out["risk_per_trade"] *= (1.0 + rng.uniform(-0.25, 0.25))
    return _fix_constraints(out, b)


def _crossover(a: Dict, b_: Dict, rng: random.Random) -> Dict:
    # Uniform crossover
    return {
        "breakout_n": a["breakout_n"] if rng.random() < 0.5 else b_["breakout_n"],
        "exit_n": a["exit_n"] if rng.random() < 0.5 else b_["exit_n"],
        "atr_n": a["atr_n"] if rng.random() < 0.5 else b_["atr_n"],
        "atr_multiple": a["atr_multiple"] if rng.random() < 0.5 else b_["atr_multiple"],
        "risk_per_trade": a["risk_per_trade"] if rng.random() < 0.5 else b_["risk_per_trade"],
    }


def _fitness(symbol: str, start: str, end: str, starting_equity: float, indiv: Dict) -> Tuple[float, Dict]:
    # Evaluate via the real backtest engine (includes ATR sizing & stops)
    params = ATRParams(
        breakout_n=int(indiv["breakout_n"]),
        exit_n=int(indiv["exit_n"]),
        atr_n=int(indiv["atr_n"]),
        atr_multiple=float(indiv["atr_multiple"]),
        risk_per_trade=float(indiv["risk_per_trade"]),
        allow_fractional=True,
        slippage_bp=5.0,
        fee_per_trade=0.0,
    )
    res = backtest_atr_breakout(symbol, start, end, float(starting_equity), params)
    metrics = res["metrics"]
    return float(metrics.get("sharpe", 0.0)), metrics


def evolve_params(
    symbol: str,
    start: str,
    end: str,
    starting_equity: float,
    bounds: Bounds,
    pop_size: int = 40,
    generations: int = 20,
    crossover_rate: float = 0.7,
    mutation_rate: float = 0.35,
    random_seed: int | None = 42,
    progress_cb: Callable[[int, int, float], None] | None = None,
) -> Tuple[Dict, Dict, List[Dict]]:
    """
    Evolves breakout_n, exit_n, atr_n, atr_multiple, risk_per_trade to maximize Sharpe.
    Returns (best_params, best_metrics, history)
    """
    rng = random.Random(random_seed)

    # Initialize population
    pop = [_random_individual(bounds, rng) for _ in range(pop_size)]

    best_indiv: Dict | None = None
    best_fit = -1e12
    best_metrics: Dict = {}
    history: List[Dict] = []

    for gen in range(generations):
        scored = []
        for indiv in pop:
            f, m = _fitness(symbol, start, end, starting_equity, indiv)
            scored.append((f, indiv, m))

        scored.sort(key=lambda x: x[0], reverse=True)
        gen_best_fit, gen_best_indiv, gen_best_metrics = scored[0]

        if gen_best_fit > best_fit:
            best_fit = gen_best_fit
            best_indiv = dict(gen_best_indiv)
            best_metrics = dict(gen_best_metrics)

        avg_fit = float(np.mean([s[0] for s in scored])) if scored else 0.0
        history.append({
            "generation": gen,
            "best_fitness": float(gen_best_fit),
            "avg_fitness": float(avg_fit),
            "best_params": dict(gen_best_indiv),
        })

        if progress_cb:
            progress_cb(gen + 1, generations, float(gen_best_fit))

        # ---- Create next generation ----
        keep = max(1, int(0.1 * pop_size))  # elitism
        next_pop = [dict(scored[i][1]) for i in range(keep)]

        # Tournament selection
        def tournament() -> Dict:
            k = 3
            picks = rng.sample(scored[: max(pop_size, 3)], k=min(k, len(scored)))
            picks.sort(key=lambda x: x[0], reverse=True)
            return dict(picks[0][1])

        while len(next_pop) < pop_size:
            p1 = tournament()
            p2 = tournament()
            child = dict(p1)
            if rng.random() < crossover_rate:
                child = _crossover(p1, p2, rng)
            if rng.random() < mutation_rate:
                child = _mutate(child, bounds, rng)
            child = _fix_constraints(child, bounds)
            next_pop.append(child)

        pop = next_pop

    return best_indiv or {}, best_metrics, history