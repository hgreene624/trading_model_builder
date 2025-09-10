# src/tuning/evolve.py
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

from src.backtest.engine import ATRParams, backtest_atr_breakout


@dataclass
class Bounds:
    # Core windows
    breakout_min: int = 20
    breakout_max: int = 120
    exit_min: int = 10
    exit_max: int = 60
    atr_min: int = 7
    atr_max: int = 30

    # Risk & stops
    atr_multiple_min: float = 1.5
    atr_multiple_max: float = 5.0
    risk_per_trade_min: float = 0.002    # 0.2%
    risk_per_trade_max: float = 0.02     # 2.0%
    tp_multiple_min: float = 0.0         # 0 => disabled is allowed
    tp_multiple_max: float = 6.0

    # Trend filter
    allow_trend_filter: bool = True
    sma_fast_min: int = 10
    sma_fast_max: int = 50
    sma_slow_min: int = 40
    sma_slow_max: int = 100
    sma_long_min: int = 100
    sma_long_max: int = 300
    long_slope_len_min: int = 10
    long_slope_len_max: int = 50

    # Time/risk management
    holding_period_min: int = 0          # 0 => disabled
    holding_period_max: int = 120

    # Costs
    cost_bps_min: float = 0.0
    cost_bps_max: float = 10.0

    # EA meta (kept here for convenience when we save/load profiles)
    pop_size: int = 40
    generations: int = 20
    crossover_rate: float = 0.7
    mutation_rate: float = 0.35


def _clip_int(x: int, lo: int, hi: int) -> int:
    return int(min(max(x, lo), hi))


def _clip_float(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def _fix(ind: Dict, b: Bounds) -> Dict:
    ind["breakout_n"] = _clip_int(ind["breakout_n"], b.breakout_min, b.breakout_max)
    ind["exit_n"] = _clip_int(ind["exit_n"], b.exit_min, b.exit_max)
    if ind["breakout_n"] <= ind["exit_n"]:
        ind["breakout_n"] = ind["exit_n"] + 1
    ind["atr_n"] = _clip_int(ind["atr_n"], b.atr_min, b.atr_max)

    ind["atr_multiple"] = _clip_float(ind["atr_multiple"], b.atr_multiple_min, b.atr_multiple_max)
    ind["risk_per_trade"] = _clip_float(ind["risk_per_trade"], b.risk_per_trade_min, b.risk_per_trade_max)
    ind["tp_multiple"] = _clip_float(ind["tp_multiple"], b.tp_multiple_min, b.tp_multiple_max)

    ind["sma_fast"] = _clip_int(ind["sma_fast"], b.sma_fast_min, b.sma_fast_max)
    ind["sma_slow"] = _clip_int(ind["sma_slow"], b.sma_slow_min, b.sma_slow_max)
    if ind["sma_fast"] >= ind["sma_slow"]:
        ind["sma_fast"] = max(b.sma_fast_min, ind["sma_slow"] - 1)
    ind["sma_long"] = _clip_int(ind["sma_long"], b.sma_long_min, b.sma_long_max)
    ind["long_slope_len"] = _clip_int(ind["long_slope_len"], b.long_slope_len_min, b.long_slope_len_max)

    ind["holding_period_limit"] = _clip_int(ind["holding_period_limit"], b.holding_period_min, b.holding_period_max)
    ind["cost_bps"] = _clip_float(ind["cost_bps"], b.cost_bps_min, b.cost_bps_max)

    if not b.allow_trend_filter:
        ind["use_trend_filter"] = False

    return ind


def _rand(b: Bounds, rng: random.Random) -> Dict:
    ind = {
        "breakout_n": rng.randint(b.breakout_min, b.breakout_max),
        "exit_n": rng.randint(b.exit_min, b.exit_max),
        "atr_n": rng.randint(b.atr_min, b.atr_max),

        "atr_multiple": rng.uniform(b.atr_multiple_min, b.atr_multiple_max),
        "risk_per_trade": rng.uniform(b.risk_per_trade_min, b.risk_per_trade_max),
        "tp_multiple": rng.uniform(b.tp_multiple_min, b.tp_multiple_max),

        "use_trend_filter": (rng.random() < 0.5) if b.allow_trend_filter else False,
        "sma_fast": rng.randint(b.sma_fast_min, b.sma_fast_max),
        "sma_slow": rng.randint(b.sma_slow_min, b.sma_slow_max),
        "sma_long": rng.randint(b.sma_long_min, b.sma_long_max),
        "long_slope_len": rng.randint(b.long_slope_len_min, b.long_slope_len_max),

        "holding_period_limit": rng.randint(b.holding_period_min, b.holding_period_max),
        "cost_bps": rng.uniform(b.cost_bps_min, b.cost_bps_max),
    }
    return _fix(ind, b)


def _mutate(ind: Dict, b: Bounds, rng: random.Random) -> Dict:
    out = dict(ind)

    # integers
    if rng.random() < 0.7: out["breakout_n"] += rng.randint(-5, 5)
    if rng.random() < 0.7: out["exit_n"] += rng.randint(-3, 3)
    if rng.random() < 0.6: out["atr_n"] += rng.randint(-2, 2)
    if rng.random() < 0.4: out["sma_fast"] += rng.randint(-3, 3)
    if rng.random() < 0.4: out["sma_slow"] += rng.randint(-3, 3)
    if rng.random() < 0.4: out["sma_long"] += rng.randint(-5, 5)
    if rng.random() < 0.4: out["long_slope_len"] += rng.randint(-2, 2)
    if rng.random() < 0.4: out["holding_period_limit"] += rng.randint(-10, 10)

    # floats (multiplicative jitter)
    if rng.random() < 0.6: out["atr_multiple"] *= (1.0 + rng.uniform(-0.2, 0.2))
    if rng.random() < 0.6: out["risk_per_trade"] *= (1.0 + rng.uniform(-0.25, 0.25))
    if rng.random() < 0.6: out["tp_multiple"] *= (1.0 + rng.uniform(-0.3, 0.3))
    if rng.random() < 0.5: out["cost_bps"] *= (1.0 + rng.uniform(-0.4, 0.4))

    # bool flip
    if b.allow_trend_filter and rng.random() < 0.2:
        out["use_trend_filter"] = not out["use_trend_filter"]

    return _fix(out, b)


def _xover(a: Dict, b_: Dict, rng: random.Random) -> Dict:
    return {
        "breakout_n": a["breakout_n"] if rng.random() < 0.5 else b_["breakout_n"],
        "exit_n": a["exit_n"] if rng.random() < 0.5 else b_["exit_n"],
        "atr_n": a["atr_n"] if rng.random() < 0.5 else b_["atr_n"],
        "atr_multiple": a["atr_multiple"] if rng.random() < 0.5 else b_["atr_multiple"],
        "risk_per_trade": a["risk_per_trade"] if rng.random() < 0.5 else b_["risk_per_trade"],
        "tp_multiple": a["tp_multiple"] if rng.random() < 0.5 else b_["tp_multiple"],
        "use_trend_filter": a["use_trend_filter"] if rng.random() < 0.5 else b_["use_trend_filter"],
        "sma_fast": a["sma_fast"] if rng.random() < 0.5 else b_["sma_fast"],
        "sma_slow": a["sma_slow"] if rng.random() < 0.5 else b_["sma_slow"],
        "sma_long": a["sma_long"] if rng.random() < 0.5 else b_["sma_long"],
        "long_slope_len": a["long_slope_len"] if rng.random() < 0.5 else b_["long_slope_len"],
        "holding_period_limit": a["holding_period_limit"] if rng.random() < 0.5 else b_["holding_period_limit"],
        "cost_bps": a["cost_bps"] if rng.random() < 0.5 else b_["cost_bps"],
    }


def _fitness(symbol: str, start: str, end: str, starting_equity: float, ind: Dict) -> Tuple[float, Dict]:
    # Map gene values into engine params
    tp_mult = None if ind.get("tp_multiple", 0.0) <= 0.0 else float(ind["tp_multiple"])
    hp_limit = None if int(ind.get("holding_period_limit", 0)) <= 0 else int(ind["holding_period_limit"])

    params = ATRParams(
        breakout_n=int(ind["breakout_n"]),
        exit_n=int(ind["exit_n"]),
        atr_n=int(ind["atr_n"]),
        atr_multiple=float(ind["atr_multiple"]),
        risk_per_trade=float(ind["risk_per_trade"]),
        allow_fractional=True,
        slippage_bp=5.0,
        cost_bps=float(ind.get("cost_bps", 1.0)),
        fee_per_trade=0.0,
        tp_multiple=tp_mult,
        use_trend_filter=bool(ind.get("use_trend_filter", False)),
        sma_fast=int(ind.get("sma_fast", 30)),
        sma_slow=int(ind.get("sma_slow", 50)),
        sma_long=int(ind.get("sma_long", 150)),
        long_slope_len=int(ind.get("long_slope_len", 15)),
        holding_period_limit=hp_limit,
    )
    res = backtest_atr_breakout(symbol, start, end, float(starting_equity), params)
    metrics = res["metrics"]

    # Blend risk & return:
    # score = 0.5*Sharpe + 0.4*TotalReturn - 0.1*DrawdownPenalty
    sharpe = float(metrics.get("sharpe", 0.0) or 0.0)
    total_return = float(metrics.get("total_return", 0.0) or 0.0)  # e.g. 0.12 = +12%
    max_dd = float(metrics.get("max_drawdown", 0.0) or 0.0)  # negative value
    dd_penalty = max(0.0, -max_dd)

    score = 0.5 * sharpe + 0.4 * total_return - 0.1 * dd_penalty
    return score, metrics


def evolve_params(
    symbol: str,
    start: str,
    end: str,
    starting_equity: float,
    bounds: Bounds,
    pop_size: int | None = None,
    generations: int | None = None,
    crossover_rate: float | None = None,
    mutation_rate: float | None = None,
    random_seed: int | None = 42,
    progress_cb: Callable[[int, int, float], None] | None = None,
) -> Tuple[Dict, Dict, List[Dict]]:
    """
    Evolves breakout/exit/atr/atr_multiple/risk_per_trade + tp_multiple + trend filter (optional) +
    SMA windows + long slope + holding-period limit + cost_bps to maximize Sharpe.
    Returns (best_params, best_metrics, history).
    """
    rng = random.Random(random_seed)

    pop_size = pop_size or bounds.pop_size
    generations = generations or bounds.generations
    crossover_rate = crossover_rate if crossover_rate is not None else bounds.crossover_rate
    mutation_rate = mutation_rate if mutation_rate is not None else bounds.mutation_rate

    pop = [_rand(bounds, rng) for _ in range(pop_size)]

    best_ind: Dict | None = None
    best_fit = -1e12
    best_metrics: Dict = {}
    history: List[Dict] = []

    for gen in range(generations):
        scored = []
        for ind in pop:
            f, m = _fitness(symbol, start, end, starting_equity, ind)
            scored.append((f, ind, m))

        scored.sort(key=lambda x: x[0], reverse=True)
        gen_best_fit, gen_best_ind, gen_best_metrics = scored[0]

        if gen_best_fit > best_fit:
            best_fit = gen_best_fit
            best_ind = dict(gen_best_ind)
            best_metrics = dict(gen_best_metrics)

        avg_fit = float(np.mean([s[0] for s in scored])) if scored else 0.0
        history.append({
            "generation": gen,
            "best_fitness": float(gen_best_fit),
            "avg_fitness": float(avg_fit),
            "best_params": dict(gen_best_ind),
        })

        if progress_cb:
            progress_cb(gen + 1, generations, float(gen_best_fit))

        # Next generation
        keep = max(1, int(0.1 * pop_size))  # elitism
        next_pop = [dict(scored[i][1]) for i in range(keep)]

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
                child = _xover(p1, p2, rng)
            if rng.random() < mutation_rate:
                child = _mutate(child, bounds, rng)
            child = _fix(child, bounds)
            next_pop.append(child)

        pop = next_pop

    return best_ind or {}, best_metrics, history