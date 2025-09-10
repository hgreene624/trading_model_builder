# src/tuning/evolve.py
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.data.cache import get_ohlcv_cached  # uses Alpaca + Streamlit cache


@dataclass
class Bounds:
    breakout_min: int
    breakout_max: int
    exit_min: int
    exit_max: int
    atr_min: int
    atr_max: int


def _wilder_atr(df: pd.DataFrame, n: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    return atr


def _evaluate_params_on_df(
    df_in: pd.DataFrame,
    breakout_n: int,
    exit_n: int,
    atr_n: int,
    starting_equity: float,
) -> Tuple[float, Dict]:
    """
    Returns (fitness, metrics) where fitness is Sharpe by default (we'll compute Sharpe here).
    Metrics contain total_return, sharpe, max_drawdown, final_equity, etc.
    """
    # Work on a copy to avoid mutating cached data
    df = df_in.copy()

    # Indicators
    df["atr"] = _wilder_atr(df, n=atr_n)
    df["breakout_high"] = (
        df["high"].rolling(window=breakout_n, min_periods=breakout_n).max().shift(1)
    )
    df["exit_low"] = (
        df["low"].rolling(window=exit_n, min_periods=exit_n).min().shift(1)
    )

    # Simple Donchian-style regime (binary position)
    in_pos = False
    pos = []
    for _, row in df.iterrows():
        if not in_pos:
            in_pos = bool(row["close"] > row["breakout_high"]) if not np.isnan(row["breakout_high"]) else False
        else:
            if not np.isnan(row["exit_low"]) and row["close"] < row["exit_low"]:
                in_pos = False
        pos.append(1.0 if in_pos else 0.0)

    df["pos"] = pd.Series(pos, index=df.index)

    ret = df["close"].pct_change().fillna(0.0)
    strat_ret = ret * df["pos"].shift(1).fillna(0.0)
    equity = (1.0 + strat_ret).cumprod() * float(starting_equity)

    total_return = (equity.iloc[-1] / starting_equity) - 1.0
    daily = strat_ret
    sharpe = (daily.mean() / (daily.std() + 1e-12)) * np.sqrt(252) if daily.std() > 0 else 0.0
    running_max = equity.cummax()
    drawdown = (equity / running_max) - 1.0
    max_dd = float(drawdown.min()) if len(drawdown) else 0.0

    metrics = {
        "total_return": float(total_return),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "final_equity": float(equity.iloc[-1]),
        "start": str(equity.index[0].date()) if len(equity) else "",
        "end": str(equity.index[-1].date()) if len(equity) else "",
    }
    # Fitness = Sharpe (higher is better)
    return float(sharpe), metrics


def _fix_constraints(indiv: Dict[str, int], b: Bounds) -> Dict[str, int]:
    # Enforce breakout > exit
    if indiv["breakout_n"] <= indiv["exit_n"]:
        indiv["breakout_n"] = min(max(indiv["exit_n"] + 1, b.breakout_min), b.breakout_max)
    # Clip all to bounds
    indiv["breakout_n"] = int(min(max(indiv["breakout_n"], b.breakout_min), b.breakout_max))
    indiv["exit_n"] = int(min(max(indiv["exit_n"], b.exit_min), b.exit_max))
    indiv["atr_n"] = int(min(max(indiv["atr_n"], b.atr_min), b.atr_max))
    return indiv


def _random_individual(b: Bounds, rng: random.Random) -> Dict[str, int]:
    indiv = {
        "breakout_n": rng.randint(b.breakout_min, b.breakout_max),
        "exit_n": rng.randint(b.exit_min, b.exit_max),
        "atr_n": rng.randint(b.atr_min, b.atr_max),
    }
    return _fix_constraints(indiv, b)


def _mutate(indiv: Dict[str, int], b: Bounds, rng: random.Random, step: Dict[str, int]) -> Dict[str, int]:
    out = dict(indiv)
    if rng.random() < 0.5:
        out["breakout_n"] += rng.randint(-step["breakout_n"], step["breakout_n"])
    if rng.random() < 0.5:
        out["exit_n"] += rng.randint(-step["exit_n"], step["exit_n"])
    if rng.random() < 0.5:
        out["atr_n"] += rng.randint(-step["atr_n"], step["atr_n"])
    return _fix_constraints(out, b)


def _crossover(a: Dict[str, int], b_: Dict[str, int], rng: random.Random) -> Dict[str, int]:
    # Uniform crossover
    return {
        "breakout_n": a["breakout_n"] if rng.random() < 0.5 else b_["breakout_n"],
        "exit_n": a["exit_n"] if rng.random() < 0.5 else b_["exit_n"],
        "atr_n": a["atr_n"] if rng.random() < 0.5 else b_["atr_n"],
    }


def evolve_params(
    symbol: str,
    start: str,
    end: str,
    starting_equity: float,
    bounds: Bounds,
    pop_size: int = 30,
    generations: int = 15,
    crossover_rate: float = 0.7,
    mutation_rate: float = 0.3,
    random_seed: int | None = 42,
    progress_cb: Callable[[int, int, float], None] | None = None,
) -> Tuple[Dict[str, int], Dict, List[Dict]]:
    """
    Returns: (best_params, best_metrics, history)
      - best_params: dict with breakout_n, exit_n, atr_n
      - best_metrics: dict with sharpe, total_return, max_drawdown, final_equity
      - history: list of per-generation summaries
    """

    rng = random.Random(random_seed)

    # Load data once for speed
    df = get_ohlcv_cached(symbol, start, end)

    # Initialize population
    pop = [_random_individual(bounds, rng) for _ in range(pop_size)]

    def fitness(indiv: Dict[str, int]) -> Tuple[float, Dict]:
        return _evaluate_params_on_df(
            df,
            breakout_n=indiv["breakout_n"],
            exit_n=indiv["exit_n"],
            atr_n=indiv["atr_n"],
            starting_equity=starting_equity,
        )

    history: List[Dict] = []
    best_indiv = None
    best_fit = -1e9
    best_metrics = {}

    for gen in range(generations):
        # Evaluate
        scored = []
        for indiv in pop:
            f, m = fitness(indiv)
            scored.append((f, indiv, m))

        # Sort by fitness (descending)
        scored.sort(key=lambda x: x[0], reverse=True)
        gen_best_fit, gen_best_indiv, gen_best_metrics = scored[0]

        if gen_best_fit > best_fit:
            best_fit = gen_best_fit
            best_indiv = dict(gen_best_indiv)
            best_metrics = dict(gen_best_metrics)

        # Record history
        avg_fit = float(np.mean([s[0] for s in scored])) if scored else 0.0
        history.append(
            {
                "generation": gen,
                "best_fitness": float(gen_best_fit),
                "avg_fitness": float(avg_fit),
                "best_params": dict(gen_best_indiv),
            }
        )

        if progress_cb:
            progress_cb(gen + 1, generations, float(gen_best_fit))

        # ----- Breed next generation -----
        # Elitism: keep top 10%
        keep = max(1, int(0.1 * pop_size))
        next_pop = [dict(scored[i][1]) for i in range(keep)]

        # Selection pool (tournament selection)
        def tournament() -> Dict[str, int]:
            k = 3
            picks = rng.sample(scored[: max(pop_size, 3)], k=min(k, len(scored)))
            picks.sort(key=lambda x: x[0], reverse=True)
            return dict(picks[0][1])

        # Fill rest of population
        while len(next_pop) < pop_size:
            parent1 = tournament()
            parent2 = tournament()
            child = dict(parent1)
            if rng.random() < crossover_rate:
                child = _crossover(parent1, parent2, rng)
            if rng.random() < mutation_rate:
                child = _mutate(
                    child,
                    bounds,
                    rng,
                    step={"breakout_n": 5, "exit_n": 3, "atr_n": 2},
                )
            child = _fix_constraints(child, bounds)
            next_pop.append(child)

        pop = next_pop

    return best_indiv or {}, best_metrics, history