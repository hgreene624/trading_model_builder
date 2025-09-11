# src/tuning/evolve.py
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

from src.backtest.engine import ATRParams, backtest_atr_breakout
import math

_rng = random.random

def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def _blend(a, b, alpha=0.5):
    # BLX/BLEND crossover for floats
    lo, hi = (a, b) if a <= b else (b, a)
    range_ = hi - lo
    lo -= alpha * range_
    hi += alpha * range_
    return random.uniform(lo, hi)

def _tournament_select(pop, fits, k=3):
    # fits[i][0] must be the scalar fitness
    idxs = [random.randrange(len(pop)) for _ in range(k)]
    best = max(idxs, key=lambda i: fits[i][0])
    return pop[best]

def _sig_from_range(lo, hi, frac=0.12):
    # Gaussian step ~12% of the range by default
    return max(1e-9, (hi - lo) * frac)

def _log_mut(val, lo, hi, sigma=0.25):
    # multiplicative/log-space mutation for positive floats
    nv = val * math.exp(random.gauss(0.0, sigma))
    return _clamp(nv, lo, hi)

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
    y = dict(ind)
    # Core windows
    y["breakout_n"] = _clip_int(int(y.get("breakout_n", 55)), b.breakout_min, b.breakout_max)
    y["exit_n"]     = _clip_int(int(y.get("exit_n", 20)), b.exit_min, min(b.exit_max, y["breakout_n"] - 1))
    if y["exit_n"] >= y["breakout_n"]:
        y["exit_n"] = max(b.exit_min, int(0.4 * y["breakout_n"]))
    y["atr_n"]      = _clip_int(int(y.get("atr_n", 14)), b.atr_min, b.atr_max)

    # Floats
    y["atr_multiple"]   = _clip_float(float(y.get("atr_multiple", 3.0)), b.atr_multiple_min, b.atr_multiple_max)
    y["risk_per_trade"] = _clip_float(float(y.get("risk_per_trade", 0.01)), b.risk_per_trade_min, b.risk_per_trade_max)
    y["tp_multiple"]    = _clip_float(float(y.get("tp_multiple", 0.0)), b.tp_multiple_min, b.tp_multiple_max)
    y["cost_bps"]       = _clip_float(float(y.get("cost_bps", 0.0)), b.cost_bps_min, b.cost_bps_max)

    # Trend filter ordering
    sf = _clip_int(int(y.get("sma_fast", 30)), b.sma_fast_min, b.sma_fast_max)
    ss = _clip_int(int(y.get("sma_slow", 50)), max(b.sma_slow_min, sf + 1), b.sma_slow_max)
    sl = _clip_int(int(y.get("sma_long", 150)), max(b.sma_long_min, ss + 1), b.sma_long_max)
    y["sma_fast"], y["sma_slow"], y["sma_long"] = sf, ss, sl

    # Slope len & holding period
    y["long_slope_len"]      = _clip_int(int(y.get("long_slope_len", 15)), b.long_slope_len_min, b.long_slope_len_max)
    y["holding_period_limit"] = _clip_int(int(y.get("holding_period_limit", 0)), b.holding_period_min, b.holding_period_max)

    # Bool
    y["use_trend_filter"] = bool(y.get("use_trend_filter", False)) and bool(b.allow_trend_filter)

    return y


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

    # ---- integers with Gaussian steps aware of ranges ----
    if rng.random() < 0.7:
        sig = max(1.0, _sig_from_range(b.breakout_min, b.breakout_max))
        out["breakout_n"] = int(round(out.get("breakout_n", 55) + rng.gauss(0, sig)))

    if rng.random() < 0.7:
        br = max(1, int(out.get("breakout_n", 55)))
        ratio = _clamp(out.get("exit_n", 20) / br, 0.2, 0.9)
        ratio += rng.gauss(0, 0.06)
        out["exit_n"] = int(round(_clamp(ratio, 0.2, 0.9) * br))

    if rng.random() < 0.6:
        sig = max(1.0, _sig_from_range(b.atr_min, b.atr_max, frac=0.15))
        out["atr_n"] = int(round(out.get("atr_n", 14) + rng.gauss(0, sig)))

    # trend filter windows
    if rng.random() < 0.5:
        out["sma_fast"] = int(round(out.get("sma_fast", 30) + rng.gauss(0, 4)))
    if rng.random() < 0.5:
        out["sma_slow"] = int(round(out.get("sma_slow", 50) + rng.gauss(0, 5)))
    if rng.random() < 0.5:
        out["sma_long"] = int(round(out.get("sma_long", 150) + rng.gauss(0, 8)))
    if rng.random() < 0.5:
        out["long_slope_len"] = int(round(out.get("long_slope_len", 15) + rng.gauss(0, 2)))

    if rng.random() < 0.4:
        out["holding_period_limit"] = int(round(out.get("holding_period_limit", 0) + rng.gauss(0, 10)))

    # ---- floats in log-space (scale-aware) or bounded Gaussian ----
    if rng.random() < 0.6:
        out["atr_multiple"] = _log_mut(out.get("atr_multiple", 3.0), b.atr_multiple_min, b.atr_multiple_max, sigma=0.18)

    if rng.random() < 0.6:
        out["risk_per_trade"] = _log_mut(out.get("risk_per_trade", 0.01), b.risk_per_trade_min, b.risk_per_trade_max, sigma=0.25)

    if rng.random() < 0.6:
        out["tp_multiple"] = _clamp(out.get("tp_multiple", 0.0) + rng.gauss(0, 0.35), b.tp_multiple_min, b.tp_multiple_max)

    if rng.random() < 0.5:
        out["cost_bps"] = _clamp(out.get("cost_bps", 0.0) + rng.gauss(0, 0.3), b.cost_bps_min, b.cost_bps_max)

    # ---- rare boolean flip ----
    if b.allow_trend_filter and rng.random() < 0.2:
        out["use_trend_filter"] = not bool(out.get("use_trend_filter", False))

    return _fix(out, b)


def _xover(a: Dict, b_: Dict, rng: random.Random, bounds: Bounds) -> Dict:
    child = {}
    # integers: coin-flip inherit
    for k in ["breakout_n","exit_n","atr_n","sma_fast","sma_slow","sma_long","long_slope_len","holding_period_limit"]:
        child[k] = a[k] if rng.random() < 0.5 else b_[k]
    # floats: blend around parents with bounds
    child["atr_multiple"]   = _clamp(_blend(a["atr_multiple"],   b_["atr_multiple"],   alpha=0.2), bounds.atr_multiple_min, bounds.atr_multiple_max)
    child["risk_per_trade"] = _clamp(_blend(a["risk_per_trade"], b_["risk_per_trade"], alpha=0.2), bounds.risk_per_trade_min, bounds.risk_per_trade_max)
    child["tp_multiple"]    = _clamp(_blend(a["tp_multiple"],    b_["tp_multiple"],    alpha=0.2), bounds.tp_multiple_min, bounds.tp_multiple_max)
    child["cost_bps"]       = _clamp(_blend(a["cost_bps"],       b_["cost_bps"],       alpha=0.2), bounds.cost_bps_min, bounds.cost_bps_max)
    # bool
    child["use_trend_filter"] = a["use_trend_filter"] if rng.random() < 0.5 else b_["use_trend_filter"]
    return _fix(child, bounds)


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
                child = _xover(p1, p2, rng, bounds)
            if rng.random() < mutation_rate:
                child = _mutate(child, bounds, rng)
            child = _fix(child, bounds)
            next_pop.append(child)

        pop = next_pop

    return best_ind or {}, best_metrics, history