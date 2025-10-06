# src/optimization/evolutionary.py
"""
Evolutionary parameter search with progress reporting + logging.

New in this revision:
- Parallel evaluation with ProcessPoolExecutor (n_jobs)
- Swing hard gate: min_avg_holding_days_gate (score=0 if avg_hold < gate)
- Trade-rate band penalty:
    trades_per_symbol_per_year = total_trades / num_symbols / years
    soft penalty outside [trade_rate_min, trade_rate_max]
- Fitness:
        base = α*CAGR + β*Calmar + γ*Sharpe + δ*TotalReturn
        Calmar is clamped to ±calmar_cap before weighting to avoid runaway ratios.
    hold_pen = λ_hold * normalized_penalty(avg_holding_days outside [min_hold, max_hold])
    rate_pen = λ_rate * normalized_penalty(trade_rate outside [rate_min, rate_max])
    score = base - min(hold_pen + rate_pen, penalty_cap)
- All knobs exposed as function args (UI-ready)
"""

from __future__ import annotations
import random
import time
import importlib
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple, Optional, Literal
from datetime import date, datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.models.general_trainer import train_general_model
from src.utils.training_logger import TrainingLogger
from src.utils.progress import ProgressCallback, console_progress

import json
import os

import sys
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[2])  # project root
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ---- Fitness config (JSON) ----------------------------------------------

def _load_fitness_config_json(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load fitness weight configuration from JSON. Returns {} if file missing or malformed.
    Default location: storage/config/ea_fitness.json
    Recognized keys:
      - alpha_cagr, beta_calmar, gamma_sharpe, delta_total_return (floats)
      - calmar_cap (float)
      - use_normalized_scoring (bool)
      - holding_penalty_weight, trade_rate_penalty_weight, penalty_cap (floats)
      - min_holding_days, max_holding_days, trade_rate_min, trade_rate_max (floats)
      - rate_penalize_upper (bool), elite_by_return_frac (float)
    """
    try:
        p = Path(path or "storage/config/ea_fitness.json")
        if not p.exists():
            return {}
        data = json.loads(p.read_text())
        if not isinstance(data, dict):
            return {}
        out: Dict[str, Any] = {}
        # Copy only known keys with safe coercion
        def _getf(k: str):
            try:
                return float(data[k])
            except Exception:
                return None
        if "alpha_cagr" in data: out["alpha_cagr"] = _getf("alpha_cagr")
        if "beta_calmar" in data: out["beta_calmar"] = _getf("beta_calmar")
        if "gamma_sharpe" in data: out["gamma_sharpe"] = _getf("gamma_sharpe")
        if "delta_total_return" in data: out["delta_total_return"] = _getf("delta_total_return")
        if "calmar_cap" in data: out["calmar_cap"] = _getf("calmar_cap")
        if "use_normalized_scoring" in data:
            v = data["use_normalized_scoring"]
            out["use_normalized_scoring"] = bool(v) if isinstance(v, bool) else str(v).strip().lower() in {"1","true","yes","on"}
        if "holding_penalty_weight" in data: out["holding_penalty_weight"] = _getf("holding_penalty_weight")
        if "trade_rate_penalty_weight" in data: out["trade_rate_penalty_weight"] = _getf("trade_rate_penalty_weight")
        if "penalty_cap" in data: out["penalty_cap"] = _getf("penalty_cap")
        if "min_holding_days" in data: out["min_holding_days"] = _getf("min_holding_days")
        if "max_holding_days" in data: out["max_holding_days"] = _getf("max_holding_days")
        if "trade_rate_min" in data: out["trade_rate_min"] = _getf("trade_rate_min")
        if "trade_rate_max" in data: out["trade_rate_max"] = _getf("trade_rate_max")
        # new optional keys
        if "elite_by_return_frac" in data: out["elite_by_return_frac"] = _getf("elite_by_return_frac")
        if "holdout_score_weight" in data: out["holdout_score_weight"] = _getf("holdout_score_weight")
        if "holdout_gap_tolerance" in data: out["holdout_gap_tolerance"] = _getf("holdout_gap_tolerance")
        if "holdout_gap_penalty" in data: out["holdout_gap_penalty"] = _getf("holdout_gap_penalty")
        if "rate_penalize_upper" in data:
            v = data["rate_penalize_upper"]
            out["rate_penalize_upper"] = bool(v) if isinstance(v, bool) else str(v).strip().lower() in {"1","true","yes","on"}
        # Drop any None-coerced floats
        for k in [
            "alpha_cagr","beta_calmar","gamma_sharpe","delta_total_return","calmar_cap",
            "holding_penalty_weight","trade_rate_penalty_weight","penalty_cap",
            "min_holding_days","max_holding_days","trade_rate_min","trade_rate_max",
            "elite_by_return_frac","holdout_score_weight","holdout_gap_tolerance","holdout_gap_penalty",
        ]:
            if k in out and out[k] is None:
                out.pop(k, None)
        return out
    except Exception:
        return {}

import multiprocessing as mp
try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass

# Worker-safe progress alias and default noop sink
ProgressCb = ProgressCallback


def _noop_progress(event: str, payload: Dict[str, Any]) -> None:
    return


# --- EA configuration ------------------------------------------------------


@dataclass
class EAConfig:
    """Typed configuration for evolutionary search (new-style knobs)."""

    pop_size: int = 64
    generations: int = 50
    selection_method: Literal["tournament", "rank", "roulette"] = "tournament"
    tournament_k: int = 3
    replacement: Literal["generational", "mu+lambda"] = "mu+lambda"
    elitism_fraction: float = 0.05
    crossover_rate: float = 0.85
    crossover_op: Literal["blend", "sbx", "one_point"] = "blend"
    mutation_rate: float = 0.10
    mutation_scale: float = 0.20
    mutation_scheme: Literal["gaussian", "polynomial", "uniform_reset"] = "gaussian"
    genewise_clip: bool = True
    anneal_mutation: bool = True
    anneal_floor: float = 0.05
    fitness_patience: int = 8
    no_improve_tol: Optional[float] = None
    seed: Optional[int] = None
    workers: Optional[int] = None
    shuffle_eval: bool = True

    def elite_count(self) -> int:
        return max(1, int(max(1, self.pop_size) * max(0.0, float(self.elitism_fraction))))

    def mutation_scale_for_gen(self, gen: int, total_gens: int) -> float:
        if not self.anneal_mutation or total_gens <= 1:
            return float(self.mutation_scale)
        start = float(self.mutation_scale)
        floor = max(0.0, float(self.anneal_floor))
        span = max(0.0, start - floor)
        frac = min(1.0, max(0.0, float(gen) / float(max(1, total_gens - 1))))
        return max(floor, start - span * frac)

    def resolved_tournament_k(self) -> int:
        return max(2, int(self.tournament_k))

    def resolved_no_improve_tol(self) -> float:
        if self.no_improve_tol is None:
            return 0.0
        return max(0.0, float(self.no_improve_tol))

    def to_log_payload(self) -> Dict[str, Any]:
        data = asdict(self)
        data["elite_count"] = self.elite_count()
        data["mutation_scale_initial"] = float(self.mutation_scale)
        data["mutation_scale_floor"] = self.mutation_scale_for_gen(
            gen=max(0, self.generations - 1), total_gens=max(1, self.generations)
        )
        return data


# Impact Map: EAConfig introduces optional new knobs; guarded usage keeps legacy
# code paths intact when `config` is omitted. Rollback: drop EAConfig usage and
# remove the conditional branch in `evolutionary_search`, reverting to legacy
# parameters-only behaviour.


# ----------------------------- Sampling ops ------------------------------

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


def _legacy_mutate(params: Dict[str, Any], param_space: Dict[str, Tuple], rate: float = 0.3) -> Dict[str, Any]:
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


def _legacy_crossover(p1: Dict[str, Any], p2: Dict[str, Any]) -> Dict[str, Any]:
    """Uniform crossover."""
    return {k: (p1[k] if random.random() < 0.5 else p2[k]) for k in p1}


# --- Config-aware operators -----------------------------------------------


def _coerce_ea_config(config: Optional[Any]) -> Optional[EAConfig]:
    if config is None:
        return None
    if isinstance(config, EAConfig):
        return config
    if isinstance(config, dict):
        allowed = {k: config[k] for k in EAConfig.__dataclass_fields__ if k in config}
        return EAConfig(**allowed)
    raise TypeError(f"Unsupported config type: {type(config)!r}")


def _select_parent(
    scored: List[Tuple[Dict[str, Any], float, float]],
    method: Literal["tournament", "rank", "roulette"],
    tournament_k: int,
) -> Dict[str, Any]:
    if not scored:
        raise ValueError("Parent selection requires non-empty scored population")
    if method == "tournament":
        k = min(max(2, tournament_k), len(scored))
        sample = random.sample(scored, k)
        sample.sort(key=lambda x: x[1], reverse=True)
        return dict(sample[0][0])
    if method == "rank":
        weights = list(range(len(scored), 0, -1))
        total = sum(weights)
        pick = random.uniform(0, total)
        upto = 0.0
        for idx, weight in enumerate(weights):
            upto += weight
            if upto >= pick:
                return dict(scored[idx][0])
        return dict(scored[0][0])
    # roulette
    scores = [max(0.0, float(rec[1])) for rec in scored]
    total = sum(scores)
    if total <= 0.0:
        return dict(random.choice(scored)[0])
    pick = random.uniform(0, total)
    upto = 0.0
    for idx, weight in enumerate(scores):
        upto += weight
        if upto >= pick:
            return dict(scored[idx][0])
    return dict(scored[-1][0])


def _crossover_configured(
    p1: Dict[str, Any],
    p2: Dict[str, Any],
    param_space: Dict[str, Tuple],
    cfg: EAConfig,
) -> Dict[str, Any]:
    if random.random() > float(cfg.crossover_rate):
        return dict(p1 if random.random() < 0.5 else p2)

    op = cfg.crossover_op
    keys = list(p1.keys())
    child: Dict[str, Any] = {}
    if op == "one_point":
        if not keys:
            return dict(p1)
        pivot = random.randint(1, len(keys))
        for idx, key in enumerate(keys):
            child[key] = p1[key] if idx < pivot else p2[key]
        return child

    for key in keys:
        v1 = p1[key]
        v2 = p2[key]
        bounds = param_space.get(key)
        is_float = isinstance(v1, float) or isinstance(v2, float) or (
            bounds and (isinstance(bounds[0], float) or isinstance(bounds[1], float))
        )
        if not bounds:
            bounds = (v1, v2)
        lo, hi = bounds
        if op == "blend":
            if is_float:
                lo_f = float(min(v1, v2))
                hi_f = float(max(v1, v2))
                if lo_f == hi_f:
                    child[key] = float(lo_f)
                else:
                    child[key] = float(random.uniform(lo_f, hi_f))
            else:
                child[key] = random.choice([v1, v2])
        elif op == "sbx":
            # Simulated binary crossover (eta=2 approximation)
            if is_float:
                v1f = float(v1)
                v2f = float(v2)
                if v1f == v2f:
                    child[key] = v1f
                else:
                    u = random.random()
                    eta = 2.0
                    span = max(1e-9, abs(v2f - v1f))
                    beta = 1.0 + (2.0 * min(v1f, v2f) - float(lo)) / span
                    beta_prime = 1.0 + (2.0 * float(hi) - 2.0 * max(v1f, v2f)) / span
                    # Guard against negative crossover coefficients which would yield complex values
                    beta = max(1e-9, beta)
                    beta_prime = max(1e-9, beta_prime)
                    if u <= 0.5:
                        beta_q = max(1e-9, u * beta) ** (1.0 / (eta + 1.0))
                    else:
                        denom = max(1e-9, 2.0 - u * beta_prime)
                        beta_q = (1.0 / denom) ** (1.0 / (eta + 1.0))
                    child_val = 0.5 * ((1 + beta_q) * v1f + (1 - beta_q) * v2f)
                    child[key] = float(child_val)
            else:
                child[key] = random.choice([v1, v2])
        else:
            # fallback to blend semantics for unsupported op types
            child[key] = random.choice([v1, v2]) if not is_float else float(
                random.uniform(float(min(v1, v2)), float(max(v1, v2)))
            )
    return child


def _mutate_configured(
    params: Dict[str, Any],
    param_space: Dict[str, Tuple],
    cfg: EAConfig,
    mutation_scale: float,
) -> Dict[str, Any]:
    new = dict(params)
    for key, bounds in param_space.items():
        if random.random() >= float(cfg.mutation_rate):
            continue
        low, high = bounds
        is_float = isinstance(low, float) or isinstance(high, float) or isinstance(new.get(key), float)
        span = float(high) - float(low)
        if cfg.mutation_scheme == "uniform_reset":
            if is_float:
                new_val = random.uniform(float(low), float(high))
            else:
                new_val = random.randint(int(low), int(high))
        elif cfg.mutation_scheme == "polynomial":
            if is_float:
                eta = 20.0
                u = random.random()
                if u < 0.5:
                    delta = (2 * u) ** (1 / (eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                new_val = float(new.get(key, low)) + delta * span * mutation_scale
            else:
                delta = random.randint(-max(1, int(span * mutation_scale)), max(1, int(span * mutation_scale)))
                new_val = int(new.get(key, low)) + delta
        else:  # gaussian default
            if is_float:
                sigma = max(1e-9, span * mutation_scale)
                new_val = float(new.get(key, low)) + random.gauss(0.0, sigma)
            else:
                step = max(1, int(round(span * mutation_scale)))
                if step <= 0:
                    step = 1
                new_val = int(new.get(key, low)) + random.randint(-step, step)

        if cfg.genewise_clip:
            if is_float:
                new_val = float(min(float(high), max(float(low), float(new_val))))
            else:
                new_val = int(min(int(high), max(int(low), int(new_val))))
        new[key] = new_val
    return new

# ------------------------------ Penalties --------------------------------

def _holding_penalty(avg_hold_days: float, lo: float, hi: float) -> float:
    """Linear penalty outside [lo, hi]. Zero inside the band."""
    if lo <= avg_hold_days <= hi:
        return 0.0
    if avg_hold_days < lo:
        return float(lo - avg_hold_days)
    return float(avg_hold_days - hi)


def _rate_penalty(trade_rate: float, lo: float, hi: float, penalize_upper: bool = True) -> float:
    """Linear penalty outside [lo, hi] for trades per symbol per year.
    If penalize_upper is False, only penalize values below `lo`."""
    if lo <= trade_rate <= hi:
        return 0.0
    if trade_rate < lo:
        return float(lo - trade_rate)
    return float(trade_rate - hi) if penalize_upper else 0.0


# ------------------------------ Fitness ----------------------------------

def _clamped_fitness(
    metrics: Dict[str, Any],
    *,
    # gates & clamps
    min_trades: int,
    min_avg_holding_days_gate: float,
    require_hold_days: bool,
    eps_mdd: float,
    eps_sharpe: float,
    # weights for core terms
    alpha_cagr: float,
    beta_calmar: float,
    gamma_sharpe: float,
    delta_total_return: float,
    # holding window preference
    min_holding_days: float,
    max_holding_days: float,
    holding_penalty_weight: float,
    # trade rate preference
    trade_rate_min: float,
    trade_rate_max: float,
    trade_rate_penalty_weight: float,
    # context for rate
    num_symbols: int,
    years: float,
    calmar_cap: float,
    penalty_cap: float = 0.50,
    rate_penalize_upper: bool = True,
    use_normalized_scoring: bool = True,
) -> float:
    """
    Robust fitness:
        base = α*CAGR + β*Calmar + γ*Sharpe
        hold_pen = λ_hold * penalty(avg_holding_days in [min_hold, max_hold])
        rate_pen = λ_rate * penalty(trade_rate in [rate_min, rate_max])
        score = base - hold_pen - rate_pen
    Safety:
        - trades < min_trades => 0
        - avg_hold < min_avg_holding_days_gate => 0
        - require_hold_days and avg_hold <= 0 => 0
        - |MDD| < eps_mdd => Calmar := 0
        - |Sharpe| < eps_sharpe => Sharpe := 0
    """
    trades = int(metrics.get("trades", 0) or 0)
    if trades < min_trades:
        return 0.0

    avg_hold = float(metrics.get("avg_holding_days", 0.0) or 0.0)
    if avg_hold < float(min_avg_holding_days_gate):
        return 0.0
    if require_hold_days and avg_hold <= 0.0:
        return 0.0

    cagr = float(metrics.get("cagr", 0.0) or 0.0)

    calmar = float(metrics.get("calmar", 0.0) or 0.0)
    mdd = float(abs(metrics.get("max_drawdown", 0.0) or 0.0))
    if mdd < eps_mdd:
        calmar = 0.0

    sharpe = float(metrics.get("sharpe", 0.0) or 0.0)
    if abs(sharpe) < eps_sharpe:
        sharpe = 0.0

    total_return = float(metrics.get("total_return", 0.0) or 0.0)

    if calmar_cap > 0:
        calmar = max(-calmar_cap, min(calmar, calmar_cap))

    # ---- Scoring (legacy vs normalized) ---------------------------------
    if not use_normalized_scoring:
        # Legacy behavior: raw weighted sum with Calmar clamped
        base = (
            (alpha_cagr * cagr)
            + (beta_calmar * calmar)
            + (gamma_sharpe * sharpe)
            + (delta_total_return * total_return)
        )
    else:
        # Normalized scoring: map metrics to comparable 0..1 ranges to avoid any
        # single term (e.g., Calmar/Sharpe) overwhelming absolute growth metrics.
        def _clip01(x: float) -> float:
            return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

        # Conservative caps; adjust later if you know your universe differs.
        # total_return: 0..50% → 0..1
        tr_n = _clip01((total_return - 0.0) / 0.50)
        # CAGR: 0..30% → 0..1
        cagr_n = _clip01((cagr - 0.0) / 0.30)
        # Sharpe: 0..3 → 0..1 (negatives clipped to 0)
        sh_n = _clip01((sharpe - 0.0) / 3.0)
        # Calmar: 0..calmar_cap → 0..1 (already clamped to ±cap above; use positive band)
        cm_n = _clip01(max(0.0, calmar) / max(1e-9, calmar_cap))

        base = (
            (alpha_cagr * cagr_n)
            + (beta_calmar * cm_n)
            + (gamma_sharpe * sh_n)
            + (delta_total_return * tr_n)
        )

    # ---- Normalized, capped penalties ----------------------------------
    _bw_hold = max(1e-9, (max_holding_days - min_holding_days))
    hold_dist = _holding_penalty(avg_hold, min_holding_days, max_holding_days)
    hold_pen  = holding_penalty_weight * (hold_dist / _bw_hold)

    trade_rate = 0.0
    if num_symbols > 0 and years > 0:
        trade_rate = float(trades) / float(num_symbols) / float(years)
    _bw_rate = max(1e-9, (trade_rate_max - trade_rate_min))
    rate_dist = _rate_penalty(trade_rate, trade_rate_min, trade_rate_max, rate_penalize_upper)
    rate_pen  = trade_rate_penalty_weight * (rate_dist / _bw_rate)

    penalty_total = min(hold_pen + rate_pen, float(penalty_cap))
    return float(base - penalty_total)


# --------------------------- Child evaluation ----------------------------

def _eval_one(
    strategy_dotted: str,
    tickers: List[str],
    start,
    end,
    starting_equity: float,
    params: Dict[str, Any],
    test_range: Optional[Tuple[Any, Any]] = None,
) -> Dict[str, Any]:
    """
    Pure function suitable for multiprocessing.
    Returns dict with metrics and elapsed seconds.
    """
    t1 = time.time()
    # prove which loader module is actually imported inside the worker process
    L = importlib.import_module("src.data.loader")
    _loader_file = getattr(L, "__file__", "<??>")
    res = train_general_model(strategy_dotted, tickers, start, end, starting_equity, params)
    metrics = res.get("aggregate", {}).get("metrics", {}) or {}

    test_metrics: Optional[Dict[str, Any]] = None
    test_error: Optional[str] = None
    if test_range and test_range[0] is not None and test_range[1] is not None:
        try:
            test_start, test_end = test_range
            test_res = train_general_model(
                strategy_dotted,
                tickers,
                test_start,
                test_end,
                starting_equity,
                params,
            )
            test_metrics = test_res.get("aggregate", {}).get("metrics", {}) or {}
        except Exception as exc:
            test_metrics = None
            test_error = f"{type(exc).__name__}: {exc}"

    payload: Dict[str, Any] = {
        "metrics": metrics,
        "elapsed_sec": time.time() - t1,
        "params": params,
        "loader_file": _loader_file,
    }
    if test_metrics is not None:
        payload["test_metrics"] = test_metrics
    if test_error is not None:
        payload["test_error"] = test_error
    return payload


# ----------------------------- Main EA loop ------------------------------

def evolutionary_search(
    strategy_dotted: str,
    tickers: List[str],
    start,
    end,
    starting_equity: float,
    param_space: Dict[str, Tuple],
    *,
    test_start: Optional[Any] = None,
    test_end: Optional[Any] = None,
    generations: int = 10,
    pop_size: int = 20,
    # EA knobs
    mutation_rate: float = 0.3,
    elite_frac: float = 0.5,            # keep top 50%
    random_inject_frac: float = 0.2,    # inject 20% fresh randoms each gen
    # Parallelism
    n_jobs: int = 1,                    # 1 = single process; >1 uses ProcessPoolExecutor
    # Fitness gates & clamps
    min_trades: int = 5,
    min_avg_holding_days_gate: float = 1.0,
    require_hold_days: bool = False,
    eps_mdd: float = 1e-4,
    eps_sharpe: float = 1e-4,
    # Fitness weights (growth vs risk)
    alpha_cagr: float = 1.0,
    beta_calmar: float = 0.2,
    gamma_sharpe: float = 0.25,
    delta_total_return: float = 1.0,
    # Holding window preference (avoid day-trading & buy/hold)
    min_holding_days: float = 3.0,
    max_holding_days: float = 30.0,
    holding_penalty_weight: float = 1.0,
    # Trade-rate preference (per symbol per year)
    trade_rate_min: float = 5.0,
    trade_rate_max: float = 50.0,
    trade_rate_penalty_weight: float = 0.5,
    calmar_cap: float = 3.0,
    penalty_cap: float = 0.50,
    rate_penalize_upper: bool = True,
    elite_by_return_frac: float = 0.10,
    use_normalized_scoring: bool = True,
    # Progress & logging
    progress_cb: Optional[ProgressCb] = None,
    log_file: str = "training.log",
    # Reproducibility
    seed: Optional[int] = None,
    # New-style config
    config: Optional[Any] = None,
) -> List[Tuple[Dict[str, Any], float]]:
    """
    Run EA and return top parameter sets [(params, score)].

    Notes:
    - When `config` is provided, the new-style EAConfig knobs drive mutation, crossover, and replacement.
      Legacy kwargs remain in effect when `config` is None to maintain backward compatibility.
    - Survivor selection indirectly enforces `min_trades` and min_avg_holding_days_gate via fitness gating.
    - Diversity maintained via `random_inject_frac`.
    - Multiprocessing evaluates individuals in parallel per generation.
    - If `test_start`/`test_end` are supplied, scoring and gating use the holdout metrics from that range,
      while training metrics remain available for logging.
    """
    resolved_cb: Optional[ProgressCb] = progress_cb or console_progress
    if resolved_cb is None:
        resolved_cb = _noop_progress
    progress_cb = resolved_cb

    holdout_weight = 0.65
    holdout_gap_tolerance = 0.15
    holdout_gap_penalty = 0.50
    holdout_shortfall_penalty = 0.35

    cfg = _coerce_ea_config(config)

    if cfg and cfg.seed is not None:
        random.seed(cfg.seed)
    elif seed is not None:
        random.seed(seed)

    if cfg and cfg.workers is not None:
        try:
            n_jobs = int(cfg.workers)
        except Exception:
            n_jobs = n_jobs

    generations_total = int(cfg.generations if cfg else generations)
    if generations_total <= 0:
        generations_total = max(1, generations)
    pop_size_effective = int(cfg.pop_size if cfg else pop_size)
    if pop_size_effective <= 0:
        pop_size_effective = max(2, pop_size)

    mutation_rate_effective = float(cfg.mutation_rate if cfg else mutation_rate)
    if cfg:
        cfg.pop_size = pop_size_effective
        cfg.generations = generations_total

    test_range: Optional[Tuple[Any, Any]] = None
    if test_start is not None and test_end is not None:
        test_range = (test_start, test_end)

    logger = TrainingLogger(log_file)
    population = [random_param(param_space) for _ in range(pop_size_effective)]
    scored: List[Tuple[Dict[str, Any], float]] = []
    t0 = time.time()
    best_run_return_rec: Optional[Tuple[Dict[str, Any], float, float]] = None  # (params, total_return, score)

    # Load fitness weights from JSON config (single source of truth)
    _cfg = _load_fitness_config_json(None)
    if _cfg:
        alpha_cagr = _cfg.get("alpha_cagr", alpha_cagr)
        beta_calmar = _cfg.get("beta_calmar", beta_calmar)
        gamma_sharpe = _cfg.get("gamma_sharpe", gamma_sharpe)
        delta_total_return = _cfg.get("delta_total_return", delta_total_return)
        calmar_cap = _cfg.get("calmar_cap", calmar_cap)
        use_normalized_scoring = _cfg.get("use_normalized_scoring", use_normalized_scoring)
        holding_penalty_weight = _cfg.get("holding_penalty_weight", holding_penalty_weight)
        trade_rate_penalty_weight = _cfg.get("trade_rate_penalty_weight", trade_rate_penalty_weight)
        penalty_cap = _cfg.get("penalty_cap", penalty_cap)
        min_holding_days = _cfg.get("min_holding_days", min_holding_days)
        max_holding_days = _cfg.get("max_holding_days", max_holding_days)
        trade_rate_min = _cfg.get("trade_rate_min", trade_rate_min)
        trade_rate_max = _cfg.get("trade_rate_max", trade_rate_max)
        rate_penalize_upper = _cfg.get("rate_penalize_upper", rate_penalize_upper)
        elite_by_return_frac = _cfg.get("elite_by_return_frac", elite_by_return_frac)
        holdout_weight = _cfg.get("holdout_score_weight", holdout_weight)
        holdout_gap_tolerance = _cfg.get("holdout_gap_tolerance", holdout_gap_tolerance)
        holdout_gap_penalty = _cfg.get("holdout_gap_penalty", holdout_gap_penalty)
        holdout_shortfall_penalty = _cfg.get(
            "holdout_shortfall_penalty", holdout_shortfall_penalty
        )
        # Emit a one-time breadcrumb so logs show which source set the weights
        logger.log("fitness_config", {
            "source": "storage/config/ea_fitness.json",
            "fitness_weights": {
                "alpha_cagr": alpha_cagr,
                "beta_calmar": beta_calmar,
                "gamma_sharpe": gamma_sharpe,
                "delta_total_return": delta_total_return,
                "calmar_cap": calmar_cap,
                "use_normalized_scoring": use_normalized_scoring,
                "holding_penalty_weight": holding_penalty_weight,
                "trade_rate_penalty_weight": trade_rate_penalty_weight,
                "penalty_cap": penalty_cap,
                "min_holding_days": min_holding_days,
                "max_holding_days": max_holding_days,
                "trade_rate_min": trade_rate_min,
                "trade_rate_max": trade_rate_max,
                "rate_penalize_upper": rate_penalize_upper,
                "elite_by_return_frac": elite_by_return_frac,
                "holdout_shortfall_penalty": holdout_shortfall_penalty,
            },
        })

    try:
        holdout_weight = max(0.0, min(1.0, float(holdout_weight)))
    except Exception:
        holdout_weight = 0.65
    try:
        holdout_gap_tolerance = max(0.0, float(holdout_gap_tolerance))
    except Exception:
        holdout_gap_tolerance = 0.15
    try:
        holdout_gap_penalty = max(0.0, float(holdout_gap_penalty))
    except Exception:
        holdout_gap_penalty = 0.50
    try:
        holdout_shortfall_penalty = max(0.0, float(holdout_shortfall_penalty))
    except Exception:
        holdout_shortfall_penalty = 0.35

    # One-time session metadata for inspector tooling
    session_meta_payload = {
        "strategy": strategy_dotted,
        "tickers": list(tickers) if isinstance(tickers, (list, tuple)) else [str(tickers)],
        "starting_equity": float(starting_equity),
        "train_start": str(start),
        "train_end": str(end),
    }
    if test_range:
        session_meta_payload["test_start"] = str(test_range[0])
        session_meta_payload["test_end"] = str(test_range[1])
    logger.log("session_meta", session_meta_payload)
    # years used for trade-rate normalization
    def _years(a, b) -> float:
        # support date/datetime/str
        def _to_dt(x):
            if isinstance(x, (date, datetime)):
                return datetime(x.year, x.month, x.day)
            return datetime.fromisoformat(str(x))
        days = (_to_dt(end) - _to_dt(start)).days
        return max(1e-9, days / 365.25)

    years = _years(start, end)
    num_symbols = max(1, len(tickers))

    def _score_metrics_block(metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        data: Dict[str, Any] = metrics or {}
        trades = int(data.get("trades", 0) or 0)
        avg_hold = float(data.get("avg_holding_days", 0.0) or 0.0)
        gated_zero = (
            trades < min_trades
            or avg_hold < float(min_avg_holding_days_gate)
            or (require_hold_days and avg_hold <= 0.0)
        )

        trade_rate = 0.0
        if num_symbols > 0 and years > 0:
            trade_rate = float(trades) / float(num_symbols) / float(years)

        base_norm = 0.0
        hold_dist = 0.0
        rate_dist = 0.0
        hold_pen = 0.0
        rate_pen = 0.0
        pen_raw = 0.0
        pen_cap = 0.0
        total_return = float(data.get("total_return", 0.0) or 0.0)

        if gated_zero:
            score = 0.0
            fitness_dbg = {
                "gated_zero": True,
                "base_norm": 0.0,
                "trade_rate": trade_rate,
                "hold_dist": 0.0,
                "rate_dist": 0.0,
                "hold_pen": 0.0,
                "rate_pen": 0.0,
                "penalty_total_raw": 0.0,
                "penalty_total_capped": 0.0,
            }
        else:
            cagr = float(data.get("cagr", 0.0) or 0.0)
            calmar = float(data.get("calmar", 0.0) or 0.0)
            mdd = float(abs(data.get("max_drawdown", 0.0) or 0.0))
            if mdd < eps_mdd:
                calmar = 0.0
            sharpe = float(data.get("sharpe", 0.0) or 0.0)
            if abs(sharpe) < eps_sharpe:
                sharpe = 0.0

            def _clip01(x: float) -> float:
                if x < 0.0:
                    return 0.0
                if x > 1.0:
                    return 1.0
                return x

            tr_n = _clip01(total_return / 0.50)
            cagr_n = _clip01(cagr / 0.30)
            sh_n = _clip01(sharpe / 3.0)
            if calmar_cap > 0:
                calmar = max(-calmar_cap, min(calmar, calmar_cap))
            cm_n = _clip01(max(0.0, calmar) / max(1e-9, calmar_cap))

            base_norm = (
                (alpha_cagr * cagr_n)
                + (beta_calmar * cm_n)
                + (gamma_sharpe * sh_n)
                + (delta_total_return * tr_n)
            )

            _bw_hold = max(1e-9, (max_holding_days - min_holding_days))
            if not (min_holding_days <= avg_hold <= max_holding_days):
                hold_dist = (min_holding_days - avg_hold) if avg_hold < min_holding_days else (avg_hold - max_holding_days)
            hold_pen = holding_penalty_weight * (hold_dist / _bw_hold)

            _bw_rate = max(1e-9, (trade_rate_max - trade_rate_min))
            if trade_rate < trade_rate_min:
                rate_dist = (trade_rate_min - trade_rate)
            elif rate_penalize_upper and trade_rate > trade_rate_max:
                rate_dist = (trade_rate - trade_rate_max)
            else:
                rate_dist = 0.0
            rate_pen = trade_rate_penalty_weight * (rate_dist / _bw_rate)

            pen_raw = hold_pen + rate_pen
            pen_cap = min(pen_raw, float(penalty_cap))

            if use_normalized_scoring:
                score = float(base_norm - pen_cap)
            else:
                score = float(
                    (alpha_cagr * cagr)
                    + (beta_calmar * calmar)
                    + (gamma_sharpe * sharpe)
                    + (delta_total_return * total_return)
                    - pen_cap
                )

            fitness_dbg = {
                "gated_zero": False,
                "base_norm": base_norm,
                "trade_rate": trade_rate,
                "hold_dist": hold_dist,
                "rate_dist": rate_dist,
                "hold_pen": hold_pen,
                "rate_pen": rate_pen,
                "penalty_total_raw": pen_raw,
                "penalty_total_capped": pen_cap,
            }

        return {
            "score": float(score),
            "debug": fitness_dbg,
            "ret": total_return,
            "trades": trades,
            "avg_hold": avg_hold,
            "metrics": data,
        }

    best_score_seen: Optional[float] = None
    stale_generations = 0
    stopped_early = False
    last_gen = -1

    for gen in range(generations_total):
        last_gen = gen
        progress_cb("generation_start", {"gen": gen, "pop_size": len(population)})
        logger.log("generation_start", {"gen": gen, "pop_size": len(population)})

        if cfg and gen == 0:
            logger.log("ea_config", cfg.to_log_payload())

        gen_scores: List[Tuple[Dict[str, Any], float, float]] = []
        gen_trades: List[int] = []
        no_trade_count = 0
        best_gen_return_rec: Optional[Tuple[Dict[str, Any], float, float]] = None  # (params, total_return, score)

        # Per-generation penalty/gating trackers (for telemetry)
        gen_penalties_capped: List[float] = []
        gen_cap_hits: int = 0
        gen_gated_zeros: int = 0

        # ---- Evaluate population (parallel or single) ----
        results: List[Dict[str, Any]] = []

        eval_population = list(population)
        if cfg and cfg.shuffle_eval:
            random.shuffle(eval_population)

        if n_jobs and n_jobs > 1:
            with ProcessPoolExecutor(max_workers=n_jobs) as ex:
                futures = [
                    ex.submit(
                        _eval_one,
                        strategy_dotted,
                        tickers,
                        start,
                        end,
                        starting_equity,
                        params,
                        test_range,
                    )
                    for params in eval_population
                ]
                for i, fut in enumerate(as_completed(futures)):
                    try:
                        out = fut.result()
                    except Exception as e:
                        # record as an error entry; continue
                        ctx = {"gen": gen, "idx": i, "params": population[i] if i < len(population) else {}}
                        logger.log_error(ctx, e)
                        out = {"metrics": {}, "elapsed_sec": 0.0, "params": ctx.get("params", {})}
                    results.append(out)
        else:
            for params in eval_population:
                try:
                    out = _eval_one(
                        strategy_dotted,
                        tickers,
                        start,
                        end,
                        starting_equity,
                        params,
                        test_range,
                    )
                except Exception as e:
                    logger.log_error({"gen": gen, "params": params}, e)
                    out = {"metrics": {}, "elapsed_sec": 0.0, "params": params}
                results.append(out)

        # Emit per-individual logs/telemetry in deterministic order of the current population
        # (for nice, predictable UI), matching index by params identity.
        for i, params in enumerate(population):
            # find first matching result for these params (simple linear scan; pop to avoid duplicates)
            hit_idx = None
            for j, r in enumerate(results):
                if r is not None and r.get("params") == params:
                    hit_idx = j
                    break
            if hit_idx is None:
                metrics_train: Dict[str, Any] = {}
                metrics_test: Dict[str, Any] = {}
                elapsed = 0.0
                loader_file = None
                test_error = None
            else:
                rec = results.pop(hit_idx)
                metrics_train = rec.get("metrics", {}) or {}
                metrics_test = rec.get("test_metrics") or {}
                if not isinstance(metrics_test, dict):
                    metrics_test = {}
                elapsed = rec.get("elapsed_sec", 0.0)
                loader_file = rec.get("loader_file")
                test_error = rec.get("test_error")

            train_eval = _score_metrics_block(metrics_train)
            test_eval = _score_metrics_block(metrics_test) if metrics_test else None

            if test_error:
                logger.log(
                    "holdout_eval_failed",
                    {
                        "gen": gen,
                        "idx": i,
                        "params": params,
                        "error": test_error,
                    },
                )

            score_metrics = (
                test_eval["metrics"] if test_eval is not None else train_eval["metrics"]
            )
            if not isinstance(score_metrics, dict):
                score_metrics = {}

            score = train_eval["score"]
            fitness_dbg = dict(train_eval["debug"])
            blend_info: Optional[Dict[str, Any]] = None

            if test_eval is not None:
                score = test_eval["score"]
                fitness_dbg = dict(test_eval["debug"])
                train_score = float(train_eval["score"])
                test_score = float(test_eval["score"])
                train_gated = bool(train_eval["debug"].get("gated_zero", False))
                test_gated = bool(test_eval["debug"].get("gated_zero", False))
                weight_train = max(0.0, min(1.0, 1.0 - holdout_weight))
                blend_info = {
                    "train_score": train_score,
                    "test_score": test_score,
                    "train_gated": train_gated,
                    "test_gated": test_gated,
                    "holdout_weight": holdout_weight,
                    "train_weight": weight_train,
                    "gap_tolerance": holdout_gap_tolerance,
                    "gap_penalty": holdout_gap_penalty,
                }

                if test_gated:
                    score = 0.0
                    fitness_dbg["gated_zero"] = True
                    blend_info["reason"] = "holdout_gated"
                else:
                    effective_train_score = 0.0 if train_gated else train_score
                    test_component = holdout_weight * test_score
                    train_bonus = weight_train * min(effective_train_score, test_score)
                    ratio = None
                    if effective_train_score > 1e-9:
                        ratio = test_score / effective_train_score
                    gap = max(0.0, effective_train_score - test_score - holdout_gap_tolerance)
                    penalty = holdout_gap_penalty * gap
                    train_shortfall = max(
                        0.0, test_score - effective_train_score - holdout_gap_tolerance
                    )
                    shortfall_penalty = holdout_shortfall_penalty * train_shortfall
                    score = max(0.0, test_component + train_bonus - penalty - shortfall_penalty)
                    fitness_dbg.update(
                        {
                            "holdout_blended": True,
                            "train_score": train_score,
                            "train_component": effective_train_score,
                            "train_gated": train_gated,
                            "holdout_weight": holdout_weight,
                            "train_bonus": train_bonus,
                            "holdout_component": test_component,
                            "blend_penalty": penalty,
                            "gap_excess": gap,
                            "train_shortfall": train_shortfall,
                            "shortfall_penalty": shortfall_penalty,
                            "blended_score": score,
                        }
                    )
                    if ratio is not None:
                        fitness_dbg["train_to_test_ratio"] = ratio
                    blend_info.update(
                        {
                            "train_bonus": train_bonus,
                            "holdout_component": test_component,
                            "gap_excess": gap,
                            "penalty": penalty,
                            "train_shortfall": train_shortfall,
                            "shortfall_penalty": shortfall_penalty,
                            "blended_score": score,
                        }
                    )
                    if ratio is not None:
                        blend_info["train_to_test_ratio"] = ratio

            # --- accumulate per-gen telemetry ---------------------------------
            if fitness_dbg.get("gated_zero", False):
                gen_gated_zeros += 1
            else:
                pen_cap_val = float(fitness_dbg.get("penalty_total_capped", 0.0) or 0.0)
                gen_penalties_capped.append(pen_cap_val)
                if pen_cap_val >= float(penalty_cap) - 1e-12:
                    gen_cap_hits += 1

            trades = int(score_metrics.get("trades", 0) or 0)
            gen_trades.append(trades)

            ret = float(score_metrics.get("total_return", 0.0) or 0.0)

            # Track best-by-return for this generation
            if (best_gen_return_rec is None) or (ret > best_gen_return_rec[1]):
                best_gen_return_rec = (params, ret, score)
            # Track best-by-return across the whole run
            if (best_run_return_rec is None) or (ret > best_run_return_rec[1]):
                best_run_return_rec = (params, ret, score)

            payload = {
                "gen": gen,
                "idx": i,
                "params": params,
                "score": score,
                "metrics": score_metrics,
                "train_metrics": metrics_train,
                "elapsed_sec": elapsed,
                "loader_file": loader_file,
                "fitness_debug": fitness_dbg,
            }
            if test_range is not None:
                payload["test_metrics"] = metrics_test
            if blend_info is not None:
                payload["score_blend"] = blend_info
            progress_cb("individual_evaluated", payload)
            logger.log("individual_evaluated", payload)

            if trades < min_trades:
                logger.log("under_min_trades", payload)
                if trades == 0 and (score_metrics.get("calmar", 0) or 0) != 0:
                    logger.log("degenerate_fitness", payload)
                no_trade_count += (1 if trades == 0 else 0)

            gen_scores.append((params, score, ret))

        # ---- Selection & breeding ----
        # Sort primarily by score, then tie-break by total_return
        gen_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
        scored.extend([(p, s) for (p, s, _ret) in gen_scores])

        elite_n = max(1, int(pop_size_effective * elite_frac)) if not cfg else min(
            pop_size_effective, cfg.elite_count()
        )
        inject_n = max(0, int(pop_size_effective * random_inject_frac))
        breed_n = pop_size_effective - elite_n - inject_n
        if breed_n < 0:
            inject_n = max(0, pop_size_effective - elite_n)
            breed_n = pop_size_effective - elite_n - inject_n

        # Mixed elites: by score and by total_return to reduce oscillation
        elite_by_return_n = max(0, min(elite_n, int(round(elite_n * elite_by_return_frac))))
        elite_by_score_n = max(0, elite_n - elite_by_return_n)

        survivors: List[Dict[str, Any]] = []
        seen = set()

        # 1) take top by score (positive scores only for legacy path)
        for p, s, _r in gen_scores:
            if elite_by_score_n <= 0:
                break
            if not cfg and s <= 0:
                continue
            key = json.dumps(p, sort_keys=True)
            if key in seen:
                continue
            survivors.append(p)
            seen.add(key)
            elite_by_score_n -= 1

        # 2) take top by return (regardless of score) until quota met
        by_return = sorted(gen_scores, key=lambda x: x[2], reverse=True)
        for p, s, _r in by_return:
            if elite_by_return_n <= 0:
                break
            key = json.dumps(p, sort_keys=True)
            if key in seen:
                continue
            survivors.append(p)
            seen.add(key)
            elite_by_return_n -= 1

        # 3) pad with remaining best by score to reach elite_n
        for p, s, _r in gen_scores:
            if len(survivors) >= elite_n:
                break
            key = json.dumps(p, sort_keys=True)
            if key in seen:
                continue
            survivors.append(p)
            seen.add(key)

        children: List[Dict[str, Any]] = []
        if cfg:
            mutation_scale_gen = cfg.mutation_scale_for_gen(gen, generations_total)
            parent_scores = gen_scores if gen_scores else [(random_param(param_space), 0.0, 0.0)]
            for _ in range(max(0, breed_n)):
                try:
                    p1 = _select_parent(parent_scores, cfg.selection_method, cfg.resolved_tournament_k())
                    p2 = _select_parent(parent_scores, cfg.selection_method, cfg.resolved_tournament_k())
                except ValueError:
                    p1 = random_param(param_space)
                    p2 = random_param(param_space)
                child = _mutate_configured(
                    _crossover_configured(p1, p2, param_space, cfg),
                    param_space,
                    cfg,
                    mutation_scale_gen,
                )
                children.append(child)
        else:
            parents = survivors if len(survivors) >= 2 else [random_param(param_space) for _ in range(2)]
            for _ in range(max(0, breed_n)):
                p1 = random.choice(parents)
                p2 = random.choice(parents)
                child = _legacy_mutate(
                    _legacy_crossover(p1, p2), param_space, rate=mutation_rate_effective
                )
                children.append(child)

        injections = [random_param(param_space) for _ in range(max(0, inject_n))]

        if cfg and cfg.replacement == "mu+lambda":
            next_population: List[Dict[str, Any]] = list(survivors)
            mu_seen = {json.dumps(p, sort_keys=True) for p in next_population}

            def _try_add(candidate: Dict[str, Any]) -> bool:
                key = json.dumps(candidate, sort_keys=True)
                if key in mu_seen:
                    return False
                next_population.append(candidate)
                mu_seen.add(key)
                return True

            for child in children:
                if len(next_population) >= pop_size_effective:
                    break
                _try_add(child)

            for inj in injections:
                if len(next_population) >= pop_size_effective:
                    break
                _try_add(inj)

            if len(next_population) < pop_size_effective:
                for p, _s, _r in gen_scores:
                    if len(next_population) >= pop_size_effective:
                        break
                    _try_add(p)

            attempts = 0
            max_attempts = max(10 * pop_size_effective, 50)
            while len(next_population) < pop_size_effective and attempts < max_attempts:
                rnd = random_param(param_space)
                if _try_add(rnd):
                    attempts += 1
                    continue
                attempts += 1

            while len(next_population) < pop_size_effective:
                next_population.append(random_param(param_space))

            population = next_population[:pop_size_effective]
        else:
            population = survivors + children + injections
            while len(population) < pop_size_effective:
                population.append(random_param(param_space))
            if len(population) > pop_size_effective:
                population = population[:pop_size_effective]

        avg_score = (sum(s for _, s, _ in gen_scores) / len(gen_scores)) if gen_scores else 0.0
        best_score = gen_scores[0][1] if gen_scores else 0.0
        avg_trades = (sum(gen_trades) / len(gen_trades)) if gen_trades else 0.0
        pct_no_trades = (no_trade_count / len(gen_trades)) if gen_trades else 0.0

        # Compute per-generation penalty/gating telemetry
        _n = max(1, len(gen_scores))
        cap_hit_rate = float(gen_cap_hits) / float(_n)
        share_gated_zero = float(gen_gated_zeros) / float(_n)
        if gen_penalties_capped:
            # safe mean and p95
            _pen_mean = float(sum(gen_penalties_capped) / len(gen_penalties_capped))
            _pen_p95 = float(sorted(gen_penalties_capped)[int(0.95 * (len(gen_penalties_capped) - 1))])
        else:
            _pen_mean = 0.0
            _pen_p95 = 0.0

        should_stop = False
        if cfg:
            tol = cfg.resolved_no_improve_tol()
            if best_score_seen is None or best_score > (best_score_seen + tol):
                best_score_seen = best_score
                stale_generations = 0
            else:
                stale_generations += 1
                if cfg.fitness_patience > 0 and stale_generations >= cfg.fitness_patience:
                    should_stop = True

        end_payload = {
            "gen": gen,
            "best_score": best_score,
            "avg_score": avg_score,
            "avg_trades": avg_trades,
            "pct_no_trades": pct_no_trades,
            "top_params": gen_scores[0][0] if gen_scores else {},
            "top_by_return_params": (best_gen_return_rec[0] if best_gen_return_rec else {}),
            "best_return": (best_gen_return_rec[1] if best_gen_return_rec else 0.0),
            "best_return_score": (best_gen_return_rec[2] if best_gen_return_rec else 0.0),
            "elite_n": elite_n,
            "breed_n": breed_n,
            "inject_n": inject_n,
            "fitness_weights": {
                "alpha_cagr": alpha_cagr,
                "beta_calmar": beta_calmar,
                "gamma_sharpe": gamma_sharpe,
                "delta_total_return": delta_total_return,
                "calmar_cap": calmar_cap,
                "use_normalized_scoring": use_normalized_scoring,
                "holding_penalty_weight": holding_penalty_weight,
                "trade_rate_penalty_weight": trade_rate_penalty_weight,
                "penalty_cap": penalty_cap,
                "min_holding_days": min_holding_days,
                "max_holding_days": max_holding_days,
                "trade_rate_min": trade_rate_min,
                "trade_rate_max": trade_rate_max,
                "rate_penalize_upper": rate_penalize_upper,
                "elite_by_return_frac": elite_by_return_frac,
            },
            "penalty_stats": {
                "cap_hit_rate": cap_hit_rate,
                "share_gated_zero": share_gated_zero,
                "penalty_mean": _pen_mean,
                "penalty_p95": _pen_p95,
            },
        }
        progress_cb("generation_end", end_payload)
        logger.log("generation_end", end_payload)

        if should_stop:
            stopped_early = True
            break

    elapsed_total = time.time() - t0
    scored.sort(key=lambda x: x[1], reverse=True)
    best = scored[0] if scored else ({}, 0.0)
    done_payload = {
        "elapsed_sec": elapsed_total,
        "best": best[0],
        "score": best[1],
        "best_by_return": (best_run_return_rec[0] if best_run_return_rec else {}),
        "best_by_return_value": (best_run_return_rec[1] if best_run_return_rec else 0.0),
        "best_by_return_score": (best_run_return_rec[2] if best_run_return_rec else 0.0),
        "fitness_weights": {
            "alpha_cagr": alpha_cagr,
            "beta_calmar": beta_calmar,
            "gamma_sharpe": gamma_sharpe,
            "delta_total_return": delta_total_return,
            "calmar_cap": calmar_cap,
            "use_normalized_scoring": use_normalized_scoring,
            "holding_penalty_weight": holding_penalty_weight,
            "trade_rate_penalty_weight": trade_rate_penalty_weight,
            "penalty_cap": penalty_cap,
            "min_holding_days": min_holding_days,
            "max_holding_days": max_holding_days,
            "trade_rate_min": trade_rate_min,
            "trade_rate_max": trade_rate_max,
            "rate_penalize_upper": rate_penalize_upper,
            "elite_by_return_frac": elite_by_return_frac,
        },
        "generations_ran": max(0, last_gen + 1),
        "stopped_early": stopped_early,
        "pop_size": pop_size_effective,
    }
    if test_range:
        done_payload["test_start"] = str(test_range[0])
        done_payload["test_end"] = str(test_range[1])
        done_payload["score_window"] = "test"
    if cfg:
        done_payload["ea_config"] = cfg.to_log_payload()
    progress_cb("done", done_payload)
    logger.log("session_end", done_payload)
    return scored[:5]
