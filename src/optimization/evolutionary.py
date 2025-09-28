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
    hold_pen = λ_hold * penalty(avg_holding_days outside [min_hold, max_hold])
    rate_pen = λ_rate * penalty(trade_rate outside [rate_min, rate_max])
    score = base - hold_pen - rate_pen
- All knobs exposed as function args (UI-ready)
"""

from __future__ import annotations
import random
import time
import importlib
from typing import Any, Dict, List, Tuple, Optional
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
        # Drop any None-coerced floats
        for k in ["alpha_cagr","beta_calmar","gamma_sharpe","delta_total_return","calmar_cap"]:
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


# ------------------------------ Penalties --------------------------------

def _holding_penalty(avg_hold_days: float, lo: float, hi: float) -> float:
    """Linear penalty outside [lo, hi]. Zero inside the band."""
    if lo <= avg_hold_days <= hi:
        return 0.0
    if avg_hold_days < lo:
        return float(lo - avg_hold_days)
    return float(avg_hold_days - hi)


def _rate_penalty(trade_rate: float, lo: float, hi: float) -> float:
    """Linear penalty outside [lo, hi] for trades per symbol per year."""
    if lo <= trade_rate <= hi:
        return 0.0
    if trade_rate < lo:
        return float(lo - trade_rate)
    return float(trade_rate - hi)


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

    hold_pen = holding_penalty_weight * _holding_penalty(avg_hold, min_holding_days, max_holding_days)

    trade_rate = 0.0
    if num_symbols > 0 and years > 0:
        trade_rate = float(trades) / float(num_symbols) / float(years)
    rate_pen = trade_rate_penalty_weight * _rate_penalty(trade_rate, trade_rate_min, trade_rate_max)

    return float(base - hold_pen - rate_pen)


# --------------------------- Child evaluation ----------------------------

def _eval_one(
    strategy_dotted: str,
    tickers: List[str],
    start,
    end,
    starting_equity: float,
    params: Dict[str, Any],
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
    return {
        "metrics": metrics,
        "elapsed_sec": time.time() - t1,
        "params": params,
        "loader_file": _loader_file,
    }


# ----------------------------- Main EA loop ------------------------------

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
    use_normalized_scoring: bool = True,
    # Progress & logging
    progress_cb: Optional[ProgressCb] = None,
    log_file: str = "training.log",
    # Reproducibility
    seed: Optional[int] = None,
) -> List[Tuple[Dict[str, Any], float]]:
    """
    Run EA and return top parameter sets [(params, score)].

    Notes:
    - Survivor selection indirectly enforces `min_trades` and min_avg_holding_days_gate via fitness gating.
    - Diversity maintained via `random_inject_frac`.
    - Multiprocessing evaluates individuals in parallel per generation.
    """
    resolved_cb: Optional[ProgressCb] = progress_cb or console_progress
    if resolved_cb is None:
        resolved_cb = _noop_progress
    progress_cb = resolved_cb

    if seed is not None:
        random.seed(seed)

    logger = TrainingLogger(log_file)
    population = [random_param(param_space) for _ in range(pop_size)]
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
            },
        })

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

    for gen in range(generations):
        progress_cb("generation_start", {"gen": gen, "pop_size": len(population)})
        logger.log("generation_start", {"gen": gen, "pop_size": len(population)})

        gen_scores: List[Tuple[Dict[str, Any], float, float]] = []
        gen_trades: List[int] = []
        no_trade_count = 0
        best_gen_return_rec: Optional[Tuple[Dict[str, Any], float, float]] = None  # (params, total_return, score)

        # ---- Evaluate population (parallel or single) ----
        results: List[Dict[str, Any]] = []

        if n_jobs and n_jobs > 1:
            with ProcessPoolExecutor(max_workers=n_jobs) as ex:
                futures = [
                    ex.submit(_eval_one, strategy_dotted, tickers, start, end, starting_equity, params)
                    for params in population
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
            for params in population:
                try:
                    out = _eval_one(strategy_dotted, tickers, start, end, starting_equity, params)
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
                metrics = {}
                elapsed = 0.0
                loader_file = None
            else:
                rec = results.pop(hit_idx)
                metrics = rec.get("metrics", {}) or {}
                elapsed = rec.get("elapsed_sec", 0.0)
                loader_file = rec.get("loader_file")

            # compute fitness
            score = _clamped_fitness(
                metrics,
                min_trades=min_trades,
                min_avg_holding_days_gate=min_avg_holding_days_gate,
                require_hold_days=require_hold_days,
                eps_mdd=eps_mdd,
                eps_sharpe=eps_sharpe,
                alpha_cagr=alpha_cagr,
                beta_calmar=beta_calmar,
                gamma_sharpe=gamma_sharpe,
                delta_total_return=delta_total_return,
                min_holding_days=min_holding_days,
                max_holding_days=max_holding_days,
                holding_penalty_weight=holding_penalty_weight,
                trade_rate_min=trade_rate_min,
                trade_rate_max=trade_rate_max,
                trade_rate_penalty_weight=trade_rate_penalty_weight,
                num_symbols=num_symbols,
                years=years,
                calmar_cap=calmar_cap,
                use_normalized_scoring=use_normalized_scoring,
            )

            trades = int(metrics.get("trades", 0) or 0)
            gen_trades.append(trades)

            ret = float(metrics.get("total_return", 0.0) or 0.0)

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
                "metrics": metrics,
                "elapsed_sec": elapsed,
                "loader_file": loader_file,
            }
            progress_cb("individual_evaluated", payload)
            logger.log("individual_evaluated", payload)

            if trades < min_trades:
                logger.log("under_min_trades", payload)
                if trades == 0 and (metrics.get("calmar", 0) or 0) != 0:
                    logger.log("degenerate_fitness", payload)
                no_trade_count += (1 if trades == 0 else 0)

            gen_scores.append((params, score, ret))

        # ---- Selection & breeding ----
        # Sort primarily by score, then tie-break by total_return
        gen_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
        scored.extend([(p, s) for (p, s, _ret) in gen_scores])

        elite_n = max(1, int(pop_size * elite_frac))
        inject_n = max(0, int(pop_size * random_inject_frac))
        breed_n = pop_size - elite_n - inject_n
        if breed_n < 0:
            inject_n = max(0, pop_size - elite_n)
            breed_n = pop_size - elite_n - inject_n

        survivors: List[Dict[str, Any]] = []
        for p, s, _r in gen_scores:
            if s > 0:
                survivors.append(p)
            if len(survivors) == elite_n:
                break

        # pad if too few survivors (keep population viable)
        idx = 0
        while len(survivors) < elite_n and idx < len(gen_scores):
            survivors.append(gen_scores[idx][0])
            idx += 1

        # breed children
        children: List[Dict[str, Any]] = []
        parents = survivors if len(survivors) >= 2 else [random_param(param_space) for _ in range(2)]
        for _ in range(max(0, breed_n)):
            p1 = random.choice(parents)
            p2 = random.choice(parents)
            child = mutate(crossover(p1, p2), param_space, rate=mutation_rate)
            children.append(child)

        injections = [random_param(param_space) for _ in range(max(0, inject_n))]

        population = survivors + children + injections
        while len(population) < pop_size:
            population.append(random_param(param_space))
        if len(population) > pop_size:
            population = population[:pop_size]

        avg_score = (sum(s for _, s, _ in gen_scores) / len(gen_scores)) if gen_scores else 0.0
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
            },
        }
        progress_cb("generation_end", end_payload)
        logger.log("generation_end", end_payload)

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
        },
    }
    progress_cb("done", done_payload)
    logger.log("session_end", done_payload)
    return scored[:5]
