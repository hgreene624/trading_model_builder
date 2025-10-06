# EA Implementation Audit

## Changelog
- 2025-10-06: Added EAConfig with mutation/elitism controls exposed through the Strategy Adapter UI, including mutation rate/scale, crossover, selection, annealing, and worker knobs. Legacy callers that omit `config` continue to use the previous defaults, and the Buy-the-Dip configuration UI has been consolidated into a single section.

## 1. Executive Summary
The platform wires two evolutionary optimizers into the Streamlit research workflow: a multi-symbol portfolio tuner (`evolutionary_search`) that loops through the general trainer/backtest stack with JSONL logging, and a single-symbol helper (`evolve_params`) that powers the legacy tuning page; both reuse the ATR breakout engine and metrics pipeline without modifying strategy code.【F:src/optimization/evolutionary.py†L318-L796】【F:src/tuning/evolve.py†L287-L367】【F:src/models/general_trainer.py†L89-L265】

**Current optimization objectives and penalties**
- Weighted growth score blending CAGR, Calmar, Sharpe, and total return with optional normalization and Calmar clamping.【F:src/optimization/evolutionary.py†L171-L270】
- Linear penalties on portfolio-average holding period and per-symbol trade rate, capped before subtracting from the score.【F:src/optimization/evolutionary.py†L272-L285】
- Hard gates that zero fitness when trades fall below `min_trades`, average holding days miss the gate, or drawdown/Sharpe are inside epsilon tolerances.【F:src/optimization/evolutionary.py†L214-L235】
- Single-symbol helper objective: `0.5*Sharpe + 0.4*TotalReturn - 0.1*DrawdownPenalty` with drawdown penalty equal to positive drawdown magnitude.【F:src/tuning/evolve.py†L276-L284】

## 2. Code Map & Call Flow
```
Streamlit UI (pages/4_Ticker_Selector_and_Tuning.py)
  └─ Collect EA bounds + run settings, call evolve_params(...) for single symbol【F:pages/4_Ticker_Selector_and_Tuning.py†L215-L396】
Streamlit Walkforward page / services
  └─ Optionally invoke evolutionary_search(...) inside walk_forward(...)【F:src/optimization/walkforward.py†L87-L197】
Evolutionary loop (src/optimization/evolutionary.py)
  ├─ random_param/mutate/crossover manage genomes【F:src/optimization/evolutionary.py†L118-L146】
  ├─ train_general_model(...) evaluates each individual【F:src/optimization/evolutionary.py†L290-L314】【F:src/models/general_trainer.py†L89-L265】
  ├─ backtest_atr_breakout(...) + metrics compute fitness inputs【F:src/backtest/engine.py†L392-L500】【F:src/backtest/metrics.py†L247-L261】
  ├─ TrainingLogger emits JSONL telemetry for inspectors【F:src/optimization/evolutionary.py†L404-L795】【F:src/utils/training_logger.py†L9-L41】
  └─ Optional walk-forward loop replays EA winners OOS【F:src/optimization/walkforward.py†L246-L370】
Storage/Inspection
  ├─ EA logs saved to configurable log file and inspected via pages/3_EA_Train_Inspector.py【F:src/optimization/evolutionary.py†L359-L795】【F:pages/3_EA_Train_Inspector.py†L17-L125】
  └─ Fitness weights optionally loaded from storage/config/ea_fitness.json.【F:src/optimization/evolutionary.py†L41-L424】
```

## 3. Genome & Parameter Space
- **Genome form (portfolio EA):** plain Python `dict` keyed by strategy parameter names; genes are sampled from `(low, high)` tuples passed in via `param_space`, inferring integer vs float by bound types.【F:src/optimization/evolutionary.py†L118-L140】
- **Genome form (single-symbol EA):** structured dict managed by the `Bounds` dataclass that defines min/max per feature plus EA run defaults.【F:src/tuning/evolve.py†L41-L152】
- **Constraint handling:** portfolio EA simply resamples mutated genes within their native bounds (uniform re-draw); single-symbol EA repairs each chromosome with `_fix`, clipping ranges, enforcing exit < breakout, ordering SMAs, non-negative buffers, and boolean guards before returning individuals.【F:src/optimization/evolutionary.py†L130-L140】【F:src/tuning/evolve.py†L99-L214】
- **Invalid values:** portfolio EA relies on sampling within bounds and does not perform post-mutation clipping, so any mutation draw stays feasible by construction; single-symbol EA clamps on every random sample, crossover, and mutation call via `_fix` and `_clip_*` helpers.【F:src/optimization/evolutionary.py†L118-L140】【F:src/tuning/evolve.py†L95-L214】

## 4. EA Mechanics (as implemented)
### Population & Budget
- Portfolio EA exposes `generations`, `pop_size`, and evaluation limits via function arguments with defaults `(10, 20)`; walk-forward forwards `ea_generations`/`ea_pop`, and the Streamlit walk-forward UI can override via kwargs.【F:src/optimization/evolutionary.py†L318-L369】【F:src/optimization/walkforward.py†L87-L197】
- Single-symbol EA pulls run-length defaults from `Bounds` (`pop_size=40`, `generations=20`) but allows slider overrides from Streamlit before calling `evolve_params`.【F:src/tuning/evolve.py†L41-L90】【F:pages/4_Ticker_Selector_and_Tuning.py†L215-L339】

### Selection
- Portfolio EA performs deterministic truncation selection with mixed criteria: keep top `elite_frac` of the population (default 50%) by score, augment with top total-return individuals per `elite_by_return_frac`, deduplicate by JSON string, then refill with next-best scores.【F:src/optimization/evolutionary.py†L330-L691】
- Single-symbol EA keeps the top 10% (`keep = max(1, 0.1*pop_size)`) each generation as elites and fills the mating pool via a fixed-size (k=3) tournament sampled from the sorted fitness list.【F:src/tuning/evolve.py†L325-L365】

### Crossover
- Portfolio EA uses uniform crossover with 0.5 per-gene swap probability and no independent crossover rate toggle; every child is crossed then mutated.【F:src/optimization/evolutionary.py†L143-L145】【F:src/optimization/evolutionary.py†L692-L699】
- Single-symbol EA applies blend crossover to floats (BLX-α with α=0.2) and coin-flip inheritance for integers/booleans whenever `rng.random() < crossover_rate` (default 0.7).【F:src/tuning/evolve.py†L217-L240】【F:src/tuning/evolve.py†L354-L363】

### Mutation
- Portfolio EA mutates each gene independently with probability `mutation_rate` (default 0.3) by resampling a fresh uniform value within the gene’s bounds (reset mutation).【F:src/optimization/evolutionary.py†L130-L140】【F:src/optimization/evolutionary.py†L692-L699】
- Single-symbol EA applies heterogeneous mutations: Gaussian steps for windows, ratio-preserving exit tuning, log-space multiplicative mutation for ATR multiple and risk, capped Gaussian tweaks for other floats, and rare boolean flips, all gated by per-gene probabilities; overall mutation occurs when `rng.random() < mutation_rate` (default 0.35).【F:src/tuning/evolve.py†L156-L214】【F:src/tuning/evolve.py†L354-L363】

### Elitism / Survivor Selection
- Portfolio EA’s `elite_frac` computes survivor count each generation, with optional `elite_by_return_frac` rebalancing between score- and return-based elites; remainder of population is rebuilt generationally (μ,λ with μ elites and λ children/injections).【F:src/optimization/evolutionary.py†L330-L707】
- Single-symbol EA is generational with fixed 10% elitism and no separate return-based quota; elites are copied verbatim into the next population.【F:src/tuning/evolve.py†L345-L365】

### Diversity & Anti-duplication
- Portfolio EA injects `random_inject_frac` (default 0.2) fresh random individuals each generation and deduplicates elites/children using JSON string keys before forming survivors.【F:src/optimization/evolutionary.py†L331-L705】
- Single-symbol EA relies on stochastic tournament sampling and Gaussian/log mutations; there is no explicit duplicate guard beyond `_fix` normalizing values.【F:src/tuning/evolve.py†L156-L214】【F:src/tuning/evolve.py†L345-L365】

### Early Stopping
- Neither EA implements patience- or threshold-based early termination; both loops always run the requested number of generations.【F:src/optimization/evolutionary.py†L447-L765】【F:src/tuning/evolve.py†L319-L367】

### Seeding & Reproducibility
- Portfolio EA seeds the global `random` module when `seed` is provided and propagates deterministic ordering when logging individuals.【F:src/optimization/evolutionary.py†L362-L632】
- Single-symbol EA constructs a dedicated `random.Random(random_seed)` RNG, defaulting to 42 from the UI call, ensuring reproducible runs given identical bounds and seed.【F:src/tuning/evolve.py†L297-L361】【F:pages/4_Ticker_Selector_and_Tuning.py†L327-L339】

### Parallelism
- Portfolio EA evaluates populations either serially or via `ProcessPoolExecutor` with `n_jobs` workers; results are reconciled in submission order to maintain deterministic logging.【F:src/optimization/evolutionary.py†L332-L505】
- Single-symbol EA runs synchronously on the main thread and does not expose worker controls.【F:src/tuning/evolve.py†L287-L367】

## 5. Fitness Function & Penalties
- **Primary score:** normalized weighted sum of CAGR, Calmar (clamped to ±`calmar_cap`), Sharpe, and total return; normalization maps each metric into [0,1] before weighting when `use_normalized_scoring` is true.【F:src/optimization/evolutionary.py†L237-L270】
- **Penalties:** holding-period deviation and trade-rate deviation are scaled by respective weights, normalized by band width, summed, and clipped by `penalty_cap` before subtracting from the score.【F:src/optimization/evolutionary.py†L272-L285】
- **Gates:** zero score returned when trades < `min_trades`, average holding days below `min_avg_holding_days_gate`, or optional hold-day requirement fails; small drawdowns/Sharpe values are zeroed via `eps_mdd`/`eps_sharpe`.【F:src/optimization/evolutionary.py†L214-L235】
- **Trade-rate context:** trade rate = total trades ÷ symbols ÷ years, where years is derived from train period length; parameters `trade_rate_min`, `trade_rate_max`, and `rate_penalize_upper` control band behavior.【F:src/optimization/evolutionary.py†L272-L285】【F:src/optimization/evolutionary.py†L434-L455】
- **Metrics sources:** `train_general_model` aggregates equal-weight portfolio metrics, injecting `trades`, `avg_holding_days`, `win_rate`, and `expectancy` so the EA can gate on activity.【F:src/models/general_trainer.py†L169-L245】
- **Cost modeling:** ATR engine applies costs/slippage through `CostModel.from_inputs`, environment overrides, and returns post-cost trade logs used by `compute_core_metrics`, ensuring fitness incorporates cost-aware equity curves when enabled.【F:src/backtest/engine.py†L392-L500】【F:src/backtest/metrics.py†L247-L261】
- **External config:** optional `storage/config/ea_fitness.json` overrides fitness weights and penalty parameters at runtime, with the applied values logged once per session.【F:src/optimization/evolutionary.py†L41-L424】
- **Single-symbol EA fitness:** returns the blended Sharpe/return/drawdown score described above; penalties rely solely on drawdown magnitude, with engine metrics feeding the calculation.【F:src/tuning/evolve.py†L243-L284】

## 6. Walk-Forward & Holdout Integration
- `walk_forward` builds rolling train/test windows, optionally invokes `evolutionary_search` on each training slice with a shrink-wrapped search space around the incoming params, and uses the best genome for out-of-sample evaluation, logging each split via `TrainingLogger`.【F:src/optimization/walkforward.py†L87-L326】
- The helper normalizes equity curves per split, computes aggregated IS/OOS metrics with `compute_core_metrics`, and writes JSONL checkpoints for inspector tooling.【F:src/optimization/walkforward.py†L246-L370】

## 7. UI Exposure & Config Path
- The Streamlit tuning page exposes sliders for population size, generations, crossover rate, and mutation rate, plus numeric inputs for every bound defined by `Bounds`; button handlers serialize these into a `Bounds` instance before calling `evolve_params` with a fixed random seed of 42.【F:pages/4_Ticker_Selector_and_Tuning.py†L215-L339】
- Profile loading applies stored EA settings (population, generations, crossover, mutation) and parameter bounds back into Streamlit session state, enabling reproducible reruns.【F:pages/4_Ticker_Selector_and_Tuning.py†L50-L214】
- EA JSONL logs are surfaced in the EA inspector page, which looks under `storage/logs/ea` and parses the `TrainingLogger` events for visualization.【F:pages/3_EA_Train_Inspector.py†L17-L125】
- Walk-forward parameters (including EA toggles) are available via keyword arguments or UI controls in the associated page (not shown here) and forwarded to `walk_forward`/`evolutionary_search`.【F:src/optimization/walkforward.py†L87-L197】

## 8. Artifacts, Logging, and Repro
- **Logging:** `TrainingLogger` writes append-only JSON lines with timestamps; the EA emits `fitness_config`, `session_meta`, `generation_start`, `individual_evaluated`, `under_min_trades`, `generation_end`, and `session_end` events plus error records when evaluations fail.【F:src/optimization/evolutionary.py†L404-L795】【F:src/utils/training_logger.py†L9-L41】
- **File locations:** default EA log file is `training.log` in the working directory unless overridden; walk-forward uses `walkforward.jsonl`. Inspector tooling expects logs beneath `storage/logs/ea` by convention, so callers typically point `log_file` there.【F:src/optimization/evolutionary.py†L359-L795】【F:pages/3_EA_Train_Inspector.py†L17-L71】
- **Artifacts saved:** besides logs, the EA returns the top five `(params, score)` tuples to the caller; metrics snapshots per individual and generation are only persisted via logging. There is no automatic storage of genomes or seeds beyond what callers record.【F:src/optimization/evolutionary.py†L362-L796】
- **Reproducibility:** seeds must be supplied by the caller (`seed` for portfolio EA, `random_seed` for single-symbol EA); logs do not currently store the RNG seed unless included in session metadata by the caller.【F:src/optimization/evolutionary.py†L362-L795】【F:src/tuning/evolve.py†L297-L361】

## 9. Gaps vs. Common EA Controls
| Control | Status | Where/Default | Notes |
| --- | --- | --- | --- |
| `mutation_rate` | Implemented | Portfolio arg default 0.3; single-symbol slider default 0.35 | UI exposes single-symbol rate; portfolio rate configurable via kwargs only.【F:src/optimization/evolutionary.py†L318-L699】【F:pages/4_Ticker_Selector_and_Tuning.py†L215-L339】 |
| `mutation_scale` / step size | Partially hard-coded | Gaussian/log sigmas fixed inside `_mutate`; portfolio uses uniform resets | No external control over sigma or reset magnitude.【F:src/tuning/evolve.py†L156-L214】【F:src/optimization/evolutionary.py†L130-L140】 |
| `elitism_fraction` | Implemented (portfolio), hard-coded (single) | Portfolio `elite_frac=0.5`; single-symbol keeps 10% | Portfolio fraction override via kwargs; UI does not expose it.【F:src/optimization/evolutionary.py†L318-L707】【F:src/tuning/evolve.py†L325-L365】 |
| `selection_method` / `tournament_k` | Fixed | Portfolio uses deterministic truncation; single-symbol tournament k=3 | No strategy toggle or UI hook.【F:src/optimization/evolutionary.py†L646-L705】【F:src/tuning/evolve.py†L344-L363】 |
| `crossover_rate` | Mixed | Portfolio lacks rate knob (always uniform); single-symbol exposes slider default 0.7 | Portfolio children always undergo uniform crossover before mutation.【F:src/optimization/evolutionary.py†L143-L699】【F:pages/4_Ticker_Selector_and_Tuning.py†L215-L339】 |
| `crossover_op` | Fixed | Portfolio uniform; single-symbol BLX/co-inheritance | Not user-selectable.【F:src/optimization/evolutionary.py†L143-L699】【F:src/tuning/evolve.py†L217-L363】 |
| `replacement policy` | Generational | Both rebuild full population from elites + offspring + injections | No steady-state option.【F:src/optimization/evolutionary.py†L646-L707】【F:src/tuning/evolve.py†L325-L365】 |
| `fitness_patience` / early stop | Missing | — | Loops always run `generations` iterations.【F:src/optimization/evolutionary.py†L447-L765】【F:src/tuning/evolve.py†L319-L367】 |
| `seed` | Implemented | `seed` argument (portfolio); `random_seed` (single) | UI hard-codes 42 for single-symbol; portfolio caller must pass explicitly.【F:src/optimization/evolutionary.py†L362-L795】【F:pages/4_Ticker_Selector_and_Tuning.py†L327-L339】 |
| `workers` | Implemented (portfolio) | `n_jobs` controls ProcessPool workers | Single-symbol runs single-threaded.【F:src/optimization/evolutionary.py†L332-L505】 |
| `duplicate_guard` | Partial | Portfolio dedupes elites via JSON keys; no global archive | Single-symbol lacks duplication checks beyond normalization.【F:src/optimization/evolutionary.py†L657-L705】【F:src/tuning/evolve.py†L99-L214】 |
| `niching_radius` | Missing | — | No niching or crowding distance support.【F:src/optimization/evolutionary.py†L646-L707】 |

## 10. Usage Examples (Current)
```python
# Walk-forward training split
space = {...}  # derived from base params
best = evolutionary_search(
    strategy_dotted,
    tickers,
    train_start,
    train_end,
    starting_equity,
    space,
    generations=ea_generations,
    pop_size=ea_pop,
    min_trades=min_trades,
    n_jobs=n_jobs,
    seed=seed,
    log_file=log_file,
    progress_cb=progress_cb,
)
```
【F:src/optimization/walkforward.py†L165-L197】

```python
# Streamlit single-symbol tuning
best_params, best_metrics, history = evolve_params(
    symbol=symbol,
    start=start.isoformat(),
    end=end.isoformat(),
    starting_equity=float(starting_equity),
    bounds=bounds,
    pop_size=pop_size,
    generations=generations,
    crossover_rate=crossover_rate,
    mutation_rate=mutation_rate,
    random_seed=42,
    progress_cb=_cb,
)
```
【F:pages/4_Ticker_Selector_and_Tuning.py†L303-L339】

## 11. Appendix A – Line-Referenced Index
| Topic | File | Function/Section | Lines | Notes |
| --- | --- | --- | --- | --- |
| Selection logic | `src/optimization/evolutionary.py` | `evolutionary_search` survivor block | 642-705 | Elite mix by score/return with dedupe and injections.【F:src/optimization/evolutionary.py†L642-L705】 |
| Selection (single) | `src/tuning/evolve.py` | `evolve_params` loop | 325-365 | 10% elitism + tournament mating.【F:src/tuning/evolve.py†L325-L365】 |
| Crossover | `src/optimization/evolutionary.py` | `crossover` helper | 143-145 | Uniform per gene.【F:src/optimization/evolutionary.py†L143-L145】 |
| Crossover (single) | `src/tuning/evolve.py` | `_xover` | 217-240 | BLX-α for floats, coin-flip ints.【F:src/tuning/evolve.py†L217-L240】 |
| Mutation | `src/optimization/evolutionary.py` | `mutate` | 130-140 | Uniform re-draw per gene.【F:src/optimization/evolutionary.py†L130-L140】 |
| Mutation (single) | `src/tuning/evolve.py` | `_mutate` | 156-214 | Gaussian/log moves per field.【F:src/tuning/evolve.py†L156-L214】 |
| Elitism | `src/optimization/evolutionary.py` | `elite_frac` handling | 646-707 | Score/return elites + injections.【F:src/optimization/evolutionary.py†L646-L707】 |
| Early stop | `src/optimization/evolutionary.py` | Main loop | 447-765 | No break before `generations` complete.【F:src/optimization/evolutionary.py†L447-L765】 |
| Fitness equation | `src/optimization/evolutionary.py` | `_clamped_fitness` | 171-285 | Weighted normalized score minus capped penalties.【F:src/optimization/evolutionary.py†L171-L285】 |
| Cost modeling | `src/backtest/engine.py` | `backtest_atr_breakout` setup | 392-500 | Cost model, env overrides, metadata.【F:src/backtest/engine.py†L392-L500】 |
| Logging | `src/utils/training_logger.py` | `TrainingLogger` | 9-41 | JSONL logger with error hook.【F:src/utils/training_logger.py†L9-L41】 |

## 12. Appendix B – Grep/Find Hints
```bash
rg -n "evolutionary|mutation|crossover|elit|tournament|fitness|walk.?forward|logger" src/ pages/
rg -n "Sharpe|expectancy|edge|drawdown|penalt|cost|slippage" src/
rg -n "ATRParams|breakout|engine|wilder_atr" src/
```
