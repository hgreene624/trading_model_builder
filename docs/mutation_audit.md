# Evolutionary Algorithm Mutation Audit

This note summarizes how mutation operates inside the project's two evolutionary optimizers (the portfolio-wide optimizer in `src/optimization/evolutionary.py` and the single-symbol tuner in `src/tuning/evolve.py`). The goal is to give an accessible, end-to-end view of when mutation triggers, how new parameter values are drawn, and what safeguards keep the search stable.

## 1. Where mutation fits in the portfolio EA

1. **Configurable knobs** – The portfolio optimizer accepts an `EAConfig` dataclass that carries the mutation rate, step size, distribution, optional gene-wise clipping, and an annealing toggle. If no config is supplied it falls back to legacy defaults. 【F:src/optimization/evolutionary.py†L126-L211】
2. **Per-generation scale** – When annealing is enabled the mutation scale is linearly decreased from the starting value down to a floor across generations, letting early exploration take bigger steps and late generations fine-tune. 【F:src/optimization/evolutionary.py†L153-L176】
3. **Main loop placement** – After survivor selection and crossover, each child goes through `_mutate_configured` with the generation-specific scale. Legacy callers still use `_legacy_mutate`, which simply resamples any gene hit by the mutation probability. 【F:src/optimization/evolutionary.py†L1194-L1219】

## 2. How `_mutate_configured` changes genes

For every gene in the search space the function flips a mutation coin using the configured rate. Genes that pass the check are mutated according to the selected scheme, with integer and float handling separated to keep types consistent. 【F:src/optimization/evolutionary.py†L341-L387】

* **Uniform reset** draws an entirely new random value within the allowed bounds (full re-sample). 【F:src/optimization/evolutionary.py†L354-L361】
* **Polynomial mutation** uses the SBX-style delta formula for floats and a bounded integer delta for discrete genes, scaled by `mutation_scale`. Candidates that overshoot their bounds are re-sampled a handful of times before falling back to a uniform draw, keeping exploration lively even when the scale is high. 【F:src/optimization/evolutionary.py†L362-L390】
* **Gaussian mutation (default)** adds Gaussian noise to floats with a standard deviation tied to the parameter range and applies a symmetric integer step for discrete variables. Like the polynomial path, repeated out-of-bounds samples trigger a re-draw so large sigmas do not collapse everything to the parameter edges. 【F:src/optimization/evolutionary.py†L392-L417】

When `genewise_clip` is enabled the result is still clamped back to the parameter bounds, but the truncated sampling reduces how often clipping activates, which preserves diversity when mutation parameters are cranked up. 【F:src/optimization/evolutionary.py†L419-L424】

## 3. Mutation rate in practice

* The **effective mutation rate** is taken from `EAConfig` when present; otherwise the legacy rate argument is used. 【F:src/optimization/evolutionary.py†L681-L1219】
* **Annealing** only changes the mutation scale, not the probability of mutating each gene. The per-gene coin flip stays at `cfg.mutation_rate` throughout a run. 【F:src/optimization/evolutionary.py†L153-L176】【F:src/optimization/evolutionary.py†L341-L387】
* **Random injections** (fresh individuals sampled from the parameter space) supply additional diversity separate from mutation. 【F:src/optimization/evolutionary.py†L1213-L1223】

## 4. Single-symbol tuner mutation

The ticker-specific EA uses a different, hand-crafted mutation routine tailored to breakout strategy parameters. Mutation occurs only when the outer loop's mutation probability check succeeds. 【F:src/tuning/evolve.py†L300-L367】

Once triggered, `_mutate` applies heterogeneous moves per gene:

* **Window lengths** (`breakout_n`, SMAs, etc.) receive Gaussian steps with probabilities tuned per field so each control evolves at its own cadence. 【F:src/tuning/evolve.py†L156-L186】
* **Exit/lookback coupling** keeps `exit_n` as a fraction of `breakout_n`, mutating the ratio instead of raw values to preserve sensible orderings. 【F:src/tuning/evolve.py†L164-L169】
* **ATR and risk sizing** use log-space multiplicative noise, maintaining positivity and scaling adjustments relative to the current value. 【F:src/tuning/evolve.py†L187-L193】
* **Other floats** (profit target, trading costs, ATR buffer) get bounded Gaussian tweaks, again gated by per-gene mutation probabilities. 【F:src/tuning/evolve.py†L194-L205】
* **Discrete persistence** mutates via small integer Gaussian steps with clipping, and the trend filter boolean occasionally flips. 【F:src/tuning/evolve.py†L206-L214】

Every mutated child runs back through `_fix`, which enforces bounds, ordering constraints between moving averages, and boolean eligibility. 【F:src/tuning/evolve.py†L99-L130】【F:src/tuning/evolve.py†L156-L214】

## 5. Key takeaways for practitioners

* **Mutation rate vs. scale** – In the portfolio EA you can treat the rate as "how many knobs" to change and the scale as "how far" to move those knobs; annealing only shrinks the latter. The sampler now retries out-of-bounds draws before clipping, so dialing the scale up really does yield larger moves. 【F:src/optimization/evolutionary.py†L138-L176】【F:src/optimization/evolutionary.py†L341-L424】
* **Scheme choice** – Gaussian mutation is local and smooth, polynomial behaves similarly but with heavier tails, and uniform reset is best when you need drastic jumps. 【F:src/optimization/evolutionary.py†L354-L379】
* **Safety nets** – Gene-wise clipping and `_fix` ensure mutated parameters remain legal, so experimentation with higher mutation rates will not produce invalid configurations. 【F:src/optimization/evolutionary.py†L381-L386】【F:src/tuning/evolve.py†L99-L130】
* **Diversity sources** – Mutation works alongside crossover and random injections (portfolio EA) or tournament selection (single-symbol EA), so tuning should consider the entire pipeline rather than mutation in isolation. 【F:src/optimization/evolutionary.py†L1194-L1223】【F:src/tuning/evolve.py†L344-L365】

With these mechanics in mind you can confidently adjust mutation settings, knowing how they influence exploration versus exploitation across both evolutionary search implementations.
