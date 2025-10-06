# Buy-the-Dip Implementation Audit

## 1. Executive Summary
*The intent.* The Strategy Adapter UI exposes an `entry_mode` toggle with Dip-specific controls (`trend_ma`, `dip_atr_from_high`, `dip_lookback_high`, `dip_rsi_max`, `dip_confirm`, `dip_cooldown_days`) alongside the standard ATR breakout knobs, signalling an experiment to layer a short-term mean-reversion entry over the breakout engine.【F:pages/2_Strategy_Adapter.py†L411-L446】【F:pages/2_Strategy_Adapter.py†L693-L740】

*Current status.* When Dip mode is enabled, those parameters are forwarded through the evolutionary search and trainer into the ATR wrapper, but the runtime still instantiates the `ATRParams` dataclass that only knows about breakout fields. The extra Dip keywords reach `ATRParams(**p)` and trigger `TypeError: unexpected keyword argument 'trend_ma'`, so Dip runs fail before the backtest starts.【F:pages/2_Strategy_Adapter.py†L915-L939】【F:src/models/atr_breakout.py†L22-L32】【F:src/backtest/engine.py†L358-L376】

## 2. Code Map & Call Flow (BtD)
1. **Streamlit UI** – Strategy Adapter builds Dip defaults/bounds and, when Dip is selected, adds the Dip keys to the EA parameter space before launching the search.【F:pages/2_Strategy_Adapter.py†L411-L446】【F:pages/2_Strategy_Adapter.py†L537-L566】【F:pages/2_Strategy_Adapter.py†L915-L939】
2. **Evolutionary search** – `evo.evolutionary_search` seeds a population with `random_param(param_space)` and evaluates candidates in parallel via `_eval_one` (ProcessPool when `n_jobs>1`).【F:src/optimization/evolutionary.py†L360-L405】【F:src/optimization/evolutionary.py†L464-L478】
3. **Worker evaluation** – `_eval_one` invokes `train_general_model` with the genome dictionary (including Dip keys).【F:src/optimization/evolutionary.py†L290-L307】
4. **Portfolio trainer** – `train_general_model` imports the selected `run_strategy` and calls it once per symbol using the same params dict; there is no Dip-specific preprocessing.【F:src/models/general_trainer.py†L112-L195】
5. **Strategy wrapper** – `atr_breakout.run_strategy` still coerces the dict into `ATRParams`, so the Dip keys cause the failure before reaching the engine.【F:src/models/atr_breakout.py†L22-L60】
6. **Backtest engine / metrics** – If instantiation succeeded the ATR engine would run and compute metrics, but Dip never reaches this stage today.【F:src/backtest/engine.py†L392-L700】【F:src/backtest/metrics.py†L1-L120】
7. **Logging/storage** – The UI wraps the EA run with `TrainingLogger` writes and session state updates (unaffected by the crash).【F:pages/2_Strategy_Adapter.py†L1048-L1180】【F:src/utils/training_logger.py†L1-L30】

```
Dip UI ──params──▶ param_space (Dip keys added)
   │                     │
   │                     └──▶ evolutionary_search ─▶ _eval_one ─▶ train_general_model ─▶ run_strategy ─┐
   │                                                                                                     │
ATR UI ─────────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        └──▶ ATRParams(**genome)  ✖ (Dip kwargs crash)
```

Dip and ATR share the entire downstream pipeline; the only divergence is the additional keys injected before `run_strategy`, which the ATR dataclass rejects.

## 3. Parameter Model & Plumbing
| Parameter | Type | Default | Range (UI/EA) | Defined in | Forwarded via | Notes |
|-----------|------|---------|---------------|------------|---------------|-------|
| `entry_mode` | string | `"breakout"` | radio options `['breakout','dip']` | UI defaults and radio setup【F:pages/2_Strategy_Adapter.py†L411-L446】【F:pages/2_Strategy_Adapter.py†L568-L579】 | Stored in base params and EA session state; merged into top results【F:pages/2_Strategy_Adapter.py†L915-L939】【F:pages/2_Strategy_Adapter.py†L1118-L1154】 | Only drives UI branching; not consumed downstream. |
| `trend_ma` | int | 200 | EA bounds min/max inputs (20–600) | Base defaults & Dip expander【F:pages/2_Strategy_Adapter.py†L411-L446】【F:pages/2_Strategy_Adapter.py†L693-L703】 | Added to EA param_space and genomes【F:pages/2_Strategy_Adapter.py†L915-L939】 | Not in `ATRParams`; causes TypeError when passed through.【F:src/models/atr_breakout.py†L22-L32】【F:src/backtest/engine.py†L358-L376】 |
| `dip_atr_from_high` | float | 2.0 | 0–20 slider bounds | Same as above for defaults and expander【F:pages/2_Strategy_Adapter.py†L411-L446】【F:pages/2_Strategy_Adapter.py†L704-L711】 | EA param_space update【F:pages/2_Strategy_Adapter.py†L915-L939】 | Extra kwargs stored only in `extra_params` if they ever reached engine; unused today.【F:src/backtest/engine.py†L416-L495】 |
| `dip_lookback_high` | int | 60 | 5–600 | Defaults & expander【F:pages/2_Strategy_Adapter.py†L411-L446】【F:pages/2_Strategy_Adapter.py†L712-L718】 | EA param_space update【F:pages/2_Strategy_Adapter.py†L915-L939】 | Same as above. |
| `dip_rsi_max` | float | 55.0 | 0–100 | Defaults & expander【F:pages/2_Strategy_Adapter.py†L411-L446】【F:pages/2_Strategy_Adapter.py†L720-L726】 | EA param_space update【F:pages/2_Strategy_Adapter.py†L915-L939】 | Same as above. |
| `dip_confirm` | bool (int in bounds) | False | 0 or 1 | Defaults & expander【F:pages/2_Strategy_Adapter.py†L411-L446】【F:pages/2_Strategy_Adapter.py†L728-L732】 | EA param_space update【F:pages/2_Strategy_Adapter.py†L915-L939】 | Stored but unused downstream. |
| `dip_cooldown_days` | int | 5 | 0–240 | Defaults & expander【F:pages/2_Strategy_Adapter.py†L411-L446】【F:pages/2_Strategy_Adapter.py†L733-L739】 | EA param_space update【F:pages/2_Strategy_Adapter.py†L915-L939】 | Stored but unused downstream. |

The Dip parameters never receive a dedicated dataclass; they are treated as extra keys in the same dict the ATR strategy expects, which is the core incompatibility.

## 4. UI Layout & Duplication
* Dip defaults are seeded in the base parameter dictionary and preserved via `setdefault` so Session State always carries the keys.【F:pages/2_Strategy_Adapter.py†L411-L446】  
* EA bounds exposes a dedicated “Dip parameter bounds” block with identical labels and defaults, mirroring the base controls.【F:pages/2_Strategy_Adapter.py†L537-L566】  
* When Dip mode is selected, a second “Dip Settings” expander re-renders the same controls for per-run defaults (separate from the EA bounds).【F:pages/2_Strategy_Adapter.py†L693-L740】  
* The leaderboard later displays the Dip columns again, so users see the same field names in at least three places (defaults, EA bounds, results), all currently wired to the same values.【F:pages/2_Strategy_Adapter.py†L1135-L1154】

## 5. Runtime Error Deep-Dive
* **Kwargs producer.** The UI merges Dip bounds into `param_space` when `entry_mode == 'dip'`, so every sampled genome contains `trend_ma`, `dip_atr_from_high`, `dip_lookback_high`, `dip_rsi_max`, `dip_confirm`, and `dip_cooldown_days`.【F:pages/2_Strategy_Adapter.py†L915-L939】
* **Call stack.** `evolutionary_search` seeds the population and evaluates each genome through `_eval_one` (possibly in a `ProcessPoolExecutor`), which calls `train_general_model` → `run_strategy`.【F:src/optimization/evolutionary.py†L360-L478】【F:src/optimization/evolutionary.py†L290-L307】【F:src/models/general_trainer.py†L112-L195】
* **Failure site.** `run_strategy` unconditionally coerces dict inputs into `ATRParams(**p)`, and `ATRParams` only declares breakout-related fields. The unexpected `trend_ma` (and other Dip keys) triggers the TypeError before the engine can filter `extra_params`.【F:src/models/atr_breakout.py†L22-L32】【F:src/backtest/engine.py†L358-L376】
* **Consumer mismatch.** `backtest_atr_breakout` *would* split known vs unknown fields if given a dict directly, but the wrapper converts to `ATRParams` first, so Dip-only keys never reach that safety net.【F:src/backtest/engine.py†L416-L423】

## 6. Signal & Entry/Exit Logic (as-is)
* The backtest engine computes classic ATR breakout signals: rolling highs/lows, ATR-derived entry thresholds, optional persistence, and handles order execution, position sizing, and exits with no reference to Dip parameters.【F:src/backtest/engine.py†L551-L700】
* Unknown fields are merely copied into `meta['extra_params']`, so even if Dip keys survived they would be metadata only.【F:src/backtest/engine.py†L488-L500】
* The only “dip” feature elsewhere is `dip_from_high_atr` used for probability-gate feature engineering, but this is separate from the entry logic and still driven by breakout parameters.【F:src/backtest/prob_gate.py†L33-L79】
* Order placement uses the same market-style fills and ATR-based stops as the breakout strategy; there is no alternative mean-reversion entry/exit path implemented today.【F:src/backtest/engine.py†L566-L700】

## 7. EA Integration (current state)
* Dip parameters participate in EA sampling only when Dip mode is active; they are added to the `param_space` passed into `evolutionary_search`. There are no Dip-specific fitness weights or constraints beyond the shared `min_trades` gate.【F:pages/2_Strategy_Adapter.py†L915-L939】【F:src/optimization/evolutionary.py†L360-L520】
* Outside of that toggle, the EA treats Dip keys just like any other hyperparameter; there is no decoder to split them before strategy execution, leading directly to the crash.

## 8. Incompatibilities & Coupling Risks
* **Dataclass mismatch:** `ATRParams` lacks every Dip field, so any Dip key causes object construction to fail.【F:src/backtest/engine.py†L358-L376】
* **Wrapper rigidity:** `run_strategy` converts dicts to `ATRParams` before the engine can discard extra fields, preventing graceful handling of optional overlays.【F:src/models/atr_breakout.py†L22-L32】【F:src/backtest/engine.py†L416-L423】
* **Shared state assumptions:** The engine caches indicators (`roll_high`, `roll_low`, `atr`) keyed to breakout logic; no structures exist for Dip-specific state such as cooldown counters, so overlaying Dip would require new state management hooks.【F:src/backtest/engine.py†L551-L700】
* **UI binding:** Dip controls live in multiple UI sections and are always stored in Session State, so any backend refactor must keep field names consistent or migrate stored profiles carefully.【F:pages/2_Strategy_Adapter.py†L411-L446】【F:pages/2_Strategy_Adapter.py†L537-L566】【F:pages/2_Strategy_Adapter.py†L693-L740】

## 9. Refactor Readiness Checklist (for later integration)
1. **Parameter schema:** Introduce a Dip-specific dataclass or schema and adjust `run_strategy` to route Dip keys appropriately instead of forcing them into `ATRParams`.【F:src/models/atr_breakout.py†L22-L32】
2. **Engine factory:** Teach `backtest_atr_breakout` (or a wrapper) to interpret Dip overlays—e.g., allow an `entry_mode` switch that combines breakout and Dip signals without breaking existing behaviour.【F:src/backtest/engine.py†L551-L700】
3. **UI plumbing:** Consolidate Dip inputs (bounds + defaults) and ensure profiles saved via `save_strategy_params` continue to round-trip once backend types change.【F:pages/2_Strategy_Adapter.py†L411-L446】【F:pages/2_Strategy_Adapter.py†L537-L566】【F:pages/2_Strategy_Adapter.py†L1118-L1154】
4. **EA bridge:** Add a transformer before `_eval_one` that splits overlay-specific keys so each strategy wrapper receives only the fields it understands.【F:src/optimization/evolutionary.py†L360-L478】
5. **Metrics attribution:** Extend the metadata payload to record Dip signal activity (cooldown hits, confirmation usage) so fitness can distinguish overlay contributions.【F:src/backtest/engine.py†L488-L500】
6. **Logging/tests:** Update `TrainingLogger` consumers and any stored EA logs to include the new schema without breaking historical viewers.【F:pages/2_Strategy_Adapter.py†L1048-L1180】【F:src/utils/training_logger.py†L1-L30】

## 10. Appendix A — Parameter Table (BtD)
| Param | Type | Default | Range (if any) | Defined in | Consumed in | Notes |
|-------|------|---------|----------------|------------|-------------|-------|
| `entry_mode` | str | `"breakout"` | radio toggle | UI defaults / radio【F:pages/2_Strategy_Adapter.py†L411-L446】【F:pages/2_Strategy_Adapter.py†L568-L579】 | Used only for UI branching, leaderboard tagging【F:pages/2_Strategy_Adapter.py†L1118-L1154】 | Controls whether Dip params enter EA. |
| `trend_ma` | int | 200 | 20–600 | Defaults / Dip expander【F:pages/2_Strategy_Adapter.py†L411-L446】【F:pages/2_Strategy_Adapter.py†L693-L703】 | Sent to EA/strategy dict【F:pages/2_Strategy_Adapter.py†L915-L939】 | Not recognised by ATR engine. |
| `dip_atr_from_high` | float | 2.0 | 0–20 | Same as above【F:pages/2_Strategy_Adapter.py†L411-L446】【F:pages/2_Strategy_Adapter.py†L704-L711】 | EA/strategy dict【F:pages/2_Strategy_Adapter.py†L915-L939】 | Unused extra param today. |
| `dip_lookback_high` | int | 60 | 5–600 | Same as above【F:pages/2_Strategy_Adapter.py†L411-L446】【F:pages/2_Strategy_Adapter.py†L712-L718】 | EA/strategy dict【F:pages/2_Strategy_Adapter.py†L915-L939】 | Unused extra param today. |
| `dip_rsi_max` | float | 55.0 | 0–100 | Same as above【F:pages/2_Strategy_Adapter.py†L411-L446】【F:pages/2_Strategy_Adapter.py†L720-L726】 | EA/strategy dict【F:pages/2_Strategy_Adapter.py†L915-L939】 | Unused extra param today. |
| `dip_confirm` | bool | False | checkbox (EA bounds 0/1) | Same as above【F:pages/2_Strategy_Adapter.py†L411-L446】【F:pages/2_Strategy_Adapter.py†L728-L732】 | EA/strategy dict【F:pages/2_Strategy_Adapter.py†L915-L939】 | Represented as bool in UI, ints in EA bounds. |
| `dip_cooldown_days` | int | 5 | 0–240 | Same as above【F:pages/2_Strategy_Adapter.py†L411-L446】【F:pages/2_Strategy_Adapter.py†L733-L739】 | EA/strategy dict【F:pages/2_Strategy_Adapter.py†L915-L939】 | Cooling-off concept not implemented in engine. |

## 11. Appendix B — Offending Kwargs Table
| Kwarg | Provided by | Present in `ATRParams`? | Status |
|-------|-------------|--------------------------|--------|
| `trend_ma` | EA genome when Dip active【F:pages/2_Strategy_Adapter.py†L915-L939】 | No (`ATRParams` fields listed)【F:src/backtest/engine.py†L358-L376】 | **Unexpected** – raises TypeError. |
| `dip_atr_from_high` | Same as above | No | **Unexpected** (would be dropped if dict reached engine). |
| `dip_lookback_high` | Same as above | No | **Unexpected**. |
| `dip_rsi_max` | Same as above | No | **Unexpected**. |
| `dip_confirm` | Same as above | No | **Unexpected**. |
| `dip_cooldown_days` | Same as above | No | **Unexpected**. |

## 12. Appendix C — Grep/Find Commands
```
rg -n "buy.?the.?dip|dip_|trend_ma|mean.?reversion|btd" src/ pages/
rg -n "ATRParams|@dataclass class ATRParams|def __init__" src/
rg -n "Strategy_Adapter|DIP|dip" pages/
rg -n "evolutionary_search|fitness|bounds|genome" src/
```
