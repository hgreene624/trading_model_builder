# Cost Model Status — Phase 1.2 Audit

## Executive Summary

- Phase 1.2 cost knobs are wired through the engine: environment-backed `CostModel.from_env` values, ATR-based slippage, and per-run metadata surfaced for downstream tools. 【F:src/backtest/engine.py†L237-L352】【F:src/backtest/engine.py†L1083-L1146】
- Harness and troubleshooting utilities respect CLI > ENV precedence, capture the effective knobs, and write structured artifacts without mutating the caller’s environment. 【F:scripts/run_smoke_backtest.py†L30-L156】【F:scripts/run_smoke_backtest.py†L191-L552】【F:src/utils/troubleshoot_costs.py†L110-L333】
- Latest troubleshoot run could not source SPY data (blocked network/API credentials), so slippage/drag metrics remain unavailable; the tools still recorded env inputs and surfaced the failure. 【ce008d†L1-L22】【F:storage/logs/troubleshoot_costs_latest.json†L1-L20】
- Next: harden precedence/unit tests and add cached data or mocks so smoke/troubleshoot runs can quantify slippage/drag without external connectivity. 【F:tests/smoke/test_costmodel.py†L1-L60】【ce008d†L1-L22】

## What’s Implemented (with references)

### Engine & Cost Model

- Environment-driven constructor honors Phase 1.2 knobs (`COST_ENABLED`, `COST_ATR_K`, `COST_MIN_HS_BPS`, `COST_USE_RANGE_IMPACT`, `CAP_RANGE_IMPACT_BPS`) and normalizes inputs. 【F:src/backtest/engine.py†L237-L259】
- ATR-scaled slippage path computes `base_bps = ATR% * atr_k`, applies a floor via `_estimate_half_spread_bps`, and respects the range-impact cap before adjusting fill prices. 【F:src/backtest/engine.py†L309-L352】
- Trades inherit either CLI-provided simple costs or the Phase 1.2 env model, with results logged in `meta['cost_inputs']` for downstream reconciliation. 【F:src/backtest/engine.py†L801-L840】【F:src/backtest/engine.py†L1136-L1146】

### Metrics & Storage

- `summarize_costs` computes turnover, weighted slippage/fees (bps), and annualized drag (bps) from pre/post curves. 【F:src/backtest/metrics.py†L200-L242】
- Engine-level summaries augment metrics with total cost aggregates and expose both annualized drag (decimal) and `annualized_drag_bps`. 【F:src/backtest/engine.py†L1098-L1134】

### Harness & Troubleshooter

- Smoke harness enforces CLI > ENV precedence, restores the original environment via `_temporary_env`, records cost inputs/metrics into `artifacts/smoke/.../summary.json`, and uses `.ffill()` aggregation to silence prior warnings. 【F:scripts/run_smoke_backtest.py†L115-L552】
- Troubleshooter snapshots the active env, runs scenario variants, and prints `ENV used per run` alongside tabular slippage/drag diagnostics while persisting JSON. 【F:src/utils/troubleshoot_costs.py†L110-L333】

### Tests

- Smoke tests validate range-impact estimation, buy/sell application of the ATR-aware cost model, and disabled-cost fallbacks. 【F:tests/smoke/test_costmodel.py†L1-L60】

## What’s Incomplete or Risky

| Gap | Risk | Suggested Remediation |
| --- | --- | --- |
| No automated coverage for CLI>ENV precedence or `meta['cost_inputs']` integrity; current tests only hit helpers. 【F:tests/smoke/test_costmodel.py†L1-L60】【F:scripts/run_smoke_backtest.py†L191-L551】 | Medium – regressions could silently revert to defaults without detection. | Add harness integration tests that assert `summary['inputs']` and engine `meta['cost_inputs']` equal the resolved knobs for representative combinations; require failure if precedence breaks. |
| `meta['costs']['summary']` mixes decimal `annualized_drag` and bps `annualized_drag_bps`, risking downstream unit confusion. 【F:src/backtest/engine.py†L1098-L1134】 | Medium – consumers may double-convert or misreport drag. | Consolidate on bps (or document both) and extend troubleshoot JSON to label units explicitly; add a regression test that reads the summary and checks expected magnitudes. |
| Troubleshooter/smoke runs depend on live data; failures leave NaNs with no cached baseline for impact tracking. 【ce008d†L1-L22】【F:storage/logs/troubleshoot_costs_latest.json†L1-L20】 | High – cannot verify cost drag/slippage without network/API access. | Ship minimal cached OHLCV fixtures or allow an offline fixture mode so Phase 1.2 metrics are always populated; acceptance: troubleshoot run yields finite slip/drag for SPY & QQQ without network. |
| Range-impact cap toggle lacks regression coverage ensuring `use_range_impact=0/1` and cap bounds change slippage as intended. 【F:src/backtest/engine.py†L309-L352】 | Medium – future refactors could break the max-of logic. | Add parametrized tests that sweep ATR %, range, and cap values verifying slip_bps outcomes (with/without toggle). |

## Observed Impact on Model Behavior

| Scenario | Slip (bps) | Annualized Drag (bps) | Turnover Ratio | Trades | Blocks | Fees (bps) | Cost Inputs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SPY, 2023-01-01 → 2023-06-30 (base) | N/A (data unavailable) | N/A (data unavailable) | N/A | 0 | error | N/A | `atr_k=0.05`, `min_half_spread_bps=0.5`, `use_range_impact=0`, `cap_range_impact_bps=10` |

The troubleshoot harness executed but could not download SPY due to blocked credentials/network, leaving the scenario with NaN cost metrics while still recording the requested env knobs. 【ce008d†L1-L22】【F:storage/logs/troubleshoot_costs_latest.json†L1-L20】 Without historical baselines, we cannot confirm that slippage now falls in the expected 5–15 bps range or that drag improved versus Phase 1.1.

Alpha Retention % (gross vs. net Sharpe/CAGR) is not derivable from current artifacts because troubleshoot outputs omit gross-vs-net Sharpe/CAGR pairs.

## Recommendations & Next Steps

1. **Precedence & Inputs Integrity Tests** — Add integration tests around the smoke harness to assert that CLI overrides win over env defaults and that engine `meta['cost_inputs']` mirrors the resolved knobs (pass when equality holds across a matrix of overrides). Scope: `scripts/run_smoke_backtest.py` + `src/backtest/engine.py`. Risk: Medium.
2. **Clarify Drag Units & Surface Alpha Retention** — Normalize cost summaries to a single unit for drag and extend troubleshoot/smoke outputs with gross vs. net Sharpe/CAGR so alpha retention can be computed (pass when troubleshoot JSON exposes both net/gross and unit docs). Risk: Medium.
3. **Offline Fixture Mode for Cost Harnesses** — Bundle cached OHLCV samples (e.g., SPY/QQQ) or stub loader hooks to enable deterministic cost metrics without external APIs (pass when troubleshoot & smoke runs succeed offline with finite slip/drag/turnover). Risk: High.
4. **Range-Impact Regression Coverage** — Add targeted unit tests verifying `_estimate_half_spread_bps` + `_apply_costs` respect `use_range_impact` and cap thresholds across edge cases (pass when tests fail if slip_bps ignores the cap or toggle). Risk: Medium.

## Appendix

- **Reviewed files:** `src/backtest/engine.py`, `src/backtest/metrics.py`, `src/models/atr_breakout.py`, `src/models/general_trainer.py`, `scripts/run_smoke_backtest.py`, `src/utils/troubleshoot_costs.py`, `tests/smoke/test_costmodel.py`. 【F:src/backtest/engine.py†L237-L1146】【F:src/backtest/metrics.py†L200-L242】【F:src/models/atr_breakout.py†L1-L86】【F:src/models/general_trainer.py†L1-L140】【F:scripts/run_smoke_backtest.py†L30-L552】【F:src/utils/troubleshoot_costs.py†L110-L333】【F:tests/smoke/test_costmodel.py†L1-L60】
- **Artifacts:** `storage/logs/troubleshoot_costs_latest.json`. 【F:storage/logs/troubleshoot_costs_latest.json†L1-L20】
- **Commands executed:** `python src/utils/troubleshoot_costs.py --tickers SPY --start 2023-01-01 --end 2023-06-30 --output storage/logs/troubleshoot_costs_latest.json`; `pytest -q tests/smoke/test_costmodel.py`. 【ce008d†L1-L22】【111785†L1-L3】
