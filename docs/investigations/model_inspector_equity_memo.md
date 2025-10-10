# Model Inspector Equity Flattening — Root Cause Memo

## Summary
*The Model Inspector is replaying EA candidates with the new risk/reward sizing defaults, while the stored EA metrics were produced before those knobs existed.*

Older EA logs only record the classic ATR parameters (breakout/exit lengths, ATR multiple, etc.). When the Inspector regenerates curves it instantiates `ATRParams(**payload)`, so any missing keys fall back to the current defaults (`size_mode="rr_portfolio"`, 0.5% base allocation, 5% cap).【F:src/models/atr_breakout.py†L25-L75】【F:src/backtest/engine.py†L451-L475】 The single-symbol backtests therefore run with tiny funded notionals and produce equity paths that wiggle by only a few dollars, even though the EA session originally sized positions as if they controlled the full account. The discrepancy shows up because the Inspector plots the newly generated curve, yet it labels it with the historical metrics (`total_return≈0.65`) that were computed under the legacy sizing path.【F:pages/3_Model_Inspector.py†L848-L904】【F:storage/logs/ea/20250926_130005_ea.jsonl†L1-L20】

## Data Flow Trace
1. **Inspector request** – `run_equity_curve(...)` is called for train/test windows when rendering the Model Inspector charts.【F:pages/3_Model_Inspector.py†L848-L940】
2. **Trainer replay** – That helper tries `train_general_model(...)`, which loops each symbol, runs `run_strategy`, and builds an equal-weight normalized equity curve (`aggregate_equity`).【F:src/models/general_trainer.py†L177-L279】
3. **Strategy runner** – `run_strategy` converts the persisted params dict into `ATRParams`. Missing sizing knobs therefore adopt the new defaults declared in `ATRParams` (portfolio-aware mode, small fractions).【F:src/models/atr_breakout.py†L25-L75】【F:src/backtest/engine.py†L451-L475】
4. **Equity formation** – Inside the engine, trade returns (`return_pct`) are still expressed as price-relative P&L, independent of account leverage.【F:src/backtest/engine.py†L1340-L1408】 The Inspector’s trade table and heatmaps therefore look numerically reasonable even though the regenerated equity curve reflects the down-sized notionals.
5. **Holdout/UI divergence** – The Strategy Adapter’s holdout tooling now ration capital across concurrent trades when computing portfolio equity.【F:pages/2_Model_Builder.py†L1182-L1556】 Those series are consistent because they use the funded notionals emitted by the updated engine. The Inspector, however, replays the older EA sessions without those additional sizing fields, so it silently switches behaviour at replay time.

## Hypotheses Review
- **H1 – Wrong series plotted**: *Ruled out.* Inspector still plots the regenerated aggregate equity from `train_general_model`; no alternate series substitution occurs.【F:pages/3_Model_Inspector.py†L848-L904】【F:src/models/general_trainer.py†L177-L279】
- **H2 – Double normalization**: *Ruled out.* `aggregate_equity` is normalized once and re-scaled by `starting_equity`, just as before.【F:src/models/general_trainer.py†L211-L233】 The muted moves stem from smaller funded returns, not extra normalization.
- **H3 – Percent vs ratio mix-up**: *Ruled out.* Trade `return_pct` and curve scaling are unchanged; the percentages simply act on much smaller notionals.【F:src/backtest/engine.py†L1340-L1408】
- **H4 – Per-share vs portfolio P&L**: *Partially relevant.* The Inspector exports per-trade `Net P&L` that is tied to share-level P&L, but the core issue is the replay using new sizing defaults rather than misinterpreting the per-trade fields.【F:src/backtest/engine.py†L1340-L1408】
- **H5 – Schema drift**: *Confirmed.* Legacy EA logs omit the new sizing keys; replay-time defaults therefore shift to the new sizing regime, flattening the regenerated curve.【F:storage/logs/ea/20250926_130005_ea.jsonl†L1-L20】【F:src/models/atr_breakout.py†L25-L75】
- **H6 – Train/test join**: *Ruled out.* The concatenation still follows the historical split; the scale issue appears even when viewing the training segment in isolation.【F:pages/3_Model_Inspector.py†L990-L1018】
- **H7 – Excess index subtraction**: *Ruled out.* The Excess Index panel still uses the normalized curve derived from the regenerated equity; no extra subtraction happens upstream.【F:pages/3_Model_Inspector.py†L1300-L1466】

## Ambiguities & Gaps
- The prompt referenced `/mnt/data/...` artifacts, but that mount is absent in this environment. Confirming with those exact datasets would further quantify the magnitude mismatch.
- It is unclear whether future EA runs will log the new sizing parameters. Once they do, the Inspector should replay faithfully; the current issue is limited to legacy logs.

## Minimal Verification Plan
Toggle-able snippets (e.g., behind a `st.checkbox("Debug sizing drift")`) that can be added temporarily to the Inspector:

```python
if debug_sizing_drift:
    params_logged = _row_params(selected_row)
    params_payload = _strategy_params_payload(strategy, params_logged, disable_warmup=True)
    missing = sorted(set(params_payload) - set(params_logged))
    st.write({
        "logged_keys": sorted(params_logged.keys()),
        "payload_defaults": {k: params_payload[k] for k in missing},
    })
```

Expected output on a legacy log: the `payload_defaults` dict will list `size_mode`, `size_base_fraction`, etc., proving the replay is injecting the new defaults.

A complementary assertion can verify that replayed equity ends near the logged return when those defaults are overridden:

```python
replayed = run_equity_curve(...)
reported_ret = selected_row["total_return"]
nav_logged = 1.0 + reported_ret
nav_replayed = float(replayed["equity"].iloc[-1] / replayed["equity"].iloc[0])
st.write({"logged": nav_logged, "replayed": nav_replayed})
```

Enabling the inspector to backfill the missing sizing fields with the legacy semantics should bring `nav_replayed` back in line with the reported return.

## Proposed Fix Outline (not yet implemented)
1. **Backfill sizing knobs during replay** – When `_row_params` returns a dict that lacks `size_mode`, inject `"legacy"` (and optionally the other sizing weights) so replays match historical behaviour. Alternatively, detect the absence and warn the user before plotting.【F:src/models/atr_breakout.py†L25-L75】
2. **Persist sizing parameters going forward** – Ensure EA logging includes the new sizing knobs so future runs replay faithfully and inspectors of mixed-era logs can branch on the presence/absence of those fields.【F:storage/logs/ea/20250926_130005_ea.jsonl†L1-L20】
3. **UI guardrail** – Surface a banner when `payload_defaults` introduces any `size_*` or `leverage_cap` fields so users know the regenerated equity differs from the logged metrics.【F:pages/3_Model_Inspector.py†L848-L940】

These are small, localized changes that avoid refactoring core engine code while restoring consistency between the Inspector’s visuals and the EA metrics.
