# EA Fitness Weight Review

## Storage and Load Behavior
- Overrides are stored in `storage/config/ea_fitness.json` and consumed at runtime by the evolutionary fitness pipeline.
- The JSON now nests related parameters by impact (return emphasis, penalties, bands, holdout) and the loader flattens them when applying overrides.
- On startup, the optimizer reads this JSON, logs the applied parameters, and uses them when blending normalized objective metrics with capped penalties.
- Editing the JSON by hand is the supported workflow for adjusting the evolutionary search behavior.

## Current Weight Configuration

### Return Emphasis (`return_emphasis`)
| Parameter | Value | Role |
| --- | --- | --- |
| `use_normalized_scoring` | true | Enables metric normalisation so weight magnitudes map predictably to impact. |
| `alpha_cagr` | 0.535 | Weight on normalized CAGR (annualised growth). |
| `beta_calmar` | 0.11 | Weight on normalized Calmar ratio (drawdown sensitivity). |
| `gamma_sharpe` | 0.124 | Weight on normalized Sharpe ratio (risk-adjusted return). |
| `delta_total_return` | 0.4029 | Weight on normalized total return. |
| `calmar_cap` | 3.0 | Caps the normalized Calmar contribution before weighting. |

### Execution Discipline Penalties (`execution_penalties`)
| Parameter | Value | Role |
| --- | --- | --- |
| `holding_penalty_weight` | 0.05 | Scales penalty for violating holding-period bounds. |
| `trade_rate_penalty_weight` | 0.02 | Scales penalty for violating trade-rate band. |
| `penalty_cap` | 0.45 | Maximum combined penalty deduction. |

### Holding Window Preferences (`holding_windows`)
| Parameter | Value | Role |
| --- | --- | --- |
| `min_holding_days` | 3.0 | Lower bound before incurring holding penalty. |
| `max_holding_days` | 30.0 | Upper bound before incurring holding penalty. |
### Trade Rate Band (`trade_rate_band`)
| Parameter | Value | Role |
| --- | --- | --- |
| `trade_rate_min` | 5.0 | Lower bound before trade-rate penalty triggers. |
| `trade_rate_max` | 50.0 | Upper bound threshold when upper-band penalties are active. |
| `rate_penalize_upper` | false | Disables penalties for exceeding `trade_rate_max`. |

### Holdout Protection (`holdout_protection`)
| Parameter | Value | Role |
| --- | --- | --- |
| `holdout_score_weight` | 0.65 | Blend weight for holdout/test metrics when both windows are available. |
| `holdout_gap_tolerance` | 0.15 | Margin allowed before penalising train vs. test score gaps. |
| `holdout_gap_penalty` | 0.50 | Penalty multiplier when the training score materially exceeds the holdout. |
| `holdout_shortfall_penalty` | 0.35 | Penalty multiplier when the training score materially lags the holdout. |

## Impact of Manual Adjustments
- **Return emphasis (`alpha_cagr`, `delta_total_return`)**: Increasing either raises preference for high-growth candidates; lowering them unlocks space for risk metrics.
- **Risk weighting (`beta_calmar`, `gamma_sharpe`)**: Raising these values amplifies drawdown and volatility discipline; current near-zero Calmar weight makes drawdown control effectively irrelevant.
- **Penalty weights and caps**: Higher `holding_penalty_weight` or `trade_rate_penalty_weight` quickly suppress rule-breaking strategies until the `penalty_cap` is reached; reducing the cap allows penalties to dominate faster.
- **Band definitions**: Tightening `min_holding_days`/`max_holding_days` or `trade_rate_min`/`trade_rate_max` narrows acceptable behavior; enabling `rate_penalize_upper` enforces the upper trade-rate ceiling symmetrically.
- **Elite fraction**: Survivor mixing is now governed strictly by the EA runtime parameters; the JSON config no longer overrides `elite_by_return_frac`.

## Observations & Recommendations
1. **Return-heavy blend remains dominant**: CAGR and total return still represent ~94% of the total scoring weight, so risk ratios remain secondary. Rebalancing toward Sharpe/Calmar would align selection with drawdown tolerance.
2. **Calmar is no longer negligible but still light**: Raising `beta_calmar` to 0.11 gives drawdown awareness, yet it trails Sharpe. Further increases or a lower `calmar_cap` would tighten downside control.
3. **Penalty authority improved**: With `penalty_cap` at 0.45, the system can meaningfully suppress rule breakers, but the low penalty weights mean only repeated violations move the needle. Increase the weights if constraints must bite faster.
4. **Upper trade rate still unchecked**: `rate_penalize_upper=false` leaves high-frequency behaviour unpenalised. Flip it to `true` or lower `trade_rate_max` if turnover is a concern.
5. **Holdout blend encourages generalisation**: The 0.65 weight and paired penalties prioritise holdout integrity. Adjust all three holdout knobs in tandem when tuning to avoid overfitting back to the training window.
6. **Grouped config keeps intent explicit**: We now organise the JSON into intent-based blocks (`return_emphasis`, `execution_penalties`, etc.). This structure helps reviewers tune related knobs together; ensure any downstream tools flatten nested keys the way the runtime loader does.
