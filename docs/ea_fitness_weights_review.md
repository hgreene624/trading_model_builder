# EA Fitness Weight Review

## Storage and Load Behavior
- Overrides are stored in `storage/config/ea_fitness.json` and consumed at runtime by the evolutionary fitness pipeline.
- On startup, the optimizer reads this JSON, logs the applied parameters, and uses them when blending normalized objective metrics with capped penalties.
- Editing the JSON by hand is the supported workflow for adjusting the evolutionary search behavior.

## Current Weight Configuration
| Parameter | Value | Role |
| --- | --- | --- |
| `alpha_cagr` | 0.4835 | Weight on normalized CAGR (annualized growth). |
| `beta_calmar` | 0.0011 | Weight on normalized Calmar ratio (drawdown sensitivity). |
| `gamma_sharpe` | 0.0124 | Weight on normalized Sharpe ratio (risk-adjusted return). |
| `delta_total_return` | 0.5029 | Weight on normalized total return. |
| `calmar_cap` | 3.0 | Caps the normalized Calmar contribution. |
| `holding_penalty_weight` | 0.05 | Scales penalty for violating holding-period bounds. |
| `trade_rate_penalty_weight` | 0.02 | Scales penalty for violating trade-rate band. |
| `penalty_cap` | 0.15 | Maximum combined penalty deduction. |
| `min_holding_days` | 3.0 | Lower bound before incurring holding penalty. |
| `max_holding_days` | 30.0 | Upper bound before incurring holding penalty. |
| `trade_rate_min` | 5.0 | Lower bound before trade-rate penalty triggers. |
| `trade_rate_max` | 50.0 | Upper bound threshold (currently inactive). |
| `rate_penalize_upper` | false | Disables penalties for exceeding `trade_rate_max`. |
| `elite_by_return_frac` | 0.10 | Fraction of top performers kept by raw return during selection. |

## Impact of Manual Adjustments
- **Return emphasis (`alpha_cagr`, `delta_total_return`)**: Increasing either raises preference for high-growth candidates; lowering them unlocks space for risk metrics.
- **Risk weighting (`beta_calmar`, `gamma_sharpe`)**: Raising these values amplifies drawdown and volatility discipline; current near-zero Calmar weight makes drawdown control effectively irrelevant.
- **Penalty weights and caps**: Higher `holding_penalty_weight` or `trade_rate_penalty_weight` quickly suppress rule-breaking strategies until the `penalty_cap` is reached; reducing the cap allows penalties to dominate faster.
- **Band definitions**: Tightening `min_holding_days`/`max_holding_days` or `trade_rate_min`/`trade_rate_max` narrows acceptable behavior; enabling `rate_penalize_upper` enforces the upper trade-rate ceiling symmetrically.
- **Elite fraction**: Larger `elite_by_return_frac` preserves more top-return solutions; shrinking it accelerates turnover and exploration.

## Observations & Recommendations
1. **Return-heavy blend**: With nearly 1.0 combined weight on CAGR/total return, the search deprioritizes risk controls. Consider shifting 10–20% of the weight toward Sharpe and/or Calmar to balance risk-adjusted performance.
2. **Calmar underweighting**: The `beta_calmar` weight (0.0011) is effectively negligible. If drawdown matters, raise it closer to Sharpe's weight or enforce a higher `calmar_cap` to reduce tolerance for deep losses.
3. **Asymmetric trade penalties**: Because `rate_penalize_upper` is `false`, only under-trading is penalized. Enable the upper bound or increase `trade_rate_min` if over-trading is a concern.
4. **Penalty slack**: The 0.15 `penalty_cap` combined with modest weights allows rule violations to persist. Tightening the cap or raising the weights will make holding/trade constraints more influential.
5. **Monitoring edits**: After manual adjustments, re-run the evolutionary search and review logged metric blends to verify the optimizer behaves as intended.
