# July 2024 Holdout Drawdown Analysis

## Summary
Model Builder trainings that target the `NASDAQ_10` portfolio repeatedly show a sharp equity dip during July 2024. This is not a simulator bug—the holdout window used across the recent EA runs always spans mid-2024, when most of the portfolio constituents sold off 10–25%. The same market regime therefore reappears in every training session, producing the consistent drawdown that appears on the holdout chart.

## Holdout configuration
Every EA session recorded under `storage/logs/ea/` during the recent work captured a holdout that begins either on 29 March 2024 or 26 April 2024 and runs into late 2025. Regardless of the exact start, the July 2024 sell-off is always part of the evaluation period, so any long-biased breakout configuration is forced to trade through that regime.【3b7f96†L1-L20】

## Price action during July 2024
The ten symbols inside `NASDAQ_10` all suffered notable declines between 15 July and 9 August 2024. Five of the high-beta names (AMD, ABNB, GOOGL, GOOG, AMZN) fell 12–25%, while even stalwarts like AAPL shed almost 8%. AEP was the lone gainer, so the equal-weighted basket still lost roughly 10.4% over the window.【7f8236†L1-L11】 Sample OHLC data from the Alpaca cache shows the same slump, e.g. AMD sliding from $174 on 8 July to the $138 area by 30 July.【4a67d8†L9-L18】【4a67d8†L21-L30】

### Why the SPY benchmark looks calmer
The SPY holdout benchmark does not exhibit the same cliff because its sector mix spreads the July shock across 500 names instead of concentrating in mega-cap tech. The cached Alpaca data shows the equal-weighted `NASDAQ_10` basket dropping about 9.3% peak-to-trough between 1 July and 7 August 2024, whereas SPY fell only ~5.1% before stabilizing. By mid-August the SPY drawdown had nearly retraced, but the tech-heavy portfolio remained down ~5.6% as the weakest members (ABNB, AMD, AMZN, the GOOG* twins) were still 13–25% below their July highs while the defensive utility AEP barely budged.【1f786b†L1-L11】 In other words, the recurring loss is a concentration problem: the EA keeps relearning how to trade a basket whose holdout period coincides with a tech-specific correction that the broader S&P 500 avoided.

## What the EA logs show
Because that sell-off dominates the holdout window, many individuals evaluated by the EA show negative test returns and large drawdowns even when their in-sample performance looked acceptable. For example, several generation-0 candidates report `test_metrics['total_return']` between -4% and -7% and `max_drawdown` worse than -17% to -23%.【d962d5†L6-L147】 Later generations manage the damage better—the final best candidate from 6 Oct 2025 still records a maximum drawdown of about -6.9% while finishing the holdout up 17.7%, but that dip is a direct imprint of the July regime.【9c3b9a†L1-L2】

## Implications
Until the training data window or the portfolio composition changes, every EA search will continue to experience that July 2024 shock. Possible mitigations include:

- Incorporating symbols or hedges that diversify the heavy mega-cap tech exposure.
- Allowing the EA to short or hold cash so it can sidestep the sell-off.
- Shifting the holdout window (or adding multiple rolling windows) to confirm the behaviour is localized to that event.

Understanding that the drawdown is data-driven rather than a simulation failure should make it easier to reason about strategy tweaks.

### How parameter sets could adapt
Because the EA currently optimizes long-only, breakout-oriented behaviours, most parameter sets are forced to stay fully invested as prices roll over. To blunt the July drawdown without altering the data, the model would need knobs that reward de-risking once the sell-off starts. Examples include:

- **Volatility-sensitive position sizing.** If the engine allowed ATR- or variance-based scaling, a parameter set that targets constant volatility (e.g., reduce gross exposure when 14-day ATR spikes) would naturally cut risk as July’s range expansion appears.
- **Trend or regime filters.** Adding moving-average alignment, slope constraints, or breadth filters as tunable parameters would let the EA exit—or never enter—longs when the basket flips into a short-term downtrend.
- **Stop or time-based exits.** Permitting hard stops, trailing stops, or “maximum bar hold” rules would hand the search space a direct way to truncate losers instead of riding them through the trough.
- **Cash or hedge allocation.** If the strategy template exposed cash targets or inverse/market-neutral hedges (e.g., short QQQ, long VIX calls) as parameters, the EA could discover defensive combinations for that regime.

In short, no single parameter tweak guarantees a win, but expanding the parameter space to include volatility targeting, trend filters, explicit exits, or hedging levers would give the EA the theoretical ability to surface parameter sets that sidestep most of the July 2024 damage.
