# Model Builder Visualization Enhancements

The existing holdout equity curve normalizes per-symbol P\&L streams by the first non-zero observation. When many strategies
report a zero balance until the first closing trade, this produces a flat "no trades" segment at the start of the chart. To give
traders a clearer picture of out-of-sample behaviour, consider replacing or augmenting that chart with the following views.

## 1. Starting-Equity Anchored Equity Curve
- Anchor the equity curve to the configured starting equity (or the first timestamp) rather than the first non-zero trade.
- Plot both the strategy equity and the benchmark on the same axes to highlight relative performance from day one.
- Optional: add shading for periods where the strategy is outperforming/underperforming the benchmark.

## 2. Rolling Performance Heatmap
- Render a 2D heatmap where the x-axis is time and the y-axis is a rolling window length (e.g., 5, 10, 20, 60 trading days).
- Each cell shows the annualized return or Sharpe ratio for that window, revealing momentum regimes at a glance.
- Useful for spotting when the strategy repeatedly stalls or accelerates in the holdout.

## 3. Trade Lifecycle Timeline
- Plot each fill as a horizontal bar spanning entry to exit, positioned vertically by symbol or strategy leg.
- Encode profit/loss with color and bar thickness with position size.
- Provides intuition on trade overlap, holding periods, and whether inactivity stems from filters or market conditions.

## 4. Drawdown vs. Exposure Scatter
- Each point represents a trading day (or hour) with x = net exposure (% of capital deployed) and y = current drawdown.
- Helps diagnose whether large drawdowns coincide with maxed-out exposure or leverage.
- Can be extended with marginal histograms to emphasize distribution tails.

## 5. Cumulative Net Exposure Chart
- Stack area chart showing long, short, and cash allocations through time.
- When paired with the equity curve, clarifies whether flat performance stems from being flat/cash or from losing trades.

## 6. Trade Frequency & Hit Rate Dashboard
- Combine two small multiples: (a) rolling trade count per period, (b) rolling win rate / expectancy.
- Signals whether the strategy is encountering fewer opportunities or simply underperforming on existing trades.

## Layout Suggestions
- **Two-column layout:** equity curve (anchored) on the left, dashboard of rolling stats on the right.
- **Tabbed interface:** separate tabs for equity, trade lifecycle, and diagnostics to avoid clutter while keeping related charts a single click away.
- **Small multiples:** align charts vertically with shared x-axis (time) for quick cross-comparison.

Implementing even one of these replacements will help traders understand whether a flat holdout segment reflects inactivity,
risk management, or simply delayed trade closures.
