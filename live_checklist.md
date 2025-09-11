# Live Trading Checklist

## Risk guardrails
- **Max drawdown budget:** ≤ 12–18%. If > 18% → pause & review.
- **Risk per trade:** 0.5–0.8% of equity.
- **Per-ticker cap:** ≤ 25–33% of equity.

## Return & quality goals (first 6–12 months, $1,000 start)
- **Green zone:** +6–12% annualized, Sharpe 0.6–0.9, ≥ 80–150 total trades.
- **Stretch:** +12–20% annualized, Sharpe 0.9–1.2.
- If long-only: target SPY +2–5%/yr with similar or lower DD.

## Consistency / robustness
- **Live vs OOS drift:** |ΔSharpe| ≤ 0.2; live equity within ±20% of OOS projection.
- **Validation:** per-ticker ≥ 8–12 trades; portfolio ≥ 50–100 trades over 6–12 months.

## Process KPIs (track weekly)
- **Execution slippage:** avg ≤ 10–20 bps; tails < 50 bps.
- **Signal compliance:** ≥ 98–99% of model signals executed.
- **Cost share:** (commissions+slippage)/gross P&L < 25–35%.
- **Data/latency errors:** none recurring; annotate anomalies.

## Scale-up rules
- After ≥ 100 trades and 3 consecutive months in Green → scale to $5k–$10k (same risk %).
- If DD > 18%, Sharpe < 0.2 over last 60–80 trades, or trade count collapses → pause and re-tune.