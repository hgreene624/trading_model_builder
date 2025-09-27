# Strategy Adapter Model Inputs

This note summarizes every input on the **Strategy Adapter** page that directly feeds the training and evaluation stack. Use it as a plain-language companion when you are configuring runs or explaining the workflow to collaborators.

## Data & Context Controls

| Control | What it does | Why it matters |
| --- | --- | --- |
| **Portfolio** | Loads a saved list of tickers and normalizes the symbols before training begins.【F:pages/2_Strategy_Adapter.py†L200-L238】 | The evolutionary search evaluates every candidate parameter set across the selected symbols. Broader portfolios increase runtime but produce more robust fitness estimates because metrics such as trades, CAGR, and drawdown are aggregated across more markets.【F:src/models/general_trainer.py†L26-L104】【F:src/optimization/evolutionary.py†L219-L320】 |
| **Strategy module** | Lets you pick the strategy wrapper that exposes `run_strategy`. The page currently offers `src.models.atr_breakout`.| The module determines which parameter names are valid and which backtest logic the EA explores.【F:pages/2_Strategy_Adapter.py†L238-L240】【F:src/models/atr_breakout.py†L1-L27】 |
| **Training window** | The page automatically defines `start = now - 365 days` and `end = now` each time you click **Train**.【F:pages/2_Strategy_Adapter.py†L526-L532】 | Every evolutionary evaluation uses this lookback. Adjusting the window in code changes how many bars the backtests consume, which affects the stability of metrics such as drawdown and CAGR.【F:src/optimization/evolutionary.py†L219-L320】 |
| **Starting equity** | Specifies the notional capital passed into each backtest run.【F:pages/2_Strategy_Adapter.py†L405-L406】【F:pages/2_Strategy_Adapter.py†L688-L702】 | Acts as the bankroll for position sizing inside `backtest_atr_breakout`. While the strategy currently allocates 100% of equity per trade, the equity curve, returns, and risk metrics scale with this input.【F:src/backtest/engine.py†L85-L140】 |
| **Jobs (processes)** | Sets the number of worker processes used during evolutionary evaluation.【F:pages/2_Strategy_Adapter.py†L410-L413】【F:pages/2_Strategy_Adapter.py†L688-L702】 | Higher parallelism shortens wall-clock time but increases data-loader pressure and CPU use. It maps directly to the `n_jobs` argument in `evolutionary_search`, which chooses between a `ProcessPoolExecutor` or single-process loop.【F:src/optimization/evolutionary.py†L219-L328】 |

## Evolutionary Search Knobs

These controls shape the evolutionary algorithm that samples, evaluates, and mutates strategy parameters.

| Control | How it is wired | Impact on training |
| --- | --- | --- |
| **Generations** | Passed to `evolutionary_search(..., generations=...)`.【F:pages/2_Strategy_Adapter.py†L283-L372】【F:pages/2_Strategy_Adapter.py†L688-L702】 | Determines how many evolutionary cycles run. More generations allow the population to refine toward higher scores at the cost of longer runtime.【F:src/optimization/evolutionary.py†L219-L360】 |
| **Population** | Passed as `pop_size` when invoking the search loop.【F:pages/2_Strategy_Adapter.py†L283-L372】【F:pages/2_Strategy_Adapter.py†L688-L702】 | Controls the number of candidate parameter sets explored per generation. Larger populations improve coverage of the parameter space but linearly increase evaluation cost.【F:src/optimization/evolutionary.py†L219-L320】 |
| **Min trades (gate)** | Sent to the fitness function via the `min_trades` argument.【F:pages/2_Strategy_Adapter.py†L283-L372】【F:pages/2_Strategy_Adapter.py†L688-L702】 | Any candidate whose aggregate trade count is below this threshold receives a zero fitness score, which filters out inactive configurations and prevents overfitting to small sample sizes.【F:src/optimization/evolutionary.py†L219-L360】 |
| **Jobs (EA)** | Overrides `ea_cfg["n_jobs"]`, which controls how many individuals the EA evaluates concurrently.【F:pages/2_Strategy_Adapter.py†L283-L372】【F:src/optimization/evolutionary.py†L303-L328】 | The setting throttles multiprocessing inside the EA itself. Use lower values when you hit API rate limits; raise it to accelerate local simulations. |

## Parameter Search Ranges (EA Inputs)

The sliders define the bounds for each tunable strategy parameter. The page clamps the bounds and feeds them into `param_space`, which the EA samples when creating or mutating individuals.【F:pages/2_Strategy_Adapter.py†L312-L573】 The underlying ATR breakout model interprets each field as documented below.【F:src/backtest/engine.py†L61-L132】 Narrow ranges shrink the search space (faster runs, less diversity) while wider ranges broaden exploration.

| Parameter | Strategy behavior | Training effect |
| --- | --- | --- |
| `breakout_n` | Lookback window used to compute the rolling high that triggers long entries.【F:src/backtest/engine.py†L61-L132】 | Higher values demand longer consolidations before entering, reducing trade frequency but often improving signal quality. Lower values react faster but can overtrade; the EA samples within the selected range to balance responsiveness and reliability. |
| `exit_n` | Lookback window for the rolling low that triggers exits.【F:src/backtest/engine.py†L61-L132】 | Tight exit windows (`exit_n` small) cut trades quickly, boosting win rate at the cost of premature exits. Larger windows allow trades more room, potentially increasing drawdowns. |
| `atr_n` | Period used in Wilder's ATR calculation.【F:src/backtest/engine.py†L73-L132】 | Longer ATR windows smooth volatility estimates, leading to wider stops and fewer signals. Shorter windows adapt faster but can whipsaw; the EA optimizes this trade-off. |
| `atr_multiple` | Multiplier applied to ATR in the entry threshold.【F:src/backtest/engine.py†L98-L132】 | Higher multiples require price to exceed the rolling high by a larger volatility-adjusted margin, filtering out noise but producing fewer trades. Lower multiples admit more breakouts and can improve responsiveness at the cost of more false signals. |
| `tp_multiple` | Optional profit target expressed in ATR units.【F:src/backtest/engine.py†L98-L132】 | Setting this above zero caps gains once the target is reached. Larger targets keep trades open longer; smaller targets realize profits quickly but may truncate big winners. |
| `holding_period_limit` | Maximum number of bars a position may remain open before a time-based exit fires.【F:src/backtest/engine.py†L98-L132】 | Lower limits enforce quick turnover, boosting sample size but potentially clipping trend-following trades. Higher limits let winners run but may tie up capital or allow reversals. |

## Session Defaults vs. Active Inputs

The **Base params** expander exposes additional fields (e.g., `risk_per_trade`, moving-average trend filter knobs). At present these values are stored in session state for future work but are **not** injected into the evolutionary param space, so they do not influence the current ATR breakout training loop.【F:pages/2_Strategy_Adapter.py†L241-L402】 Similarly, the `CV folds` and `Min trades (valid)` inputs are placeholders that are not forwarded to `evolutionary_search` yet.【F:pages/2_Strategy_Adapter.py†L403-L408】 Documenting this distinction helps prevent confusion when reviewing runs.

