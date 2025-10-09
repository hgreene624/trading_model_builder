# Training vs Holdout Behaviour

The Model Builder runs two different evaluation passes when a holdout range is supplied to the evolutionary search:

* **Training evaluations** run through `train_general_model`.  The trainer explicitly injects the `__disable_warmup_fetch__` flag into every strategy invocation so the backtest starts exactly on the requested training window without fetching any pre-history.  Because ATR-style indicators rely on long lookbacks, this means the early portion of the training sample trades with partially-initialised state, which can depress metrics during the first few months of the window.【F:src/models/general_trainer.py†L172-L177】【F:src/backtest/engine.py†L533-L667】
* **Holdout simulations** use the `_portfolio_equity_curve` helper when the UI refreshes charts.  That helper forwards the resolved parameters without forcing the warmup flag, so `backtest_atr_breakout` runs with `use_warmup=True` by default.  The engine therefore fetches the appropriate lookback before the March 2023 holdout window, allowing the strategy to begin trading immediately with fully-seeded indicators.【F:pages/2_Model_Builder.py†L918-L997】【F:src/backtest/engine.py†L533-L667】

During training the evolutionary algorithm also favours the holdout score when both training and testing ranges are present—the default blend applies a 65 % weight to the holdout metrics and only a 35 % bonus from the training score when it does not lag the test.  As a result, the “best scored” genome is the one that performs best on the March 2023+ holdout even if its pre-March training equity curve looks weaker.【F:src/optimization/evolutionary.py†L687-L1131】

These differences explain the apparent mismatch: the training chart shows a curve produced without warmup data that is also de-emphasised in the EA’s scoring, while the holdout chart reflects a fully-warmed simulation that dominates the blended fitness.
