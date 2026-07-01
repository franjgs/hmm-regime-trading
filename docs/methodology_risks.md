# Methodology Risks

This document lists methodological risks that the student must address before
using the scaffold to make empirical claims.

## Look-Ahead Bias

Look-ahead bias occurs when a model or trading rule uses information that would
not have been available at the decision time. In this project, the main risks are
feature construction, scaling, model fitting, and label interpretation.

Current scaffold status:

- `src/features.py` uses past returns, rolling volatility, and momentum.
- `src/hmm_regime.py` currently fits the scaler and HMM on the full selected
  sample in `train_hmm.py`.
- `run_walk_forward.py` is only a placeholder.

Student TODO:

- TODO: Implement train/test or walk-forward splits.
- TODO: Fit scalers, HMMs, and label conventions only on each training window.
- TODO: Shift trading signals when needed so positions use only prior
  information.

## Label Switching

HMM state numbers are arbitrary. State `0` in one model fit does not necessarily
mean the same economic regime as state `0` in another model fit. This is called
label switching.

Current scaffold status:

- `src/label_switching.py` orders states by average realized return.
- This convention is simple and useful for inspection, but it can itself create
  look-ahead bias if calculated on the evaluation period.

Student TODO:

- TODO: Define label mappings using only training-window information.
- TODO: Test whether states are stable across random seeds and time windows.
- TODO: Consider alternative interpretations based on volatility, persistence,
  drawdowns, or economic context.

## Overfitting

Overfitting occurs when the chosen features, number of states, parameters, or
trading rules fit historical noise rather than a repeatable pattern.

Current scaffold status:

- `config.yaml` defines one initial feature set and one HMM configuration.
- `train_hmm.py` trains one model on the selected benchmark asset.
- No validation results are implemented yet.

Student TODO:

- TODO: Reserve final evaluation data that is not used for design decisions.
- TODO: Compare model variants using a pre-defined validation protocol.
- TODO: Report sensitivity to the number of states, feature windows, and random
  seeds.
- TODO: Avoid adding features only because they improve one historical result.

## Multiple Comparisons

Multiple comparisons arise when many assets, windows, features, model settings,
or strategy rules are tested and only the best result is reported. This can make
random performance look meaningful.

Current scaffold status:

- `config.yaml` supports several tickers and configurable model parameters.
- The scaffold does not yet track experiment counts or selection criteria.

Student TODO:

- TODO: Define experiment groups before running large comparisons.
- TODO: Record all tested configurations, not only successful ones.
- TODO: Distinguish exploratory experiments from confirmatory evaluation.
- TODO: Use robust reporting, such as out-of-sample results and sensitivity
  tables.

## Transaction Costs

Transaction costs, slippage, bid-ask spreads, and execution delays can turn an
apparently profitable strategy into an unprofitable one. These costs are
especially important if the strategy trades frequently.

Current scaffold status:

- `run_backtest.py` is a placeholder and does not yet implement realistic
  trading costs.

Student TODO:

- TODO: Add explicit assumptions for commissions, spreads, slippage, and order
  timing.
- TODO: Report performance before and after transaction costs.
- TODO: Measure turnover and explain whether the strategy is realistic for the
  intended investor.

## Minimum Standard Before Drawing Conclusions

Before the thesis reports performance conclusions, the student should complete
at least the following:

- A causal feature pipeline.
- A documented label switching convention.
- A walk-forward or train/test evaluation design.
- A backtest with transaction costs.
- A record of tested configurations.
- A discussion of limitations and failed experiments.
