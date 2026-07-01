# TFG Index to Repository Mapping

This document maps a possible undergraduate thesis index to the current
repository scaffold. The mapping is intentionally practical: each chapter should
connect the written work to concrete files, scripts, or outputs.

## 1. Introduction

Purpose in the thesis:

- Present the motivation for adaptive algorithmic trading.
- Define the research question.
- State the scope and limitations of the project.

Repository references:

- `README.md`: project overview, environment, and sequential workflow.
- `config.yaml`: initial scope of assets, dates, features, and HMM settings.

Student TODO:

- TODO: Write the final research question and justify why regime modeling is
  relevant for the selected market assets.

## 2. State of the Art

Purpose in the thesis:

- Review market regimes, Hidden Markov Models, feature engineering, and
  algorithmic trading evaluation.
- Discuss known methodological problems in financial machine learning.

Repository references:

- `docs/methodology_risks.md`: risk checklist for the methodology.
- `src/hmm_regime.py`: HMM training scaffold.
- `src/features.py`: causal feature engineering scaffold.
- `src/label_switching.py`: label switching scaffold using explicit raw and
  semantic regime labels.

Student TODO:

- TODO: Add academic citations and explain which design choices are supported
  by the literature.

## 3. Data

Purpose in the thesis:

- Describe the selected assets, time period, data source, and data limitations.
- Explain the raw and processed data pipeline.

Repository references:

- `data_fetch.py`: downloads raw multi-asset data with `yfinance`.
- `src/data_loader.py`: configuration, CSV loading, and persistence utilities.
- `config.yaml`: tickers, dates, interval, and output paths.
- `data/raw/`: generated raw data location.

Student TODO:

- TODO: Document missing data handling, corporate actions, survivorship bias,
  and data-source limitations.

## 4. Feature Engineering

Purpose in the thesis:

- Explain each feature and why it is causal.
- Distinguish information available at decision time from future information.

Repository references:

- `build_features.py`: builds features from raw data.
- `src/features.py`: returns, rolling volatility, and momentum calculations.
- `data/processed/`: generated feature files.

Student TODO:

- TODO: Add feature diagnostics and justify all windows and transformations.

## 5. Regime Modeling Methodology

Purpose in the thesis:

- Explain Gaussian HMM assumptions, parameters, training procedure, and regime
  interpretation.
- Discuss label switching and state ordering.

Repository references:

- `train_hmm.py`: trains the initial HMM and saves regime-enriched data.
- `src/hmm_regime.py`: model preparation, training, prediction, and persistence.
- `src/label_switching.py`: state ordering convention and semantic regime
  labels.
- `models/hmm/`: generated model artifacts.

Student TODO:

- TODO: Compare alternative numbers of states and report convergence behavior.

## 6. Walk-Forward Validation

Purpose in the thesis:

- Evaluate whether the method works out of sample.
- Avoid using the same period for model development and final claims.

Repository references:

- `run_walk_forward.py`: sequential walk-forward validation driver.
- `src/walk_forward.py`: reusable chronological split and per-window HMM
  utilities.
- `data/processed/walk_forward_regimes.csv`: generated test-window regime
  predictions with `raw_hmm_state`, `regime_id`, and `regime_label`.
- `docs/methodology_risks.md`: look-ahead bias and overfitting warnings.

Student TODO:

- TODO: Compare rolling and expanding walk-forward protocols before drawing
  performance conclusions.

## 7. Trading Strategy and Backtest

Purpose in the thesis:

- Define the trading rule based on regimes.
- Include transaction costs, slippage, and risk controls.
- Report realistic performance metrics.

Repository references:

- `run_backtest.py`: placeholder for the regime-based backtest.
- `data/processed/`: expected input location for regime-enriched data.

Student TODO:

- TODO: Replace the placeholder summary with a causal strategy and full
  backtest report.

## 8. Results

Purpose in the thesis:

- Present feature diagnostics, regime diagnostics, validation results, and
  backtest results.
- Separate exploratory findings from final out-of-sample conclusions.

Repository references:

- `data/processed/`: generated intermediate outputs.
- `models/hmm/`: generated model artifacts.
- Future `reports/` or `figures/` outputs if the student adds them.

Student TODO:

- TODO: Add reproducible figures and tables generated from the scripts.

## 9. Discussion

Purpose in the thesis:

- Interpret the results.
- Explain limitations, failure modes, and sensitivity to assumptions.

Repository references:

- `docs/methodology_risks.md`: core risk checklist.
- `config.yaml`: parameters to discuss in sensitivity analysis.

Student TODO:

- TODO: Discuss whether regime labels are economically meaningful or only
  statistical clusters.

## 10. Conclusions and Future Work

Purpose in the thesis:

- Summarize what was learned.
- Identify extensions beyond the scaffold.

Repository references:

- `README.md`: current scope and student TODO list.
- All root scripts: reproducible workflow from data download to backtest
  placeholder.

Student TODO:

- TODO: State clearly which parts are implemented, which parts are exploratory,
  and which parts remain future work.
