# Adaptive Algorithmic Trading Systems Based on Market Regime Modeling

Initial scaffold for the undergraduate thesis project `hmm-regime-trading`.

The project studies adaptive trading systems that use Hidden Markov Models (HMMs)
to identify market regimes from causal financial features. This repository is a
research scaffold, not a finished trading system.

## Environment

Python version: 3.11

Conda environment name: `llm_rl_finance`

Create the environment:

```bash
conda env create -f environment.yml
conda activate llm_rl_finance
```

Or install dependencies into an existing Python 3.11 environment:

```bash
pip install -r requirements.txt
```

## Project Structure

```text
.
├── config.yaml
├── data_fetch.py
├── build_features.py
├── train_hmm.py
├── run_walk_forward.py
├── run_backtest.py
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── hmm_regime.py
│   └── label_switching.py
├── data/
│   ├── raw/
│   └── processed/
└── models/
    └── hmm/
```

Generated `data/` and `models/` contents are ignored by Git.

## Sequential Research Workflow

The root scripts are intentionally written as sequential notebooks-in-script form
with no `main()` function and no `if __name__ == "__main__"` block. This makes it
easy to run each file in Spyder and inspect intermediate variables in the Variable
Explorer.

Run the scripts in this order:

```bash
python data_fetch.py
python build_features.py
python train_hmm.py
python run_walk_forward.py
python run_backtest.py
```

## Current Scope

Implemented scaffold features:

- Download multi-asset OHLCV data with `yfinance`.
- Build causal features: returns, rolling volatility, and momentum.
- Train a `GaussianHMM` using `hmmlearn`.
- Apply a basic label switching convention.
- Save regime-enriched data to `data/processed/`.
- Save trained HMM artifacts to `models/hmm/`.

## Student TODOs

- Improve data validation and missing-value handling.
- Add robust train/test and walk-forward evaluation.
- Compare different feature sets and HMM state counts.
- Add transaction costs, slippage, and position sizing to backtests.
- Add statistical tests and visual diagnostics.
- Document all experiments and conclusions.
