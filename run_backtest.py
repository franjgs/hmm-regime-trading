#!/usr/bin/env python
"""Placeholder script for strategy backtesting based on HMM regimes."""

from pathlib import Path

from src.data_loader import load_config, load_market_data


config = load_config("config.yaml")
processed_data_dir = Path(config["paths"]["processed_data_dir"])
benchmark_ticker = config["data"]["benchmark_ticker"]
regime_data_path = processed_data_dir / f"{benchmark_ticker}_regimes.csv"

regime_data = load_market_data(regime_data_path)

# TODO: Define a causal trading rule that uses only information available at
# decision time. Include transaction costs, slippage, and risk controls.
strategy_returns = regime_data[f"return_{config['features']['return_lag']}d"].copy()

# TODO: Replace this naive summary with a full backtest report.
backtest_summary = {
    "ticker": benchmark_ticker,
    "observations": len(strategy_returns),
    "mean_return": float(strategy_returns.mean()),
    "volatility": float(strategy_returns.std()),
    "status": "placeholder",
}
