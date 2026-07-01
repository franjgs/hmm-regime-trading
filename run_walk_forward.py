#!/usr/bin/env python
"""Placeholder script for walk-forward regime modeling experiments."""

from pathlib import Path

from src.data_loader import load_config, load_market_data


config = load_config("config.yaml")
processed_data_dir = Path(config["paths"]["processed_data_dir"])
benchmark_ticker = config["data"]["benchmark_ticker"]
regime_data_path = processed_data_dir / f"{benchmark_ticker}_regimes.csv"

regime_data = load_market_data(regime_data_path)

# TODO: Replace this placeholder with expanding-window or rolling-window HMM
# training to avoid evaluating on the same data used for model fitting.
walk_forward_summary = {
    "ticker": benchmark_ticker,
    "observations": len(regime_data),
    "status": "placeholder",
}
