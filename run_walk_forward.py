#!/usr/bin/env python
"""Run a basic walk-forward HMM regime modeling experiment."""

from pathlib import Path

from src.data_loader import ensure_directory, load_config, load_market_data, save_market_data
from src.walk_forward import run_hmm_walk_forward


config = load_config("config.yaml")
processed_data_dir = ensure_directory(config["paths"]["processed_data_dir"])
benchmark_ticker = config["data"]["benchmark_ticker"]
feature_data_path = Path(processed_data_dir) / f"{benchmark_ticker}_features.csv"
output_path = Path(processed_data_dir) / "walk_forward_regimes.csv"

feature_data = load_market_data(feature_data_path)
feature_columns = config["hmm"]["feature_columns"]
return_column = f"return_{config['features']['return_lag']}d"

initial_train_size = max(config["hmm"]["min_observations"], 756)
test_size = 63
step_size = 63
expanding_window = True

# TODO(student): Move walk-forward parameters to config.yaml after the student
# defines the experimental protocol for rolling versus expanding windows.
# TODO(student): Compare rolling and expanding windows before finalizing the TFG
# methodology, but do not choose the protocol only because it performs better.
walk_forward_regimes = run_hmm_walk_forward(
    data=feature_data,
    feature_columns=feature_columns,
    return_column=return_column,
    initial_train_size=initial_train_size,
    test_size=test_size,
    step_size=step_size,
    expanding_window=expanding_window,
    n_components=config["hmm"]["n_components"],
    covariance_type=config["hmm"]["covariance_type"],
    n_iter=config["hmm"]["n_iter"],
    random_seed=config["project"]["random_seed"],
)

save_market_data(walk_forward_regimes, output_path)

# TODO(student): Add transition matrix diagnostics and regime persistence
# analysis before interpreting regimes as economically meaningful.
# TODO(student): Use this output as input for a future strategy/backtest module;
# do not add trading logic to the walk-forward regime estimation script.
walk_forward_summary = {
    "ticker": benchmark_ticker,
    "observations": len(walk_forward_regimes),
    "iterations": int(walk_forward_regimes["walk_forward_iteration"].nunique()),
    "output_path": str(output_path),
    "status": "saved",
}
