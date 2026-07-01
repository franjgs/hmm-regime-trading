#!/usr/bin/env python
"""Build causal features from downloaded market data."""

from pathlib import Path

from src.data_loader import (
    ensure_directory,
    load_config,
    load_market_data,
    save_market_data,
    select_single_ticker_frame,
)
from src.features import build_multi_asset_features


config = load_config("config.yaml")
raw_data_path = Path(config["paths"]["raw_data_dir"]) / "market_data.csv"
processed_data_dir = ensure_directory(config["paths"]["processed_data_dir"])

market_data = load_market_data(raw_data_path)
tickers = config["data"]["tickers"]

market_data_by_ticker = {
    ticker: select_single_ticker_frame(market_data, ticker)
    for ticker in tickers
}

feature_data_by_ticker = build_multi_asset_features(
    market_data_by_ticker=market_data_by_ticker,
    price_column=config["features"]["price_column"],
    return_lag=config["features"]["return_lag"],
    volatility_window=config["features"]["volatility_window"],
    momentum_window=config["features"]["momentum_window"],
)

for ticker, feature_data in feature_data_by_ticker.items():
    output_path = Path(processed_data_dir) / f"{ticker}_features.csv"
    save_market_data(feature_data, output_path)

# TODO: Add feature diagnostics and plots for exploratory analysis.
