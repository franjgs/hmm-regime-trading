#!/usr/bin/env python
"""Download raw multi-asset market data with yfinance."""

from pathlib import Path

import yfinance as yf

from src.data_loader import ensure_directory, flatten_yfinance_columns, load_config, save_market_data


config = load_config("config.yaml")
raw_data_dir = ensure_directory(config["paths"]["raw_data_dir"])

tickers = config["data"]["tickers"]
start_date = config["data"]["start_date"]
end_date = config["data"]["end_date"]
interval = config["data"]["interval"]
auto_adjust = config["data"]["auto_adjust"]

downloaded_data = yf.download(
    tickers=tickers,
    start=start_date,
    end=end_date,
    interval=interval,
    auto_adjust=auto_adjust,
    group_by="column",
    progress=False,
)

market_data = flatten_yfinance_columns(downloaded_data)
output_path = Path(raw_data_dir) / "market_data.csv"
save_market_data(market_data, output_path)

# TODO: Add validation for failed ticker downloads and missing trading days.
