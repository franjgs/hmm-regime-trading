"""Data loading and persistence utilities for market data."""

from pathlib import Path

import pandas as pd
import yaml


def load_config(config_path: str | Path = "config.yaml") -> dict:
    """Load project configuration from a YAML file."""
    with Path(config_path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not already exist and return its Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_market_data(data: pd.DataFrame, output_path: str | Path) -> None:
    """Save market data to CSV using a relative path."""
    path = Path(output_path)
    ensure_directory(path.parent)
    data.to_csv(path, index=True)


def load_market_data(input_path: str | Path) -> pd.DataFrame:
    """Load market data from CSV with a DateTime index."""
    data = pd.read_csv(input_path, index_col=0, parse_dates=True)
    data.index.name = "Date"
    return data


def flatten_yfinance_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance multi-index columns into field_ticker names."""
    if not isinstance(data.columns, pd.MultiIndex):
        return data

    flattened = data.copy()
    flattened.columns = [
        "_".join(str(part) for part in column if str(part))
        for column in flattened.columns.to_flat_index()
    ]
    return flattened


def select_single_ticker_frame(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Select one ticker from flattened yfinance data."""
    suffix = f"_{ticker}"
    selected_columns = {
        column: column.removesuffix(suffix)
        for column in data.columns
        if column.endswith(suffix)
    }

    if not selected_columns:
        raise ValueError(f"No columns found for ticker '{ticker}'.")

    selected = data.loc[:, list(selected_columns.keys())].rename(columns=selected_columns)
    selected.index.name = "Date"
    return selected
