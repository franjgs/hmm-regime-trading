"""Causal feature engineering for market regime modeling."""

import pandas as pd


def build_causal_features(
    price_data: pd.DataFrame,
    price_column: str,
    return_lag: int,
    volatility_window: int,
    momentum_window: int,
) -> pd.DataFrame:
    """Build features that use only current and past information."""
    features = price_data.copy()

    if price_column not in features.columns:
        raise ValueError(f"Expected price column '{price_column}' in input data.")

    prices = features[price_column]
    features[f"return_{return_lag}d"] = prices.pct_change(return_lag)
    features[f"volatility_{volatility_window}d"] = (
        features[f"return_{return_lag}d"]
        .rolling(window=volatility_window, min_periods=volatility_window)
        .std()
    )
    features[f"momentum_{momentum_window}d"] = prices.pct_change(momentum_window)

    # TODO: Add additional causal features and document their financial meaning.
    return features.dropna()


def build_multi_asset_features(
    market_data_by_ticker: dict[str, pd.DataFrame],
    price_column: str,
    return_lag: int,
    volatility_window: int,
    momentum_window: int,
) -> dict[str, pd.DataFrame]:
    """Build causal features independently for each ticker."""
    feature_data_by_ticker = {}

    for ticker, ticker_data in market_data_by_ticker.items():
        feature_data_by_ticker[ticker] = build_causal_features(
            price_data=ticker_data,
            price_column=price_column,
            return_lag=return_lag,
            volatility_window=volatility_window,
            momentum_window=momentum_window,
        )

    return feature_data_by_ticker
