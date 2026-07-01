#!/usr/bin/env python
"""Train a GaussianHMM and save regime-enriched data."""

from pathlib import Path

from src.data_loader import ensure_directory, load_config, load_market_data, save_market_data
from src.hmm_regime import (
    add_regime_labels,
    predict_hmm_states,
    prepare_hmm_matrix,
    save_hmm_artifact,
    train_gaussian_hmm,
)


config = load_config("config.yaml")
processed_data_dir = ensure_directory(config["paths"]["processed_data_dir"])
model_dir = ensure_directory(config["paths"]["model_dir"])

benchmark_ticker = config["data"]["benchmark_ticker"]
feature_path = Path(processed_data_dir) / f"{benchmark_ticker}_features.csv"
feature_data = load_market_data(feature_path)

feature_columns = config["hmm"]["feature_columns"]
return_column = f"return_{config['features']['return_lag']}d"
feature_matrix, aligned_data = prepare_hmm_matrix(feature_data, feature_columns)

# TODO: Split model development from final evaluation to reduce overfitting risk.
min_observations = config["hmm"]["min_observations"]
if len(aligned_data) < min_observations:
    raise ValueError(
        f"Not enough observations for HMM training: {len(aligned_data)} < {min_observations}"
    )

hmm_model, scaler = train_gaussian_hmm(
    feature_matrix=feature_matrix,
    n_components=config["hmm"]["n_components"],
    covariance_type=config["hmm"]["covariance_type"],
    n_iter=config["hmm"]["n_iter"],
    random_seed=config["project"]["random_seed"],
)

raw_states = predict_hmm_states(hmm_model, scaler, feature_matrix)
regime_data = add_regime_labels(aligned_data, raw_states, return_column)

regime_output_path = Path(processed_data_dir) / f"{benchmark_ticker}_regimes.csv"
model_output_path = Path(model_dir) / f"{benchmark_ticker}_gaussian_hmm.joblib"

save_market_data(regime_data, regime_output_path)
save_hmm_artifact(hmm_model, scaler, feature_columns, model_output_path)

# TODO: Add model diagnostics, state summaries, and convergence checks.
