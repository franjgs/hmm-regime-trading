"""Hidden Markov Model training utilities for market regime modeling."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from src.data_loader import ensure_directory
from src.label_switching import add_ordered_regime_column


def prepare_hmm_matrix(
    data: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[np.ndarray, pd.DataFrame]:
    """Create the model matrix and aligned data frame for HMM training."""
    missing_columns = [column for column in feature_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing HMM feature columns: {missing_columns}")

    aligned_data = data.dropna(subset=feature_columns).copy()
    feature_matrix = aligned_data.loc[:, feature_columns].to_numpy(dtype=float)
    return feature_matrix, aligned_data


def train_gaussian_hmm(
    feature_matrix: np.ndarray,
    n_components: int,
    covariance_type: str,
    n_iter: int,
    random_seed: int,
) -> tuple[GaussianHMM, StandardScaler]:
    """Scale features and train a GaussianHMM."""
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)

    model = GaussianHMM(
        n_components=n_components,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_seed,
    )
    model.fit(scaled_features)
    return model, scaler


def predict_hmm_states(
    model: GaussianHMM,
    scaler: StandardScaler,
    feature_matrix: np.ndarray,
) -> np.ndarray:
    """Predict hidden states from raw feature values."""
    scaled_features = scaler.transform(feature_matrix)
    states = model.predict(scaled_features)
    return states


def add_regime_labels(
    data: pd.DataFrame,
    states: np.ndarray,
    return_column: str,
) -> pd.DataFrame:
    """Add raw HMM states and ordered regime labels to a data frame."""
    enriched = data.copy()
    enriched["hmm_state"] = states
    enriched = add_ordered_regime_column(
        data=enriched,
        state_column="hmm_state",
        return_column=return_column,
        ordered_column="regime",
    )
    return enriched


def save_hmm_artifact(
    model: GaussianHMM,
    scaler: StandardScaler,
    feature_columns: list[str],
    output_path: str | Path,
) -> None:
    """Save the trained HMM, scaler, and feature metadata."""
    path = Path(output_path)
    ensure_directory(path.parent)

    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_columns": feature_columns,
    }
    joblib.dump(artifact, path)


def load_hmm_artifact(input_path: str | Path) -> dict:
    """Load a saved HMM artifact."""
    artifact = joblib.load(input_path)
    return artifact
