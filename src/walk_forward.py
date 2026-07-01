"""Walk-forward utilities for HMM regime experiments."""

from dataclasses import dataclass

import pandas as pd

from src.hmm_regime import (
    add_regime_labels,
    predict_hmm_states,
    prepare_hmm_matrix,
    train_gaussian_hmm,
)


@dataclass(frozen=True)
class WalkForwardWindow:
    """Chronological train/test split metadata for one walk-forward iteration."""

    iteration: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_start_position: int
    train_end_position: int
    test_start_position: int
    test_end_position: int


def build_walk_forward_windows(
    data: pd.DataFrame,
    initial_train_size: int,
    test_size: int,
    step_size: int,
    expanding_window: bool = True,
) -> list[WalkForwardWindow]:
    """Create chronological train/test windows for walk-forward validation.

    The default expanding-window mode grows the training sample after each
    iteration. Rolling-window mode keeps the training length fixed.
    """
    if initial_train_size <= 0:
        raise ValueError("initial_train_size must be positive.")
    if test_size <= 0:
        raise ValueError("test_size must be positive.")
    if step_size <= 0:
        raise ValueError("step_size must be positive.")
    if len(data) < initial_train_size + test_size:
        raise ValueError(
            "Not enough observations for one walk-forward split: "
            f"{len(data)} < {initial_train_size + test_size}."
        )

    windows = []
    iteration = 0
    test_start_position = initial_train_size

    while test_start_position + test_size <= len(data):
        train_start_position = 0 if expanding_window else test_start_position - initial_train_size
        train_end_position = test_start_position
        test_end_position = test_start_position + test_size

        window = WalkForwardWindow(
            iteration=iteration,
            train_start=data.index[train_start_position],
            train_end=data.index[train_end_position - 1],
            test_start=data.index[test_start_position],
            test_end=data.index[test_end_position - 1],
            train_start_position=train_start_position,
            train_end_position=train_end_position,
            test_start_position=test_start_position,
            test_end_position=test_end_position,
        )
        windows.append(window)

        iteration += 1
        test_start_position += step_size

    return windows


def run_hmm_walk_forward_iteration(
    data: pd.DataFrame,
    window: WalkForwardWindow,
    feature_columns: list[str],
    return_column: str,
    n_components: int,
    covariance_type: str,
    n_iter: int,
    random_seed: int,
) -> pd.DataFrame:
    """Train on one window and infer regimes on its test window only."""
    train_data = data.iloc[window.train_start_position : window.train_end_position].copy()
    test_data = data.iloc[window.test_start_position : window.test_end_position].copy()

    train_matrix, aligned_train_data = prepare_hmm_matrix(train_data, feature_columns)
    test_matrix, aligned_test_data = prepare_hmm_matrix(test_data, feature_columns)

    hmm_model, scaler = train_gaussian_hmm(
        feature_matrix=train_matrix,
        n_components=n_components,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_seed=random_seed,
    )

    train_raw_states = predict_hmm_states(hmm_model, scaler, train_matrix)
    test_raw_states = predict_hmm_states(hmm_model, scaler, test_matrix)

    training_regime_data = aligned_train_data.copy()
    training_regime_data["hmm_state"] = train_raw_states

    test_regime_data = add_regime_labels(
        data=aligned_test_data,
        states=test_raw_states,
        return_column=return_column,
        training_data=training_regime_data,
    )

    test_regime_data["walk_forward_iteration"] = window.iteration
    test_regime_data["train_start"] = window.train_start
    test_regime_data["train_end"] = window.train_end
    test_regime_data["test_start"] = window.test_start
    test_regime_data["test_end"] = window.test_end

    # TODO(student): Add transition matrix diagnostics for each training window.
    # TODO(student): Analyze regime persistence and compare it across windows.
    return test_regime_data


def run_hmm_walk_forward(
    data: pd.DataFrame,
    feature_columns: list[str],
    return_column: str,
    initial_train_size: int,
    test_size: int,
    step_size: int,
    expanding_window: bool,
    n_components: int,
    covariance_type: str,
    n_iter: int,
    random_seed: int,
) -> pd.DataFrame:
    """Run a basic walk-forward HMM regime experiment."""
    sorted_data = data.sort_index().copy()
    windows = build_walk_forward_windows(
        data=sorted_data,
        initial_train_size=initial_train_size,
        test_size=test_size,
        step_size=step_size,
        expanding_window=expanding_window,
    )

    regime_predictions = []
    for window in windows:
        window_predictions = run_hmm_walk_forward_iteration(
            data=sorted_data,
            window=window,
            feature_columns=feature_columns,
            return_column=return_column,
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_seed=random_seed + window.iteration,
        )
        regime_predictions.append(window_predictions)

    walk_forward_regimes = pd.concat(regime_predictions).sort_index()

    # TODO(student): Compare rolling and expanding windows as an explicit
    # research design choice, not as a post-hoc performance optimization.
    # TODO(student): Connect these regime predictions to a future strategy and
    # backtest module without adding trading rules inside this file.
    return walk_forward_regimes
