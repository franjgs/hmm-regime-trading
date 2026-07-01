"""Utilities for basic HMM label switching resolution."""

import numpy as np
import pandas as pd


def order_states_by_mean_return(
    data: pd.DataFrame,
    state_column: str,
    return_column: str,
) -> dict[int, int]:
    """Map arbitrary HMM labels to ordered labels by average realized return."""
    state_means = data.groupby(state_column)[return_column].mean().sort_values()
    mapping = {old_state: new_state for new_state, old_state in enumerate(state_means.index)}
    return mapping


def apply_state_mapping(states: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    """Apply a state-label mapping to a NumPy array of state labels."""
    mapped_states = np.array([mapping[int(state)] for state in states], dtype=int)
    return mapped_states


def add_ordered_regime_column(
    data: pd.DataFrame,
    state_column: str,
    return_column: str,
    ordered_column: str = "regime",
) -> pd.DataFrame:
    """Add a regime column with labels ordered from low-return to high-return."""
    enriched = data.copy()
    mapping = order_states_by_mean_return(enriched, state_column, return_column)
    enriched[ordered_column] = enriched[state_column].map(mapping).astype(int)

    # TODO: Evaluate more stable conventions, such as ordering by volatility or
    # transition persistence, depending on the research question.
    return enriched
