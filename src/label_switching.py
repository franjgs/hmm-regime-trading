"""Deterministic label switching utilities for three-state HMM regimes.

HMM state numbers are arbitrary. A model can label the same market condition as
state 0 in one fit and state 2 in another fit. This module provides a simple,
teaching-oriented convention for mapping raw HMM states to semantic regimes
using only training-window information.
"""

import numpy as np
import pandas as pd


SEMANTIC_REGIME_TO_ID = {
    "defensive": 0,
    "neutral": 1,
    "growth": 2,
}


def _require_three_states(states: set[int]) -> None:
    """Validate that exactly three HMM states are available."""
    if len(states) != 3:
        raise ValueError(
            "This teaching scaffold expects exactly three HMM states. "
            f"Received {len(states)} states: {sorted(states)}."
        )


def calculate_regime_summary(
    training_data: pd.DataFrame,
    state_column: str,
    return_column: str,
) -> pd.DataFrame:
    """Summarize each raw HMM state using training-window observations only.

    In walk-forward experiments, `training_data` must contain only the active
    training window. Do not include validation or test observations when
    estimating label-switching statistics.
    """
    required_columns = {state_column, return_column}
    missing_columns = required_columns.difference(training_data.columns)
    if missing_columns:
        raise ValueError(f"Missing columns for label switching: {sorted(missing_columns)}")

    clean_data = training_data.dropna(subset=[state_column, return_column]).copy()
    clean_data[state_column] = clean_data[state_column].astype(int)

    states = set(clean_data[state_column].unique())
    _require_three_states(states)

    grouped_returns = clean_data.groupby(state_column)[return_column]
    summary = pd.DataFrame(
        {
            "mean_return": grouped_returns.mean(),
            "volatility": grouped_returns.std(ddof=0),
            "drawdown_proxy": grouped_returns.apply(lambda values: values.clip(upper=0).mean()),
            "frequency": grouped_returns.size() / len(clean_data),
        }
    )

    summary.index = summary.index.astype(int)
    summary.index.name = "raw_state"
    summary["volatility"] = summary["volatility"].fillna(0.0)
    return summary.sort_index()


def assign_semantic_regimes(summary: pd.DataFrame) -> dict[int, str]:
    """Assign defensive, neutral, and growth labels from regime statistics.

    The convention is deterministic and intentionally simple:

    - defensive: weakest mean return, highest volatility, and deepest downside.
    - growth: strongest mean return, lowest downside, and sufficient frequency.
    - neutral: remaining state.

    Frequency is included as a tie-breaker so very rare states are treated
    conservatively when their other statistics are ambiguous.
    """
    _require_three_states(set(summary.index.astype(int)))

    ranked = summary.copy()
    ranked["return_rank"] = ranked["mean_return"].rank(method="first", ascending=True)
    ranked["volatility_rank"] = ranked["volatility"].rank(method="first", ascending=True)
    ranked["drawdown_rank"] = ranked["drawdown_proxy"].rank(method="first", ascending=True)
    ranked["frequency_rank"] = ranked["frequency"].rank(method="first", ascending=True)

    ranked["defensive_score"] = (
        -2.0 * ranked["return_rank"]
        + ranked["volatility_rank"]
        - ranked["drawdown_rank"]
        - 0.25 * ranked["frequency_rank"]
    )
    defensive_state = int(ranked["defensive_score"].idxmax())

    remaining = ranked.drop(index=defensive_state).copy()
    remaining["growth_score"] = (
        2.0 * remaining["return_rank"]
        - remaining["volatility_rank"]
        + remaining["drawdown_rank"]
        + 0.25 * remaining["frequency_rank"]
    )
    growth_state = int(remaining["growth_score"].idxmax())

    neutral_state = int(next(state for state in ranked.index if state not in {defensive_state, growth_state}))

    mapping = {
        defensive_state: "defensive",
        neutral_state: "neutral",
        growth_state: "growth",
    }
    return mapping


def build_three_regime_mapping(
    training_data: pd.DataFrame,
    state_column: str,
    return_column: str,
) -> dict[int, int]:
    """Build a raw-state to ordered-regime mapping from training data only.

    In walk-forward experiments, estimate this mapping only on the active
    training window, then apply it unchanged to the validation or test window.
    """
    summary = calculate_regime_summary(training_data, state_column, return_column)
    semantic_mapping = assign_semantic_regimes(summary)
    numeric_mapping = {
        raw_state: SEMANTIC_REGIME_TO_ID[semantic_label]
        for raw_state, semantic_label in semantic_mapping.items()
    }
    return numeric_mapping


def build_three_regime_semantic_mapping(
    training_data: pd.DataFrame,
    state_column: str,
    return_column: str,
) -> dict[int, str]:
    """Build a raw-state to semantic-label mapping from training data only.

    In walk-forward experiments, estimate this mapping only on the active
    training window. Using future observations to name regimes would introduce
    look-ahead bias.
    """
    summary = calculate_regime_summary(training_data, state_column, return_column)
    semantic_mapping = assign_semantic_regimes(summary)
    return semantic_mapping


def order_states_by_mean_return(
    data: pd.DataFrame,
    state_column: str,
    return_column: str,
) -> dict[int, int]:
    """Map raw HMM states to ordered regimes using the three-regime convention."""
    return build_three_regime_mapping(
        training_data=data,
        state_column=state_column,
        return_column=return_column,
    )


def apply_state_mapping(states: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    """Apply a state-label mapping to a NumPy array of raw HMM states."""
    missing_states = sorted({int(state) for state in states}.difference(mapping.keys()))
    if missing_states:
        raise ValueError(f"Missing mapping for states: {missing_states}")

    mapped_states = np.array([mapping[int(state)] for state in states], dtype=int)
    return mapped_states


def add_ordered_regime_column(
    data: pd.DataFrame,
    state_column: str,
    return_column: str,
    ordered_column: str = "regime",
    semantic_column: str = "regime_label",
    training_data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add numeric and semantic regime columns using a training-window mapping.

    If `training_data` is not provided, the input `data` is used as the training
    window. This fallback is acceptable only for exploratory full-sample
    inspection, not for final backtesting or performance claims. In
    walk-forward experiments, pass only the active training window through
    `training_data`, then apply the estimated mapping unchanged to the
    validation or test window.
    """
    # Warning: `training_data=None` uses the full input sample to estimate the
    # label mapping. This is for exploratory inspection only and must not be
    # used for final backtesting.
    label_training_data = data if training_data is None else training_data
    numeric_mapping = build_three_regime_mapping(
        training_data=label_training_data,
        state_column=state_column,
        return_column=return_column,
    )
    semantic_mapping = build_three_regime_semantic_mapping(
        training_data=label_training_data,
        state_column=state_column,
        return_column=return_column,
    )

    enriched = data.copy()
    enriched[ordered_column] = enriched[state_column].map(numeric_mapping).astype(int)
    enriched[semantic_column] = enriched[state_column].map(semantic_mapping)

    # TODO(student): In walk-forward validation, estimate these mappings only on
    # the training window and then apply them unchanged to the validation window.
    # TODO(student): Align semantic labels across consecutive windows so regime
    # names remain stable when HMM states rotate or disappear temporarily.
    return enriched


def validate_three_regime_mapping(
    training_data: pd.DataFrame,
    state_column: str,
    return_column: str,
) -> dict[str, object]:
    """Run lightweight validation checks for a training-window mapping.

    In walk-forward experiments, validate the mapping estimated from the active
    training window only. Do not validate by recomputing regime names with
    validation or test observations included.
    """
    summary = calculate_regime_summary(training_data, state_column, return_column)
    semantic_mapping = assign_semantic_regimes(summary)
    numeric_mapping = {
        raw_state: SEMANTIC_REGIME_TO_ID[semantic_label]
        for raw_state, semantic_label in semantic_mapping.items()
    }

    semantic_labels = set(semantic_mapping.values())
    numeric_labels = set(numeric_mapping.values())
    is_valid = (
        semantic_labels == set(SEMANTIC_REGIME_TO_ID)
        and numeric_labels == set(SEMANTIC_REGIME_TO_ID.values())
    )

    return {
        "is_valid": is_valid,
        "summary": summary,
        "semantic_mapping": semantic_mapping,
        "numeric_mapping": numeric_mapping,
    }
