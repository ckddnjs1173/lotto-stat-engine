from __future__ import annotations

import random
from typing import Iterable

import numpy as np
import pandas as pd

from .candidates import random_combination
from .config import ROUND_COLUMN
from .loader import dataset_fingerprint, row_numbers
from .v31_core import (
    FEATURE_NAMES,
    _add_target,
    _advance_history,
    _empty_history_state,
    _empty_stats,
    _feature_context,
    directional_raw_features,
    fair_null_transform,
    fit_subset_pairwise_ridge,
)
from .v31_exact_structural_null import hybrid_exact_structural_transform
from .v31_model_spec import FROZEN7_SPEC, V31FrozenModelSpec

REFERENCE_STREAM_OFFSET = 77_003
NEGATIVE_STREAM_OFFSET = 131_003
STREAM_DELTA_MULTIPLIER = 1_000_003


def _validate_frozen_spec(spec: V31FrozenModelSpec) -> None:
    if spec.structural_null_policy != "exact_whole_universe_midrank_for_structural_features":
        raise ValueError("frozen v3.1 requires exact structural null coordinates")
    if spec.reference_policy != "nested_deterministic_fair_sample_empirical_midrank_stream0":
        raise ValueError("unexpected frozen v3.1 reference policy")
    if spec.negative_sampling_policy != "nested_count_independent_deterministic_unique_fair_stream":
        raise ValueError("unexpected frozen v3.1 negative sampling policy")
    if int(spec.reference_stream_delta) != 0 or int(spec.negative_stream_delta) != 0:
        raise ValueError("frozen v3.1 uses stream delta 0")


def sample_nested_reference_candidates(
    round_no: int,
    count: int,
    spec: V31FrozenModelSpec = FROZEN7_SPEC,
) -> list[tuple[int, ...]]:
    count = int(count)
    if count <= 0:
        raise ValueError("reference count must be positive")
    _validate_frozen_spec(spec)
    rng = random.Random(
        int(round_no) * 310_007
        + int(spec.reference_seed_offset)
        + REFERENCE_STREAM_OFFSET
        + int(spec.reference_stream_delta) * STREAM_DELTA_MULTIPLIER
    )
    seen: set[tuple[int, ...]] = set()
    result: list[tuple[int, ...]] = []
    while len(result) < count:
        candidate = tuple(random_combination(rng))
        if candidate in seen:
            continue
        seen.add(candidate)
        result.append(candidate)
    return result


def sample_nested_reference(
    context: dict,
    round_no: int,
    count: int,
    spec: V31FrozenModelSpec = FROZEN7_SPEC,
) -> np.ndarray:
    rows = [
        directional_raw_features(candidate, context)
        for candidate in sample_nested_reference_candidates(round_no, count, spec)
    ]
    return np.sort(np.asarray(rows, dtype=float), axis=0)


def sample_nested_training_negatives(
    round_no: int,
    count: int,
    actual: tuple[int, ...],
    spec: V31FrozenModelSpec = FROZEN7_SPEC,
) -> list[tuple[int, ...]]:
    count = int(count)
    if count <= 0:
        raise ValueError("negative count must be positive")
    _validate_frozen_spec(spec)
    rng = random.Random(
        int(round_no) * 200_003
        + int(spec.training_seed_offset)
        + NEGATIVE_STREAM_OFFSET
        + int(spec.negative_stream_delta) * STREAM_DELTA_MULTIPLIER
    )
    seen = {tuple(actual)}
    result: list[tuple[int, ...]] = []
    while len(result) < count:
        candidate = tuple(random_combination(rng))
        if candidate in seen:
            continue
        seen.add(candidate)
        result.append(candidate)
    return result


def frozen_vector(
    numbers: Iterable[int],
    context: dict,
    reference: np.ndarray,
) -> np.ndarray:
    raw = directional_raw_features(numbers, context)
    sampled = fair_null_transform(raw, reference)
    return hybrid_exact_structural_transform(raw, sampled)


def _active_indices(spec: V31FrozenModelSpec) -> tuple[int, ...]:
    unknown = [name for name in spec.active_features if name not in FEATURE_NAMES]
    if unknown:
        raise ValueError(f"unknown active frozen features: {unknown}")
    return tuple(FEATURE_NAMES.index(name) for name in spec.active_features)


def build_latest_frozen_model(
    df: pd.DataFrame,
    spec: V31FrozenModelSpec = FROZEN7_SPEC,
) -> dict:
    _validate_frozen_spec(spec)
    start_index = int(spec.history_start_index)
    if len(df) <= start_index:
        raise ValueError("not enough completed draws")

    state = _empty_history_state()
    for idx in range(start_index):
        _advance_history(state, tuple(row_numbers(df.iloc[idx])), spec)

    stats = _empty_stats()
    for idx in range(start_index, len(df)):
        actual = tuple(row_numbers(df.iloc[idx]))
        round_no = int(df.iloc[idx][ROUND_COLUMN])
        context = _feature_context(state, spec)
        reference = sample_nested_reference(
            context,
            round_no,
            int(spec.fair_reference_samples_per_target),
            spec,
        )
        actual_vector = frozen_vector(actual, context, reference)
        negatives = sample_nested_training_negatives(
            round_no,
            int(spec.train_negatives_per_target),
            actual,
            spec,
        )
        negative_vectors = [
            frozen_vector(candidate, context, reference) for candidate in negatives
        ]
        _add_target(stats, actual_vector, negative_vectors)
        _advance_history(state, actual, spec)

    model = fit_subset_pairwise_ridge(
        stats,
        _active_indices(spec),
        ridge_lambda=float(spec.ridge_lambda),
    )
    latest_round = int(df.iloc[-1][ROUND_COLUMN])
    target_round = latest_round + 1
    next_context = _feature_context(state, spec)
    next_reference = sample_nested_reference(
        next_context,
        target_round,
        int(spec.fair_reference_samples_per_target),
        spec,
    )
    return {
        "model": model,
        "context": next_context,
        "reference": next_reference,
        "latest_round": latest_round,
        "target_round": target_round,
        "history_draws": int(state["history_draws"]),
        "reference_samples": int(spec.fair_reference_samples_per_target),
        "train_negatives_per_target": int(spec.train_negatives_per_target),
        "dataset_rows": int(len(df)),
        "dataset_sha256": dataset_fingerprint(df),
        "spec": spec,
    }


def frozen_model_score(numbers: Iterable[int], fitted: dict) -> float:
    vector = frozen_vector(numbers, fitted["context"], fitted["reference"])
    weights = np.asarray(fitted["model"]["effective_weights"], dtype=float)
    return float(np.dot(weights, vector))
