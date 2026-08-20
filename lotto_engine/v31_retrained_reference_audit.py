from __future__ import annotations

import numpy as np
import pandas as pd

from .candidates import generate_candidates, make_seed
from .config import ROUND_COLUMN
from .loader import dataset_fingerprint, row_numbers
from .v31_audit_utils import (
    compare_score_vectors,
    compare_weight_vectors,
    ridge_condition_diagnostics,
    runtime_identity,
)
from .v31_core import (
    FEATURE_NAMES,
    _add_target,
    _advance_history,
    _empty_history_state,
    _empty_stats,
    _feature_context,
    _sample_fair_candidates,
    _sample_reference,
    _training_rng,
    directional_raw_features,
    fair_null_transform,
    fit_subset_pairwise_ridge,
)
from .v31_model_spec import FULL11_BASELINE_SPEC, V31ModelSpec, spec_metadata
from .v31_reference_stability_audit import sample_nested_reference

AUDIT_VERSION = "v31_retrained_reference_stability_audit_v1"
DEFAULT_REFERENCE_COUNTS = (1_024, 4_096)
DEFAULT_STREAM_DELTAS = (0, 1)


def _active_indices(spec: V31ModelSpec) -> tuple[int, ...]:
    return tuple(FEATURE_NAMES.index(name) for name in spec.active_features)


def _variant_labels(
    spec: V31ModelSpec,
    reference_counts: tuple[int, ...],
    stream_deltas: tuple[int, ...],
) -> tuple[str, ...]:
    labels = [f"current_policy:{int(spec.fair_reference_samples_per_target)}"]
    for delta in stream_deltas:
        for count in reference_counts:
            labels.append(f"nested_count:{int(count)}:stream:{int(delta)}")
    return tuple(labels)


def _variant_reference(
    label: str,
    context: dict,
    round_no: int,
    spec: V31ModelSpec,
) -> np.ndarray:
    if label.startswith("current_policy:"):
        return _sample_reference(
            context,
            round_no,
            int(spec.fair_reference_samples_per_target),
            spec,
        )
    parts = label.split(":")
    if len(parts) != 4 or parts[0] != "nested_count" or parts[2] != "stream":
        raise ValueError(f"invalid reference audit variant {label!r}")
    return sample_nested_reference(
        context,
        round_no,
        int(parts[1]),
        spec,
        stream_delta=int(parts[3]),
    )


def run_retrained_reference_stability_audit(
    df: pd.DataFrame,
    spec: V31ModelSpec = FULL11_BASELINE_SPEC,
    reference_counts: tuple[int, ...] = DEFAULT_REFERENCE_COUNTS,
    stream_deltas: tuple[int, ...] = DEFAULT_STREAM_DELTAS,
    latest_candidate_count: int = 20_000,
    progress_every: int = 0,
) -> dict:
    """Retrain the same model under alternative deterministic reference streams.

    This audit does not select the reference with the best historical winner score.
    It measures whether reference Monte Carlo choices materially change fitted
    coefficients and the latest ranking after the complete training history is rerun.
    """
    counts = tuple(sorted(set(int(value) for value in reference_counts)))
    deltas = tuple(sorted(set(int(value) for value in stream_deltas)))
    latest_candidate_count = int(latest_candidate_count)
    progress_every = int(progress_every)
    if not counts or any(value <= 0 for value in counts):
        raise ValueError("reference_counts must contain positive values")
    if not deltas:
        raise ValueError("stream_deltas must be non-empty")
    if latest_candidate_count <= 0:
        raise ValueError("latest_candidate_count must be positive")
    if progress_every < 0:
        raise ValueError("progress_every must be non-negative")

    start_index = int(spec.history_start_index)
    if len(df) <= start_index:
        raise ValueError("not enough completed draws")
    labels = _variant_labels(spec, counts, deltas)
    baseline_label = labels[0]
    active = _active_indices(spec)

    state = _empty_history_state()
    for idx in range(start_index):
        _advance_history(state, tuple(row_numbers(df.iloc[idx])), spec)
    stats = {label: _empty_stats() for label in labels}

    trained_targets = 0
    for idx in range(start_index, len(df)):
        actual = tuple(row_numbers(df.iloc[idx]))
        round_no = int(df.iloc[idx][ROUND_COLUMN])
        context = _feature_context(state, spec)
        references = {
            label: _variant_reference(label, context, round_no, spec) for label in labels
        }
        actual_raw = directional_raw_features(actual, context)
        negatives = _sample_fair_candidates(
            _training_rng(round_no, spec),
            int(spec.train_negatives_per_target),
            actual,
        )
        negative_raw = [directional_raw_features(candidate, context) for candidate in negatives]

        for label in labels:
            reference = references[label]
            actual_vector = fair_null_transform(actual_raw, reference)
            negative_vectors = [
                fair_null_transform(raw, reference) for raw in negative_raw
            ]
            _add_target(stats[label], actual_vector, negative_vectors)

        _advance_history(state, actual, spec)
        trained_targets += 1
        if progress_every and trained_targets % progress_every == 0:
            print(
                f"v3.1 retrained reference audit: {trained_targets} targets through draw {round_no}"
            )

    models = {
        label: fit_subset_pairwise_ridge(
            stats[label], active, ridge_lambda=float(spec.ridge_lambda)
        )
        for label in labels
    }
    latest_round = int(df.iloc[-1][ROUND_COLUMN])
    target_round = latest_round + 1
    next_context = _feature_context(state, spec)
    next_references = {
        label: _variant_reference(label, next_context, target_round, spec) for label in labels
    }
    candidate_seed = make_seed(latest_round, 991)
    candidates = [tuple(candidate) for candidate in generate_candidates(latest_candidate_count, candidate_seed)]
    score_vectors = {label: np.empty(latest_candidate_count, dtype=float) for label in labels}

    for row_index, candidate in enumerate(candidates):
        raw = directional_raw_features(candidate, next_context)
        for label in labels:
            bounded = fair_null_transform(raw, next_references[label])
            score_vectors[label][row_index] = float(
                np.dot(models[label]["effective_weights"], bounded)
            )

    baseline_weights = np.asarray(models[baseline_label]["effective_weights"], dtype=float)
    comparisons = {}
    for label in labels[1:]:
        comparisons[label] = {
            "reference_policy": label,
            "weights": compare_weight_vectors(
                baseline_weights,
                np.asarray(models[label]["effective_weights"], dtype=float),
            ),
            "latest_sampled_ranking": compare_score_vectors(
                score_vectors[baseline_label], score_vectors[label]
            ),
            "ridge": ridge_condition_diagnostics(models[label]),
        }

    return {
        "version": AUDIT_VERSION,
        "audit_role": "numerical_retraining_stability_only_not_reference_parameter_tuning",
        "strict_target_isolation": True,
        "model_spec": spec_metadata(spec),
        "data": {"rows": int(len(df)), "sha256": dataset_fingerprint(df)},
        "runtime": runtime_identity(),
        "trained_targets": trained_targets,
        "latest_candidate_count": latest_candidate_count,
        "candidate_seed": int(candidate_seed),
        "baseline_variant": baseline_label,
        "variants": list(labels),
        "reference_counts": list(counts),
        "stream_deltas": list(deltas),
        "baseline_ridge": ridge_condition_diagnostics(models[baseline_label]),
        "comparisons_vs_current_policy": comparisons,
    }
