from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ROUND_COLUMN
from .loader import dataset_fingerprint, row_numbers
from .v31_audit_utils import ridge_condition_diagnostics, runtime_identity
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
    candidate_vector,
    fit_subset_pairwise_ridge,
)
from .v31_model_spec import FULL11_BASELINE_SPEC, V31ModelSpec, spec_metadata

AUDIT_VERSION = "v31_ridge_dependency_audit_v1"
RMS_EPSILON = 1e-12

PREDECLARED_DEPENDENCY_PAIRS = (
    ("number_full_log_lift", "pair_full_log_lift"),
    ("number_recent20_excess", "number_recent100_excess"),
    ("number_recent100_excess", "pair_recent100_excess"),
    ("number_full_log_lift", "number_recent100_excess"),
    ("pair_full_log_lift", "pair_recent100_excess"),
    ("sum_signed_center_138", "high_minus_low_zone_count"),
    ("number_range", "consecutive_pairs"),
)


def _active_indices(spec: V31ModelSpec) -> tuple[int, ...]:
    return tuple(FEATURE_NAMES.index(name) for name in spec.active_features)


def scaled_second_moment_matrix(stats: dict) -> np.ndarray:
    rows = int(stats["pair_rows"])
    if rows <= 0:
        raise ValueError("training rows required")
    second = np.asarray(stats["sum_outer"], dtype=float) / rows
    rms = np.sqrt(np.maximum(np.diag(second), 0.0) + RMS_EPSILON)
    return second / np.outer(rms, rms)


def dependency_pairs_from_matrix(matrix: np.ndarray) -> list[dict]:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (len(FEATURE_NAMES), len(FEATURE_NAMES)):
        raise ValueError("unexpected dependency matrix shape")
    result = []
    for left in range(len(FEATURE_NAMES) - 1):
        for right in range(left + 1, len(FEATURE_NAMES)):
            value = float(matrix[left, right])
            result.append(
                {
                    "left": FEATURE_NAMES[left],
                    "right": FEATURE_NAMES[right],
                    "second_moment_cosine": value,
                    "absolute_second_moment_cosine": abs(value),
                }
            )
    return sorted(
        result,
        key=lambda item: (
            -item["absolute_second_moment_cosine"],
            item["left"],
            item["right"],
        ),
    )


def _condition_summary(rows: list[dict], key: str) -> dict:
    values = [float(row[key]) for row in rows if np.isfinite(row.get(key, np.nan))]
    return {
        "tests": len(values),
        "mean": float(np.mean(values)) if values else float("nan"),
        "median": float(np.median(values)) if values else float("nan"),
        "max": float(np.max(values)) if values else float("nan"),
        "recent_100_mean": (
            float(np.mean(values[-min(100, len(values)):])) if values else float("nan")
        ),
    }


def _weight_stability(weight_rows: list[np.ndarray]) -> dict:
    if not weight_rows:
        return {}
    matrix = np.vstack(weight_rows)
    result = {}
    for index, feature in enumerate(FEATURE_NAMES):
        values = matrix[:, index]
        recent = values[-min(100, len(values)):]
        signs = np.sign(values)
        result[feature] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "recent_100_mean": float(np.mean(recent)),
            "positive_fraction": float(np.mean(values > 0.0)),
            "negative_fraction": float(np.mean(values < 0.0)),
            "sign_flip_count": int(np.sum(signs[1:] != signs[:-1])),
        }
    return result


def run_dependency_audit(
    df: pd.DataFrame,
    spec: V31ModelSpec = FULL11_BASELINE_SPEC,
    progress_every: int = 0,
) -> dict:
    progress_every = int(progress_every)
    if progress_every < 0:
        raise ValueError("progress_every must be non-negative")
    start_index = int(spec.history_start_index)
    minimum_meta = int(spec.minimum_meta_train_targets)
    if len(df) <= start_index + minimum_meta:
        raise ValueError("not enough completed draws for dependency audit")

    state = _empty_history_state()
    for idx in range(start_index):
        _advance_history(state, tuple(row_numbers(df.iloc[idx])), spec)
    stats = _empty_stats()
    active = _active_indices(spec)
    condition_rows = []
    weight_rows = []
    fitted_states = 0

    for idx in range(start_index, len(df)):
        actual = tuple(row_numbers(df.iloc[idx]))
        round_no = int(df.iloc[idx][ROUND_COLUMN])
        context = _feature_context(state, spec)
        reference = _sample_reference(
            context,
            round_no,
            int(spec.fair_reference_samples_per_target),
            spec,
        )
        actual_vector = candidate_vector(actual, context, reference)

        if int(stats["solved_targets"]) >= minimum_meta:
            model = fit_subset_pairwise_ridge(
                stats, active, ridge_lambda=float(spec.ridge_lambda)
            )
            condition_rows.append(ridge_condition_diagnostics(model))
            weight_rows.append(
                np.asarray(model["effective_weights"], dtype=float).copy()
            )
            fitted_states += 1
            if progress_every and fitted_states % progress_every == 0:
                print(
                    f"v3.1 dependency audit: {fitted_states} fitted states through draw {round_no}"
                )

        negatives = _sample_fair_candidates(
            _training_rng(round_no, spec),
            int(spec.train_negatives_per_target),
            actual,
        )
        _add_target(
            stats,
            actual_vector,
            [candidate_vector(candidate, context, reference) for candidate in negatives],
        )
        _advance_history(state, actual, spec)

    final_model = fit_subset_pairwise_ridge(
        stats, active, ridge_lambda=float(spec.ridge_lambda)
    )
    matrix = scaled_second_moment_matrix(stats)
    ranked_pairs = dependency_pairs_from_matrix(matrix)
    lookup = {
        tuple(sorted((item["left"], item["right"]))): item for item in ranked_pairs
    }
    predeclared = []
    for left, right in PREDECLARED_DEPENDENCY_PAIRS:
        item = dict(lookup[tuple(sorted((left, right)))])
        item["requested_left"] = left
        item["requested_right"] = right
        predeclared.append(item)

    return {
        "version": AUDIT_VERSION,
        "audit_role": "conditioning_and_dependency_diagnostic_not_model_selection",
        "model_spec": spec_metadata(spec),
        "data": {"rows": int(len(df)), "sha256": dataset_fingerprint(df)},
        "runtime": runtime_identity(),
        "fitted_historical_states": fitted_states,
        "historical_conditioning": {
            "unregularized_scaled_second_moment": _condition_summary(
                condition_rows, "scaled_second_moment_condition_number"
            ),
            "actual_ridge_system": _condition_summary(
                condition_rows, "ridge_system_condition_number"
            ),
        },
        "final_ridge": ridge_condition_diagnostics(final_model),
        "weight_stability": _weight_stability(weight_rows),
        "predeclared_dependency_pairs": predeclared,
        "strongest_dependency_pairs": ranked_pairs[:20],
        "dependency_metric": (
            "E[d_i d_j]/sqrt(E[d_i^2]E[d_j^2]) on pairwise actual-minus-fair training differences; "
            "this is a second-moment cosine, not a centered Pearson correlation"
        ),
    }
