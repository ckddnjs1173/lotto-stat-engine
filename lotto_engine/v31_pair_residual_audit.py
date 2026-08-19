from __future__ import annotations

import random
from itertools import combinations

import numpy as np
import pandas as pd

from .candidates import generate_candidates, make_seed, random_combination
from .config import ROUND_COLUMN
from .loader import dataset_fingerprint, row_numbers
from .pair_evidence import PAIR_COUNT, PAIR_TO_INDEX, PAIR_VOCABULARY, UNIFORM_PAIR_PROBABILITY
from .v31_audit_utils import (
    compare_score_vectors,
    paired_delta_summary,
    percentile_summary,
    ridge_condition_diagnostics,
    runtime_identity,
    tie_safe_percentile,
)
from .v31_core import (
    FEATURE_NAMES,
    _add_target,
    _advance_history,
    _empty_history_state,
    _empty_stats,
    _feature_context,
    _sample_fair_candidates,
    _training_rng,
    directional_raw_features,
    fair_null_transform,
    fit_subset_pairwise_ridge,
)
from .v31_model_spec import FULL11_BASELINE_SPEC, V31ModelSpec, spec_metadata

AUDIT_VERSION = "v31_pair_residualization_audit_v1"


def _incidence_matrix() -> np.ndarray:
    matrix = np.zeros((PAIR_COUNT, 45), dtype=float)
    for row_index, (left, right) in enumerate(PAIR_VOCABULARY):
        matrix[row_index, left - 1] = 1.0
        matrix[row_index, right - 1] = 1.0
    return matrix


PAIR_INCIDENCE = _incidence_matrix()


def residualize_pair_edges(edge_values) -> tuple[np.ndarray, dict]:
    """Remove the least-squares additive vertex main effects from 990 edge values.

    X has one column per lotto number and two ones per pair edge. A constant edge
    level is already in the column space because every row contains exactly two
    selected vertices, so a separate intercept would be redundant.
    """
    values = np.asarray(edge_values, dtype=float)
    if values.shape != (PAIR_COUNT,):
        raise ValueError(f"expected {PAIR_COUNT} pair values")
    coefficients, _, rank, singular = np.linalg.lstsq(PAIR_INCIDENCE, values, rcond=None)
    fitted = PAIR_INCIDENCE @ coefficients
    residual = values - fitted
    centered = values - float(np.mean(values))
    total_ss = float(np.dot(centered, centered))
    residual_ss = float(np.dot(residual, residual))
    explained = 1.0 - residual_ss / total_ss if total_ss > 0.0 else 1.0
    return residual, {
        "design_rank": int(rank),
        "explained_variance_fraction": float(explained),
        "original_std": float(np.std(values)),
        "residual_std": float(np.std(residual)),
        "residual_mean": float(np.mean(residual)),
        "max_absolute_residual": float(np.max(np.abs(residual))),
        "smallest_singular_value": float(np.min(singular)) if len(singular) else float("nan"),
        "largest_singular_value": float(np.max(singular)) if len(singular) else float("nan"),
    }


def pair_residual_context(context: dict) -> tuple[dict, dict]:
    full_residual, full_diag = residualize_pair_edges(context["pair_lifts"])
    recent_edges = np.asarray(context["recent100_pair_rate"], dtype=float) - UNIFORM_PAIR_PROBABILITY
    recent_residual, recent_diag = residualize_pair_edges(recent_edges)
    augmented = dict(context)
    augmented["pair_full_residual"] = full_residual
    augmented["pair_recent100_residual"] = recent_residual
    return augmented, {
        "pair_full_log_lift_vertex_projection": full_diag,
        "pair_recent100_excess_vertex_projection": recent_diag,
    }


def directional_pair_residual_features(numbers, context: dict) -> np.ndarray:
    base = directional_raw_features(numbers, context).copy()
    nums = tuple(sorted(int(number) for number in numbers))
    pair_indices = tuple(PAIR_TO_INDEX[pair] for pair in combinations(nums, 2))
    base[FEATURE_NAMES.index("pair_full_log_lift")] = float(
        np.mean([context["pair_full_residual"][index] for index in pair_indices])
    )
    base[FEATURE_NAMES.index("pair_recent100_excess")] = float(
        np.mean([context["pair_recent100_residual"][index] for index in pair_indices])
    )
    return base


def _reference_candidate_stream(
    round_no: int,
    count: int,
    spec: V31ModelSpec,
) -> list[tuple[int, ...]]:
    count = int(count)
    if count <= 0:
        raise ValueError("reference count must be positive")
    rng = random.Random(
        int(round_no) * 310_007 + count * 3_101 + int(spec.reference_seed_offset)
    )
    seen = set()
    result = []
    while len(result) < count:
        candidate = tuple(random_combination(rng))
        if candidate in seen:
            continue
        seen.add(candidate)
        result.append(candidate)
    return result


def _paired_references(
    context: dict,
    residual_context: dict,
    round_no: int,
    count: int,
    spec: V31ModelSpec,
) -> tuple[np.ndarray, np.ndarray]:
    candidates = _reference_candidate_stream(round_no, count, spec)
    original = np.sort(
        np.asarray([directional_raw_features(candidate, context) for candidate in candidates], dtype=float),
        axis=0,
    )
    residual = np.sort(
        np.asarray(
            [directional_pair_residual_features(candidate, residual_context) for candidate in candidates],
            dtype=float,
        ),
        axis=0,
    )
    return original, residual


def _active_indices(spec: V31ModelSpec) -> tuple[int, ...]:
    return tuple(FEATURE_NAMES.index(name) for name in spec.active_features)


def _evaluation_rng(round_no: int, samples: int, spec: V31ModelSpec) -> random.Random:
    return random.Random(
        int(round_no) * 100_003 + int(samples) * 1_009 + int(spec.evaluation_seed_offset) + 73_001
    )


def _vector_original(candidate, context, reference) -> np.ndarray:
    return fair_null_transform(directional_raw_features(candidate, context), reference)


def _vector_residual(candidate, context, reference) -> np.ndarray:
    return fair_null_transform(directional_pair_residual_features(candidate, context), reference)


def run_pair_residualization_audit(
    df: pd.DataFrame,
    spec: V31ModelSpec = FULL11_BASELINE_SPEC,
    baseline_samples: int = 500,
    bootstrap_reps: int = 2_000,
    latest_candidate_count: int = 20_000,
    progress_every: int = 0,
) -> dict:
    """Compare original pair features with vertex-main-effect residualized pairs.

    Both representations use the same historical state, reference candidate stream,
    training negatives, and outer fair candidates. Each representation is refit from
    its own pairwise-ridge sufficient statistics. This is diagnostic only.
    """
    baseline_samples = int(baseline_samples)
    latest_candidate_count = int(latest_candidate_count)
    progress_every = int(progress_every)
    if baseline_samples <= 0 or latest_candidate_count <= 0:
        raise ValueError("sample counts must be positive")
    if progress_every < 0:
        raise ValueError("progress_every must be non-negative")

    start_index = int(spec.history_start_index)
    minimum_meta = int(spec.minimum_meta_train_targets)
    if len(df) <= start_index + minimum_meta:
        raise ValueError("not enough completed draws for pair residual audit")

    state = _empty_history_state()
    for idx in range(start_index):
        _advance_history(state, tuple(row_numbers(df.iloc[idx])), spec)

    original_stats = _empty_stats()
    residual_stats = _empty_stats()
    percentiles = {"original": [], "pair_residualized": []}
    active = _active_indices(spec)
    projection_rows = []
    outer = 0

    for idx in range(start_index, len(df)):
        actual = tuple(row_numbers(df.iloc[idx]))
        round_no = int(df.iloc[idx][ROUND_COLUMN])
        context = _feature_context(state, spec)
        residual_context, projection = pair_residual_context(context)
        original_reference, residual_reference = _paired_references(
            context,
            residual_context,
            round_no,
            int(spec.fair_reference_samples_per_target),
            spec,
        )
        actual_original = _vector_original(actual, context, original_reference)
        actual_residual = _vector_residual(actual, residual_context, residual_reference)

        if int(original_stats["solved_targets"]) >= minimum_meta:
            original_model = fit_subset_pairwise_ridge(
                original_stats, active, ridge_lambda=float(spec.ridge_lambda)
            )
            residual_model = fit_subset_pairwise_ridge(
                residual_stats, active, ridge_lambda=float(spec.ridge_lambda)
            )
            fair_candidates = _sample_fair_candidates(
                _evaluation_rng(round_no, baseline_samples, spec),
                baseline_samples,
                actual,
            )
            original_baseline = []
            residual_baseline = []
            for candidate in fair_candidates:
                original_baseline.append(
                    float(
                        np.dot(
                            original_model["effective_weights"],
                            _vector_original(candidate, context, original_reference),
                        )
                    )
                )
                residual_baseline.append(
                    float(
                        np.dot(
                            residual_model["effective_weights"],
                            _vector_residual(candidate, residual_context, residual_reference),
                        )
                    )
                )
            percentiles["original"].append(
                tie_safe_percentile(
                    float(np.dot(original_model["effective_weights"], actual_original)),
                    original_baseline,
                )
            )
            percentiles["pair_residualized"].append(
                tie_safe_percentile(
                    float(np.dot(residual_model["effective_weights"], actual_residual)),
                    residual_baseline,
                )
            )
            projection_rows.append(projection)
            outer += 1
            if progress_every and outer % progress_every == 0:
                print(f"v3.1 pair residual audit: {outer} outer targets through draw {round_no}")

        negatives = _sample_fair_candidates(
            _training_rng(round_no, spec),
            int(spec.train_negatives_per_target),
            actual,
        )
        _add_target(
            original_stats,
            actual_original,
            [_vector_original(candidate, context, original_reference) for candidate in negatives],
        )
        _add_target(
            residual_stats,
            actual_residual,
            [_vector_residual(candidate, residual_context, residual_reference) for candidate in negatives],
        )
        _advance_history(state, actual, spec)

    original_final = fit_subset_pairwise_ridge(
        original_stats, active, ridge_lambda=float(spec.ridge_lambda)
    )
    residual_final = fit_subset_pairwise_ridge(
        residual_stats, active, ridge_lambda=float(spec.ridge_lambda)
    )
    latest_round = int(df.iloc[-1][ROUND_COLUMN])
    next_context = _feature_context(state, spec)
    next_residual_context, next_projection = pair_residual_context(next_context)
    original_reference, residual_reference = _paired_references(
        next_context,
        next_residual_context,
        latest_round + 1,
        int(spec.fair_reference_samples_per_target),
        spec,
    )
    candidate_seed = make_seed(latest_round, 731)
    candidates = [tuple(candidate) for candidate in generate_candidates(latest_candidate_count, candidate_seed)]
    original_scores = np.asarray(
        [
            np.dot(
                original_final["effective_weights"],
                _vector_original(candidate, next_context, original_reference),
            )
            for candidate in candidates
        ],
        dtype=float,
    )
    residual_scores = np.asarray(
        [
            np.dot(
                residual_final["effective_weights"],
                _vector_residual(candidate, next_residual_context, residual_reference),
            )
            for candidate in candidates
        ],
        dtype=float,
    )

    projection_summary = {}
    for key in (
        "pair_full_log_lift_vertex_projection",
        "pair_recent100_excess_vertex_projection",
    ):
        explained = [row[key]["explained_variance_fraction"] for row in projection_rows]
        projection_summary[key] = {
            "outer_tests": len(explained),
            "mean_explained_variance_fraction": float(np.mean(explained)) if explained else float("nan"),
            "recent_100_mean_explained_variance_fraction": (
                float(np.mean(explained[-min(100, len(explained)):])) if explained else float("nan")
            ),
            "latest": next_projection[key],
        }

    return {
        "version": AUDIT_VERSION,
        "audit_role": "diagnostic_pair_main_effect_decomposition_not_model_promotion",
        "strict_target_isolation": True,
        "post_selection_exploratory_validation": True,
        "model_spec": spec_metadata(spec),
        "data": {"rows": int(len(df)), "sha256": dataset_fingerprint(df)},
        "runtime": runtime_identity(),
        "baseline_samples_per_target": baseline_samples,
        "latest_candidate_count": latest_candidate_count,
        "scenario_summaries": {
            "original": percentile_summary(percentiles["original"], bootstrap_reps, 33_101),
            "pair_residualized": percentile_summary(
                percentiles["pair_residualized"], bootstrap_reps, 33_102
            ),
        },
        "pair_residualized_minus_original": paired_delta_summary(
            percentiles["pair_residualized"],
            percentiles["original"],
            bootstrap_reps,
            33_103,
            "pair_residualized_minus_original",
        ),
        "pair_vertex_projection": projection_summary,
        "final_ridge": {
            "original": ridge_condition_diagnostics(original_final),
            "pair_residualized": ridge_condition_diagnostics(residual_final),
        },
        "latest_sampled_ranking_comparison": compare_score_vectors(
            original_scores, residual_scores
        ),
    }
