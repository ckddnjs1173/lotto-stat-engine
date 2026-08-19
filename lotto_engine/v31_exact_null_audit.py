from __future__ import annotations

import random

import numpy as np
import pandas as pd

from .candidates import generate_candidates, make_seed
from .config import ROUND_COLUMN
from .loader import dataset_fingerprint, row_numbers
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
    _sample_reference,
    _training_rng,
    directional_raw_features,
    fair_null_transform,
    fit_subset_pairwise_ridge,
)
from .v31_exact_structural_null import (
    EXACT_STRUCTURAL_FEATURES,
    hybrid_exact_structural_transform,
)
from .v31_model_spec import FULL11_BASELINE_SPEC, V31ModelSpec, spec_metadata

AUDIT_VERSION = "v31_exact_structural_null_audit_v1"


def _active_indices(spec: V31ModelSpec) -> tuple[int, ...]:
    return tuple(FEATURE_NAMES.index(name) for name in spec.active_features)


def sampled_vector(numbers, context: dict, reference: np.ndarray) -> np.ndarray:
    raw = directional_raw_features(numbers, context)
    return fair_null_transform(raw, reference)


def hybrid_exact_vector(numbers, context: dict, reference: np.ndarray) -> np.ndarray:
    raw = directional_raw_features(numbers, context)
    sampled = fair_null_transform(raw, reference)
    return hybrid_exact_structural_transform(raw, sampled)


def _evaluation_rng(round_no: int, samples: int, spec: V31ModelSpec) -> random.Random:
    return random.Random(
        int(round_no) * 100_003
        + int(samples) * 1_009
        + int(spec.evaluation_seed_offset)
        + 89_003
    )


def run_exact_structural_null_audit(
    df: pd.DataFrame,
    spec: V31ModelSpec = FULL11_BASELINE_SPEC,
    baseline_samples: int = 500,
    bootstrap_reps: int = 2_000,
    latest_candidate_count: int = 20_000,
    progress_every: int = 0,
) -> dict:
    """Compare sampled-null FULL11 with a hybrid exact-structural representation.

    Evidence coordinates remain on the historical sampled empirical CDF. Only the six
    discrete structural coordinates are replaced by their exact whole-universe 6/45
    midranks. Both representations are retrained independently with the same target,
    negative, and outer-evaluation candidate streams.
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
        raise ValueError("not enough completed draws for exact-null audit")

    state = _empty_history_state()
    for idx in range(start_index):
        _advance_history(state, tuple(row_numbers(df.iloc[idx])), spec)

    sampled_stats = _empty_stats()
    exact_stats = _empty_stats()
    percentiles = {"sampled_structural_null": [], "exact_structural_null": []}
    active = _active_indices(spec)
    outer = 0

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
        actual_sampled = sampled_vector(actual, context, reference)
        actual_exact = hybrid_exact_vector(actual, context, reference)

        if int(sampled_stats["solved_targets"]) >= minimum_meta:
            sampled_model = fit_subset_pairwise_ridge(
                sampled_stats, active, ridge_lambda=float(spec.ridge_lambda)
            )
            exact_model = fit_subset_pairwise_ridge(
                exact_stats, active, ridge_lambda=float(spec.ridge_lambda)
            )
            fair_candidates = _sample_fair_candidates(
                _evaluation_rng(round_no, baseline_samples, spec),
                baseline_samples,
                actual,
            )
            sampled_baseline = []
            exact_baseline = []
            for candidate in fair_candidates:
                sampled_baseline.append(
                    float(
                        np.dot(
                            sampled_model["effective_weights"],
                            sampled_vector(candidate, context, reference),
                        )
                    )
                )
                exact_baseline.append(
                    float(
                        np.dot(
                            exact_model["effective_weights"],
                            hybrid_exact_vector(candidate, context, reference),
                        )
                    )
                )
            percentiles["sampled_structural_null"].append(
                tie_safe_percentile(
                    float(np.dot(sampled_model["effective_weights"], actual_sampled)),
                    sampled_baseline,
                )
            )
            percentiles["exact_structural_null"].append(
                tie_safe_percentile(
                    float(np.dot(exact_model["effective_weights"], actual_exact)),
                    exact_baseline,
                )
            )
            outer += 1
            if progress_every and outer % progress_every == 0:
                print(f"v3.1 exact structural null audit: {outer} outer targets through draw {round_no}")

        negatives = _sample_fair_candidates(
            _training_rng(round_no, spec),
            int(spec.train_negatives_per_target),
            actual,
        )
        _add_target(
            sampled_stats,
            actual_sampled,
            [sampled_vector(candidate, context, reference) for candidate in negatives],
        )
        _add_target(
            exact_stats,
            actual_exact,
            [hybrid_exact_vector(candidate, context, reference) for candidate in negatives],
        )
        _advance_history(state, actual, spec)

    sampled_final = fit_subset_pairwise_ridge(
        sampled_stats, active, ridge_lambda=float(spec.ridge_lambda)
    )
    exact_final = fit_subset_pairwise_ridge(
        exact_stats, active, ridge_lambda=float(spec.ridge_lambda)
    )
    latest_round = int(df.iloc[-1][ROUND_COLUMN])
    next_context = _feature_context(state, spec)
    next_reference = _sample_reference(
        next_context,
        latest_round + 1,
        int(spec.fair_reference_samples_per_target),
        spec,
    )
    candidate_seed = make_seed(latest_round, 881)
    candidates = [tuple(candidate) for candidate in generate_candidates(latest_candidate_count, candidate_seed)]
    sampled_scores = np.empty(latest_candidate_count, dtype=float)
    exact_scores = np.empty(latest_candidate_count, dtype=float)
    structural_abs_delta = {feature: 0.0 for feature in EXACT_STRUCTURAL_FEATURES}
    sampled_saturation = {feature: 0 for feature in EXACT_STRUCTURAL_FEATURES}
    exact_saturation = {feature: 0 for feature in EXACT_STRUCTURAL_FEATURES}

    for row_index, candidate in enumerate(candidates):
        sampled = sampled_vector(candidate, next_context, next_reference)
        exact = hybrid_exact_vector(candidate, next_context, next_reference)
        sampled_scores[row_index] = float(np.dot(sampled_final["effective_weights"], sampled))
        exact_scores[row_index] = float(np.dot(exact_final["effective_weights"], exact))
        for feature in EXACT_STRUCTURAL_FEATURES:
            index = FEATURE_NAMES.index(feature)
            structural_abs_delta[feature] += abs(float(exact[index] - sampled[index]))
            sampled_saturation[feature] += int(abs(float(sampled[index])) == 1.0)
            exact_saturation[feature] += int(abs(float(exact[index])) == 1.0)

    return {
        "version": AUDIT_VERSION,
        "audit_role": "representation_accuracy_audit_not_model_promotion",
        "strict_target_isolation": True,
        "post_selection_exploratory_validation": True,
        "model_spec": spec_metadata(spec),
        "data": {"rows": int(len(df)), "sha256": dataset_fingerprint(df)},
        "runtime": runtime_identity(),
        "baseline_samples_per_target": baseline_samples,
        "latest_candidate_count": latest_candidate_count,
        "exact_structural_features": list(EXACT_STRUCTURAL_FEATURES),
        "scenario_summaries": {
            "sampled_structural_null": percentile_summary(
                percentiles["sampled_structural_null"], bootstrap_reps, 34_101
            ),
            "exact_structural_null": percentile_summary(
                percentiles["exact_structural_null"], bootstrap_reps, 34_102
            ),
        },
        "exact_minus_sampled": paired_delta_summary(
            percentiles["exact_structural_null"],
            percentiles["sampled_structural_null"],
            bootstrap_reps,
            34_103,
            "exact_structural_minus_sampled_structural",
        ),
        "final_ridge": {
            "sampled_structural_null": ridge_condition_diagnostics(sampled_final),
            "exact_structural_null": ridge_condition_diagnostics(exact_final),
        },
        "latest_sampled_ranking_comparison": compare_score_vectors(
            sampled_scores, exact_scores
        ),
        "latest_structural_coordinate_diagnostics": {
            feature: {
                "mean_absolute_exact_minus_sampled_z": float(
                    structural_abs_delta[feature] / latest_candidate_count
                ),
                "sampled_saturation_rate": float(
                    sampled_saturation[feature] / latest_candidate_count
                ),
                "exact_saturation_rate": float(
                    exact_saturation[feature] / latest_candidate_count
                ),
            }
            for feature in EXACT_STRUCTURAL_FEATURES
        },
    }
