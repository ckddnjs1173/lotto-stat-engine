from __future__ import annotations

import random

import numpy as np
import pandas as pd

from .candidates import generate_candidates, make_seed, random_combination
from .config import ROUND_COLUMN
from .loader import dataset_fingerprint, row_numbers
from .v31_audit_utils import (
    compare_score_vectors,
    compare_weight_vectors,
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
from .v31_exact_structural_null import hybrid_exact_structural_transform
from .v31_model_spec import FULL11_BASELINE_SPEC, V31ModelSpec, spec_metadata
from .v31_reference_stability_audit import sample_nested_reference

AUDIT_VERSION = "v31_freeze_candidate_audit_v1"
REFERENCE_COUNT = 4_096
NEGATIVE_COUNTS = (64, 128, 256)

PAIR_FEATURES = ("pair_full_log_lift", "pair_recent100_excess")
SUM_FEATURE = "sum_signed_center_138"
ODD_FEATURE = "odd_count_signed_center_3"

CANDIDATE_FEATURE_SETS = {
    "exact_full11": tuple(FEATURE_NAMES),
    "exact_no_pair9": tuple(name for name in FEATURE_NAMES if name not in PAIR_FEATURES),
    "exact_no_pair_sum8": tuple(
        name for name in FEATURE_NAMES if name not in (*PAIR_FEATURES, SUM_FEATURE)
    ),
    "exact_no_pair_sum_odd7": tuple(
        name for name in FEATURE_NAMES if name not in (*PAIR_FEATURES, SUM_FEATURE, ODD_FEATURE)
    ),
}
FINALIST_NAMES = ("exact_no_pair_sum8", "exact_no_pair_sum_odd7")


def active_indices(feature_names) -> tuple[int, ...]:
    return tuple(FEATURE_NAMES.index(name) for name in feature_names)


def exact_vector(numbers, context: dict, reference: np.ndarray) -> np.ndarray:
    raw = directional_raw_features(numbers, context)
    sampled = fair_null_transform(raw, reference)
    return hybrid_exact_structural_transform(raw, sampled)


def _evaluation_rng(round_no: int, samples: int, spec: V31ModelSpec) -> random.Random:
    return random.Random(
        int(round_no) * 100_003
        + int(samples) * 1_009
        + int(spec.evaluation_seed_offset)
        + 97_003
    )


def sample_nested_training_negatives(
    round_no: int,
    count: int,
    actual: tuple[int, ...],
    spec: V31ModelSpec = FULL11_BASELINE_SPEC,
    stream_delta: int = 0,
) -> list[tuple[int, ...]]:
    """Count-independent negative stream; larger counts preserve smaller prefixes."""
    count = int(count)
    if count <= 0:
        raise ValueError("negative count must be positive")
    rng = random.Random(
        int(round_no) * 200_003
        + int(spec.training_seed_offset)
        + 131_003
        + int(stream_delta) * 1_000_003
    )
    seen = {tuple(actual)}
    result = []
    while len(result) < count:
        candidate = tuple(random_combination(rng))
        if candidate in seen:
            continue
        seen.add(candidate)
        result.append(candidate)
    return result


def _fit_candidate_models(stats: dict, spec: V31ModelSpec) -> dict[str, dict]:
    return {
        name: fit_subset_pairwise_ridge(
            stats,
            active_indices(features),
            ridge_lambda=float(spec.ridge_lambda),
        )
        for name, features in CANDIDATE_FEATURE_SETS.items()
    }


def run_freeze_candidate_audit(
    df: pd.DataFrame,
    spec: V31ModelSpec = FULL11_BASELINE_SPEC,
    reference_count: int = REFERENCE_COUNT,
    baseline_samples: int = 500,
    bootstrap_reps: int = 2_000,
    latest_candidate_count: int = 20_000,
    negative_counts: tuple[int, ...] = NEGATIVE_COUNTS,
    progress_every: int = 0,
) -> dict:
    """Final pre-freeze audit selected from the complete v3.1 diagnostic results.

    The prior suite showed pair-family and sum harm, an exact structural null with no
    measurable historical penalty, and an ambiguous odd-count signal. This audit
    therefore limits itself to four predeclared nested feature sets under exact
    structural coordinates and a 4096 nested reference. It also measures 64/128/256
    nested training-negative stability without choosing a count by winner performance.
    """
    reference_count = int(reference_count)
    baseline_samples = int(baseline_samples)
    latest_candidate_count = int(latest_candidate_count)
    counts = tuple(sorted(set(int(value) for value in negative_counts)))
    progress_every = int(progress_every)
    if reference_count <= 0 or baseline_samples <= 0 or latest_candidate_count <= 0:
        raise ValueError("sample counts must be positive")
    if not counts or any(value <= 0 for value in counts):
        raise ValueError("negative_counts must contain positive values")
    if progress_every < 0:
        raise ValueError("progress_every must be non-negative")

    start_index = int(spec.history_start_index)
    minimum_meta = int(spec.minimum_meta_train_targets)
    if len(df) <= start_index + minimum_meta:
        raise ValueError("not enough completed draws for freeze candidate audit")

    state = _empty_history_state()
    for idx in range(start_index):
        _advance_history(state, tuple(row_numbers(df.iloc[idx])), spec)

    selection_stats = _empty_stats()
    nested_stats = {count: _empty_stats() for count in counts}
    percentiles = {name: [] for name in CANDIDATE_FEATURE_SETS}
    outer = 0
    expected = len(df) - start_index - minimum_meta

    for idx in range(start_index, len(df)):
        actual = tuple(row_numbers(df.iloc[idx]))
        round_no = int(df.iloc[idx][ROUND_COLUMN])
        context = _feature_context(state, spec)
        reference = sample_nested_reference(
            context, round_no, reference_count, spec, stream_delta=0
        )
        actual_vector = exact_vector(actual, context, reference)

        if int(selection_stats["solved_targets"]) >= minimum_meta:
            models = _fit_candidate_models(selection_stats, spec)
            fair_candidates = _sample_fair_candidates(
                _evaluation_rng(round_no, baseline_samples, spec),
                baseline_samples,
                actual,
            )
            fair_vectors = [
                exact_vector(candidate, context, reference) for candidate in fair_candidates
            ]
            for name, model in models.items():
                weights = np.asarray(model["effective_weights"], dtype=float)
                actual_score = float(np.dot(weights, actual_vector))
                baseline_scores = np.asarray(
                    [np.dot(weights, vector) for vector in fair_vectors], dtype=float
                )
                percentiles[name].append(
                    tie_safe_percentile(actual_score, baseline_scores)
                )
            outer += 1
            if progress_every and outer % progress_every == 0:
                print(
                    f"v3.1 freeze candidate audit: {outer}/{expected} outer targets "
                    f"through draw {round_no}"
                )

        current_negatives = _sample_fair_candidates(
            _training_rng(round_no, spec),
            int(spec.train_negatives_per_target),
            actual,
        )
        current_vectors = [
            exact_vector(candidate, context, reference) for candidate in current_negatives
        ]
        _add_target(selection_stats, actual_vector, current_vectors)

        nested_negatives = sample_nested_training_negatives(
            round_no, max(counts), actual, spec, stream_delta=0
        )
        nested_vectors = [
            exact_vector(candidate, context, reference) for candidate in nested_negatives
        ]
        for count in counts:
            _add_target(nested_stats[count], actual_vector, nested_vectors[:count])

        _advance_history(state, actual, spec)

    selection_models = _fit_candidate_models(selection_stats, spec)
    latest_round = int(df.iloc[-1][ROUND_COLUMN])
    target_round = latest_round + 1
    latest_context = _feature_context(state, spec)
    latest_reference = sample_nested_reference(
        latest_context, target_round, reference_count, spec, stream_delta=0
    )
    candidate_seed = make_seed(latest_round, 1_101)
    candidates = [
        tuple(candidate)
        for candidate in generate_candidates(latest_candidate_count, candidate_seed)
    ]
    latest_vectors = [
        exact_vector(candidate, latest_context, latest_reference) for candidate in candidates
    ]
    score_vectors = {}
    for name, model in selection_models.items():
        weights = np.asarray(model["effective_weights"], dtype=float)
        score_vectors[name] = np.asarray(
            [np.dot(weights, vector) for vector in latest_vectors], dtype=float
        )

    summaries = {
        name: percentile_summary(values, bootstrap_reps, 35_100 + index)
        for index, (name, values) in enumerate(percentiles.items())
    }
    paired_vs_full = {
        name: paired_delta_summary(
            percentiles[name],
            percentiles["exact_full11"],
            bootstrap_reps,
            35_200 + index,
            f"{name}_minus_exact_full11",
        )
        for index, name in enumerate(CANDIDATE_FEATURE_SETS)
        if name != "exact_full11"
    }
    incremental = {
        "exact_no_pair_sum8_minus_exact_no_pair9": paired_delta_summary(
            percentiles["exact_no_pair_sum8"],
            percentiles["exact_no_pair9"],
            bootstrap_reps,
            35_301,
            "drop_sum_after_pair_removal",
        ),
        "exact_no_pair_sum_odd7_minus_exact_no_pair_sum8": paired_delta_summary(
            percentiles["exact_no_pair_sum_odd7"],
            percentiles["exact_no_pair_sum8"],
            bootstrap_reps,
            35_302,
            "drop_odd_after_pair_and_sum_removal",
        ),
    }
    latest_vs_full = {
        name: compare_score_vectors(score_vectors["exact_full11"], scores)
        for name, scores in score_vectors.items()
        if name != "exact_full11"
    }

    nested_models = {
        count: {
            finalist: fit_subset_pairwise_ridge(
                nested_stats[count],
                active_indices(CANDIDATE_FEATURE_SETS[finalist]),
                ridge_lambda=float(spec.ridge_lambda),
            )
            for finalist in FINALIST_NAMES
        }
        for count in counts
    }

    negative_stability = {}
    for finalist in FINALIST_NAMES:
        current_model = selection_models[finalist]
        current_weights = np.asarray(current_model["effective_weights"], dtype=float)
        current_scores = score_vectors[finalist]
        by_count = {}
        nested_scores = {}
        for count in counts:
            model = nested_models[count][finalist]
            weights = np.asarray(model["effective_weights"], dtype=float)
            scores = np.asarray(
                [np.dot(weights, vector) for vector in latest_vectors], dtype=float
            )
            nested_scores[count] = scores
            by_count[str(count)] = {
                "weights_vs_current64": compare_weight_vectors(current_weights, weights),
                "ranking_vs_current64": compare_score_vectors(current_scores, scores),
                "ridge": ridge_condition_diagnostics(model),
            }
        convergence = {}
        for left, right in zip(counts, counts[1:]):
            convergence[f"{left}->{right}"] = {
                "weights": compare_weight_vectors(
                    np.asarray(
                        nested_models[left][finalist]["effective_weights"], dtype=float
                    ),
                    np.asarray(
                        nested_models[right][finalist]["effective_weights"], dtype=float
                    ),
                ),
                "ranking": compare_score_vectors(
                    nested_scores[left], nested_scores[right]
                ),
            }
        negative_stability[finalist] = {
            "current_policy_negative_count": int(spec.train_negatives_per_target),
            "nested_counts": list(counts),
            "comparisons_vs_current_policy64": by_count,
            "nested_convergence": convergence,
        }

    return {
        "version": AUDIT_VERSION,
        "audit_role": "final_pre_freeze_diagnostic_no_automatic_promotion",
        "post_selection_exploratory_validation": True,
        "strict_target_isolation": True,
        "model_spec_source": spec_metadata(spec),
        "data": {"rows": int(len(df)), "sha256": dataset_fingerprint(df)},
        "runtime": runtime_identity(),
        "representation": {
            "structural_null": "exact_whole_universe_midrank",
            "evidence_reference": "nested_deterministic_fair_sample_empirical_midrank",
            "reference_count": reference_count,
            "reference_stream_delta": 0,
            "ridge_lambda": float(spec.ridge_lambda),
        },
        "candidate_feature_sets": {
            name: list(features) for name, features in CANDIDATE_FEATURE_SETS.items()
        },
        "historical": {
            "outer_tests": len(percentiles["exact_full11"]),
            "baseline_samples_per_target": baseline_samples,
            "scenario_summaries": summaries,
            "paired_candidate_minus_exact_full11": paired_vs_full,
            "incremental_nested_deltas": incremental,
        },
        "latest_sampled_ranking": {
            "target_draw": target_round,
            "candidate_count": latest_candidate_count,
            "candidate_seed": int(candidate_seed),
            "comparisons_vs_exact_full11": latest_vs_full,
        },
        "negative_count_stability": negative_stability,
    }
