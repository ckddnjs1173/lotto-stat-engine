from __future__ import annotations

import heapq
import random
from math import comb

import numpy as np
import pandas as pd

from .candidates import generate_candidates, iter_all_combinations, make_seed
from .config import BACKTEST_START_INDEX, DEFAULT_CANDIDATE_COUNT, ROUND_COLUMN
from .loader import row_numbers
from .v27_validation import DEFAULT_BOOTSTRAP_REPS, tie_safe_percentile
from .v28_reverse_ranking import (
    MIN_META_TRAIN_TARGETS,
    RIDGE_LAMBDA,
    TRAIN_NEGATIVES_PER_TARGET,
    _advance_history,
    _empty_history_state,
    _feature_context,
    _sample_fair_candidates,
)
from .v31_component_influence_audit import (
    _advance_training,
    _compare,
    _paired_delta,
    _pool_summary,
    _push,
    _summary,
    fit_subset_pairwise_ridge,
)
from .v31_final_directional_recommendation import (
    FEATURE_NAMES,
    REFERENCE_SAMPLES_PER_TARGET,
    _add_target,
    _empty_stats,
    _sample_reference,
    _tie_break,
    candidate_vector,
    fit_additive_pairwise_ridge,
)

AUDIT_VERSION = "v31_joint_ablation_audit_v1"

# These three terms showed large latest-pool structural distortion while their
# single-feature historical leave-one-out deltas were effectively neutral.
CLEAN3_REMOVED_FEATURES = (
    "previous_draw_overlap",
    "consecutive_pairs",
    "number_range",
)

# number_full_log_lift is kept in CLEAN3 because the single-feature audit showed
# the strongest positive historical delta of the four focused terms. CLEAN4 is a
# control that removes it as well so the joint test can verify that decision.
CLEAN4_REMOVED_FEATURES = (
    *CLEAN3_REMOVED_FEATURES,
    "number_full_log_lift",
)

SCENARIO_REMOVALS = {
    "full": (),
    "direct_without:clean3": CLEAN3_REMOVED_FEATURES,
    "retrained_without:clean3": CLEAN3_REMOVED_FEATURES,
    "direct_without:clean4": CLEAN4_REMOVED_FEATURES,
    "retrained_without:clean4": CLEAN4_REMOVED_FEATURES,
}

FAIR_EXPECTED_PREVIOUS_DRAW_OVERLAP = 36.0 / 45.0
FAIR_ANY_PREVIOUS_DRAW_OVERLAP_RATE = 1.0 - comb(39, 6) / comb(45, 6)
FAIR_EXPECTED_CONSECUTIVE_PAIRS = 2.0 / 3.0
FAIR_EXPECTED_NUMBER_RANGE = 5.0 * 46.0 / 7.0


def _indices_for_removed(removed_features) -> tuple[int, ...]:
    removed = {FEATURE_NAMES.index(name) for name in removed_features}
    return tuple(index for index in range(len(FEATURE_NAMES)) if index not in removed)


def _zero_filled_subset_weights(stats: dict, removed_features, ridge_lambda: float) -> np.ndarray:
    active = _indices_for_removed(removed_features)
    fitted = fit_subset_pairwise_ridge(stats, active, ridge_lambda=ridge_lambda)
    result = np.zeros(len(FEATURE_NAMES), dtype=float)
    result[list(active)] = np.asarray(fitted["effective_weights"], dtype=float)
    return result


def build_joint_scenario_weights(stats: dict, ridge_lambda: float = RIDGE_LAMBDA) -> dict[str, np.ndarray]:
    """Build full, direct-zero and retrained joint-ablation weight vectors."""
    full = fit_additive_pairwise_ridge(stats, ridge_lambda=ridge_lambda)
    full_weights = np.asarray(full["effective_weights"], dtype=float)
    result: dict[str, np.ndarray] = {"full": full_weights}

    for label, removed in (
        ("clean3", CLEAN3_REMOVED_FEATURES),
        ("clean4", CLEAN4_REMOVED_FEATURES),
    ):
        direct = full_weights.copy()
        for feature in removed:
            direct[FEATURE_NAMES.index(feature)] = 0.0
        result[f"direct_without:{label}"] = direct
        result[f"retrained_without:{label}"] = _zero_filled_subset_weights(
            stats, removed, float(ridge_lambda)
        )
    return result


def _score_all(vector: np.ndarray, weights: dict[str, np.ndarray]) -> dict[str, float]:
    value = np.asarray(vector, dtype=float)
    return {name: float(np.dot(weight, value)) for name, weight in weights.items()}


def run_historical_joint_ablation(
    df: pd.DataFrame,
    start_index: int = BACKTEST_START_INDEX,
    min_meta_train_targets: int = MIN_META_TRAIN_TARGETS,
    train_negatives_per_target: int = TRAIN_NEGATIVES_PER_TARGET,
    baseline_samples: int = 500,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    ridge_lambda: float = RIDGE_LAMBDA,
    reference_samples: int = REFERENCE_SAMPLES_PER_TARGET,
    progress_every: int = 0,
) -> dict:
    """Strict walk-forward comparison of full vs joint retrained ablations."""
    state = _empty_history_state()
    for idx in range(int(start_index)):
        _advance_history(state, tuple(row_numbers(df.iloc[idx])))

    stats = _empty_stats()
    scenario_names = ("full", "retrained_without:clean3", "retrained_without:clean4")
    percentiles = {name: [] for name in scenario_names}
    outer = 0
    expected = max(0, len(df) - int(start_index) - int(min_meta_train_targets))

    for idx in range(int(start_index), len(df)):
        actual = tuple(row_numbers(df.iloc[idx]))
        round_no = int(df.iloc[idx][ROUND_COLUMN])
        context = _feature_context(state)
        reference = _sample_reference(context, round_no, int(reference_samples))
        actual_vector = candidate_vector(actual, context, reference)

        if int(stats["solved_targets"]) >= int(min_meta_train_targets):
            weights = build_joint_scenario_weights(stats, float(ridge_lambda))
            actual_scores = _score_all(actual_vector, weights)
            rng = random.Random(
                round_no * 100_003
                + int(baseline_samples) * 1_009
                + int(min_meta_train_targets) * 37
                + 28_001
            )
            fair_eval = _sample_fair_candidates(rng, int(baseline_samples), actual)
            baselines = {name: [] for name in scenario_names}
            for candidate in fair_eval:
                vector = candidate_vector(candidate, context, reference)
                scores = _score_all(vector, weights)
                for name in scenario_names:
                    baselines[name].append(scores[name])
            for name in scenario_names:
                percentiles[name].append(
                    tie_safe_percentile(actual_scores[name], baselines[name])
                )
            outer += 1
            if progress_every and outer % int(progress_every) == 0:
                print(
                    f"v3.1 joint historical audit: {outer}/{expected} "
                    f"targets through draw {round_no}"
                )

        train_rng = random.Random(
            round_no * 200_003
            + int(train_negatives_per_target) * 2_009
            + 31_101
        )
        fair_train = _sample_fair_candidates(
            train_rng, int(train_negatives_per_target), actual
        )
        negative_vectors = [
            candidate_vector(candidate, context, reference) for candidate in fair_train
        ]
        _add_target(stats, actual_vector, negative_vectors)
        _advance_history(state, actual)

    summaries = {
        name: _summary(values, int(bootstrap_reps), 31_900 + index)
        for index, (name, values) in enumerate(percentiles.items())
    }
    paired = {
        "clean3": _paired_delta(
            percentiles["full"],
            percentiles["retrained_without:clean3"],
            int(bootstrap_reps),
            31_950,
        ),
        "clean4": _paired_delta(
            percentiles["full"],
            percentiles["retrained_without:clean4"],
            int(bootstrap_reps),
            31_951,
        ),
    }
    return {
        "strict_nested_walk_forward": True,
        "outer_tests": len(percentiles["full"]),
        "baseline_samples_per_target": int(baseline_samples),
        "summaries": summaries,
        "paired_full_minus_without": paired,
    }


def _latest_training_bundle(df: pd.DataFrame) -> dict:
    state = _empty_history_state()
    for idx in range(BACKTEST_START_INDEX):
        _advance_history(state, tuple(row_numbers(df.iloc[idx])))
    stats = _empty_stats()
    for idx in range(BACKTEST_START_INDEX, len(df)):
        actual = tuple(row_numbers(df.iloc[idx]))
        round_no = int(df.iloc[idx][ROUND_COLUMN])
        _advance_training(
            stats,
            state,
            actual,
            round_no,
            TRAIN_NEGATIVES_PER_TARGET,
            REFERENCE_SAMPLES_PER_TARGET,
        )
    latest_round = int(df.iloc[-1][ROUND_COLUMN])
    context = _feature_context(state)
    reference = _sample_reference(
        context, latest_round + 1, REFERENCE_SAMPLES_PER_TARGET
    )
    return {
        "stats": stats,
        "context": context,
        "reference": reference,
        "latest_round": latest_round,
        "target_round": latest_round + 1,
    }


def _fair_distance(summary: dict) -> dict:
    return {
        "previous_draw_overlap_minus_fair": (
            float(summary["mean_previous_draw_overlap"])
            - FAIR_EXPECTED_PREVIOUS_DRAW_OVERLAP
        ),
        "any_previous_draw_overlap_rate_minus_fair": (
            float(summary["any_previous_draw_overlap_rate"])
            - FAIR_ANY_PREVIOUS_DRAW_OVERLAP_RATE
        ),
        "consecutive_pairs_minus_fair": (
            float(summary["mean_consecutive_pairs"])
            - FAIR_EXPECTED_CONSECUTIVE_PAIRS
        ),
        "number_range_minus_fair": (
            float(summary["mean_number_range"])
            - FAIR_EXPECTED_NUMBER_RANGE
        ),
    }


def run_latest_joint_ablation(
    df: pd.DataFrame,
    exhaustive: bool = True,
    candidate_count: int = DEFAULT_CANDIDATE_COUNT,
    pool_size: int = 1_000,
    top_k: int = 10,
    seed_offset: int = 0,
    progress_every: int = 0,
) -> dict:
    """Rank full and joint-ablation scenarios in one candidate pass."""
    bundle = _latest_training_bundle(df)
    weights = build_joint_scenario_weights(bundle["stats"], RIDGE_LAMBDA)
    seed = make_seed(int(bundle["latest_round"]), int(seed_offset))

    if exhaustive:
        candidates = iter_all_combinations()
        available = 8_145_060
        mode = "exhaustive_all_8,145,060"
    else:
        candidates = generate_candidates(int(candidate_count), seed)
        available = int(candidate_count)
        mode = "sampled_candidates"

    keep = min(available, max(1, int(pool_size)))
    heaps = {name: [] for name in weights}
    evaluated = 0

    for evaluated, candidate in enumerate(candidates, 1):
        combo = tuple(sorted(int(number) for number in candidate))
        vector = candidate_vector(combo, bundle["context"], bundle["reference"])
        scores = _score_all(vector, weights)
        tie = _tie_break(combo, seed)
        for name, score in scores.items():
            _push(heaps[name], (score, tie, combo), keep)
        if progress_every and evaluated % int(progress_every) == 0:
            print(f"v3.1 joint latest audit: {evaluated:,} candidates evaluated")

    ranked = {
        name: sorted(heap, key=lambda entry: (entry[0], entry[1]), reverse=True)
        for name, heap in heaps.items()
    }
    previous_draw = frozenset(bundle["context"]["previous_draw"])
    summaries = {
        name: _pool_summary(entries, previous_draw, top_k)
        for name, entries in ranked.items()
    }
    comparisons = {
        name: _compare(summaries["full"], summaries[name], ranked["full"], entries)
        for name, entries in ranked.items()
        if name != "full"
    }
    fair_distance = {name: _fair_distance(summary) for name, summary in summaries.items()}

    return {
        "latest_draw": int(bundle["latest_round"]),
        "target_draw": int(bundle["target_round"]),
        "evaluation_mode": mode,
        "evaluated_count": int(evaluated),
        "pool_size": int(keep),
        "fair_references": {
            "expected_previous_draw_overlap": FAIR_EXPECTED_PREVIOUS_DRAW_OVERLAP,
            "any_previous_draw_overlap_rate": FAIR_ANY_PREVIOUS_DRAW_OVERLAP_RATE,
            "expected_consecutive_pairs": FAIR_EXPECTED_CONSECUTIVE_PAIRS,
            "expected_number_range": FAIR_EXPECTED_NUMBER_RANGE,
        },
        "scenario_summaries": summaries,
        "scenario_fair_distance": fair_distance,
        "scenario_comparisons_vs_full": comparisons,
    }


def run_joint_ablation_audit(
    df: pd.DataFrame,
    historical_baseline_samples: int = 500,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    latest_exhaustive: bool = True,
    latest_candidate_count: int = DEFAULT_CANDIDATE_COUNT,
    latest_pool_size: int = 1_000,
    top_k: int = 10,
    historical_progress_every: int = 0,
    latest_progress_every: int = 0,
) -> dict:
    return {
        "version": AUDIT_VERSION,
        "model_version_under_audit": "v31_directional_fair_null_additive_reverse_ridge_v1",
        "prediction_formula_modified": False,
        "clean3_removed_features": list(CLEAN3_REMOVED_FEATURES),
        "clean4_removed_features": list(CLEAN4_REMOVED_FEATURES),
        "historical_oos": run_historical_joint_ablation(
            df,
            baseline_samples=int(historical_baseline_samples),
            bootstrap_reps=int(bootstrap_reps),
            progress_every=int(historical_progress_every),
        ),
        "latest_ranking_influence": run_latest_joint_ablation(
            df,
            exhaustive=bool(latest_exhaustive),
            candidate_count=int(latest_candidate_count),
            pool_size=int(latest_pool_size),
            top_k=int(top_k),
            progress_every=int(latest_progress_every),
        ),
    }
