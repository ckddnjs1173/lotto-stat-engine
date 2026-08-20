from __future__ import annotations

import random
from itertools import combinations

import numpy as np
import pandas as pd

from .candidates import random_combination
from .config import BACKTEST_START_INDEX, ROUND_COLUMN
from .loader import row_numbers
from .v27_validation import DEFAULT_BOOTSTRAP_REPS, _block_bootstrap_mean_ci, tie_safe_percentile
from .v28_reverse_ranking import (
    FEATURE_NAMES as RAW_BASE_FEATURE_NAMES,
    MIN_META_TRAIN_TARGETS,
    RIDGE_LAMBDA,
    SCREENING_BASELINE_SAMPLES,
    TRAIN_NEGATIVES_PER_TARGET,
    _advance_history,
    _empty_history_state,
    _feature_context,
    _sample_fair_candidates,
    candidate_feature_vector,
)
from .v29_quadratic_reverse_ranking import (
    CONFIRMATION_BASELINE_SAMPLES,
    _add_solved_target,
    _empty_training_stats,
    fit_quadratic_pairwise_ridge,
    score_quadratic_vector,
)

MODEL_VERSION = "v30_null_bounded_quadratic_reverse_nested_ridge_v1"
REFERENCE_SAMPLES_PER_TARGET = 1024
REFERENCE_SEED_OFFSET = 30_301

BOUNDED_BASE_FEATURE_NAMES = tuple(
    f"null_midrank:{name}" for name in RAW_BASE_FEATURE_NAMES
)


def _quadratic_feature_names() -> tuple[str, ...]:
    names: list[str] = []
    names.extend(f"linear:{name}" for name in BOUNDED_BASE_FEATURE_NAMES)
    names.extend(f"square:{name}" for name in BOUNDED_BASE_FEATURE_NAMES)
    for left, right in combinations(BOUNDED_BASE_FEATURE_NAMES, 2):
        names.append(f"interaction:{left}*{right}")
    return tuple(names)


FEATURE_NAMES = _quadratic_feature_names()
if len(FEATURE_NAMES) != 65:  # pragma: no cover
    raise RuntimeError("v3.0 bounded quadratic basis must contain exactly 65 terms")


def _sample_null_reference(
    rng: random.Random,
    count: int,
) -> list[tuple[int, ...]]:
    """Sample unique fair combinations without using the target answer.

    Unlike evaluation-negative sampling, this reference distribution must be
    constructible before a target is revealed. Therefore no actual winner is passed
    to or excluded by this function.
    """
    count = int(count)
    if count <= 0:
        raise ValueError("reference count must be positive")
    seen: set[tuple[int, ...]] = set()
    result: list[tuple[int, ...]] = []
    attempts = 0
    max_attempts = max(2_000, count * 20)
    while len(result) < count and attempts < max_attempts:
        attempts += 1
        candidate = tuple(random_combination(rng))
        if candidate in seen:
            continue
        seen.add(candidate)
        result.append(candidate)
    if len(result) != count:
        raise RuntimeError(f"could not sample {count} unique null-reference candidates")
    return result


def build_null_reference_matrix(
    context: dict,
    round_no: int,
    reference_samples: int = REFERENCE_SAMPLES_PER_TARGET,
) -> np.ndarray:
    """Build sorted per-feature fair-null reference columns for one target.

    Cross-feature row identity is not used by the midrank transform, so each feature
    column is sorted once here. Candidate transformations can then use binary search
    without re-sorting the same 1,024 values for every scored candidate.
    """
    reference_samples = int(reference_samples)
    rng = random.Random(
        int(round_no) * 300_007
        + reference_samples * 3_011
        + REFERENCE_SEED_OFFSET
    )
    candidates = _sample_null_reference(rng, reference_samples)
    matrix = np.asarray(
        [candidate_feature_vector(candidate, context) for candidate in candidates],
        dtype=float,
    )
    return np.sort(matrix, axis=0)


def null_midrank_transform(
    raw_vector: np.ndarray,
    reference_matrix: np.ndarray,
) -> np.ndarray:
    """Map each raw feature to a bounded fair-null midrank coordinate in [-1, 1].

    `reference_matrix` is expected to contain independently sorted feature columns,
    as returned by `build_null_reference_matrix`.

    For feature j, let F_j be the empirical fair-reference distribution available
    before the target is revealed. The transformed coordinate is

        z_j = 2 * P_midrank(X_j <= x_j | fair null) - 1.

    Exact ties receive half credit. Values beyond the sampled support saturate at
    -1 or +1 instead of growing without bound. The transformation is monotone and
    does not encode any preferred direction such as balanced odd/even or centered
    sums.
    """
    raw = np.asarray(raw_vector, dtype=float)
    reference = np.asarray(reference_matrix, dtype=float)
    if raw.shape != (len(RAW_BASE_FEATURE_NAMES),):
        raise ValueError(
            f"raw vector must have shape ({len(RAW_BASE_FEATURE_NAMES)},)"
        )
    if reference.ndim != 2 or reference.shape[1] != len(RAW_BASE_FEATURE_NAMES):
        raise ValueError("reference matrix has unexpected shape")
    if reference.shape[0] <= 0:
        raise ValueError("reference matrix must contain rows")

    transformed = np.empty_like(raw, dtype=float)
    n = reference.shape[0]
    for index, value in enumerate(raw):
        column = reference[:, index]
        left = int(np.searchsorted(column, value, side="left"))
        right = int(np.searchsorted(column, value, side="right"))
        midrank_cdf = (left + right) / (2.0 * n)
        transformed[index] = 2.0 * midrank_cdf - 1.0
    return transformed


def bounded_quadratic_basis(bounded_vector: np.ndarray) -> np.ndarray:
    """Complete degree-2 basis of ten bounded null coordinates."""
    bounded = np.asarray(bounded_vector, dtype=float)
    if bounded.shape != (len(BOUNDED_BASE_FEATURE_NAMES),):
        raise ValueError(
            f"bounded vector must have shape ({len(BOUNDED_BASE_FEATURE_NAMES)},)"
        )
    if np.any(bounded < -1.0000000001) or np.any(bounded > 1.0000000001):
        raise ValueError("bounded coordinates must lie in [-1, 1]")

    values: list[float] = [float(value) for value in bounded]
    values.extend(float(value * value) for value in bounded)
    for left, right in combinations(range(len(bounded)), 2):
        values.append(float(bounded[left] * bounded[right]))
    result = np.asarray(values, dtype=float)
    if result.shape != (len(FEATURE_NAMES),):
        raise RuntimeError("unexpected bounded quadratic basis width")
    return result


def bounded_candidate_vector(
    numbers,
    context: dict,
    reference_matrix: np.ndarray,
) -> np.ndarray:
    raw = candidate_feature_vector(numbers, context)
    bounded = null_midrank_transform(raw, reference_matrix)
    return bounded_quadratic_basis(bounded)


def _mean(values: list[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _summary(values: list[float], bootstrap_reps: int) -> dict:
    recent300 = values[-min(300, len(values)):]
    recent100 = values[-min(100, len(values)):]
    centered = [float(value) - 50.0 for value in values]
    low, high = _block_bootstrap_mean_ci(
        centered,
        reps=int(bootstrap_reps),
        seed=83_001,
    )
    result = {
        "tests": int(len(values)),
        "mean_percentile": _mean(values),
        "recent_300_mean_percentile": _mean(recent300),
        "recent_100_mean_percentile": _mean(recent100),
        "above_random_median_ratio": (
            float(np.mean([value > 50.0 for value in values]))
            if values else float("nan")
        ),
        "percentile_minus_50_block_bootstrap_95_ci": [float(low), float(high)],
    }
    gate_pass = bool(
        result["tests"] > 0
        and float(result["mean_percentile"]) > 50.0
        and float(low) > 0.0
        and float(result["recent_300_mean_percentile"]) >= 50.0
        and float(result["recent_100_mean_percentile"]) >= 50.0
    )
    result["historical_gate_pass"] = gate_pass
    result["exploratory_screening_candidate"] = gate_pass
    return result


def _model_diagnostics(model: dict) -> dict:
    return {
        "ridge_lambda": float(model["ridge_lambda"]),
        "pair_rows": int(model["pair_rows"]),
        "solved_targets": int(model["solved_targets"]),
        "basis_width": len(FEATURE_NAMES),
        "features": {
            name: {
                "effective_weight": float(model["effective_weights"][index]),
                "scaled_weight": float(model["scaled_weights"][index]),
                "rms": float(model["rms"][index]),
            }
            for index, name in enumerate(FEATURE_NAMES)
        },
    }


def run_v30_null_bounded_reverse_ranking_audit(
    df: pd.DataFrame,
    start_index: int = BACKTEST_START_INDEX,
    min_meta_train_targets: int = MIN_META_TRAIN_TARGETS,
    train_negatives_per_target: int = TRAIN_NEGATIVES_PER_TARGET,
    baseline_samples: int = SCREENING_BASELINE_SAMPLES,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    ridge_lambda: float = RIDGE_LAMBDA,
    reference_samples: int = REFERENCE_SAMPLES_PER_TARGET,
    progress_every: int = 0,
    include_rows: bool = False,
) -> dict:
    """Strict nested screen for the bounded fair-null quadratic representation."""
    start_index = int(start_index)
    min_meta_train_targets = int(min_meta_train_targets)
    train_negatives_per_target = int(train_negatives_per_target)
    baseline_samples = int(baseline_samples)
    reference_samples = int(reference_samples)
    if len(df) <= start_index:
        raise ValueError("v3.0 reverse ranking requires targets after start_index")
    if min_meta_train_targets <= 0:
        raise ValueError("min_meta_train_targets must be positive")
    if train_negatives_per_target <= 0 or baseline_samples <= 0:
        raise ValueError("training and evaluation candidate counts must be positive")
    if reference_samples <= 0:
        raise ValueError("reference_samples must be positive")

    state = _empty_history_state()
    for idx in range(start_index):
        _advance_history(state, tuple(row_numbers(df.iloc[idx])))

    stats = _empty_training_stats()
    percentiles: list[float] = []
    rows: list[dict] = []
    total_solved_targets = len(df) - start_index
    expected_outer_tests = max(0, total_solved_targets - min_meta_train_targets)
    outer_completed = 0

    for idx in range(start_index, len(df)):
        actual = tuple(row_numbers(df.iloc[idx]))
        round_no = int(df.iloc[idx][ROUND_COLUMN])
        context = _feature_context(state)

        # This distribution uses only the pre-target context and an independent fair
        # RNG path. The target answer is not an input to reference construction.
        reference_matrix = build_null_reference_matrix(
            context,
            round_no,
            reference_samples=reference_samples,
        )
        actual_raw = candidate_feature_vector(actual, context)
        actual_bounded = null_midrank_transform(actual_raw, reference_matrix)
        actual_vector = bounded_quadratic_basis(actual_bounded)

        if int(stats["solved_targets"]) >= min_meta_train_targets:
            model = fit_quadratic_pairwise_ridge(stats, ridge_lambda=ridge_lambda)
            actual_score = score_quadratic_vector(actual_vector, model)

            # Reuse the v2.8/v2.9 evaluation seed path so model classes are compared
            # against the same fair alternatives for each historical target.
            eval_rng = random.Random(
                round_no * 100_003
                + baseline_samples * 1_009
                + min_meta_train_targets * 37
                + 28_001
            )
            fair_eval = _sample_fair_candidates(eval_rng, baseline_samples, actual)
            baseline_scores = [
                score_quadratic_vector(
                    bounded_candidate_vector(candidate, context, reference_matrix),
                    model,
                )
                for candidate in fair_eval
            ]
            percentile = tie_safe_percentile(actual_score, baseline_scores)
            percentiles.append(float(percentile))
            outer_completed += 1

            if include_rows:
                rows.append({
                    "target_index": int(idx),
                    "target_round": round_no,
                    "history_draws": int(state["history_draws"]),
                    "prior_solved_targets": int(stats["solved_targets"]),
                    "prior_pair_rows": int(stats["pair_rows"]),
                    "actual_numbers": list(actual),
                    "actual_raw_features": [float(value) for value in actual_raw],
                    "actual_bounded_features": [float(value) for value in actual_bounded],
                    "actual_score": float(actual_score),
                    "percentile": float(percentile),
                    "model": _model_diagnostics(model),
                })

            if progress_every and outer_completed % int(progress_every) == 0:
                print(
                    f"v3.0 null-bounded reverse ranking progress: "
                    f"{outer_completed}/{expected_outer_tests} outer targets "
                    f"(through draw {round_no})"
                )

        # Target becomes training evidence only after its outer score is fixed.
        train_rng = random.Random(
            round_no * 200_003
            + train_negatives_per_target * 2_009
            + 28_101
        )
        fair_train = _sample_fair_candidates(
            train_rng,
            train_negatives_per_target,
            actual,
        )
        negative_vectors = [
            bounded_candidate_vector(candidate, context, reference_matrix)
            for candidate in fair_train
        ]
        _add_solved_target(stats, actual_vector, negative_vectors)
        _advance_history(state, actual)

    summary = _summary(percentiles, int(bootstrap_reps))
    latest_model = fit_quadratic_pairwise_ridge(stats, ridge_lambda=ridge_lambda)
    payload = {
        "version": MODEL_VERSION,
        "strict_nested_walk_forward": True,
        "research_status": "exploratory_boundary_correction_after_v29_diagnostic",
        "target": "actual_winning_combination_rank_vs_fair_random_combinations",
        "start_index": start_index,
        "min_meta_train_targets": min_meta_train_targets,
        "train_negatives_per_target": train_negatives_per_target,
        "baseline_samples_per_outer_target": baseline_samples,
        "reference_samples_per_target": reference_samples,
        "bootstrap_reps": int(bootstrap_reps),
        "ridge_lambda": float(ridge_lambda),
        "raw_base_feature_names": list(RAW_BASE_FEATURE_NAMES),
        "bounded_base_feature_names": list(BOUNDED_BASE_FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "basis_width": len(FEATURE_NAMES),
        "outer_summary": summary,
        "latest_model_for_next_draw": _model_diagnostics(latest_model),
        "notes": {
            "pattern_type": "not used",
            "fair_null_coordinate": "per-target deterministic empirical midrank mapped to [-1,1]",
            "reference_target_independence": (
                "reference candidates are sampled without receiving or excluding the target answer"
            ),
            "target_reveal": (
                "target contributes training rows only after its outer score is recorded"
            ),
            "same_training_and_evaluation_sampling_as_v29": True,
            "quadratic_bound": "all linear/square/interaction basis values are in [-1,1]",
            "screen_confirmation_samples": CONFIRMATION_BASELINE_SAMPLES,
            "production_change": "none",
            "purpose": (
                "test whether v2.9 ranking information survives after removing polynomial "
                "magnitude extrapolation while preserving monotone fair-null ordering"
            ),
        },
    }
    if include_rows:
        payload["rows"] = rows
    return payload
