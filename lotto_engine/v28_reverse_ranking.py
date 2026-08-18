from __future__ import annotations

import random
from collections import deque
from itertools import combinations

import numpy as np
import pandas as pd

from .candidates import random_combination
from .config import BACKTEST_START_INDEX, ROUND_COLUMN
from .evidence_scoring import NUMBER_PRODUCTION_SPEC, PAIR_PRODUCTION_SPEC
from .loader import row_numbers
from .number_evidence import UNIFORM_NUMBER_PROBABILITY
from .pair_evidence import PAIR_COUNT, PAIR_TO_INDEX, UNIFORM_PAIR_PROBABILITY
from .v27_validation import DEFAULT_BOOTSTRAP_REPS, _block_bootstrap_mean_ci, tie_safe_percentile

MODEL_VERSION = "v28_reverse_nested_ridge_v1"
FEATURE_NAMES = (
    "number_full_log_lift",
    "pair_full_log_lift",
    "number_recent20_excess",
    "number_recent100_excess",
    "pair_recent100_excess",
    "previous_draw_overlap",
    "sum_abs_deviation_138",
    "odd_imbalance",
    "number_range",
    "consecutive_pairs",
)

MIN_META_TRAIN_TARGETS = 100
TRAIN_NEGATIVES_PER_TARGET = 64
SCREENING_BASELINE_SAMPLES = 500
CONFIRMATION_BASELINE_SAMPLES = 2_000
RIDGE_LAMBDA = 2.0
RMS_EPSILON = 1e-12


def _pair_indices(numbers: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(PAIR_TO_INDEX[(left, right)] for left, right in combinations(numbers, 2))


def _empty_history_state() -> dict:
    return {
        "history_draws": 0,
        "number_hits": np.zeros(46, dtype=float),
        "pair_hits": np.zeros(PAIR_COUNT, dtype=float),
        "recent20_number_hits": np.zeros(46, dtype=float),
        "recent100_number_hits": np.zeros(46, dtype=float),
        "recent100_pair_hits": np.zeros(PAIR_COUNT, dtype=float),
        "recent20_draws": deque(),
        "recent100_draws": deque(),
        "previous_draw": None,
    }


def _add_number_hits(numbers: tuple[int, ...], target: np.ndarray, sign: float) -> None:
    for number in numbers:
        target[number] += sign


def _add_pair_hits(numbers: tuple[int, ...], target: np.ndarray, sign: float) -> None:
    for index in _pair_indices(numbers):
        target[index] += sign


def _advance_history(state: dict, numbers: tuple[int, ...]) -> None:
    numbers = tuple(sorted(int(number) for number in numbers))

    if len(state["recent20_draws"]) >= 20:
        old = state["recent20_draws"].popleft()
        _add_number_hits(old, state["recent20_number_hits"], -1.0)
    state["recent20_draws"].append(numbers)
    _add_number_hits(numbers, state["recent20_number_hits"], 1.0)

    if len(state["recent100_draws"]) >= 100:
        old = state["recent100_draws"].popleft()
        _add_number_hits(old, state["recent100_number_hits"], -1.0)
        _add_pair_hits(old, state["recent100_pair_hits"], -1.0)
    state["recent100_draws"].append(numbers)
    _add_number_hits(numbers, state["recent100_number_hits"], 1.0)
    _add_pair_hits(numbers, state["recent100_pair_hits"], 1.0)

    _add_number_hits(numbers, state["number_hits"], 1.0)
    _add_pair_hits(numbers, state["pair_hits"], 1.0)
    state["history_draws"] += 1
    state["previous_draw"] = numbers


def _feature_context(state: dict) -> dict:
    draws = int(state["history_draws"])
    if draws <= 0:
        raise ValueError("reverse ranking requires at least one history draw")

    number_alpha = float(NUMBER_PRODUCTION_SPEC["prior_strength"])
    pair_alpha = float(PAIR_PRODUCTION_SPEC["prior_strength"])

    number_posterior = (
        state["number_hits"][1:] + number_alpha * UNIFORM_NUMBER_PROBABILITY
    ) / (draws + number_alpha)
    number_lifts = np.zeros(46, dtype=float)
    number_lifts[1:] = np.log(
        np.maximum(number_posterior, 1e-15) / UNIFORM_NUMBER_PROBABILITY
    )

    pair_posterior = (
        state["pair_hits"] + pair_alpha * UNIFORM_PAIR_PROBABILITY
    ) / (draws + pair_alpha)
    pair_lifts = np.log(
        np.maximum(pair_posterior, 1e-15) / UNIFORM_PAIR_PROBABILITY
    )

    recent20_draws = max(1, len(state["recent20_draws"]))
    recent100_draws = max(1, len(state["recent100_draws"]))

    return {
        "number_lifts": number_lifts,
        "pair_lifts": pair_lifts,
        "recent20_number_rate": state["recent20_number_hits"] / recent20_draws,
        "recent100_number_rate": state["recent100_number_hits"] / recent100_draws,
        "recent100_pair_rate": state["recent100_pair_hits"] / recent100_draws,
        "previous_draw": frozenset(state["previous_draw"] or ()),
    }


def candidate_feature_vector(numbers, context: dict) -> np.ndarray:
    nums = tuple(sorted(int(number) for number in numbers))
    if len(nums) != 6 or len(set(nums)) != 6 or nums[0] < 1 or nums[-1] > 45:
        raise ValueError("reverse ranking candidate must contain six unique numbers in 1..45")

    pair_indices = _pair_indices(nums)
    gaps = [right - left for left, right in zip(nums, nums[1:])]
    odd_count = sum(number % 2 for number in nums)

    return np.asarray([
        float(np.mean([context["number_lifts"][number] for number in nums])),
        float(np.mean([context["pair_lifts"][index] for index in pair_indices])),
        float(np.mean([context["recent20_number_rate"][number] for number in nums]) - UNIFORM_NUMBER_PROBABILITY),
        float(np.mean([context["recent100_number_rate"][number] for number in nums]) - UNIFORM_NUMBER_PROBABILITY),
        float(np.mean([context["recent100_pair_rate"][index] for index in pair_indices]) - UNIFORM_PAIR_PROBABILITY),
        float(len(set(nums) & context["previous_draw"])),
        float(abs(sum(nums) - 138)),
        float(abs(odd_count - 3)),
        float(nums[-1] - nums[0]),
        float(sum(gap == 1 for gap in gaps)),
    ], dtype=float)


def _sample_fair_candidates(
    rng: random.Random,
    count: int,
    actual: tuple[int, ...],
) -> list[tuple[int, ...]]:
    count = int(count)
    if count <= 0:
        raise ValueError("fair candidate count must be positive")
    seen = {tuple(actual)}
    result: list[tuple[int, ...]] = []
    attempts = 0
    max_attempts = max(1_000, count * 20)
    while len(result) < count and attempts < max_attempts:
        attempts += 1
        candidate = tuple(random_combination(rng))
        if candidate in seen:
            continue
        seen.add(candidate)
        result.append(candidate)
    if len(result) != count:
        raise RuntimeError(f"could not sample {count} unique fair candidates")
    return result


def _empty_training_stats() -> dict:
    width = len(FEATURE_NAMES)
    return {
        "sum_outer": np.zeros((width, width), dtype=float),
        "sum_diff": np.zeros(width, dtype=float),
        "pair_rows": 0,
        "solved_targets": 0,
    }


def _add_solved_target(
    stats: dict,
    actual_vector: np.ndarray,
    negative_vectors: list[np.ndarray],
) -> None:
    for negative in negative_vectors:
        diff = np.asarray(actual_vector - negative, dtype=float)
        stats["sum_outer"] += np.outer(diff, diff)
        stats["sum_diff"] += diff
        stats["pair_rows"] += 1
    stats["solved_targets"] += 1


def fit_pairwise_ridge_ranker(stats: dict, ridge_lambda: float = RIDGE_LAMBDA) -> dict:
    rows = int(stats["pair_rows"])
    if rows <= 0:
        raise ValueError("pairwise ridge ranker requires prior solved training rows")
    ridge_lambda = float(ridge_lambda)
    if ridge_lambda <= 0.0:
        raise ValueError("ridge_lambda must be positive")

    second_moment = np.asarray(stats["sum_outer"], dtype=float) / rows
    mean_diff = np.asarray(stats["sum_diff"], dtype=float) / rows
    rms = np.sqrt(np.maximum(np.diag(second_moment), 0.0) + RMS_EPSILON)
    scaled_second = second_moment / np.outer(rms, rms)
    scaled_mean = mean_diff / rms
    system = scaled_second + ridge_lambda * np.eye(len(FEATURE_NAMES), dtype=float)
    try:
        scaled_weights = np.linalg.solve(system, scaled_mean)
    except np.linalg.LinAlgError:
        scaled_weights = np.linalg.pinv(system) @ scaled_mean
    effective_weights = scaled_weights / rms

    return {
        "ridge_lambda": ridge_lambda,
        "rms": rms,
        "scaled_weights": scaled_weights,
        "effective_weights": effective_weights,
        "pair_rows": rows,
        "solved_targets": int(stats["solved_targets"]),
    }


def score_feature_vector(vector: np.ndarray, model: dict) -> float:
    return float(np.dot(model["effective_weights"], np.asarray(vector, dtype=float)))


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
        seed=82_801,
    )
    result = {
        "tests": int(len(values)),
        "mean_percentile": _mean(values),
        "recent_300_mean_percentile": _mean(recent300),
        "recent_100_mean_percentile": _mean(recent100),
        "above_random_median_ratio": float(np.mean([value > 50.0 for value in values])) if values else float("nan"),
        "percentile_minus_50_block_bootstrap_95_ci": [float(low), float(high)],
    }
    result["screening_candidate"] = bool(
        result["tests"] > 0
        and float(result["mean_percentile"]) > 50.0
        and float(low) > 0.0
        and float(result["recent_300_mean_percentile"]) >= 50.0
        and float(result["recent_100_mean_percentile"]) >= 50.0
    )
    return result


def _model_diagnostics(model: dict) -> dict:
    return {
        "ridge_lambda": float(model["ridge_lambda"]),
        "pair_rows": int(model["pair_rows"]),
        "solved_targets": int(model["solved_targets"]),
        "features": {
            name: {
                "effective_weight": float(model["effective_weights"][index]),
                "scaled_weight": float(model["scaled_weights"][index]),
                "rms": float(model["rms"][index]),
            }
            for index, name in enumerate(FEATURE_NAMES)
        },
    }


def run_v28_reverse_ranking_audit(
    df: pd.DataFrame,
    start_index: int = BACKTEST_START_INDEX,
    min_meta_train_targets: int = MIN_META_TRAIN_TARGETS,
    train_negatives_per_target: int = TRAIN_NEGATIVES_PER_TARGET,
    baseline_samples: int = SCREENING_BASELINE_SAMPLES,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    ridge_lambda: float = RIDGE_LAMBDA,
    progress_every: int = 0,
    include_rows: bool = False,
) -> dict:
    """Nested rolling reverse-learning ranking audit with strict target isolation."""
    start_index = int(start_index)
    min_meta_train_targets = int(min_meta_train_targets)
    train_negatives_per_target = int(train_negatives_per_target)
    baseline_samples = int(baseline_samples)
    if len(df) <= start_index:
        raise ValueError("v2.8 reverse ranking requires targets after start_index")
    if min_meta_train_targets <= 0:
        raise ValueError("min_meta_train_targets must be positive")
    if train_negatives_per_target <= 0 or baseline_samples <= 0:
        raise ValueError("training and evaluation candidate counts must be positive")

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
        actual_vector = candidate_feature_vector(actual, context)

        if int(stats["solved_targets"]) >= min_meta_train_targets:
            model = fit_pairwise_ridge_ranker(stats, ridge_lambda=ridge_lambda)
            actual_score = score_feature_vector(actual_vector, model)
            eval_rng = random.Random(
                round_no * 100_003
                + baseline_samples * 1_009
                + min_meta_train_targets * 37
                + 28_001
            )
            fair_eval = _sample_fair_candidates(eval_rng, baseline_samples, actual)
            baseline_scores = [
                score_feature_vector(candidate_feature_vector(candidate, context), model)
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
                    "actual_score": float(actual_score),
                    "percentile": float(percentile),
                    "model": _model_diagnostics(model),
                })

            if progress_every and outer_completed % int(progress_every) == 0:
                print(
                    f"v2.8 reverse ranking progress: {outer_completed}/{expected_outer_tests} "
                    f"outer targets (through draw {round_no})"
                )

        # Reveal target only after its outer score is fixed. These examples can
        # influence t+1 and later, never target t itself.
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
            candidate_feature_vector(candidate, context)
            for candidate in fair_train
        ]
        _add_solved_target(stats, actual_vector, negative_vectors)
        _advance_history(state, actual)

    summary = _summary(percentiles, int(bootstrap_reps))
    latest_model = fit_pairwise_ridge_ranker(stats, ridge_lambda=ridge_lambda)
    payload = {
        "version": MODEL_VERSION,
        "strict_nested_walk_forward": True,
        "target": "actual_winning_combination_rank_vs_fair_random_combinations",
        "start_index": start_index,
        "min_meta_train_targets": min_meta_train_targets,
        "train_negatives_per_target": train_negatives_per_target,
        "baseline_samples_per_outer_target": baseline_samples,
        "bootstrap_reps": int(bootstrap_reps),
        "ridge_lambda": float(ridge_lambda),
        "feature_names": list(FEATURE_NAMES),
        "outer_summary": summary,
        "latest_model_for_next_draw": _model_diagnostics(latest_model),
        "notes": {
            "pattern_type": "not used",
            "target_reveal": "target contributes training rows only after its outer score is recorded",
            "screen_confirmation_samples": CONFIRMATION_BASELINE_SAMPLES,
            "production_change": "none until frozen screen and confirmation both pass",
        },
    }
    if include_rows:
        payload["rows"] = rows
    return payload
