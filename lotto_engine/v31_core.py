from __future__ import annotations

import random
from collections import deque
from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd

from .candidates import random_combination
from .config import ROUND_COLUMN
from .loader import dataset_fingerprint, row_numbers
from .number_evidence import UNIFORM_NUMBER_PROBABILITY
from .pair_evidence import PAIR_COUNT, PAIR_TO_INDEX, UNIFORM_PAIR_PROBABILITY
from .v31_model_spec import BASE_FEATURE_NAMES, FULL11_BASELINE_SPEC, V31ModelSpec

FEATURE_NAMES = BASE_FEATURE_NAMES
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


def _advance_history(
    state: dict,
    numbers: tuple[int, ...],
    spec: V31ModelSpec = FULL11_BASELINE_SPEC,
) -> None:
    numbers = tuple(sorted(int(number) for number in numbers))
    short_window = int(spec.recent_number_short_window)
    recent_window = int(spec.recent_window)
    if short_window != 20 or recent_window != 100:
        raise ValueError("v31 history state currently supports the frozen 20/100 windows only")

    if len(state["recent20_draws"]) >= short_window:
        old = state["recent20_draws"].popleft()
        _add_number_hits(old, state["recent20_number_hits"], -1.0)
    state["recent20_draws"].append(numbers)
    _add_number_hits(numbers, state["recent20_number_hits"], 1.0)

    if len(state["recent100_draws"]) >= recent_window:
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


def _feature_context(
    state: dict,
    spec: V31ModelSpec = FULL11_BASELINE_SPEC,
) -> dict:
    draws = int(state["history_draws"])
    if draws <= 0:
        raise ValueError("reverse ranking requires at least one history draw")

    number_alpha = float(spec.number_prior_strength)
    pair_alpha = float(spec.pair_prior_strength)

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


def directional_raw_features(numbers: Iterable[int], context: dict) -> np.ndarray:
    nums = tuple(sorted(int(number) for number in numbers))
    if len(nums) != 6 or len(set(nums)) != 6 or nums[0] < 1 or nums[-1] > 45:
        raise ValueError("candidate must contain six unique numbers in 1..45")
    pair_indices = _pair_indices(nums)
    gaps = [right - left for left, right in zip(nums, nums[1:])]
    odd_count = sum(number % 2 for number in nums)
    low_count = sum(number <= 15 for number in nums)
    high_count = sum(number >= 31 for number in nums)
    return np.asarray([
        float(np.mean([context["number_lifts"][number] for number in nums])),
        float(np.mean([context["pair_lifts"][index] for index in pair_indices])),
        float(np.mean([context["recent20_number_rate"][number] for number in nums]) - UNIFORM_NUMBER_PROBABILITY),
        float(np.mean([context["recent100_number_rate"][number] for number in nums]) - UNIFORM_NUMBER_PROBABILITY),
        float(np.mean([context["recent100_pair_rate"][index] for index in pair_indices]) - UNIFORM_PAIR_PROBABILITY),
        float(len(set(nums) & context["previous_draw"])),
        float(sum(nums) - 138),
        float(high_count - low_count),
        float(odd_count - 3),
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


def _sample_reference(
    context: dict,
    round_no: int,
    count: int,
    spec: V31ModelSpec = FULL11_BASELINE_SPEC,
) -> np.ndarray:
    count = int(count)
    if count <= 0:
        raise ValueError("reference count must be positive")
    rng = random.Random(
        int(round_no) * 310_007 + count * 3_101 + int(spec.reference_seed_offset)
    )
    seen: set[tuple[int, ...]] = set()
    rows: list[np.ndarray] = []
    while len(rows) < count:
        candidate = tuple(random_combination(rng))
        if candidate in seen:
            continue
        seen.add(candidate)
        rows.append(directional_raw_features(candidate, context))
    return np.sort(np.asarray(rows, dtype=float), axis=0)


def fair_null_transform(raw_vector: np.ndarray, sorted_reference: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw_vector, dtype=float)
    reference = np.asarray(sorted_reference, dtype=float)
    if raw.shape != (len(FEATURE_NAMES),):
        raise ValueError("unexpected raw feature width")
    if reference.ndim != 2 or reference.shape[1] != len(FEATURE_NAMES) or reference.shape[0] <= 0:
        raise ValueError("unexpected reference shape")
    n = reference.shape[0]
    result = np.empty(len(FEATURE_NAMES), dtype=float)
    for index, value in enumerate(raw):
        column = reference[:, index]
        left = int(np.searchsorted(column, value, side="left"))
        right = int(np.searchsorted(column, value, side="right"))
        result[index] = 2.0 * ((left + right) / (2.0 * n)) - 1.0
    return result


def candidate_vector(numbers: Iterable[int], context: dict, reference: np.ndarray) -> np.ndarray:
    return fair_null_transform(directional_raw_features(numbers, context), reference)


def _empty_stats() -> dict:
    width = len(FEATURE_NAMES)
    return {
        "sum_outer": np.zeros((width, width), dtype=float),
        "sum_diff": np.zeros(width, dtype=float),
        "pair_rows": 0,
        "solved_targets": 0,
    }


def _add_target(stats: dict, actual: np.ndarray, negatives: list[np.ndarray]) -> None:
    for negative in negatives:
        diff = np.asarray(actual - negative, dtype=float)
        stats["sum_outer"] += np.outer(diff, diff)
        stats["sum_diff"] += diff
        stats["pair_rows"] += 1
    stats["solved_targets"] += 1


def fit_subset_pairwise_ridge(
    stats: dict,
    active_indices: Iterable[int],
    ridge_lambda: float,
) -> dict:
    rows = int(stats["pair_rows"])
    if rows <= 0:
        raise ValueError("training rows required")
    active = tuple(int(index) for index in active_indices)
    if not active or len(set(active)) != len(active):
        raise ValueError("active feature indices must be non-empty and unique")
    if min(active) < 0 or max(active) >= len(FEATURE_NAMES):
        raise ValueError("active feature index out of range")
    ridge_lambda = float(ridge_lambda)
    if ridge_lambda <= 0.0:
        raise ValueError("ridge_lambda must be positive")

    second_full = np.asarray(stats["sum_outer"], dtype=float) / rows
    mean_full = np.asarray(stats["sum_diff"], dtype=float) / rows
    second = second_full[np.ix_(active, active)]
    mean_diff = mean_full[list(active)]
    rms = np.sqrt(np.maximum(np.diag(second), 0.0) + RMS_EPSILON)
    scaled_second = second / np.outer(rms, rms)
    scaled_mean = mean_diff / rms
    system = scaled_second + ridge_lambda * np.eye(len(active), dtype=float)
    try:
        scaled_weights = np.linalg.solve(system, scaled_mean)
    except np.linalg.LinAlgError:
        scaled_weights = np.linalg.pinv(system) @ scaled_mean

    effective_active = scaled_weights / rms
    effective_full = np.zeros(len(FEATURE_NAMES), dtype=float)
    scaled_full = np.zeros(len(FEATURE_NAMES), dtype=float)
    rms_full = np.zeros(len(FEATURE_NAMES), dtype=float)
    effective_full[list(active)] = effective_active
    scaled_full[list(active)] = scaled_weights
    rms_full[list(active)] = rms

    eigenvalues = np.linalg.eigvalsh(scaled_second)
    positive = eigenvalues[eigenvalues > RMS_EPSILON]
    condition_number = (
        float(np.max(positive) / np.min(positive)) if len(positive) else float("inf")
    )

    return {
        "ridge_lambda": ridge_lambda,
        "rms": rms_full,
        "scaled_weights": scaled_full,
        "effective_weights": effective_full,
        "active_indices": active,
        "pair_rows": rows,
        "solved_targets": int(stats["solved_targets"]),
        "scaled_second_moment_eigenvalues": eigenvalues,
        "scaled_second_moment_condition_number": condition_number,
    }


def fit_additive_pairwise_ridge(stats: dict, ridge_lambda: float) -> dict:
    return fit_subset_pairwise_ridge(
        stats,
        range(len(FEATURE_NAMES)),
        ridge_lambda=float(ridge_lambda),
    )


def _active_indices(spec: V31ModelSpec) -> tuple[int, ...]:
    names = tuple(spec.active_features)
    unknown = [name for name in names if name not in FEATURE_NAMES]
    if unknown:
        raise ValueError(f"unknown active features in model spec: {unknown}")
    return tuple(FEATURE_NAMES.index(name) for name in names)


def _training_rng(round_no: int, spec: V31ModelSpec) -> random.Random:
    return random.Random(
        int(round_no) * 200_003
        + int(spec.train_negatives_per_target) * 2_009
        + int(spec.training_seed_offset)
    )


def build_latest_model(df: pd.DataFrame, spec: V31ModelSpec) -> dict:
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
        reference = _sample_reference(
            context,
            round_no,
            int(spec.fair_reference_samples_per_target),
            spec,
        )
        actual_vector = candidate_vector(actual, context, reference)
        negatives = _sample_fair_candidates(
            _training_rng(round_no, spec),
            int(spec.train_negatives_per_target),
            actual,
        )
        negative_vectors = [
            candidate_vector(candidate, context, reference) for candidate in negatives
        ]
        _add_target(stats, actual_vector, negative_vectors)
        _advance_history(state, actual, spec)

    model = fit_subset_pairwise_ridge(
        stats,
        _active_indices(spec),
        ridge_lambda=float(spec.ridge_lambda),
    )
    latest_round = int(df.iloc[-1][ROUND_COLUMN])
    next_context = _feature_context(state, spec)
    next_reference = _sample_reference(
        next_context,
        latest_round + 1,
        int(spec.fair_reference_samples_per_target),
        spec,
    )
    return {
        "model": model,
        "context": next_context,
        "reference": next_reference,
        "latest_round": latest_round,
        "target_round": latest_round + 1,
        "history_draws": int(state["history_draws"]),
        "reference_samples": int(spec.fair_reference_samples_per_target),
        "train_negatives_per_target": int(spec.train_negatives_per_target),
        "dataset_rows": int(len(df)),
        "dataset_sha256": dataset_fingerprint(df),
        "spec": spec,
    }


def model_score(numbers: Iterable[int], fitted: dict) -> float:
    vector = candidate_vector(numbers, fitted["context"], fitted["reference"])
    return float(np.dot(fitted["model"]["effective_weights"], vector))


def _combination_code(numbers: Iterable[int]) -> int:
    code = 0
    nums = tuple(int(number) for number in numbers)
    if len(nums) != 6:
        raise ValueError("tie key requires exactly six numbers")
    for number in nums:
        code = code * 46 + number
    return code


def _tie_break(numbers: Iterable[int], seed: int | None = None) -> int:
    """Fixed, RNG-free tie key. Lexicographically smaller combination wins ties."""
    del seed
    return -_combination_code(tuple(sorted(int(number) for number in numbers)))
