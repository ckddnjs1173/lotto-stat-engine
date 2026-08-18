from __future__ import annotations

import hashlib
import heapq
import json
import random
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .candidates import generate_candidates, iter_all_combinations, make_seed, random_combination
from .config import BACKTEST_START_INDEX, DEFAULT_CANDIDATE_COUNT, ROUND_COLUMN, TOP_K_RECOMMENDATIONS
from .loader import load_lotto_data, row_numbers
from .mixed_scoring import structure_record
from .profiles import build_profile
from .number_evidence import UNIFORM_NUMBER_PROBABILITY
from .pair_evidence import PAIR_TO_INDEX, UNIFORM_PAIR_PROBABILITY
from .v28_reverse_ranking import (
    TRAIN_NEGATIVES_PER_TARGET,
    RIDGE_LAMBDA,
    _advance_history,
    _empty_history_state,
    _feature_context,
    _sample_fair_candidates,
)

MODEL_VERSION = "v31_directional_fair_null_additive_reverse_ridge_v1"
REFERENCE_SAMPLES_PER_TARGET = 1024
REFERENCE_SEED_OFFSET = 31_301
SELECTION_STRATEGY = "v31_directional_model_score_top_k"
AUDIT_POOL_SIZE = 1000

FEATURE_NAMES = (
    "number_full_log_lift",
    "pair_full_log_lift",
    "number_recent20_excess",
    "number_recent100_excess",
    "pair_recent100_excess",
    "previous_draw_overlap",
    "sum_signed_center_138",
    "high_minus_low_zone_count",
    "odd_count_signed_center_3",
    "number_range",
    "consecutive_pairs",
)


def _pair_indices(numbers: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(PAIR_TO_INDEX[(left, right)] for left, right in combinations(numbers, 2))


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


def _sample_reference(context: dict, round_no: int, count: int) -> np.ndarray:
    count = int(count)
    if count <= 0:
        raise ValueError("reference count must be positive")
    rng = random.Random(int(round_no) * 310_007 + count * 3_101 + REFERENCE_SEED_OFFSET)
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


def fit_additive_pairwise_ridge(stats: dict, ridge_lambda: float = RIDGE_LAMBDA) -> dict:
    rows = int(stats["pair_rows"])
    if rows <= 0:
        raise ValueError("training rows required")
    second = np.asarray(stats["sum_outer"], dtype=float) / rows
    mean_diff = np.asarray(stats["sum_diff"], dtype=float) / rows
    rms = np.sqrt(np.maximum(np.diag(second), 0.0) + 1e-12)
    scaled_second = second / np.outer(rms, rms)
    scaled_mean = mean_diff / rms
    system = scaled_second + float(ridge_lambda) * np.eye(len(FEATURE_NAMES), dtype=float)
    try:
        scaled_weights = np.linalg.solve(system, scaled_mean)
    except np.linalg.LinAlgError:
        scaled_weights = np.linalg.pinv(system) @ scaled_mean
    return {
        "ridge_lambda": float(ridge_lambda),
        "rms": rms,
        "scaled_weights": scaled_weights,
        "effective_weights": scaled_weights / rms,
        "pair_rows": rows,
        "solved_targets": int(stats["solved_targets"]),
    }


def build_latest_model(
    df: pd.DataFrame,
    start_index: int = BACKTEST_START_INDEX,
    train_negatives_per_target: int = TRAIN_NEGATIVES_PER_TARGET,
    reference_samples: int = REFERENCE_SAMPLES_PER_TARGET,
    ridge_lambda: float = RIDGE_LAMBDA,
) -> dict:
    start_index = int(start_index)
    if len(df) <= start_index:
        raise ValueError("not enough completed draws")
    state = _empty_history_state()
    for idx in range(start_index):
        _advance_history(state, tuple(row_numbers(df.iloc[idx])))
    stats = _empty_stats()
    for idx in range(start_index, len(df)):
        actual = tuple(row_numbers(df.iloc[idx]))
        round_no = int(df.iloc[idx][ROUND_COLUMN])
        context = _feature_context(state)
        reference = _sample_reference(context, round_no, reference_samples)
        actual_vector = candidate_vector(actual, context, reference)
        train_rng = random.Random(round_no * 200_003 + int(train_negatives_per_target) * 2_009 + 31_101)
        negatives = _sample_fair_candidates(train_rng, int(train_negatives_per_target), actual)
        negative_vectors = [candidate_vector(candidate, context, reference) for candidate in negatives]
        _add_target(stats, actual_vector, negative_vectors)
        _advance_history(state, actual)
    model = fit_additive_pairwise_ridge(stats, ridge_lambda)
    next_context = _feature_context(state)
    latest_round = int(df.iloc[-1][ROUND_COLUMN])
    next_reference = _sample_reference(next_context, latest_round + 1, reference_samples)
    return {
        "model": model,
        "context": next_context,
        "reference": next_reference,
        "latest_round": latest_round,
        "target_round": latest_round + 1,
        "history_draws": int(state["history_draws"]),
        "start_index": start_index,
        "reference_samples": int(reference_samples),
        "train_negatives_per_target": int(train_negatives_per_target),
    }


def model_score(numbers: Iterable[int], fitted: dict) -> float:
    vector = candidate_vector(numbers, fitted["context"], fitted["reference"])
    return float(np.dot(fitted["model"]["effective_weights"], vector))


def _tie_break(numbers: Iterable[int], seed: int) -> int:
    encoded = f"{int(seed)}:" + ",".join(str(int(n)) for n in numbers)
    return int.from_bytes(hashlib.blake2b(encoded.encode("ascii"), digest_size=8).digest(), "big")


def explain_candidate(numbers: Iterable[int], fitted: dict) -> dict:
    nums = tuple(sorted(int(n) for n in numbers))
    raw = directional_raw_features(nums, fitted["context"])
    bounded = fair_null_transform(raw, fitted["reference"])
    weights = np.asarray(fitted["model"]["effective_weights"], dtype=float)
    contributions = weights * bounded
    order = np.argsort(np.abs(contributions))[::-1]
    return {
        "numbers": list(nums),
        "model_score": float(np.sum(contributions)),
        "raw_features": {name: float(raw[i]) for i, name in enumerate(FEATURE_NAMES)},
        "bounded_features": {name: float(bounded[i]) for i, name in enumerate(FEATURE_NAMES)},
        "feature_contributions": [
            {
                "feature": FEATURE_NAMES[int(i)],
                "bounded_value": float(bounded[int(i)]),
                "weight": float(weights[int(i)]),
                "contribution": float(contributions[int(i)]),
            }
            for i in order
        ],
    }


def _bias_audit(entries: list[tuple[float, int, tuple[int, ...]]]) -> dict:
    numbers = [entry[2] for entry in entries]
    if not numbers:
        return {}
    counts = {str(n): 0 for n in range(1, 46)}
    sums = []
    one_digit = 0
    low = mid = high = 0
    for combo in numbers:
        sums.append(sum(combo))
        one_digit += int(any(n <= 9 for n in combo))
        for n in combo:
            counts[str(n)] += 1
            low += int(n <= 15)
            mid += int(16 <= n <= 30)
            high += int(n >= 31)
    total_slots = 6 * len(numbers)
    return {
        "pool_size": len(numbers),
        "fair_expected_number_inclusion_rate": 6.0 / 45.0,
        "number_inclusion_rates": {k: v / len(numbers) for k, v in counts.items()},
        "one_digit_present_rate": one_digit / len(numbers),
        "mean_sum": float(np.mean(sums)),
        "min_sum": int(min(sums)),
        "max_sum": int(max(sums)),
        "zone_slot_rates": {
            "1_15": low / total_slots,
            "16_30": mid / total_slots,
            "31_45": high / total_slots,
        },
    }


def generate_recommendations(
    exhaustive: bool = True,
    candidate_count: int = DEFAULT_CANDIDATE_COUNT,
    top_k: int = TOP_K_RECOMMENDATIONS,
    seed_offset: int = 0,
    progress_every: int = 0,
) -> dict:
    df = load_lotto_data()
    profile = build_profile(df)
    fitted = build_latest_model(df)
    seed = make_seed(int(fitted["latest_round"]), int(seed_offset))
    candidates = iter_all_combinations() if exhaustive else generate_candidates(int(candidate_count), seed)
    mode = "exhaustive_all_8,145,060" if exhaustive else "sampled_candidates"
    keep = max(int(top_k), min(AUDIT_POOL_SIZE, 8145060 if exhaustive else int(candidate_count)))
    heap: list[tuple[float, int, tuple[int, ...]]] = []
    evaluated = 0
    for evaluated, candidate in enumerate(candidates, 1):
        nums = tuple(sorted(int(n) for n in candidate))
        score = model_score(nums, fitted)
        entry = (score, _tie_break(nums, seed), nums)
        if len(heap) < keep:
            heapq.heappush(heap, entry)
        elif entry[:2] > heap[0][:2]:
            heapq.heapreplace(heap, entry)
        if progress_every and evaluated % int(progress_every) == 0:
            print(f"v3.1 final ranking progress: {evaluated:,} candidates evaluated")
    ranked = sorted(heap, key=lambda item: (item[0], item[1]), reverse=True)
    selected = [explain_candidate(entry[2], fitted) for entry in ranked[: int(top_k)]]
    for rank, item in enumerate(selected, 1):
        item["rank"] = rank
        record = structure_record(item["numbers"], str(profile["latest_pattern_type"]))
        item["pattern_type"] = str(record["pattern_type"])
        item["structure_metadata"] = {
            "sum": int(record["sum"]),
            "odd_count": int(record["odd_count"]),
            "number_range": int(record["number_range"]),
            "section_distribution": list(record["section_distribution"]),
        }
    return {
        "meta": {
            "model_version": MODEL_VERSION,
            "latest_draw": int(fitted["latest_round"]),
            "target_draw": int(fitted["target_round"]),
            "evaluation_mode": mode,
            "evaluated_count": int(evaluated),
            "selection_strategy": SELECTION_STRATEGY,
            "formula": "score(c)=w^T z(c), z_j=2*F_mid,j(x_j)-1",
            "feature_names": list(FEATURE_NAMES),
            "feature_policy": "direction preserved; fair-null bounded; additive only; no quadratic interactions",
            "candidate_policy": "all valid 6-of-45 combinations; no hard filters or pattern quotas",
            "pattern_type_role": "metadata_only_not_used_in_ranking",
            "training": {
                "history_draws": int(fitted["history_draws"]),
                "solved_targets": int(fitted["model"]["solved_targets"]),
                "pair_rows": int(fitted["model"]["pair_rows"]),
                "reference_samples_per_target": int(fitted["reference_samples"]),
                "train_negatives_per_target": int(fitted["train_negatives_per_target"]),
                "ridge_lambda": float(fitted["model"]["ridge_lambda"]),
            },
        },
        "bias_audit_top_pool": _bias_audit(ranked),
        "recommendations": selected,
    }


def write_json(payload: dict, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path.resolve()
