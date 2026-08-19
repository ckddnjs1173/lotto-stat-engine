from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from .candidates import TOTAL_COMBINATION_COUNT
from .v31_core import FEATURE_NAMES

EXACT_STRUCTURAL_FEATURES = (
    "previous_draw_overlap",
    "sum_signed_center_138",
    "high_minus_low_zone_count",
    "odd_count_signed_center_3",
    "number_range",
    "consecutive_pairs",
)


def _validated_distribution(counts: dict[int, int], name: str) -> dict[int, int]:
    cleaned = {int(value): int(count) for value, count in counts.items() if int(count) > 0}
    total = sum(cleaned.values())
    if total != TOTAL_COMBINATION_COUNT:
        raise RuntimeError(
            f"exact null distribution {name} totals {total:,}, expected {TOTAL_COMBINATION_COUNT:,}"
        )
    return dict(sorted(cleaned.items()))


@lru_cache(maxsize=1)
def previous_draw_overlap_distribution() -> dict[int, int]:
    return _validated_distribution(
        {
            overlap: math.comb(6, overlap) * math.comb(39, 6 - overlap)
            for overlap in range(7)
        },
        "previous_draw_overlap",
    )


@lru_cache(maxsize=1)
def odd_count_distribution() -> dict[int, int]:
    return _validated_distribution(
        {
            odd: math.comb(23, odd) * math.comb(22, 6 - odd)
            for odd in range(7)
            if odd <= 23 and 6 - odd <= 22
        },
        "odd_count",
    )


@lru_cache(maxsize=1)
def high_minus_low_distribution() -> dict[int, int]:
    counts: dict[int, int] = {}
    for low in range(7):
        for mid in range(7 - low):
            high = 6 - low - mid
            if low > 15 or mid > 15 or high > 15:
                continue
            value = high - low
            count = math.comb(15, low) * math.comb(15, mid) * math.comb(15, high)
            counts[value] = counts.get(value, 0) + count
    return _validated_distribution(counts, "high_minus_low_zone_count")


@lru_cache(maxsize=1)
def number_range_distribution() -> dict[int, int]:
    counts = {
        number_range: (45 - number_range) * math.comb(number_range - 1, 4)
        for number_range in range(5, 45)
    }
    return _validated_distribution(counts, "number_range")


@lru_cache(maxsize=1)
def consecutive_pairs_distribution() -> dict[int, int]:
    # A 6-number subset with a adjacent selected pairs has r=6-a runs of ones.
    # Number of binary strings of length 45, weight 6, with r runs is
    # C(5, r-1) * C(40, r).
    counts = {}
    for adjacent_pairs in range(6):
        runs = 6 - adjacent_pairs
        counts[adjacent_pairs] = math.comb(5, runs - 1) * math.comb(40, runs)
    return _validated_distribution(counts, "consecutive_pairs")


@lru_cache(maxsize=1)
def sum_distribution() -> dict[int, int]:
    # Exact DP for choosing six distinct values from 1..45 by total sum.
    max_sum = sum(range(40, 46))
    dp = [[0] * (max_sum + 1) for _ in range(7)]
    dp[0][0] = 1
    for number in range(1, 46):
        for chosen in range(5, -1, -1):
            row = dp[chosen]
            target = dp[chosen + 1]
            for total, count in enumerate(row):
                if count and total + number <= max_sum:
                    target[total + number] += count
    counts = {total: count for total, count in enumerate(dp[6]) if count}
    return _validated_distribution(counts, "sum")


def exact_distribution_for_feature(feature: str) -> dict[int, int]:
    if feature == "previous_draw_overlap":
        return previous_draw_overlap_distribution()
    if feature == "sum_signed_center_138":
        return {total - 138: count for total, count in sum_distribution().items()}
    if feature == "high_minus_low_zone_count":
        return high_minus_low_distribution()
    if feature == "odd_count_signed_center_3":
        return {odd - 3: count for odd, count in odd_count_distribution().items()}
    if feature == "number_range":
        return number_range_distribution()
    if feature == "consecutive_pairs":
        return consecutive_pairs_distribution()
    raise ValueError(f"feature does not have an exact structural null: {feature}")


def exact_midrank_z(value: float, counts: dict[int, int]) -> float:
    if not float(value).is_integer():
        raise ValueError("exact structural midrank requires an integer-valued feature")
    target = int(value)
    less = sum(count for observed, count in counts.items() if observed < target)
    equal = int(counts.get(target, 0))
    if equal == 0:
        if target < min(counts):
            return -1.0
        if target > max(counts):
            return 1.0
        raise ValueError(f"value {target} is outside the discrete support gap")
    cdf_mid = (less + 0.5 * equal) / TOTAL_COMBINATION_COUNT
    return 2.0 * cdf_mid - 1.0


def exact_structural_coordinates(raw_vector: np.ndarray) -> dict[str, float]:
    raw = np.asarray(raw_vector, dtype=float)
    if raw.shape != (len(FEATURE_NAMES),):
        raise ValueError("unexpected raw feature width")
    result = {}
    for feature in EXACT_STRUCTURAL_FEATURES:
        index = FEATURE_NAMES.index(feature)
        result[feature] = exact_midrank_z(
            raw[index], exact_distribution_for_feature(feature)
        )
    return result


def hybrid_exact_structural_transform(
    raw_vector: np.ndarray,
    sampled_bounded_vector: np.ndarray,
) -> np.ndarray:
    """Replace only structural coordinates with exact fair-null midranks.

    This function is deliberately not wired into the historical FULL11/CLEAN3
    baseline path. It exists for a separately audited representation scenario.
    """
    raw = np.asarray(raw_vector, dtype=float)
    bounded = np.asarray(sampled_bounded_vector, dtype=float).copy()
    if raw.shape != (len(FEATURE_NAMES),) or bounded.shape != (len(FEATURE_NAMES),):
        raise ValueError("unexpected v31 vector width")
    for feature, value in exact_structural_coordinates(raw).items():
        bounded[FEATURE_NAMES.index(feature)] = value
    return bounded
