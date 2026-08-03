from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

import numpy as np

PRIMES = frozenset({2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43})
PATTERN_FLAG_KEYS = (
    "odd_even_extreme", "sum_low", "sum_high", "sum_extreme",
    "range_narrow", "range_wide", "max_gap_extreme", "has_consecutive_pairs",
)

def normalize_numbers(numbers: Iterable[int]) -> list[int]:
    values = sorted(int(n) for n in numbers)
    if len(values) != 6:
        raise ValueError("로또 번호는 정확히 6개여야 합니다.")
    if len(set(values)) != 6:
        raise ValueError("로또 번호 6개는 중복될 수 없습니다.")
    if min(values) < 1 or max(values) > 45:
        raise ValueError("로또 번호는 1~45 범위여야 합니다.")
    return values


def _entropy(values: list[int]) -> float:
    if not values:
        return 0.0
    total = len(values)
    counts = Counter(values)
    return float(-sum((count / total) * math.log2(count / total) for count in counts.values()))


def _entropy_from_counts(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    return float(-sum((count / total) * math.log2(count / total) for count in counts if count > 0))


def _max_consecutive_run(numbers: list[int]) -> int:
    best = 1
    current = 1
    for a, b in zip(numbers, numbers[1:]):
        if b == a + 1:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def _gap_bucket(gap: int) -> int:
    if gap <= 3:
        return 1
    if gap <= 6:
        return 2
    if gap <= 10:
        return 3
    return 4


def extract_features(numbers: Iterable[int], previous_draw: Iterable[int] | None = None) -> dict[str, float | int]:
    nums = normalize_numbers(numbers)
    gaps = [b - a for a, b in zip(nums, nums[1:])]
    endings = [n % 10 for n in nums]

    odd_count = sum(1 for n in nums if n % 2 == 1)
    low_count = sum(1 for n in nums if 1 <= n <= 15)
    mid_count = sum(1 for n in nums if 16 <= n <= 30)
    high_count = sum(1 for n in nums if 31 <= n <= 45)

    section_1 = sum(1 for n in nums if 1 <= n <= 10)
    section_2 = sum(1 for n in nums if 11 <= n <= 20)
    section_3 = sum(1 for n in nums if 21 <= n <= 30)
    section_4 = sum(1 for n in nums if 31 <= n <= 40)
    section_5 = sum(1 for n in nums if 41 <= n <= 45)
    section_counts = [section_1, section_2, section_3, section_4, section_5]
    total = int(sum(nums))
    number_range = int(max(nums) - min(nums))
    max_gap = int(max(gaps))
    consecutive_pairs = int(sum(1 for gap in gaps if gap == 1))
    ending_duplicate_count = int(sum(count - 1 for count in Counter(endings).values() if count > 1))
    empty_section_count = int(sum(count == 0 for count in section_counts))
    previous = set(normalize_numbers(previous_draw)) if previous_draw is not None else set()

    return {
        "sum": total,
        "mean": float(np.mean(nums)),
        "std": float(np.std(nums)),
        "range": number_range,
        "odd_count": int(odd_count),
        "even_count": int(6 - odd_count),
        "low_count": int(low_count),
        "mid_count": int(mid_count),
        "high_count": int(high_count),
        "section_1": int(section_1),
        "section_2": int(section_2),
        "section_3": int(section_3),
        "section_4": int(section_4),
        "section_5": int(section_5),
        "avg_gap": float(np.mean(gaps)),
        "min_gap": int(min(gaps)),
        "max_gap": max_gap,
        "gap_std": float(np.std(gaps)),
        "consecutive_pairs": consecutive_pairs,
        "max_consecutive_run": int(_max_consecutive_run(nums)),
        "duplicate_endings": ending_duplicate_count,
        "ending_duplicate_count": ending_duplicate_count,
        "empty_section_count": empty_section_count,
        "prime_count": int(sum(number in PRIMES for number in nums)),
        "previous_draw_proximity": int(len(set(nums) & previous)),
        "odd_even_extreme": int(odd_count in {0, 1, 5, 6}),
        "sum_low": int(total <= 105),
        "sum_high": int(total >= 170),
        "sum_extreme": int(total <= 90 or total >= 185),
        "range_narrow": int(number_range <= 20),
        "range_wide": int(number_range >= 41),
        "max_gap_extreme": int(max_gap >= 20),
        "has_consecutive_pairs": int(consecutive_pairs > 0),
        "ending_digit_entropy": float(_entropy(endings)),
        "section_entropy": float(_entropy_from_counts(section_counts)),
        "low_mid_high_entropy": float(_entropy_from_counts([low_count, mid_count, high_count])),
        "gap_entropy": float(_entropy([_gap_bucket(gap) for gap in gaps])),
    }


def pattern_flags(features: dict[str, float | int]) -> dict[str, int]:
    return {key: int(features[key]) for key in PATTERN_FLAG_KEYS}


def pattern_type(features: dict[str, float | int]) -> str:
    extreme_count = sum(int(features[key]) for key in (
        "odd_even_extreme", "sum_extreme", "range_narrow", "range_wide", "max_gap_extreme",
    ))
    secondary_count = int(features["has_consecutive_pairs"]) + int(features["empty_section_count"] >= 2)
    if extreme_count >= 2 or (extreme_count == 1 and secondary_count >= 2):
        return "outlier"
    if extreme_count == 0 and secondary_count == 0:
        return "normal"
    return "mixed"


def structure_type(features: dict[str, float | int]) -> str:
    total = float(features["sum"])
    std = float(features["std"])
    odd = int(features["odd_count"])

    if total <= 125:
        sum_band = "low"
    elif total <= 155:
        sum_band = "mid"
    else:
        sum_band = "high"

    if std <= 9:
        variance_band = "compact"
    elif std >= 14:
        variance_band = "wide"
    else:
        variance_band = "normal"

    return f"{sum_band}|{variance_band}|odd{odd}"
