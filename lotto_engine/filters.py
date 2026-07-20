from __future__ import annotations

from collections import Counter

from .features import extract_features, normalize_numbers


def is_arithmetic_sequence(numbers: list[int]) -> bool:
    gaps = [b - a for a, b in zip(numbers, numbers[1:])]
    return len(set(gaps)) == 1


def passes_hard_filter(numbers: list[int], profile: dict | None = None) -> bool:
    try:
        nums = normalize_numbers(numbers)
    except ValueError:
        return False

    features = extract_features(nums)
    odd = int(features["odd_count"])
    if odd in {0, 6}:
        return False

    if int(features["max_consecutive_run"]) >= 4:
        return False

    endings = [n % 10 for n in nums]
    if max(Counter(endings).values()) >= 4:
        return False

    if is_arithmetic_sequence(nums):
        return False

    total = int(features["sum"])
    if profile:
        lower = float(profile.get("sum_min", 70.0))
        upper = float(profile.get("sum_max", 190.0))
    else:
        lower, upper = 70.0, 190.0
    if total < lower or total > upper:
        return False

    return True
