from __future__ import annotations

from .features import normalize_numbers


def is_arithmetic_sequence(numbers: list[int]) -> bool:
    gaps = [b - a for a, b in zip(numbers, numbers[1:])]
    return len(set(gaps)) == 1


def passes_hard_filter(numbers: list[int], profile: dict | None = None) -> bool:
    """Deprecated compatibility shim.

    v2.2 does not structurally reject any valid 6/45 combination.  This function
    now performs input validation only and must not be used to prune candidates.
    """
    try:
        normalize_numbers(numbers)
    except ValueError:
        return False
    return True
