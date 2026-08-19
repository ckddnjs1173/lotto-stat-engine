from __future__ import annotations

import math
from collections import Counter

PORTFOLIO_MAX_SHARED_NUMBERS = 2
PORTFOLIO_MAX_NUMBER_EXPOSURE_RATE = 0.40
FAIR_RANDOM_OVERLAP_GE3_RATE = 0.023834078570323606


def _fair_binomial_tail(trials: int, probability: float, at_least: int) -> float:
    return float(
        sum(
            math.comb(trials, hits)
            * probability**hits
            * (1.0 - probability) ** (trials - hits)
            for hits in range(at_least, trials + 1)
        )
    )


def select_strict_portfolio_entries(
    ranked: list[tuple[float, int, tuple[int, ...]]],
    top_k: int,
    max_shared_numbers: int = PORTFOLIO_MAX_SHARED_NUMBERS,
    max_number_exposure_rate: float = PORTFOLIO_MAX_NUMBER_EXPOSURE_RATE,
) -> tuple[list[tuple[float, int, tuple[int, ...]]], dict]:
    """Select in raw-score order without changing scores or relaxing constraints."""
    top_k = int(top_k)
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    max_shared_numbers = int(max_shared_numbers)
    if max_shared_numbers < 0 or max_shared_numbers > 6:
        raise ValueError("max_shared_numbers must be in 0..6")
    max_number_exposure_rate = float(max_number_exposure_rate)
    if not 0.0 < max_number_exposure_rate <= 1.0:
        raise ValueError("max_number_exposure_rate must be in (0,1]")

    exposure_cap = max(1, int(math.floor(top_k * max_number_exposure_rate)))
    selected: list[tuple[float, int, tuple[int, ...]]] = []
    selected_sets: list[set[int]] = []
    number_counts: Counter[int] = Counter()

    for entry in ranked:
        combo = tuple(entry[2])
        combo_set = set(combo)
        if any(len(combo_set & prior) > max_shared_numbers for prior in selected_sets):
            continue
        if any(number_counts[number] >= exposure_cap for number in combo):
            continue
        selected.append(entry)
        selected_sets.append(combo_set)
        number_counts.update(combo)
        if len(selected) >= top_k:
            break

    metadata = {
        "complete": len(selected) == top_k,
        "selected_count": len(selected),
        "requested_count": top_k,
        "max_shared_numbers": max_shared_numbers,
        "max_number_exposure_rate": max_number_exposure_rate,
        "max_number_ticket_count": exposure_cap,
        "fair_random_pair_overlap_ge_3_rate": FAIR_RANDOM_OVERLAP_GE3_RATE,
        "fair_random_fixed_number_exceeds_exposure_cap_rate": _fair_binomial_tail(
            top_k, 6.0 / 45.0, exposure_cap + 1
        ),
        "fallback_relaxed": False,
        "ranking_score_modified": False,
    }
    return selected, metadata
