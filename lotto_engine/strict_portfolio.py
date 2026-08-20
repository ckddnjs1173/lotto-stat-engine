from __future__ import annotations

import math
from collections import Counter
from itertools import combinations

PORTFOLIO_MAX_SHARED_NUMBERS = 2
PORTFOLIO_MAX_NUMBER_EXPOSURE_RATE = 0.40
PORTFOLIO_REPAIR_MAX_REMOVALS = 5
PORTFOLIO_REPAIR_ATTEMPT_LIMIT = 512
FAIR_RANDOM_OVERLAP_GE3_RATE = 0.023834078570323606

Entry = tuple[float, int, tuple[int, ...]]


def _fair_binomial_tail(trials: int, probability: float, at_least: int) -> float:
    return float(
        sum(
            math.comb(trials, hits)
            * probability**hits
            * (1.0 - probability) ** (trials - hits)
            for hits in range(at_least, trials + 1)
        )
    )


def _exposure_cap(ticket_count: int, rate: float) -> int:
    """Maximum appearances of one number for the *actual* ticket count."""
    return int(math.floor(int(ticket_count) * float(rate) + 1e-12))


def _is_compatible(
    combo_set: frozenset[int],
    selected_sets: list[frozenset[int]],
    number_counts: Counter[int],
    exposure_cap: int,
    max_shared_numbers: int,
) -> bool:
    if exposure_cap <= 0:
        return False
    if any(len(combo_set & prior) > max_shared_numbers for prior in selected_sets):
        return False
    if any(number_counts[number] >= exposure_cap for number in combo_set):
        return False
    return True


def _fill_from_fixed(
    ranked: list[Entry],
    combo_sets: list[frozenset[int]],
    target_count: int,
    exposure_cap: int,
    max_shared_numbers: int,
    fixed_indices: tuple[int, ...] = (),
    blocked_indices: frozenset[int] = frozenset(),
) -> list[int]:
    """Greedily refill around a fixed compatible subset in raw-rank order."""
    fixed = tuple(sorted(int(index) for index in fixed_indices))
    fixed_set = set(fixed)
    selected_indices: list[int] = []
    selected_sets: list[frozenset[int]] = []
    number_counts: Counter[int] = Counter()

    for index in fixed:
        combo_set = combo_sets[index]
        if not _is_compatible(
            combo_set,
            selected_sets,
            number_counts,
            exposure_cap,
            max_shared_numbers,
        ):
            raise ValueError("fixed portfolio subset is not internally feasible")
        selected_indices.append(index)
        selected_sets.append(combo_set)
        number_counts.update(combo_set)

    for index, combo_set in enumerate(combo_sets):
        if index in fixed_set or index in blocked_indices:
            continue
        if not _is_compatible(
            combo_set,
            selected_sets,
            number_counts,
            exposure_cap,
            max_shared_numbers,
        ):
            continue
        selected_indices.append(index)
        selected_sets.append(combo_set)
        number_counts.update(combo_set)
        if len(selected_indices) >= target_count:
            break

    return sorted(selected_indices)


def _search_target_count(
    ranked: list[Entry],
    combo_sets: list[frozenset[int]],
    target_count: int,
    exposure_cap: int,
    max_shared_numbers: int,
    repair_max_removals: int,
    repair_attempt_limit: int,
) -> tuple[list[int], dict]:
    """Greedy first, then deterministic minimum-removal repair.

    Repair never alters scores or constraints. At each removal depth it exhaustively
    tests subsets of the greedy selection until the attempt limit is reached, blocks
    the intentionally removed greedy entries, and greedily refills from the full raw
    ranked source pool. The first removal depth that can complete the target wins;
    ties at that depth use the lexicographically smallest raw-rank tuple.
    """
    baseline = _fill_from_fixed(
        ranked,
        combo_sets,
        target_count,
        exposure_cap,
        max_shared_numbers,
    )
    if len(baseline) >= target_count:
        return baseline[:target_count], {
            "greedy_selected_count": int(target_count),
            "repair_used": False,
            "repair_removed_from_greedy": 0,
            "repair_search_attempts": 0,
            "repair_attempt_limit_reached": False,
        }

    attempts = 0
    limit_reached = False
    max_remove = min(int(repair_max_removals), len(baseline))
    for remove_count in range(1, max_remove + 1):
        best: list[int] | None = None
        for removed_positions in combinations(range(len(baseline)), remove_count):
            if attempts >= int(repair_attempt_limit):
                limit_reached = True
                break
            attempts += 1
            removed_indices = frozenset(baseline[position] for position in removed_positions)
            fixed_indices = tuple(
                index for position, index in enumerate(baseline)
                if position not in removed_positions
            )
            trial = _fill_from_fixed(
                ranked,
                combo_sets,
                target_count,
                exposure_cap,
                max_shared_numbers,
                fixed_indices=fixed_indices,
                blocked_indices=removed_indices,
            )
            if len(trial) < target_count:
                continue
            trial = trial[:target_count]
            if best is None or tuple(trial) < tuple(best):
                best = trial
        if best is not None:
            return best, {
                "greedy_selected_count": len(baseline),
                "repair_used": True,
                "repair_removed_from_greedy": remove_count,
                "repair_search_attempts": attempts,
                "repair_attempt_limit_reached": limit_reached,
            }
        if limit_reached:
            break

    return baseline, {
        "greedy_selected_count": len(baseline),
        "repair_used": attempts > 0,
        "repair_removed_from_greedy": None,
        "repair_search_attempts": attempts,
        "repair_attempt_limit_reached": limit_reached,
    }


def _observed_portfolio_metrics(selected: list[Entry]) -> dict:
    if not selected:
        return {
            "observed_max_number_ticket_count": 0,
            "observed_max_number_exposure_rate": 0.0,
            "observed_max_pairwise_shared_numbers": 0,
        }
    counts = Counter(number for entry in selected for number in entry[2])
    combo_sets = [set(entry[2]) for entry in selected]
    max_shared = 0
    for index, current in enumerate(combo_sets):
        for prior in combo_sets[:index]:
            max_shared = max(max_shared, len(current & prior))
    max_count = max(counts.values()) if counts else 0
    return {
        "observed_max_number_ticket_count": int(max_count),
        "observed_max_number_exposure_rate": float(max_count / len(selected)),
        "observed_max_pairwise_shared_numbers": int(max_shared),
    }


def select_strict_portfolio_entries(
    ranked: list[Entry],
    top_k: int,
    max_shared_numbers: int = PORTFOLIO_MAX_SHARED_NUMBERS,
    max_number_exposure_rate: float = PORTFOLIO_MAX_NUMBER_EXPOSURE_RATE,
    repair_max_removals: int = PORTFOLIO_REPAIR_MAX_REMOVALS,
    repair_attempt_limit: int = PORTFOLIO_REPAIR_ATTEMPT_LIMIT,
) -> tuple[list[Entry], dict]:
    """Select a strict diversified portfolio without changing raw model scores.

    The requested-size search uses the exposure cap implied by ``top_k``. If greedy
    selection cannot fill the request, a bounded deterministic repair search replaces
    a minimum number of greedy choices and refills from the same raw-ranked source
    pool. If the requested size still cannot be completed, smaller sizes are searched
    again using *their own* exposure cap. A partial greedy set is never returned just
    because a larger denominator would have made its exposure look acceptable.
    """
    top_k = int(top_k)
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    max_shared_numbers = int(max_shared_numbers)
    if max_shared_numbers < 0 or max_shared_numbers > 6:
        raise ValueError("max_shared_numbers must be in 0..6")
    max_number_exposure_rate = float(max_number_exposure_rate)
    if not 0.0 < max_number_exposure_rate <= 1.0:
        raise ValueError("max_number_exposure_rate must be in (0,1]")
    repair_max_removals = int(repair_max_removals)
    repair_attempt_limit = int(repair_attempt_limit)
    if repair_max_removals < 0 or repair_attempt_limit < 0:
        raise ValueError("repair search limits must be non-negative")

    ranked = list(ranked)
    combo_sets = [frozenset(int(number) for number in entry[2]) for entry in ranked]
    requested_cap = _exposure_cap(top_k, max_number_exposure_rate)

    selected_indices: list[int] = []
    search_meta = {
        "greedy_selected_count": 0,
        "repair_used": False,
        "repair_removed_from_greedy": 0,
        "repair_search_attempts": 0,
        "repair_attempt_limit_reached": False,
    }
    selected_target = 0

    max_target = min(top_k, len(ranked))
    for target_count in range(max_target, 0, -1):
        exposure_cap = _exposure_cap(target_count, max_number_exposure_rate)
        if exposure_cap <= 0:
            continue
        indices, current_meta = _search_target_count(
            ranked,
            combo_sets,
            target_count,
            exposure_cap,
            max_shared_numbers,
            repair_max_removals,
            repair_attempt_limit,
        )
        if len(indices) >= target_count:
            selected_indices = indices[:target_count]
            search_meta = current_meta
            selected_target = target_count
            break

    selected = [ranked[index] for index in sorted(selected_indices)]
    observed = _observed_portfolio_metrics(selected)
    actual_count = len(selected)
    actual_cap = _exposure_cap(actual_count, max_number_exposure_rate) if actual_count else 0
    exposure_ok = (
        observed["observed_max_number_ticket_count"] <= actual_cap
        and observed["observed_max_number_exposure_rate"] <= max_number_exposure_rate + 1e-12
    ) if actual_count else True
    overlap_ok = observed["observed_max_pairwise_shared_numbers"] <= max_shared_numbers

    metadata = {
        "complete": actual_count == top_k,
        "selected_count": actual_count,
        "requested_count": top_k,
        "selected_target_count": int(selected_target),
        "no_feasible_strict_size_found": actual_count == 0,
        "max_shared_numbers": max_shared_numbers,
        "max_number_exposure_rate": max_number_exposure_rate,
        "requested_max_number_ticket_count": requested_cap,
        "max_number_ticket_count": actual_cap,
        "actual_denominator_exposure_check": True,
        "exposure_constraint_satisfied": bool(exposure_ok),
        "overlap_constraint_satisfied": bool(overlap_ok),
        "selection_strategy": (
            "raw_rank_greedy_then_minimum-removal_repair; "
            "incomplete sizes are revalidated with their actual denominator"
        ),
        "repair_max_removals": repair_max_removals,
        "repair_attempt_limit": repair_attempt_limit,
        "source_pool_size_checked": len(ranked),
        **search_meta,
        **observed,
        "fair_random_pair_overlap_ge_3_rate": FAIR_RANDOM_OVERLAP_GE3_RATE,
        "fair_random_fixed_number_exceeds_exposure_cap_rate": (
            _fair_binomial_tail(actual_count, 6.0 / 45.0, actual_cap + 1)
            if actual_count else 0.0
        ),
        "fallback_relaxed": False,
        "ranking_score_modified": False,
        "portfolio_entries_sorted_by_raw_rank": True,
    }
    return selected, metadata
