from __future__ import annotations

import heapq
from collections import Counter

import numpy as np

from .candidates import (
    TOTAL_COMBINATION_COUNT,
    generate_candidates,
    iter_all_combinations,
    make_seed,
)
from .config import DEFAULT_CANDIDATE_COUNT, TOP_K_RECOMMENDATIONS
from .loader import load_lotto_data
from .strict_portfolio import select_strict_portfolio_entries
from .v31_core import (
    FEATURE_NAMES,
    _tie_break,
    build_latest_model,
    directional_raw_features,
    fair_null_transform,
    model_score,
)
from .v31_model_spec import (
    CLEAN3_CANDIDATE_SPEC,
    FULL11_BASELINE_SPEC,
    V31ModelSpec,
    spec_metadata,
)

PORTFOLIO_POOL_SIZE = 50_000
DIAGNOSTIC_POOL_SIZE = 1_000

SCENARIO_SPECS: dict[str, V31ModelSpec] = {
    "full11": FULL11_BASELINE_SPEC,
    "clean3": CLEAN3_CANDIDATE_SPEC,
}


def scenario_spec(name: str) -> V31ModelSpec:
    key = str(name).strip().lower()
    if key not in SCENARIO_SPECS:
        raise ValueError(f"unknown scenario {name!r}; choose one of {sorted(SCENARIO_SPECS)}")
    return SCENARIO_SPECS[key]


def _structure_metadata(numbers: tuple[int, ...]) -> dict:
    sections = [
        sum(1 <= number <= 10 for number in numbers),
        sum(11 <= number <= 20 for number in numbers),
        sum(21 <= number <= 30 for number in numbers),
        sum(31 <= number <= 40 for number in numbers),
        sum(41 <= number <= 45 for number in numbers),
    ]
    return {
        "sum": int(sum(numbers)),
        "odd_count": int(sum(number % 2 for number in numbers)),
        "number_range": int(numbers[-1] - numbers[0]),
        "consecutive_pair_count": int(
            sum(right - left == 1 for left, right in zip(numbers, numbers[1:]))
        ),
        "section_distribution": sections,
    }


def explain_candidate(numbers: tuple[int, ...], fitted: dict, spec: V31ModelSpec) -> dict:
    nums = tuple(sorted(int(number) for number in numbers))
    raw = directional_raw_features(nums, fitted["context"])
    bounded = fair_null_transform(raw, fitted["reference"])
    weights = np.asarray(fitted["model"]["effective_weights"], dtype=float)
    contributions = weights * bounded
    order = np.argsort(np.abs(contributions))[::-1]
    active = set(spec.active_features)
    return {
        "numbers": list(nums),
        "model_score": float(np.sum(contributions)),
        "structure_metadata": _structure_metadata(nums),
        "raw_features": {name: float(raw[i]) for i, name in enumerate(FEATURE_NAMES)},
        "bounded_features": {name: float(bounded[i]) for i, name in enumerate(FEATURE_NAMES)},
        "feature_contributions": [
            {
                "feature": FEATURE_NAMES[int(i)],
                "active": FEATURE_NAMES[int(i)] in active,
                "bounded_value": float(bounded[int(i)]),
                "weight": float(weights[int(i)]),
                "contribution": float(contributions[int(i)]),
            }
            for i in order
        ],
    }


def _ranking_distribution_diagnostics(
    ranked: list[tuple[float, int, tuple[int, ...]]],
    fitted: dict,
    pool_size: int = DIAGNOSTIC_POOL_SIZE,
) -> dict:
    entries = ranked[: min(int(pool_size), len(ranked))]
    if not entries:
        return {}

    combos = [entry[2] for entry in entries]
    counts = Counter(number for combo in combos for number in combo)
    previous = frozenset(fitted["context"]["previous_draw"])
    overlaps = [len(set(combo) & previous) for combo in combos]
    consecutive = [
        sum(right - left == 1 for left, right in zip(combo, combo[1:]))
        for combo in combos
    ]
    ranges = [combo[-1] - combo[0] for combo in combos]
    sums = [sum(combo) for combo in combos]

    saturation_counts = {name: 0 for name in FEATURE_NAMES}
    saturation_positive = {name: 0 for name in FEATURE_NAMES}
    saturation_negative = {name: 0 for name in FEATURE_NAMES}
    for combo in combos:
        bounded = fair_null_transform(
            directional_raw_features(combo, fitted["context"]),
            fitted["reference"],
        )
        for index, name in enumerate(FEATURE_NAMES):
            value = float(bounded[index])
            if value == 1.0 or value == -1.0:
                saturation_counts[name] += 1
            if value == 1.0:
                saturation_positive[name] += 1
            if value == -1.0:
                saturation_negative[name] += 1

    number_rates = {
        str(number): counts[number] / len(combos) for number in range(1, 46)
    }
    dominant = sorted(number_rates.items(), key=lambda item: (-item[1], int(item[0])))[:10]
    return {
        "role": "diagnostic_only_not_feature_deletion_gate",
        "pool_size": len(combos),
        "dominant_numbers": [
            {"number": int(number), "inclusion_rate": float(rate)}
            for number, rate in dominant
        ],
        "number_inclusion_rates": number_rates,
        "mean_previous_draw_overlap": float(np.mean(overlaps)),
        "any_previous_draw_overlap_rate": float(np.mean([value > 0 for value in overlaps])),
        "mean_consecutive_pairs": float(np.mean(consecutive)),
        "mean_number_range": float(np.mean(ranges)),
        "mean_sum": float(np.mean(sums)),
        "feature_saturation_rates": {
            name: saturation_counts[name] / len(combos) for name in FEATURE_NAMES
        },
        "feature_positive_saturation_rates": {
            name: saturation_positive[name] / len(combos) for name in FEATURE_NAMES
        },
        "feature_negative_saturation_rates": {
            name: saturation_negative[name] / len(combos) for name in FEATURE_NAMES
        },
    }


def _tie_diagnostics(
    ranked: list[tuple[float, int, tuple[int, ...]]],
    top_k: int,
) -> dict:
    if not ranked:
        return {}
    cutoff_index = min(int(top_k), len(ranked)) - 1
    cutoff_score = float(ranked[cutoff_index][0])
    equal_in_retained = sum(float(entry[0]) == cutoff_score for entry in ranked)
    top_scores = [float(entry[0]) for entry in ranked[: int(top_k)]]
    return {
        "tie_policy": FULL11_BASELINE_SPEC.tie_policy,
        "raw_top_k_cutoff_score": cutoff_score,
        "equal_cutoff_score_count_within_retained_pool": int(equal_in_retained),
        "exact_score_duplicate_count_inside_raw_top_k": int(
            len(top_scores) - len(set(top_scores))
        ),
    }


def generate_scenario_recommendations(
    scenario: str,
    exhaustive: bool = True,
    candidate_count: int = DEFAULT_CANDIDATE_COUNT,
    top_k: int = TOP_K_RECOMMENDATIONS,
    seed_offset: int = 0,
    progress_every: int = 0,
) -> dict:
    spec = scenario_spec(scenario)
    top_k = int(top_k)
    candidate_count = int(candidate_count)
    progress_every = int(progress_every)
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if progress_every < 0:
        raise ValueError("progress_every must be non-negative")

    df = load_lotto_data()
    fitted = build_latest_model(df, spec)
    sampling_seed = make_seed(int(fitted["latest_round"]), int(seed_offset))

    if exhaustive:
        candidates = iter_all_combinations()
        available = TOTAL_COMBINATION_COUNT
        mode = "exhaustive_all_8,145,060"
    else:
        candidates = generate_candidates(candidate_count, sampling_seed)
        available = len(candidates)
        mode = "sampled_candidates"
    if top_k > available:
        raise ValueError("top_k cannot exceed available candidate count")

    retained_count = min(
        available,
        max(top_k, DIAGNOSTIC_POOL_SIZE, PORTFOLIO_POOL_SIZE),
    )
    heap: list[tuple[float, int, tuple[int, ...]]] = []
    evaluated = 0
    for evaluated, candidate in enumerate(candidates, 1):
        nums = tuple(sorted(int(number) for number in candidate))
        entry = (model_score(nums, fitted), _tie_break(nums), nums)
        if len(heap) < retained_count:
            heapq.heappush(heap, entry)
        elif entry[:2] > heap[0][:2]:
            heapq.heapreplace(heap, entry)
        if progress_every and evaluated % progress_every == 0:
            print(
                f"v3.1 {scenario} experimental ranking progress: "
                f"{evaluated:,} candidates evaluated"
            )

    ranked = sorted(heap, key=lambda item: (item[0], item[1]), reverse=True)
    raw_entries = ranked[:top_k]
    raw_recommendations = [
        explain_candidate(entry[2], fitted, spec) for entry in raw_entries
    ]
    for rank, item in enumerate(raw_recommendations, 1):
        item["rank"] = rank

    portfolio_entries, portfolio_meta = select_strict_portfolio_entries(ranked, top_k)
    raw_rank_lookup = {entry[2]: rank for rank, entry in enumerate(ranked, 1)}
    portfolio_recommendations = [
        explain_candidate(entry[2], fitted, spec) for entry in portfolio_entries
    ]
    for rank, item in enumerate(portfolio_recommendations, 1):
        item["portfolio_rank"] = rank
        item["raw_rank_within_retained_pool"] = int(
            raw_rank_lookup[tuple(item["numbers"])]
        )

    return {
        "meta": {
            "scenario": str(scenario).lower(),
            "model_version": spec.name,
            "model_status": spec.status,
            "promotion_state": "historical_experimental_scenario_not_current_frozen_model",
            "model_spec": spec_metadata(spec),
            "latest_draw": int(fitted["latest_round"]),
            "target_draw": int(fitted["target_round"]),
            "evaluation_mode": mode,
            "evaluated_count": int(evaluated),
            "candidate_sampling_seed": int(sampling_seed) if not exhaustive else None,
            "formula": "score(c)=w_A^T z_A(c), z_j=2*F_mid,j(x_j)-1",
            "active_feature_names": list(spec.active_features),
            "all_feature_names": list(FEATURE_NAMES),
            "candidate_policy": "all valid 6-of-45 combinations remain eligible",
            "tie_policy": spec.tie_policy,
            "data": {
                "rows": int(fitted["dataset_rows"]),
                "sha256": str(fitted["dataset_sha256"]),
            },
            "training": {
                "history_draws": int(fitted["history_draws"]),
                "solved_targets": int(fitted["model"]["solved_targets"]),
                "pair_rows": int(fitted["model"]["pair_rows"]),
                "reference_samples_per_target": int(fitted["reference_samples"]),
                "train_negatives_per_target": int(fitted["train_negatives_per_target"]),
                "ridge_lambda": float(fitted["model"]["ridge_lambda"]),
                "scaled_second_moment_condition_number": float(
                    fitted["model"]["scaled_second_moment_condition_number"]
                ),
            },
            "portfolio": {
                "source_pool_size": int(len(ranked)),
                **portfolio_meta,
            },
        },
        "tie_diagnostics": _tie_diagnostics(ranked, top_k),
        "ranking_distribution_diagnostics": _ranking_distribution_diagnostics(
            ranked, fitted, DIAGNOSTIC_POOL_SIZE
        ),
        "recommendations": raw_recommendations,
        "portfolio_recommendations": portfolio_recommendations,
    }
