from __future__ import annotations

import heapq
import math
from collections import Counter

from .candidates import generate_candidates, iter_all_combinations, make_seed
from .config import DEFAULT_CANDIDATE_COUNT, TOP_K_RECOMMENDATIONS
from .loader import load_lotto_data
from .profiles import build_profile
from .v31_final_directional_recommendation import (
    AUDIT_POOL_SIZE,
    FAIR_RANDOM_OVERLAP_GE3_RATE,
    PORTFOLIO_MAX_SHARED_NUMBERS,
    _attach_structure,
    _bias_audit,
    _tie_break,
    build_latest_model,
    explain_candidate,
    model_score,
    write_json,
)

PORTFOLIO_POOL_SIZE = 50_000
PORTFOLIO_MAX_NUMBER_EXPOSURE_RATE = 0.40
PORTFOLIO_STRATEGY = (
    "highest_raw_score_subject_to_pairwise_shared_numbers_le_2_"
    "and_per_number_exposure_cap"
)


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
    """Select tickets without ever relaxing diversification constraints.

    Candidates are scanned in descending raw model-score order. A candidate is kept
    only when both of these conditions hold:

    1. it shares at most ``max_shared_numbers`` with every already selected ticket;
    2. adding it would not make any individual number appear on more than the
       exposure cap across the requested ticket set.

    The raw model score is never changed. If the retained source pool cannot supply
    ``top_k`` tickets, the function returns fewer tickets and marks the portfolio
    incomplete instead of silently relaxing the constraints.
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

    fair_number_probability = 6.0 / 45.0
    exceed_probability = _fair_binomial_tail(
        top_k,
        fair_number_probability,
        exposure_cap + 1,
    )
    metadata = {
        "complete": len(selected) == top_k,
        "selected_count": len(selected),
        "requested_count": top_k,
        "max_shared_numbers": max_shared_numbers,
        "max_number_exposure_rate": max_number_exposure_rate,
        "max_number_ticket_count": exposure_cap,
        "fair_random_pair_overlap_ge_3_rate": FAIR_RANDOM_OVERLAP_GE3_RATE,
        "fair_random_fixed_number_exceeds_exposure_cap_rate": exceed_probability,
        "fallback_relaxed": False,
        "ranking_score_modified": False,
    }
    return selected, metadata


def generate_recommendations(
    exhaustive: bool = True,
    candidate_count: int = DEFAULT_CANDIDATE_COUNT,
    top_k: int = TOP_K_RECOMMENDATIONS,
    seed_offset: int = 0,
    progress_every: int = 0,
) -> dict:
    """Run frozen v3.1 scoring and emit raw plus strictly diversified tickets."""
    df = load_lotto_data()
    profile = build_profile(df)
    fitted = build_latest_model(df)
    seed = make_seed(int(fitted["latest_round"]), int(seed_offset))

    if exhaustive:
        candidates = iter_all_combinations()
        mode = "exhaustive_all_8,145,060"
        available = 8_145_060
    else:
        candidates = generate_candidates(int(candidate_count), seed)
        mode = "sampled_candidates"
        available = int(candidate_count)

    retained_count = min(
        available,
        max(int(top_k), AUDIT_POOL_SIZE, PORTFOLIO_POOL_SIZE),
    )
    heap: list[tuple[float, int, tuple[int, ...]]] = []
    evaluated = 0
    for evaluated, candidate in enumerate(candidates, 1):
        nums = tuple(sorted(int(number) for number in candidate))
        score = model_score(nums, fitted)
        entry = (score, _tie_break(nums, seed), nums)
        if len(heap) < retained_count:
            heapq.heappush(heap, entry)
        elif entry[:2] > heap[0][:2]:
            heapq.heapreplace(heap, entry)
        if progress_every and evaluated % int(progress_every) == 0:
            print(f"v3.1 final ranking progress: {evaluated:,} candidates evaluated")

    ranked = sorted(heap, key=lambda item: (item[0], item[1]), reverse=True)
    latest_pattern_type = str(profile["latest_pattern_type"])
    raw_rank_lookup = {entry[2]: rank for rank, entry in enumerate(ranked, 1)}

    raw_entries = ranked[: int(top_k)]
    raw_recommendations = [explain_candidate(entry[2], fitted) for entry in raw_entries]
    for rank, item in enumerate(raw_recommendations, 1):
        item["rank"] = rank
        _attach_structure(item, latest_pattern_type)

    portfolio_entries, portfolio_meta = select_strict_portfolio_entries(
        ranked,
        int(top_k),
    )
    portfolio_recommendations = [
        explain_candidate(entry[2], fitted) for entry in portfolio_entries
    ]
    for rank, item in enumerate(portfolio_recommendations, 1):
        item["portfolio_rank"] = rank
        item["raw_rank_within_retained_pool"] = int(
            raw_rank_lookup[tuple(item["numbers"])]
        )
        _attach_structure(item, latest_pattern_type)

    audit_entries = ranked[: min(AUDIT_POOL_SIZE, len(ranked))]
    return {
        "meta": {
            "model_version": "v31_directional_fair_null_additive_reverse_ridge_v1",
            "latest_draw": int(fitted["latest_round"]),
            "target_draw": int(fitted["target_round"]),
            "evaluation_mode": mode,
            "evaluated_count": int(evaluated),
            "formula": "score(c)=w^T z(c), z_j=2*F_mid,j(x_j)-1",
            "feature_policy": (
                "direction preserved; fair-null bounded; additive only; "
                "no quadratic interactions"
            ),
            "candidate_policy": (
                "all valid 6-of-45 combinations; no hard filters or pattern quotas"
            ),
            "pattern_type_role": "metadata_only_not_used_in_ranking",
            "raw_ranking_modified": False,
            "portfolio": {
                "strategy": PORTFOLIO_STRATEGY,
                "source_pool_size": int(len(ranked)),
                **portfolio_meta,
            },
            "training": {
                "history_draws": int(fitted["history_draws"]),
                "solved_targets": int(fitted["model"]["solved_targets"]),
                "pair_rows": int(fitted["model"]["pair_rows"]),
                "reference_samples_per_target": int(fitted["reference_samples"]),
                "train_negatives_per_target": int(
                    fitted["train_negatives_per_target"]
                ),
                "ridge_lambda": float(fitted["model"]["ridge_lambda"]),
            },
        },
        "bias_audit_top_pool": _bias_audit(audit_entries),
        "portfolio_bias_audit": _bias_audit(portfolio_entries),
        "recommendations": raw_recommendations,
        "portfolio_recommendations": portfolio_recommendations,
    }
