from __future__ import annotations

import heapq
from collections import Counter
from itertools import combinations

from .backtest import run_walk_forward_backtest
from .candidates import generate_candidates, iter_all_combinations, make_seed
from .config import DEFAULT_CANDIDATE_COUNT, DEFAULT_EXHAUSTIVE_RECOMMENDATION, TOP_K_RECOMMENDATIONS
from .loader import load_lotto_data
from .mixed_scoring import (
    build_all_combination_baseline,
    build_draw_structure_records,
    build_mixed_profile,
    score_mixed_record,
    structure_record,
)
from .mixed_subtypes import (
    build_exact_mixed_subtype_baseline,
    build_historical_subtype_records,
    latest_target_draw,
    mixed_subtype_fit_components,
    signature_family_diagnostics,
    subtype_information_diagnostics,
    suggest_mixed_signature_allocation,
    suggest_mixed_subtype_allocation,
)
from .profiles import build_profile
from .scoring import score_candidate, score_candidates
from .weights import load_weight_payload

MIXED_POOL_MULTIPLIER = 250
MIXED_DIVERSITY_STRENGTH = 8.0
MIXED_SUBTYPE_FIT_STRENGTH = 6.0
MIXED_FAMILY_OVERFILL_PENALTY = 3.0
MIXED_DUPLICATE_SIGNATURE_PENALTY = 2.0


def _mixed_similarity(left: dict, right: dict) -> float:
    """Statistical-structure similarity in [0, 1] for soft portfolio scoring."""
    a, b = left["structure_record"], right["structure_record"]
    overlap = len(set(left["numbers"]) & set(right["numbers"])) / 6.0
    high_a = frozenset(n for n in left["numbers"] if n >= 41)
    high_b = frozenset(n for n in right["numbers"] if n >= 41)
    high_cluster = 1.0 if high_a and high_a == high_b else 0.0
    parts = (
        (0.20, a["extreme_signature"] == b["extreme_signature"]),
        (0.18, a["section_distribution"] == b["section_distribution"]),
        (0.14, a["low_mid_high_distribution"] == b["low_mid_high_distribution"]),
        (0.20, overlap),
        (0.12, high_cluster),
        (0.08, a["sum_bin"] == b["sum_bin"]),
        (0.08, a["range_bin"] == b["range_bin"]),
    )
    return sum(weight * float(value) for weight, value in parts)


def _select_diverse_mixed(
    pool: list[dict],
    quota: int,
    family_allocation: dict[str, int] | None = None,
    subtype_diagnostics: dict[str, dict] | None = None,
    family_diagnostics: dict[str, dict] | None = None,
) -> list[dict]:
    family_allocation = family_allocation or {}
    subtype_diagnostics = subtype_diagnostics or {}
    family_diagnostics = family_diagnostics or {}
    remaining = sorted(pool, key=lambda item: float(item["mixed_slot_score"]), reverse=True)
    selected: list[dict] = []
    selected_families: Counter = Counter()
    selected_signatures: Counter = Counter()
    while remaining and len(selected) < quota:
        choices = []
        for index, candidate in enumerate(remaining):
            fit = mixed_subtype_fit_components(
                candidate, subtype_diagnostics, family_diagnostics,
                family_allocation, selected_families,
            )
            diversity_penalty = (MIXED_DIVERSITY_STRENGTH * sum(
                _mixed_similarity(candidate, prior) for prior in selected
            ) / len(selected) if selected else 0.0)
            family = fit["selected_family"]
            overfill_penalty = (
                MIXED_FAMILY_OVERFILL_PENALTY
                if selected_families[family] >= family_allocation.get(family, 0) else 0.0
            )
            signature_penalty = MIXED_DUPLICATE_SIGNATURE_PENALTY * selected_signatures[candidate["subtype_signature"]]
            portfolio_score = (
                float(candidate["mixed_slot_score"])
                + MIXED_SUBTYPE_FIT_STRENGTH * fit["mixed_subtype_fit_score"] / 100.0
                - diversity_penalty - overfill_penalty - signature_penalty
            )
            choices.append((portfolio_score, float(candidate["mixed_slot_score"]), index, fit,
                            diversity_penalty, overfill_penalty, signature_penalty))
        _, _, best_index, fit, penalty, overfill_penalty, signature_penalty = max(choices)
        item = remaining.pop(best_index)
        item.update(fit)
        item["mixed_diversity_penalty"] = round(penalty, 4)
        item["mixed_family_overfill_penalty"] = round(overfill_penalty, 4)
        item["mixed_duplicate_signature_penalty"] = round(signature_penalty, 4)
        item["v25_mixed_slot_score"] = round(
            0.90 * float(item["mixed_slot_score"]) + 0.10 * fit["mixed_subtype_fit_score"], 4
        )
        item["portfolio_selection_score"] = round(
            float(item["mixed_slot_score"])
            + MIXED_SUBTYPE_FIT_STRENGTH * fit["mixed_subtype_fit_score"] / 100.0
            - penalty - overfill_penalty - signature_penalty, 4
        )
        selected.append(item)
        selected_families[fit["selected_family"]] += 1
        selected_signatures[item["subtype_signature"]] += 1
    return selected


def _mixed_diagnostics(items: list[dict], family_allocation: dict[str, int] | None = None) -> dict:
    pairs = list(combinations(items, 2))
    overlaps = [len(set(a["numbers"]) & set(b["numbers"])) for a, b in pairs]
    frequencies = Counter(number for item in items for number in item["numbers"])
    tag_overlaps = [len(set(a["subtype_tags"]) & set(b["subtype_tags"])) for a, b in pairs]
    selected_families = Counter(item.get("selected_family", "unallocated") for item in items)
    return {
        "unique_extreme_signature_count": len({tuple(i["extreme_signature"]) for i in items}),
        "unique_section_distribution_count": len({tuple(i["structure_record"]["section_distribution"]) for i in items}),
        "average_pairwise_number_overlap": round(sum(overlaps) / len(overlaps), 4) if overlaps else 0.0,
        "average_subtype_tag_overlap": round(sum(tag_overlaps) / len(tag_overlaps), 4) if tag_overlaps else 0.0,
        "max_repeated_number_frequency": max(frequencies.values(), default=0),
        "mixed_diversity_penalty_applied": any(i.get("mixed_diversity_penalty", 0) > 0 for i in items),
        "unique_primary_subtype_count": len({item["primary_subtype"] for item in items}),
        "unique_subtype_signature_count": len({item["subtype_signature"] for item in items}),
        "family_allocation_target": dict(family_allocation or {}),
        "family_allocation_selected": dict(selected_families),
    }


def _top_k_stream(candidates, profile: dict, weights: dict[str, float], k: int) -> tuple[list[dict], int]:
    heap: list[tuple[float, int, dict]] = []
    evaluated = 0

    for evaluated, candidate in enumerate(candidates, 1):
        item = score_candidate(candidate, profile, weights)
        score = float(item["prediction_score"])
        entry = (score, evaluated, item)
        if len(heap) < k:
            heapq.heappush(heap, entry)
        elif score > heap[0][0]:
            heapq.heapreplace(heap, entry)

    top_items = [entry[2] for entry in sorted(heap, key=lambda x: x[0], reverse=True)]
    for idx, item in enumerate(top_items, 1):
        item["strategy"] = "TOP PREDICTION SCORE"
        item["rank"] = idx
    return top_items, evaluated


def _portfolio_allocation(latest_pattern_type: str, top_k: int) -> dict[str, int]:
    if top_k == 10 and latest_pattern_type == "outlier":
        return {"normal": 2, "mixed": 6, "outlier": 2}
    mixed = max(1, round(top_k * 0.60))
    remaining = top_k - mixed
    normal = remaining // 2
    return {"normal": normal, "mixed": mixed, "outlier": remaining - normal}


def _top_portfolio_stream(
    candidates,
    profile: dict,
    mixed_profile: dict,
    weights: dict[str, float],
    k: int,
    family_allocation: dict[str, int] | None = None,
    subtype_diagnostics: dict[str, dict] | None = None,
    family_diagnostics: dict[str, dict] | None = None,
) -> tuple[list[dict], int, dict[str, int]]:
    allocation = _portfolio_allocation(profile["latest_pattern_type"], k)
    heaps: dict[str, list[tuple[float, int, dict]]] = {
        candidate_type: [] for candidate_type in allocation
    }
    evaluated = 0
    for evaluated, candidate in enumerate(candidates, 1):
        record = structure_record(candidate, profile["latest_pattern_type"])
        candidate_type = record["pattern_type"]
        quota = allocation.get(candidate_type, 0)
        if quota <= 0:
            continue
        if candidate_type == "mixed":
            item = score_mixed_record(record, mixed_profile)
            ranking_score = float(item["mixed_slot_score"])
            item["features"] = record
            item["prediction_score"] = ranking_score
            item["score_breakdown"] = {
                key: item[key] for key in (
                    "mixed_lift_score",
                    "mixed_interaction_score",
                    "normal_backbone_score",
                    "controlled_extreme_score",
                    "recency_consistency_score",
                )
            }
        else:
            item = score_candidate(candidate, profile, weights)
            ranking_score = float(item["prediction_score"])
        entry = (ranking_score, evaluated, item)
        heap = heaps[candidate_type]
        pool_size = quota * MIXED_POOL_MULTIPLIER if candidate_type == "mixed" else quota
        if len(heap) < pool_size:
            heapq.heappush(heap, entry)
        elif ranking_score > heap[0][0]:
            heapq.heapreplace(heap, entry)

    selected = []
    for candidate_type in ("normal", "mixed", "outlier"):
        pool = [entry[2] for entry in sorted(heaps[candidate_type], key=lambda value: value[0], reverse=True)]
        selected.extend(_select_diverse_mixed(
            pool, allocation[candidate_type], family_allocation,
            subtype_diagnostics, family_diagnostics,
        ) if candidate_type == "mixed" else pool)
    selected.sort(key=lambda item: (
        0 if item["pattern_type"] == "mixed" else 1,
        -float(item.get("mixed_slot_score", item["prediction_score"])),
    ))
    for rank, item in enumerate(selected, 1):
        item["strategy"] = (
            "MIXED EMPIRICAL LIFT" if item["pattern_type"] == "mixed"
            else f"{item['pattern_type'].upper()} PORTFOLIO SLOT"
        )
        item["rank"] = rank
    return selected, evaluated, allocation


def build_recommendations(scored: list[dict], top_k: int = TOP_K_RECOMMENDATIONS) -> list[dict]:
    selected = sorted(scored, key=lambda item: item["prediction_score"], reverse=True)[:top_k]
    for idx, item in enumerate(selected, 1):
        item["strategy"] = "TOP PREDICTION SCORE"
        item["rank"] = idx
    return selected


def generate_recommendations(
    seed_offset: int = 0,
    candidate_count: int = DEFAULT_CANDIDATE_COUNT,
    exhaustive: bool = DEFAULT_EXHAUSTIVE_RECOMMENDATION,
    top_k: int = TOP_K_RECOMMENDATIONS,
) -> dict:
    df = load_lotto_data()
    profile = build_profile(df)
    records = build_draw_structure_records(df)
    subtype_records = build_historical_subtype_records(df)
    subtype_baseline = build_exact_mixed_subtype_baseline()
    family_allocation = suggest_mixed_signature_allocation(subtype_records, baseline=subtype_baseline)
    subtype_diagnostics = subtype_information_diagnostics(subtype_records, subtype_baseline)
    family_diagnostics = signature_family_diagnostics(subtype_records, subtype_baseline)
    mixed_profile = build_mixed_profile(records, build_all_combination_baseline())
    weight_payload = load_weight_payload()
    if weight_payload is None or "final_weights" not in weight_payload:
        weight_payload = run_walk_forward_backtest(df)
    weights = weight_payload["final_weights"]

    latest_round = int(profile["latest_round"])
    latest_draw, target_draw = latest_target_draw(df)
    seed = make_seed(latest_round, seed_offset)

    if exhaustive:
        recommendations, evaluated_count, allocation = _top_portfolio_stream(
            iter_all_combinations(), profile, mixed_profile, weights, top_k,
            family_allocation, subtype_diagnostics, family_diagnostics,
        )
        mode = "exhaustive_all_8,145,060"
    else:
        candidates = generate_candidates(candidate_count, seed)
        recommendations, evaluated_count, allocation = _top_portfolio_stream(
            candidates, profile, mixed_profile, weights, top_k,
            family_allocation, subtype_diagnostics, family_diagnostics,
        )
        mode = "sampled_candidates"

    return {
        "meta": {
            "latest_round": latest_round,
            "target_round": latest_round + 1,
            "latest_draw": latest_draw,
            "target_draw": target_draw,
            "seed": seed,
            "seed_offset": seed_offset,
            "candidate_count": candidate_count,
            "evaluated_count": evaluated_count,
            "recommendation_mode": mode,
            "weight_mode": "component backtest calibrated fixed blend",
            "mixed_model_version": "v2.3.1",
            "analysis_version": "v2.5",
            "portfolio_allocation": allocation,
            "mixed_diagnostics": _mixed_diagnostics(
                [item for item in recommendations if item["pattern_type"] == "mixed"], family_allocation
            ),
            "mixed_subtype_allocation_suggestion": suggest_mixed_subtype_allocation(subtype_records),
            "score_name": "prediction_score",
            "score_disclaimer": "prediction_score는 실제 당첨확률이 아니라 내부 예측확률점수입니다.",
        },
        "weights": weights,
        "weight_payload": weight_payload,
        "recommendations": recommendations,
    }


def print_recommendations(payload: dict) -> None:
    meta = payload["meta"]
    print("=" * 72)
    print("LOTTO STAT ENGINE v2.5 - MIXED SUBTYPE ALLOCATION")
    print("=" * 72)
    print(f"latest reflected draw: {meta['latest_draw']}")
    print(f"target draw: {meta['target_draw']}")
    print(f"평가 방식: {meta['recommendation_mode']}")
    print(f"평가 조합 수: {meta['evaluated_count']}")
    print(f"가중치 방식: {meta['weight_mode']}")
    print()
    print(meta["score_disclaimer"])
    diagnostics = meta["mixed_diagnostics"]
    print("MIXED PORTFOLIO DIAGNOSTICS")
    print(f"unique extreme_signature count: {diagnostics['unique_extreme_signature_count']}")
    print(f"unique section_distribution count: {diagnostics['unique_section_distribution_count']}")
    print(f"average pairwise number overlap: {diagnostics['average_pairwise_number_overlap']:.4f}")
    print(f"average subtype tag overlap: {diagnostics['average_subtype_tag_overlap']:.4f}")
    print(f"unique primary_subtype count: {diagnostics['unique_primary_subtype_count']}")
    print(f"unique subtype_signature count: {diagnostics['unique_subtype_signature_count']}")
    print(f"family allocation target: {diagnostics['family_allocation_target']}")
    print(f"family allocation selected: {diagnostics['family_allocation_selected']}")
    print(f"max repeated number frequency: {diagnostics['max_repeated_number_frequency']}")
    print(f"mixed diversity penalty applied: {'yes' if diagnostics['mixed_diversity_penalty_applied'] else 'no'}")
    print(f"mixed subtype allocation suggestion: {meta['mixed_subtype_allocation_suggestion']}")

    for idx, item in enumerate(payload["recommendations"], 1):
        f = item["features"]
        nums = " ".join(str(n) for n in item["numbers"])
        print()
        print(f"[{idx}] {item['strategy']}")
        print(nums)
        print(f"prediction_score: {item['prediction_score']}")
        print(f"pattern_type: {item['pattern_type']}")
        if item["pattern_type"] == "mixed":
            print(f"mixed_slot_score: {item['mixed_slot_score']:.4f}")
            print(f"extreme_count: {item['extreme_count']}")
            print(f"extreme_signature: {', '.join(item['extreme_signature'])}")
            print(f"subtype_tags: {', '.join(item['subtype_tags']) or 'none'}")
            print(f"subtype_signature: {item['subtype_signature']}")
            print(f"selected_family: {item['selected_family']}")
            print(f"mixed_subtype_fit_score: {item['mixed_subtype_fit_score']:.4f}")
        print("score breakdown:")
        for key, value in item["score_breakdown"].items():
            print(f"  {key}: {value:.4f}")
        if item["pattern_type"] == "mixed":
            for key in ("family_match_score", "subtype_information_score", "signature_support_score", "family_allocation_fit_score"):
                print(f"  {key}: {item[key]:.4f}")
        if item["pattern_type"] == "mixed":
            print(
                f"sum={f['sum']}, odd={f['odd_count']}, even={6 - f['odd_count']}, "
                f"range={f['number_range']}, max_gap={f['max_gap']}, "
                f"min_gap={f['min_gap']}, sections={f['section_distribution']}"
            )
        else:
            print(
                f"sum={f['sum']}, odd={f['odd_count']}, even={f['even_count']}, "
                f"gap_std={round(float(f['gap_std']), 4)}, "
                f"entropy={round(float(f['ending_digit_entropy']), 4)}, "
                f"section_entropy={round(float(f['section_entropy']), 4)}, "
                f"gap_entropy={round(float(f['gap_entropy']), 4)}"
            )

    weight_payload = payload.get("weight_payload") or {}
    if "percentile_scores" in weight_payload:
        print()
        print("FEATURE VALIDATION")
        for key in payload["weights"]:
            percentile = weight_payload["percentile_scores"].get(key, 0.0)
            stability = weight_payload["stability_scores"].get(key, 0.0)
            weight = payload["weights"].get(key, 0.0)
            print(f"{key}: percentile={percentile:.2f}%, stability={stability:.3f}, weight={weight:.6f}")
