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
    family_dynamic_scores,
    mixed_subtype_fit_components,
    signature_family_diagnostics,
    subtype_information_diagnostics,
    suggest_mixed_signature_allocation,
    suggest_mixed_subtype_allocation,
    signature_family,
)
from .profiles import build_profile
from .scoring import score_candidate, score_candidates
from .weights import load_weight_payload

MIXED_POOL_MULTIPLIER = 250
MIXED_DIVERSITY_STRENGTH = 8.0
MIXED_SUBTYPE_FIT_STRENGTH = 6.0
MIXED_FAMILY_OVERFILL_PENALTY = 3.0
MIXED_DUPLICATE_SIGNATURE_PENALTY = 2.0
PATTERN_TYPES = ("normal", "mixed", "outlier")


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
    selected_families = Counter(
        item.get("selected_family", "unallocated")
        for item in items if item["pattern_type"] == "mixed"
    )
    return {
        "unique_extreme_signature_count": len({tuple(i.get("extreme_signature", i["structure_record"]["extreme_signature"])) for i in items}),
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


def _portfolio_allocation(profile: dict, top_k: int) -> dict[str, int]:
    probabilities = profile["transition_probs"].get(
        profile["latest_pattern_type"], profile["pattern_type_probs"]
    )
    raw = {key: top_k * probabilities.get(key, 0.0) for key in PATTERN_TYPES}
    allocation = {key: int(value) for key, value in raw.items()}
    for key in sorted(raw, key=lambda item: raw[item] - allocation[item], reverse=True)[:top_k - sum(allocation.values())]:
        allocation[key] += 1
    return allocation


def _type_pool_sizes(allocation: dict[str, int], k: int) -> dict[str, int]:
    """Reserve independent score pools so one pattern type cannot erase the others.

    Allocation remains a soft portfolio preference: even a type with a zero
    target keeps a small reserve and can still win slots on prediction score.
    """
    return {
        candidate_type: max(k, max(1, allocation.get(candidate_type, 0)) * MIXED_POOL_MULTIPLIER)
        for candidate_type in PATTERN_TYPES
    }


def _push_type_candidate(
    heaps: dict[str, list[tuple[float, int, dict]]],
    pool_sizes: dict[str, int],
    candidate_type: str,
    entry: tuple[float, int, dict],
) -> None:
    heap = heaps.setdefault(candidate_type, [])
    pool_size = pool_sizes.get(candidate_type, max(1, len(heap)))
    if len(heap) < pool_size:
        heapq.heappush(heap, entry)
    elif entry[0] > heap[0][0]:
        heapq.heapreplace(heap, entry)


def _flatten_type_heaps(heaps: dict[str, list[tuple[float, int, dict]]]) -> list[dict]:
    entries = [entry for heap in heaps.values() for entry in heap]
    return [entry[2] for entry in sorted(entries, key=lambda value: value[0], reverse=True)]


def _score_structure_record(record: dict, profile: dict, mixed_profile: dict, weights: dict) -> dict:
    family = signature_family(record["subtype_signature"])
    if record["pattern_type"] == "mixed":
        item = score_mixed_record(record, mixed_profile)
        item["features"] = record
        item["prediction_score"] = item["mixed_slot_score"]
        item["score_breakdown"] = {
            key: item[key] for key in (
                "mixed_lift_score", "mixed_interaction_score", "normal_backbone_score",
                "controlled_extreme_score", "recency_consistency_score",
            )
        }
    else:
        item = score_candidate(list(record["numbers"]), profile, weights)
        base_score = float(item["prediction_score"])
        transition_score, momentum_score = family_dynamic_scores(
            family, mixed_profile["dynamic_markov_decay"]
        )
        item.update({
            "base_score": round(base_score, 4),
            "transition_lift_score": round(transition_score, 4),
            "momentum_lift_score": round(momentum_score, 4),
            "prediction_score": round(0.70 * base_score + 0.15 * transition_score + 0.15 * momentum_score, 4),
            "features": record,
        })
    item.setdefault("base_score", item.get("mixed_slot_score", item["prediction_score"]))
    item.setdefault("transition_lift_score", 50.0)
    item.setdefault("momentum_lift_score", 50.0)
    item.update({
        "primary_subtype": record["primary_subtype"], "subtype_tags": list(record["subtype_tags"]),
        "subtype_signature": record["subtype_signature"], "selected_family": family,
        "structure_record": record,
    })
    return item


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
    allocation = _portfolio_allocation(profile, k)
    pool_sizes = _type_pool_sizes(allocation, k)
    heaps: dict[str, list[tuple[float, int, dict]]] = {
        candidate_type: [] for candidate_type in PATTERN_TYPES
    }
    evaluated = 0
    for evaluated, candidate in enumerate(candidates, 1):
        record = structure_record(candidate, profile["latest_pattern_type"])
        item = _score_structure_record(record, profile, mixed_profile, weights)
        ranking_score = float(item["prediction_score"])
        entry = (ranking_score, evaluated, item)
        _push_type_candidate(
            heaps, pool_sizes, item["pattern_type"], entry
        )

    remaining = _flatten_type_heaps(heaps)
    selected, type_counts, family_counts = [], Counter(), Counter()
    while remaining and len(selected) < k:
        choices = []
        for index, item in enumerate(remaining):
            family = item["selected_family"]
            type_gap = allocation.get(item["pattern_type"], 0) - type_counts[item["pattern_type"]]
            family_gap = (family_allocation or {}).get(family, 0) - family_counts[family]
            overlap = sum(len(set(item["numbers"]) & set(prior["numbers"])) for prior in selected)
            signature_duplicates = sum(item["subtype_signature"] == prior["subtype_signature"] for prior in selected)
            section_duplicates = sum(item["structure_record"]["section_distribution"] == prior["structure_record"]["section_distribution"] for prior in selected)
            adjustment = max(-4.0, min(4.0, 1.5 * type_gap)) + max(-2.0, min(2.0, family_gap))
            adjustment -= min(6.0, 0.8 * overlap + signature_duplicates + 0.5 * section_duplicates)
            choices.append((float(item["prediction_score"]) + adjustment, -index, index))
        portfolio_score, _, best_index = max(choices)
        item = remaining.pop(best_index)
        item["portfolio_selection_score"] = round(portfolio_score, 4)
        selected.append(item)
        type_counts[item["pattern_type"]] += 1
        family_counts[item["selected_family"]] += 1
    selected.sort(key=lambda item: float(item["prediction_score"]), reverse=True)
    for rank, item in enumerate(selected, 1):
        item["strategy"] = "FINAL ENSEMBLE PORTFOLIO"
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
    type_allocation = _portfolio_allocation(profile, top_k)
    family_allocation = suggest_mixed_signature_allocation(
        subtype_records, slots=type_allocation.get("mixed", 0), baseline=subtype_baseline
    )
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
            "historical_draw_count": len(df),
            "recommendation_mode": mode,
            "weight_mode": "component backtest calibrated fixed blend",
            "mixed_model_version": "v2.3.1",
            "analysis_version": "v2.5",
            "portfolio_allocation": allocation,
            "portfolio_selected": dict(Counter(item["pattern_type"] for item in recommendations)),
            "mixed_diagnostics": _mixed_diagnostics(
                recommendations, family_allocation
            ),
            "mixed_subtype_allocation_suggestion": suggest_mixed_subtype_allocation(subtype_records),
            "score_name": "prediction_score",
            "score_disclaimer": "prediction_score는 실제 당첨확률이 아닙니다. 최신 회차까지의 데이터 기반으로 계산한 내부 경험적 순위 점수입니다.",
        },
        "weights": weights,
        "weight_payload": weight_payload,
        "recommendations": recommendations,
    }


def print_recommendations(payload: dict) -> None:
    meta = payload["meta"]
    print("=" * 72)
    print("LOTTO STAT ENGINE FINAL - LATEST-DATA DYNAMIC STRUCTURE")
    print("=" * 72)
    print(f"latest reflected draw: {meta['latest_draw']}")
    print(f"target draw: {meta['target_draw']}")
    print(f"historical draw count: {meta['historical_draw_count']}")
    print(f"evaluation mode: {meta['recommendation_mode']}")
    print(f"evaluated combination count: {meta['evaluated_count']}")
    print()
    print(meta["score_disclaimer"])
    print(f"type allocation target vs selected: {meta['portfolio_allocation']} vs {meta['portfolio_selected']}")
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
        print(f"base_score: {item['base_score']}")
        print(f"transition_lift_score: {item['transition_lift_score']}")
        print(f"momentum_lift_score: {item['momentum_lift_score']}")
        print(f"pattern_type: {item['pattern_type']}")
        print(f"primary_subtype: {item['primary_subtype']}")
        print(f"subtype_tags: {', '.join(item['subtype_tags']) or 'none'}")
        print(f"subtype_signature: {item['subtype_signature']}")
        print(f"selected_family: {item['selected_family']}")
        if item["pattern_type"] == "mixed":
            print(f"mixed_slot_score: {item['mixed_slot_score']:.4f}")
            print(f"extreme_count: {item['extreme_count']}")
            print(f"extreme_signature: {', '.join(item['extreme_signature'])}")
            print(f"subtype_tags: {', '.join(item['subtype_tags']) or 'none'}")
            print(f"subtype_signature: {item['subtype_signature']}")
            print(f"selected_family: {item['selected_family']}")
        print("score breakdown:")
        for key, value in item["score_breakdown"].items():
            print(f"  {key}: {value:.4f}")
        print(
            f"sum={f['sum']}, odd={f['odd_count']}, even={6 - f['odd_count']}, "
            f"range={f['number_range']}, max_gap={f['max_gap']}, "
            f"min_gap={f['min_gap']}, sections={f['section_distribution']}"
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
