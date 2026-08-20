from __future__ import annotations

import hashlib
import heapq
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .candidates import generate_candidates, iter_all_combinations, make_seed
from .config import (
    BASE_WEIGHTS,
    DEFAULT_CANDIDATE_COUNT,
    DEFAULT_EXHAUSTIVE_RECOMMENDATION,
    TOP_K_RECOMMENDATIONS,
)
from .evidence_scoring import (
    build_unified_evidence_profile,
    public_evidence_diagnostics,
    score_unified_evidence,
)
from .loader import load_lotto_data
from .mixed_scoring import (
    build_all_combination_baseline,
    build_draw_structure_records,
    build_mixed_profile,
    score_mixed_record,
    structure_record,
)
from .profiles import build_profile
from .scoring import SCORE_WEIGHTS, score_candidate

MODEL_VERSION = "v2.7.1"
RELEASE_STATUS = "research_candidate"
SELECTION_STRATEGY = "unified_bayesian_evidence_top_k"
KST = timezone(timedelta(hours=9), name="KST")
SCORE_DISCLAIMER = (
    "score는 실제 당첨확률이 아닙니다. 동일한 6/45 fair-null을 기준으로 번호와 pair의 "
    "Bayesian posterior evidence를 계산하고, strict walk-forward Brier skill로 영향력을 "
    "자동 축소한 내부 순위 점수입니다."
)


# ---------------------------------------------------------------------------
# Legacy v2.7 static helpers are retained for reproducible audit/tests only.
# The production recommendation entry point below no longer calls them.
# ---------------------------------------------------------------------------

def without_dynamic_context(profile: dict) -> dict:
    """Return a shallow static-only Mixed profile for legacy v2.7 audit."""
    static_profile = dict(profile)
    static_profile.pop("dynamic_markov_decay", None)
    return static_profile


def renormalized_static_score(breakdown: dict[str, float]) -> tuple[float, dict[str, float]]:
    """Remove legacy transition contribution and renormalize remaining positive weights."""
    active_weights = {
        name: float(weight)
        for name, weight in SCORE_WEIGHTS.items()
        if float(weight) > 0.0 and name != "transition_score"
    }
    denominator = sum(active_weights.values())
    if denominator <= 0.0:
        raise ValueError("v2.7 static scorer requires at least one positive component weight")
    active_breakdown = {name: float(breakdown[name]) for name in active_weights}
    score = sum(active_weights[name] * active_breakdown[name] for name in active_weights) / denominator
    return float(score), active_breakdown


def score_static_record(
    record: dict,
    profile: dict,
    mixed_profile: dict,
    feature_weights: dict[str, float],
) -> dict:
    """Score one combination with the frozen legacy v2.7 static policy."""
    if record["pattern_type"] == "mixed":
        scored = score_mixed_record(record, mixed_profile)
        score = float(scored["mixed_slot_score"])
        breakdown = {
            name: float(scored[name])
            for name in (
                "mixed_lift_score",
                "mixed_interaction_score",
                "normal_backbone_score",
                "controlled_extreme_score",
                "recency_consistency_score",
            )
        }
        origin = "mixed_static_base"
    else:
        scored = score_candidate(list(record["numbers"]), profile, feature_weights)
        score, breakdown = renormalized_static_score(scored["score_breakdown"])
        origin = "normal_outlier_static_without_transition"

    return {
        "numbers": list(record["numbers"]),
        "pattern_type": str(record["pattern_type"]),
        "prediction_score": round(float(score), 4),
        "score_origin": origin,
        "score_breakdown": {name: round(float(value), 4) for name, value in breakdown.items()},
        "features": {
            "sum": int(record["sum"]),
            "odd_count": int(record["odd_count"]),
            "number_range": int(record["number_range"]),
            "max_gap": int(record["max_gap"]),
            "min_gap": int(record["min_gap"]),
            "section_distribution": list(record["section_distribution"]),
        },
    }


def top_static_stream(
    candidates,
    profile: dict,
    mixed_profile: dict,
    feature_weights: dict[str, float],
    top_k: int,
) -> tuple[list[dict], int]:
    """Legacy static TOP-K helper retained for audit reproducibility."""
    heap: list[tuple[float, tuple[int, ...], dict]] = []
    evaluated = 0
    for evaluated, candidate in enumerate(candidates, 1):
        record = structure_record(candidate, profile["latest_pattern_type"])
        item = score_static_record(record, profile, mixed_profile, feature_weights)
        entry = (float(item["prediction_score"]), tuple(item["numbers"]), item)
        if len(heap) < int(top_k):
            heapq.heappush(heap, entry)
        elif entry[:2] > heap[0][:2]:
            heapq.heapreplace(heap, entry)

    selected = [entry[2] for entry in sorted(heap, key=lambda value: (value[0], value[1]), reverse=True)]
    for rank, item in enumerate(selected, 1):
        item["rank"] = rank
        item["strategy"] = "V2.7 STATIC TOP SCORE"
    return selected, evaluated


# ---------------------------------------------------------------------------
# v2.7.1 unified evidence production/research-candidate path.
# ---------------------------------------------------------------------------

def _seeded_fair_tiebreak(numbers, seed: int) -> int:
    """Structure-neutral deterministic tie-break used only for exact score ties."""
    encoded = f"{int(seed)}:" + ",".join(str(int(number)) for number in numbers)
    digest = hashlib.blake2b(encoded.encode("ascii"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


def _attach_structure_metadata(item: dict, profile: dict) -> dict:
    """Classify a selected item after ranking; structure never changes its score."""
    record = structure_record(item["numbers"], profile["latest_pattern_type"])
    item["pattern_type"] = str(record["pattern_type"])
    item["features"] = {
        "sum": int(record["sum"]),
        "odd_count": int(record["odd_count"]),
        "number_range": int(record["number_range"]),
        "max_gap": int(record["max_gap"]),
        "min_gap": int(record["min_gap"]),
        "section_distribution": list(record["section_distribution"]),
    }
    return item


def top_evidence_stream(
    candidates,
    profile: dict,
    evidence_profile: dict,
    top_k: int,
    seed: int,
) -> tuple[list[dict], int]:
    """Rank every candidate with one common number+pair evidence equation."""
    heap: list[tuple[float, int, tuple[int, ...], dict]] = []
    evaluated = 0
    for evaluated, candidate in enumerate(candidates, 1):
        item = score_unified_evidence(candidate, evidence_profile)
        ranking_score = float(item["ranking_score"])
        numbers = tuple(item["numbers"])
        tie_break = _seeded_fair_tiebreak(numbers, seed)
        entry = (ranking_score, tie_break, numbers, item)
        if len(heap) < int(top_k):
            heapq.heappush(heap, entry)
        elif entry[:3] > heap[0][:3]:
            heapq.heapreplace(heap, entry)

    selected = [
        entry[3]
        for entry in sorted(heap, key=lambda value: (value[0], value[1], value[2]), reverse=True)
    ]
    for rank, item in enumerate(selected, 1):
        _attach_structure_metadata(item, profile)
        item["prediction_score"] = round(float(item["prediction_score"]), 8)
        item["ranking_score"] = round(float(item["ranking_score"]), 12)
        item["score_breakdown"] = {
            name: round(float(value), 12) for name, value in item["score_breakdown"].items()
        }
        item["rank"] = rank
        item["strategy"] = "V2.7.1 UNIFIED BAYESIAN EVIDENCE"
    return selected, evaluated


def generate_release_recommendations(
    seed_offset: int = 0,
    candidate_count: int = DEFAULT_CANDIDATE_COUNT,
    exhaustive: bool = DEFAULT_EXHAUSTIVE_RECOMMENDATION,
    top_k: int = TOP_K_RECOMMENDATIONS,
) -> dict:
    """Generate recommendations from current local data using unified evidence."""
    df = load_lotto_data()
    profile = build_profile(df)

    # This is the previously missing research -> production bridge. Reliability is
    # recalculated from the exact local data used for the recommendation, then the
    # next-draw posterior is fit on all completed draws.
    evidence_profile = build_unified_evidence_profile(df)
    evidence_diagnostics = public_evidence_diagnostics(evidence_profile)

    latest_draw = int(profile["latest_round"])
    target_draw = latest_draw + 1
    seed = make_seed(latest_draw, seed_offset)

    if exhaustive:
        candidates = iter_all_combinations()
        mode = "exhaustive_all_8,145,060"
    else:
        candidates = generate_candidates(int(candidate_count), seed)
        mode = "sampled_candidates"

    recommendations, evaluated_count = top_evidence_stream(
        candidates,
        profile,
        evidence_profile,
        int(top_k),
        seed,
    )

    return {
        "meta": {
            "model_version": MODEL_VERSION,
            "release_status": RELEASE_STATUS,
            "generated_at_kst": datetime.now(KST).isoformat(timespec="seconds"),
            "latest_draw": latest_draw,
            "target_draw": target_draw,
            "historical_draw_count": int(len(df)),
            "recommendation_mode": mode,
            "evaluated_count": int(evaluated_count),
            "top_k": int(top_k),
            "seed": int(seed),
            "seed_offset": int(seed_offset),
            "candidate_count": int(candidate_count),
            "selection_strategy": SELECTION_STRATEGY,
            "candidate_policy": "all valid 6-of-45 combinations; no hard filters",
            "dynamic_components": {
                "transition": "disabled_after_v2.7_validation",
                "momentum": "disabled_after_v2.7_validation",
            },
            "static_structure_components": "diagnostic_only_after_phase5a_no_survivors",
            "cross_type_calibration": "not_required_single_common_evidence_equation",
            "pattern_type_role": "metadata_only_not_used_for_ranking",
            "tie_break_policy": "seeded_structure_neutral_only_for_exact_evidence_ties",
            "evidence_models": evidence_diagnostics,
            "score_name": "prediction_score",
            "score_disclaimer": SCORE_DISCLAIMER,
        },
        # Retained only so older consumers do not break. These legacy feature
        # weights no longer drive v2.7.1 ranking.
        "feature_weights": dict(BASE_WEIGHTS),
        "recommendations": recommendations,
    }


def public_recommendation_payload(payload: dict) -> dict:
    """Stable JSON contract for the site or an HTTP wrapper."""
    meta = payload["meta"]
    return {
        "model_version": meta["model_version"],
        "release_status": meta["release_status"],
        "generated_at_kst": meta["generated_at_kst"],
        "latest_draw": meta["latest_draw"],
        "target_draw": meta["target_draw"],
        "recommendation_mode": meta["recommendation_mode"],
        "evaluated_count": meta["evaluated_count"],
        "selection_strategy": meta["selection_strategy"],
        "candidate_policy": meta["candidate_policy"],
        "dynamic_components": meta["dynamic_components"],
        "static_structure_components": meta.get("static_structure_components"),
        "cross_type_calibration": meta.get("cross_type_calibration"),
        "pattern_type_role": meta.get("pattern_type_role"),
        "tie_break_policy": meta.get("tie_break_policy"),
        "evidence_models": meta.get("evidence_models", {}),
        "score_disclaimer": meta["score_disclaimer"],
        "recommendations": [
            {
                "rank": int(item["rank"]),
                "numbers": list(item["numbers"]),
                "score": float(item["prediction_score"]),
                "ranking_score": float(item.get("ranking_score", item["prediction_score"])),
                "pattern_type": item.get("pattern_type", "unknown"),
                "score_origin": item["score_origin"],
                "components": dict(item["score_breakdown"]),
            }
            for item in payload["recommendations"]
        ],
    }


def write_public_json(payload: dict, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(public_recommendation_payload(payload), handle, ensure_ascii=False, indent=2)
    return path.resolve()


def print_release_recommendations(payload: dict) -> None:
    meta = payload["meta"]
    print("=" * 88)
    print("LOTTO STAT ENGINE v2.7.1 - UNIFIED BAYESIAN EVIDENCE RESEARCH CANDIDATE")
    print("=" * 88)
    print(f"latest reflected draw: {meta['latest_draw']}")
    print(f"target draw: {meta['target_draw']}")
    print(f"evaluation mode: {meta['recommendation_mode']}")
    print(f"evaluated combinations: {meta['evaluated_count']}")
    print(f"selection: {meta['selection_strategy']}")
    print("pattern type: metadata only; no type-specific ranking branch")
    print("dynamic components: transition=disabled, momentum=disabled")

    evidence = meta.get("evidence_models", {})
    for label in ("number", "pair"):
        detail = evidence.get(label, {})
        if not detail:
            continue
        skills = detail.get("brier_skill_vs_uniform", {})
        print(
            f"{label} evidence: reliability={float(detail.get('reliability', 0.0)):.10f} "
            f"brier_skill overall={float(skills.get('overall', 0.0)):.10f} "
            f"recent300={float(skills.get('recent300', 0.0)):.10f} "
            f"recent100={float(skills.get('recent100', 0.0)):.10f}"
        )
    print(meta["score_disclaimer"])

    for item in payload["recommendations"]:
        numbers = " ".join(str(number) for number in item["numbers"])
        print()
        print(f"#{item['rank']}  {numbers}")
        print(
            f"score={item['prediction_score']:.8f} "
            f"ranking_evidence={item['ranking_score']:.12f} "
            f"type={item['pattern_type']} origin={item['score_origin']}"
        )
        for name, value in item["score_breakdown"].items():
            print(f"  {name}: {value:.12f}")
