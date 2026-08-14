from __future__ import annotations

import heapq
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .backtest import run_walk_forward_backtest
from .candidates import generate_candidates, iter_all_combinations, make_seed
from .config import (
    DEFAULT_CANDIDATE_COUNT,
    DEFAULT_EXHAUSTIVE_RECOMMENDATION,
    TOP_K_RECOMMENDATIONS,
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
from .weights import load_weight_payload

MODEL_VERSION = "v2.7"
RELEASE_STATUS = "release_candidate"
SELECTION_STRATEGY = "pure_static_score_top_k"
DISABLED_DYNAMIC_COMPONENTS = ("transition", "momentum")
SCORE_DISCLAIMER = (
    "score는 실제 당첨확률이 아닙니다. 최신 반영 회차까지의 데이터로 계산한 "
    "정적 통계 순위 점수이며, 모든 유효 6/45 조합은 후보가 될 수 있습니다."
)


def without_dynamic_context(profile: dict) -> dict:
    """Return a shallow static-only Mixed profile for v2.7 production ranking."""
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
    active_breakdown = {
        name: float(breakdown[name])
        for name in active_weights
    }
    score = sum(active_weights[name] * active_breakdown[name] for name in active_weights) / denominator
    return float(score), active_breakdown


def score_static_record(
    record: dict,
    profile: dict,
    mixed_profile: dict,
    feature_weights: dict[str, float],
) -> dict:
    """Score one valid combination with the frozen v2.7 static production policy."""
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
    """Keep only the globally highest static scores while streaming candidates."""
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

    selected = [
        entry[2]
        for entry in sorted(heap, key=lambda value: (value[0], value[1]), reverse=True)
    ]
    for rank, item in enumerate(selected, 1):
        item["rank"] = rank
        item["strategy"] = "V2.7 STATIC TOP SCORE"
    return selected, evaluated


def _resolve_feature_weights(df) -> tuple[dict[str, float], dict]:
    payload = load_weight_payload()
    if payload is None or "final_weights" not in payload:
        payload = run_walk_forward_backtest(df)
    return dict(payload["final_weights"]), payload


def generate_release_recommendations(
    seed_offset: int = 0,
    candidate_count: int = DEFAULT_CANDIDATE_COUNT,
    exhaustive: bool = DEFAULT_EXHAUSTIVE_RECOMMENDATION,
    top_k: int = TOP_K_RECOMMENDATIONS,
) -> dict:
    """Generate the v2.7 release-candidate recommendation payload."""
    df = load_lotto_data()
    profile = build_profile(df)
    records = build_draw_structure_records(df)
    mixed_profile = without_dynamic_context(
        build_mixed_profile(records, build_all_combination_baseline())
    )
    feature_weights, weight_payload = _resolve_feature_weights(df)

    latest_draw = int(profile["latest_round"])
    target_draw = latest_draw + 1
    seed = make_seed(latest_draw, seed_offset)

    if exhaustive:
        candidates = iter_all_combinations()
        mode = "exhaustive_all_8,145,060"
    else:
        candidates = generate_candidates(int(candidate_count), seed)
        mode = "sampled_candidates"

    recommendations, evaluated_count = top_static_stream(
        candidates,
        profile,
        mixed_profile,
        feature_weights,
        int(top_k),
    )

    return {
        "meta": {
            "model_version": MODEL_VERSION,
            "release_status": RELEASE_STATUS,
            "generated_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds"),
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
            "cross_type_calibration": "disabled_after_phase2_rejection",
            "score_name": "prediction_score",
            "score_disclaimer": SCORE_DISCLAIMER,
        },
        "feature_weights": feature_weights,
        "weight_payload": weight_payload,
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
        "score_disclaimer": meta["score_disclaimer"],
        "recommendations": [
            {
                "rank": int(item["rank"]),
                "numbers": list(item["numbers"]),
                "score": float(item["prediction_score"]),
                "pattern_type": item["pattern_type"],
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
    print("=" * 80)
    print("LOTTO STAT ENGINE v2.7 - STATIC RELEASE CANDIDATE")
    print("=" * 80)
    print(f"latest reflected draw: {meta['latest_draw']}")
    print(f"target draw: {meta['target_draw']}")
    print(f"evaluation mode: {meta['recommendation_mode']}")
    print(f"evaluated combinations: {meta['evaluated_count']}")
    print(f"selection: {meta['selection_strategy']}")
    print("dynamic components: transition=disabled, momentum=disabled")
    print(meta["score_disclaimer"])

    for item in payload["recommendations"]:
        numbers = " ".join(str(number) for number in item["numbers"])
        print()
        print(f"#{item['rank']}  {numbers}")
        print(
            f"score={item['prediction_score']:.4f} "
            f"type={item['pattern_type']} origin={item['score_origin']}"
        )
        for name, value in item["score_breakdown"].items():
            print(f"  {name}: {value:.4f}")
