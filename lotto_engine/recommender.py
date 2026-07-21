from __future__ import annotations

import heapq

from .backtest import run_walk_forward_backtest
from .candidates import generate_candidates, iter_all_combinations, make_seed
from .config import DEFAULT_CANDIDATE_COUNT, DEFAULT_EXHAUSTIVE_RECOMMENDATION, TOP_K_RECOMMENDATIONS
from .loader import load_lotto_data
from .profiles import build_profile
from .scoring import score_candidate, score_candidates
from .weights import load_weight_payload


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
    weight_payload = load_weight_payload()
    if weight_payload is None or weight_payload.get("version") != "v2_actual_vs_random":
        weight_payload = run_walk_forward_backtest(df)
    weights = weight_payload["final_weights"]

    latest_round = int(profile["latest_round"])
    seed = make_seed(latest_round, seed_offset)

    if exhaustive:
        recommendations, evaluated_count = _top_k_stream(iter_all_combinations(), profile, weights, top_k)
        mode = "exhaustive_all_8,145,060"
    else:
        candidates = generate_candidates(candidate_count, seed)
        scored = score_candidates(candidates, profile, weights)
        recommendations = build_recommendations(scored, top_k)
        evaluated_count = len(candidates)
        mode = "sampled_candidates"

    return {
        "meta": {
            "latest_round": latest_round,
            "target_round": latest_round + 1,
            "seed": seed,
            "seed_offset": seed_offset,
            "candidate_count": candidate_count,
            "evaluated_count": evaluated_count,
            "recommendation_mode": mode,
            "weight_mode": "base 70% + actual-vs-random verified 30%",
            "score_name": "prediction_score",
        },
        "weights": weights,
        "weight_payload": weight_payload,
        "recommendations": recommendations,
    }


def print_recommendations(payload: dict) -> None:
    meta = payload["meta"]
    print("=" * 72)
    print("LOTTO FORCED PREDICTION ENGINE")
    print("=" * 72)
    print(f"최신 반영 회차: {meta['latest_round']}")
    print(f"예측 대상 회차: {meta['target_round']}")
    print(f"평가 방식: {meta['recommendation_mode']}")
    print(f"평가 조합 수: {meta['evaluated_count']}")
    print(f"가중치 방식: {meta['weight_mode']}")
    print()
    print("※ prediction_score는 실제 당첨확률이 아니라, 과거 당첨번호 구조와 랜덤 기준선 검증으로 만든 내부 예측확률점수입니다.")

    for idx, item in enumerate(payload["recommendations"], 1):
        f = item["features"]
        nums = " ".join(str(n) for n in item["numbers"])
        print()
        print(f"[{idx}] {item['strategy']}")
        print(nums)
        print(f"prediction_score: {item['prediction_score']}")
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
