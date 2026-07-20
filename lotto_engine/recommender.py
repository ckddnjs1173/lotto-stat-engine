from __future__ import annotations

import random
from typing import Callable

from .backtest import run_walk_forward_backtest
from .candidates import generate_candidates, make_seed
from .config import DEFAULT_CANDIDATE_COUNT
from .features import structure_type
from .filters import passes_hard_filter
from .loader import load_lotto_data
from .portfolio import portfolio_report
from .profiles import build_profile
from .scoring import adjusted_for_overlap, score_candidates
from .weights import load_weight_payload

STRATEGIES = [
    "CORE STRUCTURE",
    "CORE STRUCTURE",
    "BALANCED STRUCTURE",
    "HIGH ENTROPY",
    "WIDE GAP",
    "RECENT SOFT MATCH",
    "CLUSTER DIVERSITY",
    "WEIGHTED LUCK",
    "WEIGHTED LUCK",
    "COVERAGE OPTIMIZER",
]


def _pick_best(pool: list[dict], selected: list[dict], predicate: Callable[[dict], bool] | None = None) -> dict | None:
    candidates = [item for item in pool if item not in selected and (predicate(item) if predicate else True)]
    if not candidates:
        return None
    return max(candidates, key=lambda item: adjusted_for_overlap(item, selected))


def _pick_weighted_luck(pool: list[dict], selected: list[dict], rng: random.Random) -> dict | None:
    candidates = [item for item in pool if item not in selected]
    if not candidates:
        return None
    sample_size = min(3000, len(candidates))
    sample = rng.sample(candidates, sample_size)
    weights = [max(1.0, float(item["score"])) for item in sample]
    return rng.choices(sample, weights=weights, k=1)[0]


def _balanced(item: dict) -> bool:
    f = item["features"]
    return int(f["odd_count"]) in {2, 3, 4} and 115 <= int(f["sum"]) <= 165 and max(int(f["low_count"]), int(f["mid_count"]), int(f["high_count"])) <= 3


def _high_entropy(item: dict) -> bool:
    return float(item["features"]["ending_digit_entropy"]) >= 2.25


def _wide_gap(item: dict) -> bool:
    f = item["features"]
    return float(f["gap_std"]) >= 5.0 or int(f["range"]) >= 38


def _recent_match(item: dict) -> bool:
    return float(item["components"]["recent"]) >= 60.0


def _cluster_diverse(item: dict, selected: list[dict]) -> bool:
    used = {structure_type(other["features"]) for other in selected}
    return structure_type(item["features"]) not in used


def _coverage_score(item: dict, selected: list[dict]) -> float:
    nums = set(item["numbers"])
    used = set(n for other in selected for n in other["numbers"])
    new_count = len(nums - used)
    overlap = sum(len(nums & set(other["numbers"])) for other in selected)
    return float(item["score"]) * 0.45 + new_count * 10.0 - overlap * 3.0


def build_recommendations(scored: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed + 777)
    pool = sorted(scored, key=lambda item: item["score"], reverse=True)
    selected: list[dict] = []

    for strategy in STRATEGIES:
        if strategy == "CORE STRUCTURE":
            pick = _pick_best(pool, selected)
        elif strategy == "BALANCED STRUCTURE":
            pick = _pick_best(pool, selected, _balanced)
        elif strategy == "HIGH ENTROPY":
            pick = _pick_best(pool, selected, _high_entropy)
        elif strategy == "WIDE GAP":
            pick = _pick_best(pool, selected, _wide_gap)
        elif strategy == "RECENT SOFT MATCH":
            pick = _pick_best(pool, selected, _recent_match)
        elif strategy == "CLUSTER DIVERSITY":
            pick = _pick_best(pool, selected, lambda item: _cluster_diverse(item, selected))
        elif strategy == "WEIGHTED LUCK":
            pick = _pick_weighted_luck(pool, selected, rng)
        elif strategy == "COVERAGE OPTIMIZER":
            remaining = [item for item in pool if item not in selected]
            pick = max(remaining, key=lambda item: _coverage_score(item, selected)) if remaining else None
        else:
            pick = _pick_best(pool, selected)

        if pick is None:
            pick = _pick_best(pool, selected)
        if pick is not None:
            item = dict(pick)
            item["strategy"] = strategy
            selected.append(item)

    return selected[:10]


def generate_recommendations(seed_offset: int = 0, candidate_count: int = DEFAULT_CANDIDATE_COUNT) -> dict:
    df = load_lotto_data()
    profile = build_profile(df)
    weight_payload = load_weight_payload()
    if weight_payload is None:
        weight_payload = run_walk_forward_backtest(df)
    weights = weight_payload["final_weights"]

    latest_round = int(profile["latest_round"])
    seed = make_seed(latest_round, seed_offset)
    candidates = generate_candidates(candidate_count, seed)
    valid = [candidate for candidate in candidates if passes_hard_filter(candidate, profile)]
    scored = score_candidates(valid, profile, weights)
    recommendations = build_recommendations(scored, seed)
    report = portfolio_report(recommendations)

    return {
        "meta": {
            "latest_round": latest_round,
            "seed": seed,
            "seed_offset": seed_offset,
            "candidate_count": candidate_count,
            "passed_count": len(valid),
            "weight_mode": "base 70% + walk-forward 30%",
        },
        "weights": weights,
        "recommendations": recommendations,
        "portfolio_report": report,
    }


def print_recommendations(payload: dict) -> None:
    meta = payload["meta"]
    print("=" * 60)
    print("LOTTO STAT ENGINE RECOMMENDATIONS")
    print("=" * 60)
    print(f"최신 회차: {meta['latest_round']}")
    print(f"Seed: {meta['seed']}")
    print(f"Seed Offset: {meta['seed_offset']}")
    print(f"후보 생성 수: {meta['candidate_count']}")
    print(f"필터 통과 수: {meta['passed_count']}")
    print(f"가중치 방식: {meta['weight_mode']}")

    for idx, item in enumerate(payload["recommendations"], 1):
        f = item["features"]
        nums = " ".join(str(n) for n in item["numbers"])
        print()
        print(f"[{idx}] {item['strategy']}")
        print(nums)
        print(f"score: {item['score']}")
        print(
            f"sum={f['sum']}, odd={f['odd_count']}, even={f['even_count']}, "
            f"gap_std={round(float(f['gap_std']), 4)}, entropy={round(float(f['ending_digit_entropy']), 4)}"
        )

    print()
    print("PORTFOLIO REPORT")
    report = payload["portfolio_report"]
    print(f"고유 번호 수: {report['unique_number_count']}")
    print(f"평균 겹침 수: {report['average_overlap']}")
    print(f"최대 겹침 수: {report['max_overlap']}")
