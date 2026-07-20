from __future__ import annotations

import numpy as np

from .features import extract_features
from .profiles import GROUPS, group_score
from .weights import load_weight_payload


def score_candidate(numbers: list[int], profile: dict, weights: dict[str, float] | None = None) -> dict:
    if weights is None:
        payload = load_weight_payload()
        weights = payload["final_weights"] if payload else None
    if weights is None:
        from .config import BASE_WEIGHTS
        weights = BASE_WEIGHTS

    features = extract_features(numbers)
    components = {}
    for group in GROUPS:
        components[group] = group_score(features, profile, group, recent=(group == "recent"))

    hybrid = float(sum(float(weights.get(group, 0.0)) * components[group] for group in GROUPS))
    return {
        "numbers": numbers,
        "features": features,
        "components": components,
        "score": round(hybrid, 4),
    }


def score_candidates(candidates: list[list[int]], profile: dict, weights: dict[str, float]) -> list[dict]:
    return [score_candidate(candidate, profile, weights) for candidate in candidates]


def adjusted_for_overlap(item: dict, selected: list[dict]) -> float:
    if not selected:
        return float(item["score"])
    nums = set(item["numbers"])
    overlap_penalty = sum(len(nums & set(other["numbers"])) for other in selected) * 2.5
    return float(item["score"]) - overlap_penalty
