from __future__ import annotations

from .features import extract_features
from .profiles import FEATURE_GROUPS, group_score
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
    groups = list(FEATURE_GROUPS.keys())
    for group in groups:
        components[group] = group_score(features, profile, group, recent=(group == "recent"))

    prediction_score = float(sum(float(weights.get(group, 0.0)) * components[group] for group in groups))
    rounded = round(prediction_score, 4)
    return {
        "numbers": numbers,
        "features": features,
        "components": components,
        "prediction_score": rounded,
        # 구버전 UI/출력 호환용 alias입니다. 의미는 prediction_score와 같습니다.
        "score": rounded,
    }


def score_candidates(candidates: list[list[int]], profile: dict, weights: dict[str, float]) -> list[dict]:
    return [score_candidate(candidate, profile, weights) for candidate in candidates]
