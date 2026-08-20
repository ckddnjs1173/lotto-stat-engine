from __future__ import annotations

import math

import numpy as np

from .features import PATTERN_FLAG_KEYS, extract_features, pattern_type
from .profiles import FEATURE_GROUPS, group_score
from .weights import load_weight_payload

SCORE_WEIGHTS = {
    # Calibrated from the v2.2 actual-vs-random component backtest.  Components
    # which did not beat random are retained in the breakdown for diagnostics,
    # but are not allowed to dilute the final empirical score.
    "outlier_survival_score": 0.45,
    "type_balance_score": 0.25,
    "transition_score": 0.15,
    "normal_structure_score": 0.15,
    "historical_pattern_score": 0.0,
    "number_dynamics_score": 0.0,
}


def _historical_pattern_score(features: dict, profile: dict) -> float:
    scores = []
    for key in PATTERN_FLAG_KEYS:
        probability = min(0.99, max(0.01, float(profile["pattern_frequencies"][key])))
        observed = probability if int(features[key]) else 1.0 - probability
        scores.append(max(0.0, 100.0 + 20.0 * math.log(observed)))
    for key, values in profile["pattern_value_profiles"].items():
        z = abs(float(features[key]) - values["mean"]) / max(values["std"], 1e-6)
        scores.append(max(0.0, 100.0 - z * 18.0))
    return float(np.mean(scores))


def _outlier_survival_score(
    features: dict, profile: dict, historical_score: float, normal_score: float
) -> float:
    active = [key for key in PATTERN_FLAG_KEYS if int(features[key])]
    if not active:
        support_score = 55.0
    else:
        supports = [
            min(1.0, float(profile["pattern_frequencies"][key]) / 0.10)
            for key in active
        ]
        # Use both average and weakest support.  This avoids the former 90-point
        # plateau when every active flag exceeded the 10% support threshold.
        support = 0.70 * float(np.mean(supports)) + 0.30 * float(min(supports))
        support_score = 45.0 + 45.0 * support - max(0, len(active) - 3) * 5.0
    # Continuous structure fit supplies within-flag discrimination without
    # changing the historical survival signal into a hard filter.
    return float(np.clip(
        0.70 * support_score + 0.20 * historical_score + 0.10 * normal_score,
        0.0,
        100.0,
    ))


def _type_balance_score(
    candidate_type: str, profile: dict, historical_score: float, normal_score: float
) -> float:
    probabilities = profile["pattern_type_probs"]
    maximum = max(probabilities.values()) or 1.0
    rarity_score = 35.0 + 50.0 * (
        1.0 - probabilities.get(candidate_type, 0.0) / maximum
    )
    return float(np.clip(
        0.85 * rarity_score + 0.05 * historical_score + 0.10 * normal_score,
        0.0,
        100.0,
    ))


def _transition_score(
    candidate_type: str, profile: dict, historical_score: float, normal_score: float
) -> float:
    probabilities = profile["transition_probs"].get(profile["latest_pattern_type"], {})
    maximum = max(probabilities.values()) if probabilities else 1.0
    transition_fit = 15.0 + 70.0 * probabilities.get(candidate_type, 0.0) / maximum
    return float(np.clip(
        0.85 * transition_fit + 0.05 * historical_score + 0.10 * normal_score,
        0.0,
        100.0,
    ))


def _number_dynamics_score(numbers: list[int], profile: dict) -> float:
    dynamics = profile["number_dynamics"]
    historical = [dynamics[n]["historical_rate"] for n in range(1, 46)]
    recent = [dynamics[n]["recent_rate"] for n in range(1, 46)]
    gaps = [dynamics[n]["draws_since_seen"] for n in range(1, 46)]

    def percentile(value: float, population: list[float]) -> float:
        return sum(item <= value for item in population) / len(population) * 100.0

    values = []
    for number in numbers:
        item = dynamics[number]
        values.append(
            0.35 * percentile(item["historical_rate"], historical)
            + 0.35 * percentile(item["recent_rate"], recent)
            + 0.30 * percentile(item["draws_since_seen"], gaps)
        )
    return float(np.mean(values))


def score_candidate(numbers: list[int], profile: dict, weights: dict[str, float] | None = None) -> dict:
    if weights is None:
        payload = load_weight_payload()
        weights = payload.get("final_weights") if payload else None
    if weights is None:
        from .config import BASE_WEIGHTS
        weights = BASE_WEIGHTS

    features = extract_features(numbers, previous_draw=profile.get("latest_numbers"))
    components = {
        group: group_score(features, profile, group, recent=(group == "recent"))
        for group in FEATURE_GROUPS
    }
    normal_score = sum(float(weights.get(group, 0.0)) * value for group, value in components.items())
    candidate_type = pattern_type(features)
    historical_score = _historical_pattern_score(features, profile)
    breakdown = {
        "normal_structure_score": float(normal_score),
        "outlier_survival_score": _outlier_survival_score(
            features, profile, historical_score, normal_score
        ),
        "historical_pattern_score": historical_score,
        "type_balance_score": _type_balance_score(
            candidate_type, profile, historical_score, normal_score
        ),
        "transition_score": _transition_score(
            candidate_type, profile, historical_score, normal_score
        ),
        "number_dynamics_score": _number_dynamics_score(numbers, profile),
    }
    prediction_score = sum(SCORE_WEIGHTS[key] * value for key, value in breakdown.items())
    rounded = round(float(prediction_score), 4)
    return {
        "numbers": numbers, "features": features, "components": components,
        "score_breakdown": {key: round(value, 4) for key, value in breakdown.items()},
        "pattern_type": candidate_type, "prediction_score": rounded, "score": rounded,
    }


def score_candidates(candidates: list[list[int]], profile: dict, weights: dict[str, float]) -> list[dict]:
    return [score_candidate(candidate, profile, weights) for candidate in candidates]
