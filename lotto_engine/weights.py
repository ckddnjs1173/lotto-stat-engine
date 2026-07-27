from __future__ import annotations

import json
from pathlib import Path

from .config import BASE_WEIGHTS, WEIGHTS_PATH, WEIGHT_CAPS


def normalize_weights(scores: dict[str, float]) -> dict[str, float]:
    usable = {key: max(0.0, float(value)) for key, value in scores.items() if key in BASE_WEIGHTS}
    total = sum(usable.values())
    if total <= 0:
        return dict(BASE_WEIGHTS)
    return {key: usable.get(key, 0.0) / total for key in BASE_WEIGHTS}


def apply_weight_caps(weights: dict[str, float], caps: dict[str, float] | None = None) -> dict[str, float]:
    """Cap selected feature weights and redistribute the excess to uncapped groups.

    This keeps the final weights summing to 1.0 while preventing a high-variance feature,
    such as odd/even balance, from dominating the prediction_score after one favorable
    actual-vs-random backtest result.
    """
    caps = caps or WEIGHT_CAPS
    capped = normalize_weights(weights)
    excess = 0.0

    for key, cap in caps.items():
        if key not in capped:
            continue
        cap_value = max(0.0, float(cap))
        if capped[key] > cap_value:
            excess += capped[key] - cap_value
            capped[key] = cap_value

    if excess <= 0:
        return normalize_weights(capped)

    recipients = [key for key in BASE_WEIGHTS if key not in caps]
    recipient_total = sum(capped[key] for key in recipients)

    if recipient_total <= 0:
        # Defensive fallback: distribute evenly to uncapped groups if something unusual happens.
        share = excess / max(1, len(recipients))
        for key in recipients:
            capped[key] += share
    else:
        for key in recipients:
            capped[key] += excess * (capped[key] / recipient_total)

    return normalize_weights(capped)


def blend_weights(learned_weights: dict[str, float], base_ratio: float = 0.70) -> dict[str, float]:
    learned = normalize_weights(learned_weights)
    mixed = {
        key: BASE_WEIGHTS[key] * base_ratio + learned.get(key, 0.0) * (1.0 - base_ratio)
        for key in BASE_WEIGHTS
    }
    return apply_weight_caps(mixed)


def save_weight_payload(payload: dict, path: Path = WEIGHTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_weight_payload(path: Path = WEIGHTS_PATH) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
