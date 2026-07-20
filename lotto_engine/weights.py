from __future__ import annotations

import json
from pathlib import Path

from .config import BASE_WEIGHTS, WEIGHTS_PATH


def normalize_weights(scores: dict[str, float]) -> dict[str, float]:
    usable = {key: max(0.0, float(value)) for key, value in scores.items() if key in BASE_WEIGHTS}
    total = sum(usable.values())
    if total <= 0:
        return dict(BASE_WEIGHTS)
    return {key: usable.get(key, 0.0) / total for key in BASE_WEIGHTS}


def blend_weights(learned_weights: dict[str, float], base_ratio: float = 0.70) -> dict[str, float]:
    learned = normalize_weights(learned_weights)
    mixed = {
        key: BASE_WEIGHTS[key] * base_ratio + learned.get(key, 0.0) * (1.0 - base_ratio)
        for key in BASE_WEIGHTS
    }
    return normalize_weights(mixed)


def save_weight_payload(payload: dict, path: Path = WEIGHTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_weight_payload(path: Path = WEIGHTS_PATH) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
