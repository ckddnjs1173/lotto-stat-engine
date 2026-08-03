from __future__ import annotations

import random
from statistics import pstdev

import numpy as np
import pandas as pd

from .candidates import random_combination
from .config import BACKTEST_START_INDEX, BASELINE_SAMPLE_COUNT, BASE_WEIGHTS
from .loader import ROUND_COLUMN, row_numbers
from .profiles import build_profile
from .scoring import SCORE_WEIGHTS, score_candidate
from .weights import save_weight_payload

COMPONENTS = list(SCORE_WEIGHTS)


def _percentile_rank(actual_score: float, baseline_scores: list[float]) -> float:
    if not baseline_scores:
        return 50.0
    return sum(score <= actual_score for score in baseline_scores) / len(baseline_scores) * 100.0


def _stability(percentiles: list[float], chunks: int = 4) -> float:
    if len(percentiles) < chunks:
        return 0.0
    chunk_means = [
        float(np.mean(chunk)) for chunk in np.array_split(np.asarray(percentiles), chunks) if len(chunk)
    ]
    volatility = min(20.0, float(pstdev(chunk_means))) / 20.0
    positive_ratio = sum(value > 50.0 for value in chunk_means) / len(chunk_means)
    return float(max(0.0, min(1.0, (1.0 - volatility) * positive_ratio)))


def _window_summary(values: list[float], window: int) -> dict:
    selected = values[-min(window, len(values)):]
    return {
        "tests": len(selected),
        "mean_percentile": float(np.mean(selected)) if selected else 0.0,
        "percentile_std": float(np.std(selected)) if selected else 0.0,
        "stability": _stability(selected),
    }


def run_walk_forward_backtest(
    df: pd.DataFrame,
    start_index: int = BACKTEST_START_INDEX,
    baseline_samples: int = BASELINE_SAMPLE_COUNT,
) -> dict:
    if len(df) <= start_index:
        raise ValueError(f"백테스트에는 최소 {start_index + 1}개 회차가 필요합니다.")
    keys = ["prediction_score", *COMPONENTS]
    percentiles = {key: [] for key in keys}
    raw_scores = {key: [] for key in keys}

    for idx in range(start_index, len(df)):
        profile = build_profile(df.iloc[:idx].copy())
        target = df.iloc[idx]
        actual = score_candidate(row_numbers(target), profile, BASE_WEIGHTS)
        actual_values = {"prediction_score": actual["prediction_score"], **actual["score_breakdown"]}
        for key in keys:
            raw_scores[key].append(float(actual_values[key]))

        rng = random.Random(int(target[ROUND_COLUMN]) * 10007 + int(baseline_samples))
        random_values = {key: [] for key in keys}
        for _ in range(int(baseline_samples)):
            scored = score_candidate(random_combination(rng), profile, BASE_WEIGHTS)
            values = {"prediction_score": scored["prediction_score"], **scored["score_breakdown"]}
            for key in keys:
                random_values[key].append(float(values[key]))
        for key in keys:
            percentiles[key].append(_percentile_rank(actual_values[key], random_values[key]))

    payload = {
        "version": "v2.2_mixed_structure",
        "total_tests": len(df) - start_index,
        "start_index": int(start_index),
        "baseline_samples": int(baseline_samples),
        "score_name": "prediction_score",
        "score_disclaimer": "prediction_score는 실제 당첨확률이 아니라 내부 예측확률점수입니다.",
        "prediction_percentile": float(np.mean(percentiles["prediction_score"])),
        "component_percentiles": {
            key: float(np.mean(percentiles[key])) for key in COMPONENTS
        },
        "component_stability": {key: _stability(percentiles[key]) for key in COMPONENTS},
        "stability_windows": {
            str(window): {key: _window_summary(percentiles[key], window) for key in keys}
            for window in (100, 300)
        },
        "raw_score_means": {key: float(np.mean(values)) for key, values in raw_scores.items()},
        "score_weights": dict(SCORE_WEIGHTS),
        "final_weights": dict(BASE_WEIGHTS),
    }
    save_weight_payload(payload)
    return payload
