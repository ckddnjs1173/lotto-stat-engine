from __future__ import annotations

import random
from statistics import pstdev

import numpy as np
import pandas as pd

from .candidates import random_combination
from .config import BACKTEST_START_INDEX, BASELINE_SAMPLE_COUNT
from .features import extract_features
from .loader import row_numbers
from .profiles import FEATURE_GROUPS, build_profile, group_score
from .weights import blend_weights, normalize_weights, save_weight_payload

GROUPS = list(FEATURE_GROUPS.keys())


def _percentile_rank(actual_score: float, baseline_scores: list[float]) -> float:
    if not baseline_scores:
        return 50.0
    below_or_equal = sum(1 for score in baseline_scores if score <= actual_score)
    return float(below_or_equal / len(baseline_scores) * 100.0)


def _stability(percentiles: list[float], chunks: int = 4) -> float:
    if len(percentiles) < chunks:
        return 0.0
    size = max(1, len(percentiles) // chunks)
    chunk_means = []
    for idx in range(chunks):
        start = idx * size
        end = len(percentiles) if idx == chunks - 1 else (idx + 1) * size
        values = percentiles[start:end]
        if values:
            chunk_means.append(float(np.mean(values)))
    if len(chunk_means) <= 1:
        return 0.0
    # percentile 표준편차가 0에 가까울수록 안정적, 20%p 이상이면 불안정으로 본다.
    volatility = min(20.0, float(pstdev(chunk_means))) / 20.0
    # 50%를 넘긴 구간 비율도 함께 본다.
    positive_ratio = sum(1 for value in chunk_means if value > 50.0) / len(chunk_means)
    return float(max(0.0, min(1.0, (1.0 - volatility) * positive_ratio)))


def _verified_strength(mean_percentile: float, stability: float) -> float:
    # 50%는 랜덤과 동일한 기준선이다. 그 이상만 예측식 가중치의 근거로 쓴다.
    advantage = max(0.0, (float(mean_percentile) - 50.0) / 50.0)
    return float(advantage * (0.5 + 0.5 * float(stability)))


def run_walk_forward_backtest(
    df: pd.DataFrame,
    start_index: int = BACKTEST_START_INDEX,
    baseline_samples: int = BASELINE_SAMPLE_COUNT,
) -> dict:
    if len(df) <= start_index:
        raise ValueError(f"워크포워드 백테스트에는 최소 {start_index + 1}개 회차가 필요합니다.")

    group_scores = {group: [] for group in GROUPS}
    percentile_scores = {group: [] for group in GROUPS}

    for idx in range(start_index, len(df)):
        train_df = df.iloc[:idx].copy()
        target = df.iloc[idx]
        target_round = int(target["회차"])
        profile = build_profile(train_df)
        actual_features = extract_features(row_numbers(target))

        actual_group_scores = {
            group: group_score(actual_features, profile, group, recent=(group == "recent"))
            for group in GROUPS
        }
        for group, score in actual_group_scores.items():
            group_scores[group].append(score)

        rng = random.Random(target_round * 10007 + int(baseline_samples))
        random_group_scores = {group: [] for group in GROUPS}
        for _ in range(int(baseline_samples)):
            candidate = random_combination(rng)
            candidate_features = extract_features(candidate)
            for group in GROUPS:
                random_group_scores[group].append(
                    group_score(candidate_features, profile, group, recent=(group == "recent"))
                )

        for group in GROUPS:
            percentile_scores[group].append(
                _percentile_rank(actual_group_scores[group], random_group_scores[group])
            )

    feature_scores = {group: float(np.mean(values)) for group, values in group_scores.items()}
    percentile_means = {group: float(np.mean(values)) for group, values in percentile_scores.items()}
    stability_scores = {group: _stability(values) for group, values in percentile_scores.items()}
    verified_strengths = {
        group: _verified_strength(percentile_means[group], stability_scores[group]) for group in GROUPS
    }

    verified_weights = normalize_weights(verified_strengths)
    final_weights = blend_weights(verified_weights, base_ratio=0.70)

    payload = {
        "version": "v2_actual_vs_random",
        "total_tests": int(len(df) - start_index),
        "start_index": int(start_index),
        "baseline_samples": int(baseline_samples),
        "feature_scores": feature_scores,
        "percentile_scores": percentile_means,
        "stability_scores": stability_scores,
        "verified_strengths": verified_strengths,
        "verified_weights": verified_weights,
        "final_weights": final_weights,
        "score_name": "prediction_score",
    }
    save_weight_payload(payload)
    return payload
