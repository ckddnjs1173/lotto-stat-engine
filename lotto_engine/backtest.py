from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BACKTEST_START_INDEX
from .features import extract_features
from .loader import row_numbers
from .profiles import build_profile, group_score
from .weights import blend_weights, normalize_weights, save_weight_payload

GROUPS = ["sum", "odd_even", "section", "gap", "entropy", "consecutive", "ending", "recent", "cluster"]


def run_walk_forward_backtest(df: pd.DataFrame, start_index: int = BACKTEST_START_INDEX) -> dict:
    if len(df) <= start_index:
        raise ValueError(f"워크포워드 백테스트에는 최소 {start_index + 1}개 회차가 필요합니다.")

    group_scores = {group: [] for group in GROUPS}

    for idx in range(start_index, len(df)):
        train_df = df.iloc[:idx].copy()
        target = df.iloc[idx]
        profile = build_profile(train_df)
        features = extract_features(row_numbers(target))

        for group in GROUPS:
            score = group_score(features, profile, group, recent=(group == "recent"))
            group_scores[group].append(score)

    feature_scores = {group: float(np.mean(values)) for group, values in group_scores.items()}
    learned_weights = normalize_weights(feature_scores)
    final_weights = blend_weights(learned_weights, base_ratio=0.70)

    payload = {
        "total_tests": int(len(df) - start_index),
        "start_index": int(start_index),
        "feature_scores": feature_scores,
        "learned_weights": learned_weights,
        "final_weights": final_weights,
    }
    save_weight_payload(payload)
    return payload
