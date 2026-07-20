from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd

from .config import NUMBER_COLUMNS, RECENT_WINDOW
from .features import extract_features, structure_type
from .loader import row_numbers

FEATURE_GROUPS = {
    "sum": ["sum"],
    "odd_even": ["odd_count", "even_count"],
    "section": ["section_1", "section_2", "section_3", "section_4", "section_5"],
    "gap": ["avg_gap", "min_gap", "max_gap", "gap_std", "range"],
    "entropy": ["ending_digit_entropy"],
    "consecutive": ["consecutive_pairs", "max_consecutive_run"],
    "ending": ["duplicate_endings"],
    "recent": ["sum", "odd_count", "gap_std", "ending_digit_entropy"],
    "cluster": [],
}


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, row in df.iterrows():
        features = extract_features(row_numbers(row))
        features["round"] = int(row["회차"])
        features["structure_type"] = structure_type(features)
        rows.append(features)
    return pd.DataFrame(rows)


def build_profile(df: pd.DataFrame) -> dict:
    feature_df = build_feature_frame(df)
    numeric_cols = [c for c in feature_df.columns if c not in {"round", "structure_type"}]
    recent_df = feature_df.tail(min(RECENT_WINDOW, len(feature_df)))

    means = {col: float(feature_df[col].mean()) for col in numeric_cols}
    stds = {col: float(feature_df[col].std(ddof=0) or 1.0) for col in numeric_cols}
    recent_means = {col: float(recent_df[col].mean()) for col in numeric_cols}
    recent_stds = {col: float(recent_df[col].std(ddof=0) or stds[col] or 1.0) for col in numeric_cols}

    structure_counts = Counter(feature_df["structure_type"].tolist())
    total = max(1, sum(structure_counts.values()))
    structure_probs = {key: value / total for key, value in structure_counts.items()}

    sums = feature_df["sum"].to_numpy(dtype=float)
    if len(sums) >= 100:
        sum_min = float(np.percentile(sums, 1))
        sum_max = float(np.percentile(sums, 99))
    else:
        sum_min, sum_max = 70.0, 190.0

    return {
        "feature_frame": feature_df,
        "means": means,
        "stds": stds,
        "recent_means": recent_means,
        "recent_stds": recent_stds,
        "structure_probs": structure_probs,
        "sum_min": sum_min,
        "sum_max": sum_max,
        "latest_round": int(df["회차"].max()),
    }


def similarity(value: float, mean: float, std: float) -> float:
    std = max(float(std), 1e-6)
    z = abs(float(value) - float(mean)) / std
    return max(0.0, 100.0 - z * 22.0)


def group_score(features: dict, profile: dict, group: str, recent: bool = False) -> float:
    if group == "cluster":
        st = structure_type(features)
        prob = profile["structure_probs"].get(st, 0.0)
        return min(100.0, 35.0 + prob * 500.0)

    keys = FEATURE_GROUPS[group]
    if not keys:
        return 50.0

    means_key = "recent_means" if recent else "means"
    stds_key = "recent_stds" if recent else "stds"
    scores = [similarity(float(features[k]), profile[means_key][k], profile[stds_key][k]) for k in keys]
    return float(np.mean(scores))
