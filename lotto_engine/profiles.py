from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .config import RECENT_WINDOW
from .features import extract_features, structure_type
from .loader import row_numbers

FEATURE_GROUPS = {
    "sum": ["sum"],
    "odd_even": ["odd_count", "even_count"],
    "section": ["section_1", "section_2", "section_3", "section_4", "section_5"],
    "gap": ["avg_gap", "min_gap", "max_gap", "gap_std", "range"],
    "entropy": ["ending_digit_entropy", "section_entropy", "low_mid_high_entropy", "gap_entropy"],
    "consecutive": ["consecutive_pairs", "max_consecutive_run"],
    "ending": ["duplicate_endings"],
    "recent": ["sum", "odd_count", "gap_std", "ending_digit_entropy", "section_entropy", "gap_entropy"],
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

    quantiles = {}
    for col in numeric_cols:
        values = feature_df[col].to_numpy(dtype=float)
        quantiles[col] = {
            "p01": float(np.percentile(values, 1)) if len(values) >= 100 else float(np.min(values)),
            "p05": float(np.percentile(values, 5)) if len(values) >= 100 else float(np.min(values)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)) if len(values) >= 100 else float(np.max(values)),
            "p99": float(np.percentile(values, 99)) if len(values) >= 100 else float(np.max(values)),
        }

    return {
        "feature_frame": feature_df,
        "means": means,
        "stds": stds,
        "recent_means": recent_means,
        "recent_stds": recent_stds,
        "structure_probs": structure_probs,
        "quantiles": quantiles,
        "latest_round": int(df["회차"].max()),
    }


def similarity(value: float, mean: float, std: float) -> float:
    """Profile 적합도 점수입니다. 낮아도 탈락시키지 않고 점수만 낮춥니다."""
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
