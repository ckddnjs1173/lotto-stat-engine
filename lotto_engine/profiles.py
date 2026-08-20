from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .config import RECENT_WINDOW
from .features import PATTERN_FLAG_KEYS, extract_features, pattern_type, structure_type
from .loader import ROUND_COLUMN, row_numbers

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
PATTERN_TYPES = ("normal", "mixed", "outlier")
RECENT_WINDOWS = (20, 50, 100, 300)


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    previous_numbers = None
    for _, row in df.iterrows():
        numbers = row_numbers(row)
        features = extract_features(numbers, previous_draw=previous_numbers)
        features["round"] = int(row[ROUND_COLUMN])
        features["structure_type"] = structure_type(features)
        features["pattern_type"] = pattern_type(features)
        rows.append(features)
        previous_numbers = numbers
    return pd.DataFrame(rows)


def build_profile(df: pd.DataFrame) -> dict:
    feature_df = build_feature_frame(df)
    excluded = {"round", "structure_type", "pattern_type"}
    numeric_cols = [column for column in feature_df.columns if column not in excluded]
    recent_df = feature_df.tail(min(RECENT_WINDOW, len(feature_df)))
    means = {column: float(feature_df[column].mean()) for column in numeric_cols}
    stds = {column: float(feature_df[column].std(ddof=0) or 1.0) for column in numeric_cols}
    recent_means = {column: float(recent_df[column].mean()) for column in numeric_cols}
    recent_stds = {
        column: float(recent_df[column].std(ddof=0) or stds[column] or 1.0)
        for column in numeric_cols
    }

    structure_counts = Counter(feature_df["structure_type"])
    total = max(1, len(feature_df))
    structure_probs = {key: count / total for key, count in structure_counts.items()}
    pattern_counts = Counter(feature_df["pattern_type"])
    pattern_type_probs = {key: pattern_counts[key] / total for key in PATTERN_TYPES}
    pattern_frequencies = {key: float(feature_df[key].mean()) for key in PATTERN_FLAG_KEYS}
    pattern_value_profiles = {
        key: {"mean": means[key], "std": stds[key]}
        for key in ("ending_duplicate_count", "empty_section_count", "prime_count", "previous_draw_proximity")
    }
    recent_distributions = {}
    for window in RECENT_WINDOWS:
        window_df = feature_df.tail(min(window, len(feature_df)))
        recent_distributions[str(window)] = {
            "pattern_frequencies": {key: float(window_df[key].mean()) for key in PATTERN_FLAG_KEYS},
            "pattern_type_probs": {
                key: float((window_df["pattern_type"] == key).mean()) for key in PATTERN_TYPES
            },
        }

    transition_counts = {
        source: {target: 0 for target in PATTERN_TYPES} for source in PATTERN_TYPES
    }
    types = feature_df["pattern_type"].tolist()
    for source, target in zip(types, types[1:]):
        transition_counts[source][target] += 1
    transition_probs = {}
    for source, counts in transition_counts.items():
        denominator = sum(counts.values()) + len(PATTERN_TYPES)
        transition_probs[source] = {
            target: (count + 1) / denominator for target, count in counts.items()
        }

    draws = [row_numbers(row) for _, row in df.iterrows()]
    all_counts = Counter(number for draw in draws for number in draw)
    recent_draws = draws[-min(RECENT_WINDOW, len(draws)):]
    recent_counts = Counter(number for draw in recent_draws for number in draw)
    draws_since_seen = {number: len(draws) for number in range(1, 46)}
    for offset, draw in enumerate(reversed(draws)):
        for number in draw:
            draws_since_seen[number] = min(draws_since_seen[number], offset)
    number_dynamics = {
        number: {
            "historical_rate": all_counts[number] / total,
            "recent_rate": recent_counts[number] / max(1, len(recent_draws)),
            "draws_since_seen": draws_since_seen[number],
        }
        for number in range(1, 46)
    }

    quantiles = {}
    for column in numeric_cols:
        values = feature_df[column].to_numpy(dtype=float)
        quantiles[column] = {
            "p01": float(np.percentile(values, 1)) if len(values) >= 100 else float(np.min(values)),
            "p05": float(np.percentile(values, 5)) if len(values) >= 100 else float(np.min(values)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)) if len(values) >= 100 else float(np.max(values)),
            "p99": float(np.percentile(values, 99)) if len(values) >= 100 else float(np.max(values)),
        }

    return {
        "feature_frame": feature_df, "means": means, "stds": stds,
        "recent_means": recent_means, "recent_stds": recent_stds,
        "structure_probs": structure_probs, "quantiles": quantiles,
        "pattern_frequencies": pattern_frequencies,
        "pattern_value_profiles": pattern_value_profiles,
        "pattern_type_probs": pattern_type_probs,
        "recent_distributions": recent_distributions,
        "transition_counts": transition_counts, "transition_probs": transition_probs,
        "latest_pattern_type": types[-1], "latest_numbers": draws[-1],
        "number_dynamics": number_dynamics,
        "latest_round": int(df[ROUND_COLUMN].max()),
    }


def similarity(value: float, mean: float, std: float) -> float:
    std = max(float(std), 1e-6)
    z = abs(float(value) - float(mean)) / std
    return max(0.0, 100.0 - z * 22.0)


def group_score(features: dict, profile: dict, group: str, recent: bool = False) -> float:
    if group == "cluster":
        probability = profile["structure_probs"].get(structure_type(features), 0.0)
        return min(100.0, 35.0 + probability * 500.0)
    keys = FEATURE_GROUPS[group]
    if not keys:
        return 50.0
    means_key = "recent_means" if recent else "means"
    stds_key = "recent_stds" if recent else "stds"
    return float(np.mean([
        similarity(float(features[key]), profile[means_key][key], profile[stds_key][key])
        for key in keys
    ]))
