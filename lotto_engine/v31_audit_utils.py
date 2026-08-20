from __future__ import annotations

import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np


def tie_safe_percentile(actual_score: float, baseline_scores) -> float:
    baseline = np.asarray(baseline_scores, dtype=float)
    if len(baseline) == 0:
        return float("nan")
    actual = float(actual_score)
    less = int(np.sum(baseline < actual))
    equal = int(np.sum(baseline == actual))
    return 100.0 * (less + 0.5 * equal) / len(baseline)


def block_bootstrap_mean_ci(
    values,
    reps: int = 2_000,
    block_size: int = 20,
    seed: int = 31_811,
) -> tuple[float, float]:
    data = np.asarray([float(value) for value in values if np.isfinite(value)], dtype=float)
    n = len(data)
    if n == 0:
        return float("nan"), float("nan")
    if n == 1 or int(reps) <= 0:
        value = float(np.mean(data))
        return value, value
    block = max(1, min(int(block_size), n))
    blocks_needed = (n + block - 1) // block
    offsets = np.arange(block, dtype=int)
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(reps), dtype=float)
    for rep in range(int(reps)):
        starts = rng.integers(0, n, size=blocks_needed)
        indices = np.concatenate(
            [(int(start) + offsets) % n for start in starts]
        )[:n]
        means[rep] = float(np.mean(data[indices]))
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def percentile_summary(values, bootstrap_reps: int = 2_000, seed: int = 31_811) -> dict:
    items = [float(value) for value in values]
    recent300 = items[-min(300, len(items)):]
    recent100 = items[-min(100, len(items)):]
    centered = [value - 50.0 for value in items]
    low, high = block_bootstrap_mean_ci(centered, reps=int(bootstrap_reps), seed=int(seed))
    return {
        "tests": len(items),
        "mean_percentile": float(np.mean(items)) if items else float("nan"),
        "recent_300_mean_percentile": float(np.mean(recent300)) if recent300 else float("nan"),
        "recent_100_mean_percentile": float(np.mean(recent100)) if recent100 else float("nan"),
        "percentile_minus_50_block_bootstrap_95_ci": [float(low), float(high)],
    }


def paired_delta_summary(
    left,
    right,
    bootstrap_reps: int = 2_000,
    seed: int = 31_901,
    label: str = "left_minus_right",
) -> dict:
    delta = [float(a) - float(b) for a, b in zip(left, right)]
    recent300 = delta[-min(300, len(delta)):]
    recent100 = delta[-min(100, len(delta)):]
    low, high = block_bootstrap_mean_ci(delta, reps=int(bootstrap_reps), seed=int(seed))
    mean = float(np.mean(delta)) if delta else float("nan")
    r300 = float(np.mean(recent300)) if recent300 else float("nan")
    r100 = float(np.mean(recent100)) if recent100 else float("nan")
    if delta and low > 0.0 and r300 >= 0.0 and r100 >= 0.0:
        direction = f"{label}_positive_consistent"
    elif delta and high < 0.0 and r300 <= 0.0 and r100 <= 0.0:
        direction = f"{label}_negative_consistent"
    else:
        direction = "inconclusive"
    return {
        "label": label,
        "mean_percentile_delta": mean,
        "recent_300_delta": r300,
        "recent_100_delta": r100,
        "block_bootstrap_95_ci": [float(low), float(high)],
        "direction": direction,
    }


def average_ranks(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        average = 0.5 * (start + end - 1)
        ranks[order[start:end]] = average
        start = end
    return ranks


def correlation(left, right) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if len(left) != len(right) or len(left) == 0:
        return float("nan")
    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return 1.0 if np.array_equal(left, right) else 0.0
    return float(np.corrcoef(left, right)[0, 1])


def top_indices(scores, count: int) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    count = min(int(count), len(values))
    if count <= 0:
        return np.asarray([], dtype=int)
    return np.lexsort((np.arange(len(values)), -values))[:count]


def jaccard_indices(left, right) -> float:
    a = set(int(value) for value in left)
    b = set(int(value) for value in right)
    union = len(a | b)
    return len(a & b) / union if union else 1.0


def compare_score_vectors(base, other) -> dict:
    base = np.asarray(base, dtype=float)
    other = np.asarray(other, dtype=float)
    if len(base) != len(other) or len(base) == 0:
        raise ValueError("score vectors must be non-empty and equal length")
    delta = other - base
    result = {
        "pearson_score_correlation": correlation(base, other),
        "spearman_rank_correlation": correlation(average_ranks(base), average_ranks(other)),
        "mean_absolute_score_delta": float(np.mean(np.abs(delta))),
        "max_absolute_score_delta": float(np.max(np.abs(delta))),
        "top_jaccard": {},
    }
    for top_n in (10, 100, 1_000):
        if len(base) >= top_n:
            result["top_jaccard"][str(top_n)] = jaccard_indices(
                top_indices(base, top_n), top_indices(other, top_n)
            )
    return result


def compare_weight_vectors(base, other) -> dict:
    base = np.asarray(base, dtype=float)
    other = np.asarray(other, dtype=float)
    if base.shape != other.shape or base.ndim != 1:
        raise ValueError("weight vectors must have the same one-dimensional shape")
    nonzero = (base != 0.0) | (other != 0.0)
    sign_disagreements = int(np.sum(np.sign(base[nonzero]) != np.sign(other[nonzero])))
    denominator = float(np.linalg.norm(base))
    return {
        "pearson_weight_correlation": correlation(base, other),
        "l2_absolute_delta": float(np.linalg.norm(other - base)),
        "l2_relative_delta": (
            float(np.linalg.norm(other - base) / denominator) if denominator > 0.0 else float("nan")
        ),
        "max_absolute_weight_delta": float(np.max(np.abs(other - base))),
        "sign_disagreement_count": sign_disagreements,
    }


def ridge_condition_diagnostics(model: dict) -> dict:
    eigenvalues = np.asarray(model.get("scaled_second_moment_eigenvalues", []), dtype=float)
    ridge_lambda = float(model.get("ridge_lambda", float("nan")))
    finite = eigenvalues[np.isfinite(eigenvalues)]
    positive = finite[finite > 1e-12]
    raw_condition = (
        float(np.max(positive) / np.min(positive)) if len(positive) else float("inf")
    )
    if len(finite) and math.isfinite(ridge_lambda) and ridge_lambda > 0.0:
        ridge_eigenvalues = finite + ridge_lambda
        ridge_condition = float(np.max(ridge_eigenvalues) / np.min(ridge_eigenvalues))
        ridge_min = float(np.min(ridge_eigenvalues))
        ridge_max = float(np.max(ridge_eigenvalues))
    else:
        ridge_condition = float("nan")
        ridge_min = float("nan")
        ridge_max = float("nan")
    return {
        "scaled_second_moment_condition_number": raw_condition,
        "ridge_system_condition_number": ridge_condition,
        "ridge_system_min_eigenvalue": ridge_min,
        "ridge_system_max_eigenvalue": ridge_max,
        "ridge_lambda": ridge_lambda,
    }


def runtime_identity() -> dict:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "platform": sys.platform,
    }


def json_safe(value: Any):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(payload: dict, output_path: Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
    return path.resolve()
