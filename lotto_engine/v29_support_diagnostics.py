from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Iterable

import numpy as np

from .candidates import random_combination
from .config import ROUND_COLUMN
from .loader import load_lotto_data
from .v28_reverse_ranking import FEATURE_NAMES as BASE_FEATURE_NAMES, candidate_feature_vector
from .v29_quadratic_reverse_ranking import (
    FEATURE_NAMES as QUADRATIC_FEATURE_NAMES,
    quadratic_basis_from_base,
)
from .v29_raw_recommendation import build_latest_v29_model

DEFAULT_TOP_JSON = Path("data/cache/v29_raw_personal.json")
DEFAULT_REFERENCE_SAMPLES = 200_000
DEFAULT_REFERENCE_SEED = 29_901


def empirical_midrank_percentile(value: float, reference: np.ndarray) -> float:
    """Empirical percentile with half credit for exact ties."""
    values = np.asarray(reference, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("reference must be a non-empty one-dimensional array")
    less = int(np.count_nonzero(values < float(value)))
    equal = int(np.count_nonzero(values == float(value)))
    return 100.0 * (less + 0.5 * equal) / values.size


def _quantiles(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("quantile input must be a non-empty one-dimensional array")
    probs = (0.001, 0.01, 0.05, 0.50, 0.95, 0.99, 0.999)
    return {
        f"p{prob * 100:g}": float(np.quantile(array, prob))
        for prob in probs
    }


def _robust_z(value: float, reference: np.ndarray) -> float:
    array = np.asarray(reference, dtype=float)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    if mad > 1e-12:
        return float(0.6744897501960817 * (float(value) - median) / mad)
    std = float(np.std(array))
    if std > 1e-12:
        return float((float(value) - float(np.mean(array))) / std)
    return 0.0


def _candidate_diagnostics(
    numbers: Iterable[int],
    model: dict,
    context: dict,
    reference_base: np.ndarray,
    reference_scores: np.ndarray,
) -> dict:
    nums = tuple(sorted(int(number) for number in numbers))
    base = candidate_feature_vector(nums, context)
    basis = quadratic_basis_from_base(base)
    weights = np.asarray(model["effective_weights"], dtype=float)
    contributions = weights * basis
    raw_score = float(np.sum(contributions))

    feature_support = {}
    outside_99_count = 0
    outside_999_count = 0
    for index, name in enumerate(BASE_FEATURE_NAMES):
        value = float(base[index])
        column = reference_base[:, index]
        percentile = empirical_midrank_percentile(value, column)
        outside_99 = bool(percentile < 0.5 or percentile > 99.5)
        outside_999 = bool(percentile < 0.05 or percentile > 99.95)
        outside_99_count += int(outside_99)
        outside_999_count += int(outside_999)
        feature_support[str(name)] = {
            "value": value,
            "percentile_vs_fair_reference": float(percentile),
            "robust_z": _robust_z(value, column),
            "outside_central_99_percent": outside_99,
            "outside_central_99_9_percent": outside_999,
        }

    abs_contrib = np.abs(contributions)
    abs_total = float(np.sum(abs_contrib))
    order = np.argsort(abs_contrib)[::-1]
    top5_abs = float(np.sum(abs_contrib[order[:5]]))
    top10_abs = float(np.sum(abs_contrib[order[:10]]))

    return {
        "numbers": list(nums),
        "raw_model_score": raw_score,
        "raw_score_percentile_vs_fair_reference": empirical_midrank_percentile(
            raw_score, reference_scores
        ),
        "outside_central_99_feature_count": int(outside_99_count),
        "outside_central_99_9_feature_count": int(outside_999_count),
        "feature_support": feature_support,
        "contribution_concentration": {
            "top5_abs_share": float(top5_abs / abs_total) if abs_total > 0.0 else 0.0,
            "top10_abs_share": float(top10_abs / abs_total) if abs_total > 0.0 else 0.0,
            "absolute_contribution_total": abs_total,
        },
        "strongest_terms": [
            {
                "term": str(QUADRATIC_FEATURE_NAMES[int(index)]),
                "value": float(basis[int(index)]),
                "weight": float(weights[int(index)]),
                "contribution": float(contributions[int(index)]),
            }
            for index in order[:10]
        ],
    }


def _bucket_means(values: np.ndarray, scores: np.ndarray, edges: list[float]) -> list[dict]:
    result: list[dict] = []
    for left, right in zip(edges[:-1], edges[1:]):
        if math.isinf(right):
            mask = values >= left
            label = f"[{left:g}, inf)"
        else:
            mask = (values >= left) & (values < right)
            label = f"[{left:g}, {right:g})"
        count = int(np.count_nonzero(mask))
        if count == 0:
            continue
        result.append({
            "bucket": label,
            "count": count,
            "mean_raw_score": float(np.mean(scores[mask])),
            "max_raw_score": float(np.max(scores[mask])),
        })
    return result


def _load_top_numbers(path: str | Path) -> list[list[int]]:
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"top recommendation JSON not found: {json_path}")
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    recommendations = payload.get("recommendations", [])
    if not recommendations:
        raise ValueError("top recommendation JSON contains no recommendations")
    return [list(item["numbers"]) for item in recommendations]


def run_v29_support_diagnostics(
    top_json: str | Path = DEFAULT_TOP_JSON,
    reference_samples: int = DEFAULT_REFERENCE_SAMPLES,
    seed: int = DEFAULT_REFERENCE_SEED,
    progress_every: int = 0,
) -> dict:
    """Measure whether raw v2.9 TOP-K candidates lie outside current fair support.

    This diagnostic never changes, clips, rescales, or reranks v2.9 scores. It only
    compares the already-selected candidates with a large structure-neutral fair
    reference sample evaluated in the exact same next-draw context.
    """
    reference_samples = int(reference_samples)
    if reference_samples <= 0:
        raise ValueError("reference_samples must be positive")

    df = load_lotto_data()
    fitted = build_latest_v29_model(df)
    model = fitted["model"]
    context = fitted["context"]
    top_numbers = _load_top_numbers(top_json)

    rng = random.Random(int(seed))
    seen: set[tuple[int, ...]] = set()
    base_rows = np.empty((reference_samples, len(BASE_FEATURE_NAMES)), dtype=float)
    scores = np.empty(reference_samples, dtype=float)
    completed = 0

    while completed < reference_samples:
        candidate = tuple(random_combination(rng))
        if candidate in seen:
            continue
        seen.add(candidate)
        base = candidate_feature_vector(candidate, context)
        basis = quadratic_basis_from_base(base)
        score = float(np.dot(model["effective_weights"], basis))
        base_rows[completed, :] = base
        scores[completed] = score
        completed += 1
        if progress_every and completed % int(progress_every) == 0:
            print(f"v2.9 support diagnostic progress: {completed:,}/{reference_samples:,}")

    feature_reference = {
        str(name): {
            "mean": float(np.mean(base_rows[:, index])),
            "std": float(np.std(base_rows[:, index])),
            "min": float(np.min(base_rows[:, index])),
            "max": float(np.max(base_rows[:, index])),
            "quantiles": _quantiles(base_rows[:, index]),
            "pearson_with_raw_score": float(
                np.corrcoef(base_rows[:, index], scores)[0, 1]
            ) if float(np.std(base_rows[:, index])) > 1e-12 else 0.0,
        }
        for index, name in enumerate(BASE_FEATURE_NAMES)
    }

    sum_index = BASE_FEATURE_NAMES.index("sum_abs_deviation_138")
    consecutive_index = BASE_FEATURE_NAMES.index("consecutive_pairs")
    sum_values = base_rows[:, sum_index]
    consecutive_values = base_rows[:, consecutive_index]

    consecutive_buckets = []
    for value in sorted(set(int(v) for v in consecutive_values.tolist())):
        mask = consecutive_values == value
        consecutive_buckets.append({
            "consecutive_pairs": int(value),
            "count": int(np.count_nonzero(mask)),
            "mean_raw_score": float(np.mean(scores[mask])),
            "max_raw_score": float(np.max(scores[mask])),
        })

    return {
        "version": "v29_support_extrapolation_diagnostic_v1",
        "diagnostic_only": True,
        "score_modified": False,
        "latest_draw": int(df.iloc[-1][ROUND_COLUMN]),
        "reference_samples": reference_samples,
        "reference_seed": int(seed),
        "reference_policy": "unique fair 6-of-45 combinations in the same next-draw context",
        "raw_score_reference": {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "quantiles": _quantiles(scores),
        },
        "feature_reference": feature_reference,
        "score_shape": {
            "sum_abs_deviation_buckets": _bucket_means(
                sum_values,
                scores,
                [0.0, 20.0, 40.0, 60.0, 80.0, 100.0, math.inf],
            ),
            "consecutive_pair_buckets": consecutive_buckets,
        },
        "top_candidate_diagnostics": [
            _candidate_diagnostics(numbers, model, context, base_rows, scores)
            for numbers in top_numbers
        ],
        "interpretation_rule": (
            "Features outside the central 99%/99.9% fair reference support and a raw score "
            "driven by concentrated quadratic terms indicate extrapolation pressure. This is "
            "diagnostic evidence only and must not be used to silently clip or rerank candidates."
        ),
    }


def write_v29_support_diagnostics(payload: dict, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path.resolve()
