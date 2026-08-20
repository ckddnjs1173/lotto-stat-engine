from __future__ import annotations

import random
from collections import Counter

import numpy as np
import pandas as pd

from .candidates import random_combination
from .mixed_scoring import build_draw_structure_records, structure_record

SCREENING_FAIR_SAMPLES = 20_000
SCREENING_PERMUTATION_REPS = 500
CONFIRMATION_FAIR_SAMPLES = 100_000
CONFIRMATION_PERMUTATION_REPS = 5_000
DEFAULT_MAX_LAG = 10
DEFAULT_RECENT_WINDOW = 300

SCALAR_FEATURES = (
    "sum",
    "number_range",
    "odd_count",
    "max_gap",
    "min_gap",
    "consecutive_pair_count",
    "ending_duplicate_count",
    "empty_decade_section_count",
    "extreme_count",
)
PATTERN_TYPES = ("normal", "mixed", "outlier")


def lag_acf(values, max_lag: int = DEFAULT_MAX_LAG) -> list[float]:
    data = np.asarray(values, dtype=float)
    if data.size < 3:
        return []
    centered = data - float(data.mean())
    denominator = float(np.dot(centered, centered))
    if denominator <= 0.0:
        return [0.0] * min(int(max_lag), data.size - 1)
    result = []
    for lag in range(1, min(int(max_lag), data.size - 1) + 1):
        result.append(float(np.dot(centered[:-lag], centered[lag:]) / denominator))
    return result


def _recent_shift_stat(values, recent_window: int) -> float:
    data = np.asarray(values, dtype=float)
    if data.size < 4:
        return 0.0
    window = min(int(recent_window), max(1, data.size // 2))
    prior = data[:-window]
    recent = data[-window:]
    scale = float(np.std(data, ddof=1))
    if scale <= 0.0 or prior.size == 0:
        return 0.0
    return float(abs(float(recent.mean()) - float(prior.mean())) / scale)


def _cusum_stat(values) -> float:
    data = np.asarray(values, dtype=float)
    if data.size < 2:
        return 0.0
    scale = float(np.std(data, ddof=1))
    if scale <= 0.0:
        return 0.0
    centered = data - float(data.mean())
    return float(np.max(np.abs(np.cumsum(centered))) / (scale * np.sqrt(data.size)))


def permutation_series_screen(
    values,
    reps: int = SCREENING_PERMUTATION_REPS,
    max_lag: int = DEFAULT_MAX_LAG,
    recent_window: int = DEFAULT_RECENT_WINDOW,
    seed: int = 27,
) -> dict:
    data = np.asarray(values, dtype=float)
    observed_acf = lag_acf(data, max_lag)
    observed_max_acf = max((abs(value) for value in observed_acf), default=0.0)
    observed_shift = _recent_shift_stat(data, recent_window)
    observed_cusum = _cusum_stat(data)

    reps = int(reps)
    if reps <= 0:
        return {
            "lag_acf": observed_acf,
            "max_abs_acf": observed_max_acf,
            "recent_shift_stat": observed_shift,
            "cusum_stat": observed_cusum,
            "p_max_abs_acf": float("nan"),
            "p_recent_shift": float("nan"),
            "p_cusum": float("nan"),
        }

    rng = np.random.default_rng(int(seed))
    exceed_acf = 0
    exceed_shift = 0
    exceed_cusum = 0
    for _ in range(reps):
        permuted = rng.permutation(data)
        permuted_acf = lag_acf(permuted, max_lag)
        max_acf = max((abs(value) for value in permuted_acf), default=0.0)
        exceed_acf += max_acf >= observed_max_acf
        exceed_shift += _recent_shift_stat(permuted, recent_window) >= observed_shift
        exceed_cusum += _cusum_stat(permuted) >= observed_cusum

    denominator = reps + 1
    return {
        "lag_acf": observed_acf,
        "max_abs_acf": observed_max_acf,
        "recent_shift_stat": observed_shift,
        "cusum_stat": observed_cusum,
        "p_max_abs_acf": float((exceed_acf + 1) / denominator),
        "p_recent_shift": float((exceed_shift + 1) / denominator),
        "p_cusum": float((exceed_cusum + 1) / denominator),
    }


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    finite = [
        (name, float(value))
        for name, value in pvalues.items()
        if np.isfinite(value)
    ]
    ordered = sorted(finite, key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        candidate = min(1.0, (total - rank) * value)
        running = max(running, candidate)
        adjusted[name] = float(running)
    for name in pvalues:
        adjusted.setdefault(name, float("nan"))
    return adjusted


def pattern_transition_mutual_information(states) -> float:
    sequence = [str(value) for value in states]
    if len(sequence) < 2:
        return 0.0
    pair_counts = Counter(zip(sequence[:-1], sequence[1:]))
    left_counts = Counter(sequence[:-1])
    right_counts = Counter(sequence[1:])
    total = float(len(sequence) - 1)
    mi = 0.0
    for (left, right), count in pair_counts.items():
        joint = count / total
        p_left = left_counts[left] / total
        p_right = right_counts[right] / total
        if joint > 0.0 and p_left > 0.0 and p_right > 0.0:
            mi += joint * np.log(joint / (p_left * p_right))
    return float(mi)


def permutation_transition_screen(
    states,
    reps: int = SCREENING_PERMUTATION_REPS,
    seed: int = 27003,
) -> dict:
    sequence = np.asarray([str(value) for value in states], dtype=object)
    observed = pattern_transition_mutual_information(sequence)
    reps = int(reps)
    if reps <= 0:
        return {"mutual_information": observed, "p_value": float("nan")}
    rng = np.random.default_rng(int(seed))
    exceed = 0
    for _ in range(reps):
        permuted = rng.permutation(sequence)
        exceed += pattern_transition_mutual_information(permuted) >= observed
    return {
        "mutual_information": observed,
        "p_value": float((exceed + 1) / (reps + 1)),
    }


def _fair_records(samples: int, seed: int = 27001) -> list[dict]:
    rng = random.Random(int(seed))
    return [structure_record(random_combination(rng)) for _ in range(int(samples))]


def _pattern_proportions(records: list[dict]) -> dict[str, float]:
    counts = Counter(record["pattern_type"] for record in records)
    total = float(len(records))
    return {
        pattern: (float(counts[pattern] / total) if total else float("nan"))
        for pattern in PATTERN_TYPES
    }


def _mean_z_vs_fair(history_values, fair_values) -> tuple[float, float]:
    """Compare means while accounting for Monte Carlo error in the fair-null mean."""
    history = np.asarray(history_values, dtype=float)
    fair = np.asarray(fair_values, dtype=float)
    if history.size == 0 or fair.size < 2:
        return float("nan"), float("nan")
    fair_sd = float(np.std(fair, ddof=1))
    if fair_sd <= 0.0:
        return 0.0, 0.0
    standard_error = fair_sd * np.sqrt((1.0 / history.size) + (1.0 / fair.size))
    mean_z = (float(history.mean()) - float(fair.mean())) / standard_error
    return float(mean_z), float(standard_error)


def run_v27_null_stationarity_screen(
    df: pd.DataFrame,
    fair_samples: int = SCREENING_FAIR_SAMPLES,
    permutation_reps: int = SCREENING_PERMUTATION_REPS,
    max_lag: int = DEFAULT_MAX_LAG,
    recent_window: int = DEFAULT_RECENT_WINDOW,
) -> dict:
    if len(df) < 20:
        raise ValueError("v2.7 null/stationarity screen requires at least 20 draws")
    if int(fair_samples) <= 0:
        raise ValueError("fair_samples must be positive")
    if int(permutation_reps) <= 0:
        raise ValueError("permutation_reps must be positive")

    historical = build_draw_structure_records(df)
    fair = _fair_records(int(fair_samples))

    marginal = {}
    series_results = {}
    for index, feature in enumerate(SCALAR_FEATURES):
        history_values = np.asarray(
            [float(record[feature]) for record in historical], dtype=float
        )
        fair_values = np.asarray(
            [float(record[feature]) for record in fair], dtype=float
        )
        fair_sd = float(np.std(fair_values, ddof=1))
        mean_z, standard_error = _mean_z_vs_fair(history_values, fair_values)
        marginal[feature] = {
            "historical_mean": float(history_values.mean()),
            "fair_mean": float(fair_values.mean()),
            "fair_sd": fair_sd,
            "mean_difference_standard_error": standard_error,
            "historical_mean_z_vs_fair": float(mean_z),
        }
        series_results[feature] = permutation_series_screen(
            history_values,
            reps=int(permutation_reps),
            max_lag=int(max_lag),
            recent_window=int(recent_window),
            seed=27_100 + index * 101,
        )

    acf_adjusted = holm_adjust({
        feature: result["p_max_abs_acf"]
        for feature, result in series_results.items()
    })
    shift_adjusted = holm_adjust({
        feature: result["p_recent_shift"]
        for feature, result in series_results.items()
    })
    cusum_adjusted = holm_adjust({
        feature: result["p_cusum"]
        for feature, result in series_results.items()
    })
    for feature, result in series_results.items():
        result["holm_p_max_abs_acf"] = acf_adjusted[feature]
        result["holm_p_recent_shift"] = shift_adjusted[feature]
        result["holm_p_cusum"] = cusum_adjusted[feature]

    historical_types = [record["pattern_type"] for record in historical]
    transition = permutation_transition_screen(
        historical_types,
        reps=int(permutation_reps),
    )
    historical_pattern = _pattern_proportions(historical)
    fair_pattern = _pattern_proportions(fair)
    pattern_marginal = {
        pattern: {
            "historical": historical_pattern[pattern],
            "fair": fair_pattern[pattern],
            "difference": float(historical_pattern[pattern] - fair_pattern[pattern]),
        }
        for pattern in PATTERN_TYPES
    }

    alpha = 0.05
    serial_candidates = [
        feature
        for feature, result in series_results.items()
        if result["holm_p_max_abs_acf"] <= alpha
    ]
    shift_candidates = [
        feature
        for feature, result in series_results.items()
        if result["holm_p_recent_shift"] <= alpha
    ]
    change_point_candidates = [
        feature
        for feature, result in series_results.items()
        if result["holm_p_cusum"] <= alpha
    ]

    return {
        "version": "v27_null_stationarity_screen_v1",
        "screening_only": True,
        "draws": int(len(df)),
        "fair_samples": int(fair_samples),
        "permutation_reps": int(permutation_reps),
        "max_lag": int(max_lag),
        "recent_window": int(recent_window),
        "scalar_marginal_vs_fair": marginal,
        "series_screen": series_results,
        "pattern_type_marginal_vs_fair": pattern_marginal,
        "pattern_transition": transition,
        "screening_candidates": {
            "serial_dependence": serial_candidates,
            "recent_distribution_shift": shift_candidates,
            "change_point": change_point_candidates,
            "pattern_transition_dependence": bool(transition["p_value"] <= alpha),
        },
        "notes": {
            "multiple_testing": "Holm adjustment is applied separately to ACF, recent-shift, and CUSUM feature families.",
            "fair_draw_null": "Each fair sample is an independent exact 6-of-45 draw generated without replacement within the draw.",
            "mean_z_standard_error": "The fair-null mean comparison includes both historical sampling error and Monte Carlo error from the finite fair sample.",
            "promotion_rule": "Screening significance creates a confirmation candidate only; it is not production evidence.",
        },
    }
