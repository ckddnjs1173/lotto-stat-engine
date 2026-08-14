from __future__ import annotations

import random

import numpy as np
import pandas as pd

from .candidates import random_combination
from .config import BACKTEST_START_INDEX, BASE_WEIGHTS, ROUND_COLUMN
from .loader import row_numbers
from .mixed_scoring import (
    build_all_combination_baseline,
    build_draw_structure_records,
    build_mixed_profile,
)
from .profiles import build_profile
from .v27_validation import (
    _block_bootstrap_mean_ci,
    _window_mean,
    score_v27_audit_candidate,
    tie_safe_percentile,
)

SCREENING_CALIBRATION_SAMPLES = 1_000
SCREENING_EVALUATION_SAMPLES = 500
CONFIRMATION_CALIBRATION_SAMPLES = 3_000
CONFIRMATION_EVALUATION_SAMPLES = 2_000
DEFAULT_BOOTSTRAP_REPS = 2_000
CALIBRATION_MODELS = ("base_only", "v26_full")


def empirical_type_score(raw_score: float, calibration_scores: list[float]) -> float:
    """Monotonic empirical-CDF mapping on an independent same-type candidate sample."""
    return tie_safe_percentile(float(raw_score), [float(value) for value in calibration_scores])


def _sample_unique_candidates(
    rng: random.Random,
    count: int,
    forbidden: set[tuple[int, ...]] | None = None,
) -> tuple[list[list[int]], set[tuple[int, ...]]]:
    seen = set(forbidden or set())
    created: list[list[int]] = []
    local: set[tuple[int, ...]] = set()
    while len(created) < int(count):
        candidate = random_combination(rng)
        key = tuple(candidate)
        if key in seen or key in local:
            continue
        local.add(key)
        created.append(candidate)
    return created, seen | local


def _calibration_buckets(scored: list[dict], model_name: str) -> dict[str, list[float]]:
    buckets = {"normal": [], "mixed": [], "outlier": []}
    for item in scored:
        buckets[item["pattern_type"]].append(float(item["scores"][model_name]))
    return buckets


def _calibrated_scores(
    scored: list[dict],
    model_name: str,
    buckets: dict[str, list[float]],
) -> list[float]:
    return [
        empirical_type_score(
            float(item["scores"][model_name]),
            buckets[item["pattern_type"]],
        )
        for item in scored
    ]


def _summary(
    raw_percentiles: list[float],
    calibrated_percentiles: list[float],
    actual_types: list[str],
    calibration_min_type_counts: list[int],
    model_name: str,
    bootstrap_reps: int,
) -> dict:
    deltas = [
        float(calibrated) - float(raw)
        for raw, calibrated in zip(raw_percentiles, calibrated_percentiles)
        if np.isfinite(raw) and np.isfinite(calibrated)
    ]
    low, high = _block_bootstrap_mean_ci(
        deltas,
        reps=int(bootstrap_reps),
        seed=27_700 + sum((index + 1) * ord(char) for index, char in enumerate(model_name)),
    )

    by_type = {}
    for pattern in ("normal", "mixed", "outlier"):
        raw_values = [
            value for value, actual_type in zip(raw_percentiles, actual_types)
            if actual_type == pattern
        ]
        calibrated_values = [
            value for value, actual_type in zip(calibrated_percentiles, actual_types)
            if actual_type == pattern
        ]
        by_type[pattern] = {
            "tests": int(len(raw_values)),
            "raw_mean_percentile": _window_mean(raw_values),
            "calibrated_mean_percentile": _window_mean(calibrated_values),
            "mean_delta": _window_mean(
                [float(c) - float(r) for r, c in zip(raw_values, calibrated_values)]
            ),
        }

    finite_calibrated = [
        value for value in calibrated_percentiles if np.isfinite(value)
    ]
    return {
        "raw_mean_percentile": _window_mean(raw_percentiles),
        "raw_recent_300": _window_mean(raw_percentiles, 300),
        "raw_recent_100": _window_mean(raw_percentiles, 100),
        "calibrated_mean_percentile": _window_mean(calibrated_percentiles),
        "calibrated_recent_300": _window_mean(calibrated_percentiles, 300),
        "calibrated_recent_100": _window_mean(calibrated_percentiles, 100),
        "calibrated_above_random_median_ratio": (
            float(np.mean([value > 50.0 for value in finite_calibrated]))
            if finite_calibrated else float("nan")
        ),
        "calibrated_minus_raw": {
            "overall": _window_mean(deltas),
            "recent_300": _window_mean(deltas, 300),
            "recent_100": _window_mean(deltas, 100),
            "block_bootstrap_95_ci": [low, high],
        },
        "minimum_calibration_type_count": int(min(calibration_min_type_counts, default=0)),
        "mean_minimum_calibration_type_count": (
            float(np.mean(calibration_min_type_counts)) if calibration_min_type_counts else 0.0
        ),
        "by_actual_pattern_type": by_type,
    }


def run_v27_calibration_audit(
    df: pd.DataFrame,
    start_index: int = BACKTEST_START_INDEX,
    calibration_samples: int = SCREENING_CALIBRATION_SAMPLES,
    evaluation_samples: int = SCREENING_EVALUATION_SAMPLES,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    progress_every: int = 0,
) -> dict:
    """Strict walk-forward test of independent type-wise empirical score calibration."""
    start_index = int(start_index)
    calibration_samples = int(calibration_samples)
    evaluation_samples = int(evaluation_samples)
    if len(df) <= start_index:
        raise ValueError(f"v2.7 calibration audit requires at least {start_index + 1} draws")
    if calibration_samples <= 0 or evaluation_samples <= 0:
        raise ValueError("calibration_samples and evaluation_samples must be positive")

    all_combination_baseline = build_all_combination_baseline()
    raw_percentiles = {name: [] for name in CALIBRATION_MODELS}
    calibrated_percentiles = {name: [] for name in CALIBRATION_MODELS}
    actual_types: list[str] = []
    calibration_min_type_counts: list[int] = []

    for idx in range(start_index, len(df)):
        history = df.iloc[:idx].copy()
        target = df.iloc[idx]
        profile = build_profile(history)
        records = build_draw_structure_records(history)
        mixed_profile = build_mixed_profile(records, all_combination_baseline)
        actual = score_v27_audit_candidate(
            row_numbers(target), profile, mixed_profile, BASE_WEIGHTS
        )

        round_no = int(target[ROUND_COLUMN])
        calibration_rng = random.Random(
            round_no * 10007 + calibration_samples * 101 + 2701
        )
        evaluation_rng = random.Random(
            round_no * 10009 + evaluation_samples * 103 + 2702
        )
        calibration_candidates, seen = _sample_unique_candidates(
            calibration_rng, calibration_samples
        )
        evaluation_candidates, _ = _sample_unique_candidates(
            evaluation_rng, evaluation_samples, forbidden=seen
        )

        calibration_scored = [
            score_v27_audit_candidate(candidate, profile, mixed_profile, BASE_WEIGHTS)
            for candidate in calibration_candidates
        ]
        evaluation_scored = [
            score_v27_audit_candidate(candidate, profile, mixed_profile, BASE_WEIGHTS)
            for candidate in evaluation_candidates
        ]

        actual_types.append(actual["pattern_type"])
        type_counts = {
            pattern: sum(item["pattern_type"] == pattern for item in calibration_scored)
            for pattern in ("normal", "mixed", "outlier")
        }
        calibration_min_type_counts.append(min(type_counts.values()))

        for model_name in CALIBRATION_MODELS:
            raw_eval = [float(item["scores"][model_name]) for item in evaluation_scored]
            raw_percentiles[model_name].append(
                tie_safe_percentile(actual["scores"][model_name], raw_eval)
            )

            buckets = _calibration_buckets(calibration_scored, model_name)
            if any(not buckets[pattern] for pattern in ("normal", "mixed", "outlier")):
                counts = {key: len(value) for key, value in buckets.items()}
                raise RuntimeError(
                    f"calibration sample missing pattern type at draw {round_no}: {counts}"
                )

            actual_calibrated = empirical_type_score(
                float(actual["scores"][model_name]),
                buckets[actual["pattern_type"]],
            )
            evaluation_calibrated = _calibrated_scores(
                evaluation_scored, model_name, buckets
            )
            calibrated_percentiles[model_name].append(
                tie_safe_percentile(actual_calibrated, evaluation_calibrated)
            )

        if progress_every and ((idx - start_index + 1) % int(progress_every) == 0):
            print(
                f"v2.7 calibration progress: {idx - start_index + 1}/"
                f"{len(df) - start_index} targets (through draw {round_no})"
            )

    return {
        "version": "v27_type_calibration_audit_v1",
        "benchmark_commit": "ecbbd02235b9ed8f6940ae47aa46e7d4c0e53499",
        "strict_walk_forward": True,
        "calibration_method": "independent same-type empirical mid-rank CDF",
        "start_index": start_index,
        "total_tests": int(len(df) - start_index),
        "calibration_samples_per_target": calibration_samples,
        "evaluation_samples_per_target": evaluation_samples,
        "bootstrap_reps": int(bootstrap_reps),
        "models": {
            name: _summary(
                raw_percentiles[name],
                calibrated_percentiles[name],
                actual_types,
                calibration_min_type_counts,
                name,
                bootstrap_reps,
            )
            for name in CALIBRATION_MODELS
        },
        "notes": {
            "calibration_sample_is_independent_from_evaluation_sample": True,
            "candidate_eligibility_changed": False,
            "production_score_changed": False,
            "screening_calibration_samples": SCREENING_CALIBRATION_SAMPLES,
            "screening_evaluation_samples": SCREENING_EVALUATION_SAMPLES,
            "confirmation_calibration_samples": CONFIRMATION_CALIBRATION_SAMPLES,
            "confirmation_evaluation_samples": CONFIRMATION_EVALUATION_SAMPLES,
        },
    }
