from __future__ import annotations

import math
import random

import numpy as np
import pandas as pd

from .candidates import random_combination
from .config import BACKTEST_START_INDEX, ROUND_COLUMN
from .mixed_scoring import build_draw_structure_records, structure_record
from .mixed_subtypes import (
    _shrink_probability,
    build_dynamic_family_model,
    signature_family,
)
from .v27_validation import (
    DEFAULT_BOOTSTRAP_REPS,
    _block_bootstrap_mean_ci,
    _window_mean,
    tie_safe_percentile,
)

SCREENING_BASELINE_SAMPLES = 200
CONFIRMATION_BASELINE_SAMPLES = 1_000

DYNAMIC_COMPONENTS = (
    "transition_type",
    "transition_coarse",
    "transition_exact",
    "momentum",
)
PRIMARY_COMPONENTS = ("transition_exact", "momentum")


def raw_dynamic_lifts(family: str, model: dict) -> dict[str, float]:
    """Mirror production hierarchy before nonlinear lift-to-score mapping."""
    families = model.get("families", ())
    alpha = float(model.get("alpha", 0.1))
    fallback_prior = alpha / max(1.0, 1.0 + alpha * len(families))
    prior = float(model.get("priors", {}).get(family, fallback_prior))
    prior = max(prior, 1e-12)
    strength = float(model.get("transition_prior_strength", 12.0))

    type_counts = model.get("type_counts", {}).get(
        model.get("latest_pattern_type"), {}
    )
    type_probability = _shrink_probability(
        type_counts, family, prior, strength
    )

    coarse_counts = model.get("coarse_counts", {}).get(
        model.get("latest_coarse_family"), {}
    )
    coarse_probability = _shrink_probability(
        coarse_counts, family, type_probability, strength
    )

    exact_counts = model.get("exact_counts", {}).get(
        model.get("latest_family"), {}
    )
    exact_probability = _shrink_probability(
        exact_counts, family, coarse_probability, strength
    )

    momentum_lift = float(model.get("momentum_lifts", {}).get(family, 1.0))
    return {
        "transition_type": float(type_probability / prior),
        "transition_coarse": float(coarse_probability / prior),
        "transition_exact": float(exact_probability / prior),
        "momentum": momentum_lift,
    }


def _sample_unique_combinations(rng: random.Random, count: int) -> list[list[int]]:
    seen: set[tuple[int, ...]] = set()
    sampled: list[list[int]] = []
    while len(sampled) < int(count):
        candidate = random_combination(rng)
        key = tuple(candidate)
        if key in seen:
            continue
        seen.add(key)
        sampled.append(candidate)
    return sampled


def _component_summary(
    percentiles: list[float],
    log_lifts: list[float],
    actual_types: list[str],
    name: str,
    bootstrap_reps: int,
) -> dict:
    centered = [
        float(value) - 50.0
        for value in percentiles
        if np.isfinite(value)
    ]
    low, high = _block_bootstrap_mean_ci(
        centered,
        reps=int(bootstrap_reps),
        seed=28_000 + sum((index + 1) * ord(char) for index, char in enumerate(name)),
    )
    finite = [float(value) for value in percentiles if np.isfinite(value)]
    by_type = {}
    for pattern in ("normal", "mixed", "outlier"):
        selected = [
            float(value)
            for value, actual_type in zip(percentiles, actual_types)
            if actual_type == pattern and np.isfinite(value)
        ]
        by_type[pattern] = {
            "tests": int(len(selected)),
            "mean_percentile": _window_mean(selected),
        }

    result = {
        "tests": int(len(finite)),
        "mean_percentile": _window_mean(percentiles),
        "recent_300_mean_percentile": _window_mean(percentiles, 300),
        "recent_100_mean_percentile": _window_mean(percentiles, 100),
        "above_random_median_ratio": (
            float(np.mean([value > 50.0 for value in finite]))
            if finite
            else float("nan")
        ),
        "mean_actual_log_lift": _window_mean(log_lifts),
        "recent_300_actual_log_lift": _window_mean(log_lifts, 300),
        "recent_100_actual_log_lift": _window_mean(log_lifts, 100),
        "percentile_minus_50_block_bootstrap_95_ci": [low, high],
        "by_actual_pattern_type": by_type,
    }
    result["screening_candidate"] = bool(
        name in PRIMARY_COMPONENTS
        and result["mean_percentile"] > 50.0
        and low > 0.0
        and result["recent_300_mean_percentile"] >= 50.0
        and result["recent_100_mean_percentile"] >= 50.0
    )
    return result


def _paired_percentile_summary(
    left: list[float],
    right: list[float],
    name: str,
    bootstrap_reps: int,
) -> dict:
    differences = [
        float(a) - float(b)
        for a, b in zip(left, right)
        if np.isfinite(a) and np.isfinite(b)
    ]
    low, high = _block_bootstrap_mean_ci(
        differences,
        reps=int(bootstrap_reps),
        seed=28_500 + sum((index + 1) * ord(char) for index, char in enumerate(name)),
    )
    return {
        "tests": int(len(differences)),
        "mean_percentile_delta": _window_mean(differences),
        "recent_300_delta": _window_mean(differences, 300),
        "recent_100_delta": _window_mean(differences, 100),
        "block_bootstrap_95_ci": [low, high],
    }


def run_v27_dynamic_evidence_audit(
    df: pd.DataFrame,
    start_index: int = BACKTEST_START_INDEX,
    baseline_samples: int = SCREENING_BASELINE_SAMPLES,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    progress_every: int = 0,
) -> dict:
    """Strict walk-forward raw transition/momentum ranking against fair tickets."""
    start_index = int(start_index)
    baseline_samples = int(baseline_samples)
    if len(df) <= start_index:
        raise ValueError(
            f"v2.7 dynamic evidence audit requires at least {start_index + 1} draws"
        )
    if baseline_samples <= 0:
        raise ValueError("baseline_samples must be positive")

    records = build_draw_structure_records(df)
    percentiles = {name: [] for name in DYNAMIC_COMPONENTS}
    log_lifts = {name: [] for name in DYNAMIC_COMPONENTS}
    actual_types: list[str] = []

    total_targets = len(records) - start_index
    for idx in range(start_index, len(records)):
        history = records[:idx]
        target = records[idx]
        model = build_dynamic_family_model(history)
        actual_family = signature_family(target["subtype_signature"])
        actual = raw_dynamic_lifts(actual_family, model)

        round_no = int(target["draw_no"])
        rng = random.Random(round_no * 10037 + baseline_samples * 109 + 2801)
        candidates = _sample_unique_combinations(rng, baseline_samples)
        baseline_lifts = {name: [] for name in DYNAMIC_COMPONENTS}
        for candidate in candidates:
            record = structure_record(candidate)
            family = signature_family(record["subtype_signature"])
            lifts = raw_dynamic_lifts(family, model)
            for name in DYNAMIC_COMPONENTS:
                baseline_lifts[name].append(float(lifts[name]))

        actual_types.append(str(target["pattern_type"]))
        for name in DYNAMIC_COMPONENTS:
            actual_lift = float(actual[name])
            percentiles[name].append(
                tie_safe_percentile(actual_lift, baseline_lifts[name])
            )
            log_lifts[name].append(math.log(max(actual_lift, 1e-12)))

        completed = idx - start_index + 1
        if progress_every and completed % int(progress_every) == 0:
            print(
                f"v2.7 dynamic evidence progress: {completed}/{total_targets} "
                f"targets (through draw {round_no})"
            )

    summaries = {
        name: _component_summary(
            percentiles[name],
            log_lifts[name],
            actual_types,
            name,
            int(bootstrap_reps),
        )
        for name in DYNAMIC_COMPONENTS
    }

    comparisons = {
        "coarse_minus_type": _paired_percentile_summary(
            percentiles["transition_coarse"],
            percentiles["transition_type"],
            "coarse_minus_type",
            int(bootstrap_reps),
        ),
        "exact_minus_coarse": _paired_percentile_summary(
            percentiles["transition_exact"],
            percentiles["transition_coarse"],
            "exact_minus_coarse",
            int(bootstrap_reps),
        ),
        "exact_minus_type": _paired_percentile_summary(
            percentiles["transition_exact"],
            percentiles["transition_type"],
            "exact_minus_type",
            int(bootstrap_reps),
        ),
    }

    return {
        "version": "v27_dynamic_evidence_screen_v1",
        "strict_walk_forward": True,
        "start_index": start_index,
        "total_tests": int(total_targets),
        "baseline_samples_per_target": baseline_samples,
        "bootstrap_reps": int(bootstrap_reps),
        "components": summaries,
        "transition_level_comparisons": comparisons,
        "screening_candidates": {
            name: bool(summaries[name]["screening_candidate"])
            for name in PRIMARY_COMPONENTS
        },
        "notes": {
            "baseline": "deterministic unique fair 6-of-45 combinations scored under the same historical dynamic model as the target",
            "primary_hypotheses": list(PRIMARY_COMPONENTS),
            "diagnostic_transition_levels": ["transition_type", "transition_coarse"],
            "confirmation_baseline_samples": CONFIRMATION_BASELINE_SAMPLES,
            "screening_rule": "primary component requires CI(percentile-50)>0 and overall/recent300/recent100 percentile >= 50",
        },
    }
