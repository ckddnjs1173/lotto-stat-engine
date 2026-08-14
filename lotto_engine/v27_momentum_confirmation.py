from __future__ import annotations

import math
import random

import numpy as np
import pandas as pd

from .config import BACKTEST_START_INDEX
from .mixed_scoring import build_draw_structure_records, structure_record
from .mixed_subtypes import _calculate_all_family_decay_momentum, signature_family
from .v27_dynamic_evidence import _sample_unique_combinations
from .v27_validation import (
    DEFAULT_BOOTSTRAP_REPS,
    _block_bootstrap_mean_ci,
    _window_mean,
    tie_safe_percentile,
)

CONFIRMATION_BASELINE_SAMPLES = 1_000
DEFAULT_ALPHA = 0.1
DEFAULT_DECAY_RATE = 0.05
DEFAULT_PRIOR_STRENGTH = 12.0


def momentum_lifts_for_history(records: list[dict]) -> dict[str, float]:
    """Mirror production all-draw momentum without transition computation."""
    if not records:
        return {}
    return _calculate_all_family_decay_momentum(
        records,
        decay_rate=DEFAULT_DECAY_RATE,
        alpha=DEFAULT_ALPHA,
        prior_strength=DEFAULT_PRIOR_STRENGTH,
        latest_draw=int(records[-1]["draw_no"]),
    )


def momentum_lookup(family: str, lifts: dict[str, float]) -> tuple[float, bool]:
    seen = family in lifts
    return float(lifts.get(family, 1.0)), bool(seen)


def _centered_ci(percentiles: list[float], bootstrap_reps: int, seed: int) -> list[float]:
    centered = [float(value) - 50.0 for value in percentiles if np.isfinite(value)]
    low, high = _block_bootstrap_mean_ci(centered, reps=int(bootstrap_reps), seed=int(seed))
    return [low, high]


def _by_type(percentiles: list[float], actual_types: list[str]) -> dict[str, dict]:
    result = {}
    for pattern in ("normal", "mixed", "outlier"):
        selected = [
            float(value)
            for value, actual_type in zip(percentiles, actual_types)
            if actual_type == pattern and np.isfinite(value)
        ]
        result[pattern] = {"tests": int(len(selected)), "mean_percentile": _window_mean(selected)}
    return result


def summarize_momentum_confirmation(
    full_percentiles: list[float],
    seen_percentiles: list[float],
    actual_types: list[str],
    actual_seen_flags: list[bool],
    baseline_seen_ratios: list[float],
    baseline_seen_counts: list[int],
    actual_log_lifts: list[float],
    bootstrap_reps: int,
) -> dict:
    full_ci = _centered_ci(full_percentiles, bootstrap_reps, seed=29_101)
    seen_ci = _centered_ci(seen_percentiles, bootstrap_reps, seed=29_102)
    finite_full = [float(value) for value in full_percentiles if np.isfinite(value)]
    finite_seen = [float(value) for value in seen_percentiles if np.isfinite(value)]
    actual_seen_numeric = [1.0 if value else 0.0 for value in actual_seen_flags]

    full = {
        "tests": int(len(finite_full)),
        "mean_percentile": _window_mean(full_percentiles),
        "recent_300_mean_percentile": _window_mean(full_percentiles, 300),
        "recent_100_mean_percentile": _window_mean(full_percentiles, 100),
        "above_random_median_ratio": (
            float(np.mean([value > 50.0 for value in finite_full])) if finite_full else float("nan")
        ),
        "percentile_minus_50_block_bootstrap_95_ci": full_ci,
        "by_actual_pattern_type": _by_type(full_percentiles, actual_types),
    }
    seen = {
        "tests": int(len(finite_seen)),
        "mean_percentile": _window_mean(seen_percentiles),
        "recent_300_target_window_mean_percentile": _window_mean(seen_percentiles, 300),
        "recent_100_target_window_mean_percentile": _window_mean(seen_percentiles, 100),
        "percentile_minus_50_block_bootstrap_95_ci": seen_ci,
    }
    coverage = {
        "actual_family_seen_ratio": _window_mean(actual_seen_numeric),
        "actual_family_seen_recent_300": _window_mean(actual_seen_numeric, 300),
        "actual_family_seen_recent_100": _window_mean(actual_seen_numeric, 100),
        "mean_baseline_family_seen_ratio": _window_mean(baseline_seen_ratios),
        "mean_baseline_family_seen_count": _window_mean([float(value) for value in baseline_seen_counts]),
        "minimum_baseline_family_seen_count": int(min(baseline_seen_counts, default=0)),
    }
    lift = {
        "mean_actual_log_lift": _window_mean(actual_log_lifts),
        "recent_300_actual_log_lift": _window_mean(actual_log_lifts, 300),
        "recent_100_actual_log_lift": _window_mean(actual_log_lifts, 100),
    }

    confirmed = bool(
        full["mean_percentile"] > 50.0
        and float(full_ci[0]) > 0.0
        and full["recent_300_mean_percentile"] >= 50.0
        and full["recent_100_mean_percentile"] >= 50.0
        and seen["mean_percentile"] > 50.0
        and float(seen_ci[0]) > 0.0
    )
    return {
        "full_fair_baseline": full,
        "seen_family_robustness": seen,
        "family_coverage": coverage,
        "actual_lift": lift,
        "confirmed": confirmed,
    }


def run_v27_momentum_confirmation(
    df: pd.DataFrame,
    start_index: int = BACKTEST_START_INDEX,
    baseline_samples: int = CONFIRMATION_BASELINE_SAMPLES,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    progress_every: int = 0,
) -> dict:
    """Strict walk-forward confirmation of the Phase-4 momentum survivor only."""
    start_index = int(start_index)
    baseline_samples = int(baseline_samples)
    if len(df) <= start_index:
        raise ValueError(f"v2.7 momentum confirmation requires at least {start_index + 1} draws")
    if baseline_samples <= 0:
        raise ValueError("baseline_samples must be positive")

    records = build_draw_structure_records(df)
    full_percentiles: list[float] = []
    seen_percentiles: list[float] = []
    actual_types: list[str] = []
    actual_seen_flags: list[bool] = []
    baseline_seen_ratios: list[float] = []
    baseline_seen_counts: list[int] = []
    actual_log_lifts: list[float] = []

    total_targets = len(records) - start_index
    for idx in range(start_index, len(records)):
        history = records[:idx]
        target = records[idx]
        lifts = momentum_lifts_for_history(history)
        actual_family = signature_family(target["subtype_signature"])
        actual_lift, actual_seen = momentum_lookup(actual_family, lifts)

        round_no = int(target["draw_no"])
        rng = random.Random(round_no * 10039 + baseline_samples * 113 + 2901)
        candidates = _sample_unique_combinations(rng, baseline_samples)
        baseline_lifts: list[float] = []
        seen_baseline_lifts: list[float] = []
        for candidate in candidates:
            record = structure_record(candidate)
            family = signature_family(record["subtype_signature"])
            lift, seen = momentum_lookup(family, lifts)
            baseline_lifts.append(lift)
            if seen:
                seen_baseline_lifts.append(lift)

        full_percentiles.append(tie_safe_percentile(actual_lift, baseline_lifts))
        seen_percentiles.append(
            tie_safe_percentile(actual_lift, seen_baseline_lifts) if actual_seen else float("nan")
        )
        actual_types.append(str(target["pattern_type"]))
        actual_seen_flags.append(actual_seen)
        baseline_seen_counts.append(len(seen_baseline_lifts))
        baseline_seen_ratios.append(float(len(seen_baseline_lifts) / baseline_samples))
        actual_log_lifts.append(math.log(max(actual_lift, 1e-12)))

        completed = idx - start_index + 1
        if progress_every and completed % int(progress_every) == 0:
            print(
                f"v2.7 momentum confirmation progress: {completed}/{total_targets} "
                f"targets (through draw {round_no})"
            )

    summary = summarize_momentum_confirmation(
        full_percentiles,
        seen_percentiles,
        actual_types,
        actual_seen_flags,
        baseline_seen_ratios,
        baseline_seen_counts,
        actual_log_lifts,
        int(bootstrap_reps),
    )
    return {
        "version": "v27_momentum_confirmation_v1",
        "strict_walk_forward": True,
        "survivor": "momentum",
        "start_index": start_index,
        "total_tests": int(total_targets),
        "baseline_samples_per_target": baseline_samples,
        "bootstrap_reps": int(bootstrap_reps),
        **summary,
        "notes": {
            "independent_baseline_seed": "confirmation uses a deterministic seed family distinct from the 200-sample screening",
            "production_equivalence": "momentum lifts call the same all-draw decay/shrinkage routine used by build_dynamic_family_model",
            "unseen_family": "an exact family absent from prior draws receives the production-equivalent neutral lift 1.0",
            "confirmation_rule": "full fair-baseline CI must be >0 with recent300/recent100 >=50, and seen-family conditional CI must also be >0",
        },
    }
