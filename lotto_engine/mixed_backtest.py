from __future__ import annotations

import random
from collections import Counter

import numpy as np
import pandas as pd

from .candidates import random_combination
from .config import BACKTEST_START_INDEX, BASELINE_SAMPLE_COUNT
from .loader import ROUND_COLUMN
from .mixed_scoring import (
    build_all_combination_baseline,
    build_draw_structure_records,
    build_mixed_profile,
    score_mixed_record,
    structure_record,
)
from .mixed_subtypes import (
    build_exact_mixed_subtype_baseline,
    mixed_subtype_fit_components,
    signature_family_diagnostics,
    subtype_information_diagnostics,
    suggest_mixed_signature_allocation,
)


def tie_safe_percentile(actual_score: float, baseline_scores: list[float]) -> float:
    if not baseline_scores:
        return 50.0
    strictly_below = sum(score < actual_score for score in baseline_scores)
    equal = sum(score == actual_score for score in baseline_scores)
    return (strictly_below + 0.5 * equal) / len(baseline_scores) * 100.0


def run_mixed_walk_forward_backtest(
    df: pd.DataFrame,
    start_index: int = BACKTEST_START_INDEX,
    baseline_samples: int = BASELINE_SAMPLE_COUNT,
) -> dict:
    if len(df) <= start_index:
        raise ValueError(f"Mixed backtest requires at least {start_index + 1} draws.")
    all_distribution = build_all_combination_baseline()
    all_records = build_draw_structure_records(df)
    percentiles_v231: list[float] = []
    percentiles_v25: list[float] = []
    tested_rounds: list[int] = []
    subtype_baseline = build_exact_mixed_subtype_baseline()
    family_target_counts: Counter = Counter()
    family_actual_counts: Counter = Counter()

    for idx in range(start_index, len(df)):
        target = all_records[idx]
        if target["pattern_type"] != "mixed":
            continue
        history = all_records[:idx]
        profile = build_mixed_profile(history, all_distribution)
        family_allocation = suggest_mixed_signature_allocation(history, baseline=subtype_baseline)
        subtype_diagnostics = subtype_information_diagnostics(history, subtype_baseline)
        family_diagnostics = signature_family_diagnostics(history, subtype_baseline)
        actual_score = score_mixed_record(target, profile)["mixed_slot_score"]
        actual_fit = mixed_subtype_fit_components(
            target, subtype_diagnostics, family_diagnostics, family_allocation
        )
        actual_v25_score = 0.90 * actual_score + 0.10 * actual_fit["mixed_subtype_fit_score"]
        family_target_counts.update(family_allocation)
        family_actual_counts[actual_fit["selected_family"]] += 1
        rng = random.Random(int(df.iloc[idx][ROUND_COLUMN]) * 23003 + baseline_samples)
        baseline_scores_v231: list[float] = []
        baseline_scores_v25: list[float] = []
        while len(baseline_scores_v231) < baseline_samples:
            record = structure_record(
                random_combination(rng),
                history[-1]["pattern_type"],
            )
            if record["pattern_type"] != "mixed":
                continue
            base_score = score_mixed_record(record, profile)["mixed_slot_score"]
            fit = mixed_subtype_fit_components(
                record, subtype_diagnostics, family_diagnostics, family_allocation
            )["mixed_subtype_fit_score"]
            baseline_scores_v231.append(base_score)
            baseline_scores_v25.append(0.90 * base_score + 0.10 * fit)
        percentiles_v231.append(tie_safe_percentile(actual_score, baseline_scores_v231))
        percentiles_v25.append(tie_safe_percentile(actual_v25_score, baseline_scores_v25))
        tested_rounds.append(int(df.iloc[idx][ROUND_COLUMN]))

    return {
        "version": "v2.5_mixed_subtype_allocation",
        "score_name": "mixed_slot_score",
        "score_disclaimer": (
            "mixed_slot_score is an internal empirical ranking score, "
            "not an actual winning probability."
        ),
        "mixed_tests": len(percentiles_v231),
        "baseline_samples": int(baseline_samples),
        "v231": _percentile_summary(percentiles_v231),
        "v25": _percentile_summary(percentiles_v25),
        "recent_stability": {
            "v231_recent_300": _percentile_summary(percentiles_v231[-300:]),
            "v231_recent_100": _percentile_summary(percentiles_v231[-100:]),
            "v25_recent_300": _percentile_summary(percentiles_v25[-300:]),
            "v25_recent_100": _percentile_summary(percentiles_v25[-100:]),
        },
        "family_coverage": {
            family: {"target_slots": target_count, "actual_matches": family_actual_counts[family]}
            for family, target_count in family_target_counts.items()
        },
        "tested_rounds": tested_rounds,
    }


def _percentile_summary(percentiles: list[float]) -> dict:
    return {
        "mean_percentile": float(np.mean(percentiles)) if percentiles else 0.0,
        "median_percentile": float(np.median(percentiles)) if percentiles else 0.0,
        "percentile_std": float(np.std(percentiles)) if percentiles else 0.0,
        "above_random_ratio": sum(value > 50.0 for value in percentiles) / len(percentiles) if percentiles else 0.0,
    }
