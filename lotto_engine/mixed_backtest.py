from __future__ import annotations

import random

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
    percentiles: list[float] = []
    tested_rounds: list[int] = []

    for idx in range(start_index, len(df)):
        target = all_records[idx]
        if target["pattern_type"] != "mixed":
            continue
        history = all_records[:idx]
        profile = build_mixed_profile(history, all_distribution)
        actual_score = score_mixed_record(target, profile)["mixed_slot_score"]
        rng = random.Random(int(df.iloc[idx][ROUND_COLUMN]) * 23003 + baseline_samples)
        baseline_scores: list[float] = []
        while len(baseline_scores) < baseline_samples:
            record = structure_record(
                random_combination(rng),
                history[-1]["pattern_type"],
            )
            if record["pattern_type"] != "mixed":
                continue
            baseline_scores.append(
                score_mixed_record(record, profile)["mixed_slot_score"]
            )
        percentiles.append(tie_safe_percentile(actual_score, baseline_scores))
        tested_rounds.append(int(df.iloc[idx][ROUND_COLUMN]))

    return {
        "version": "v2.3.1_portfolio_mixed_slot",
        "score_name": "mixed_slot_score",
        "score_disclaimer": (
            "mixed_slot_score is an internal empirical ranking score, "
            "not an actual winning probability."
        ),
        "mixed_tests": len(percentiles),
        "baseline_samples": int(baseline_samples),
        "mean_percentile": float(np.mean(percentiles)) if percentiles else 0.0,
        "median_percentile": float(np.median(percentiles)) if percentiles else 0.0,
        "percentile_std": float(np.std(percentiles)) if percentiles else 0.0,
        "above_random_ratio": (
            sum(value > 50.0 for value in percentiles) / len(percentiles)
            if percentiles else 0.0
        ),
        "tested_rounds": tested_rounds,
    }
