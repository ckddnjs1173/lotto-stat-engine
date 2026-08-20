from __future__ import annotations

import random

import numpy as np
import pandas as pd

from .candidates import random_combination
from .config import BACKTEST_START_INDEX, BASE_WEIGHTS
from .loader import row_numbers
from .mixed_scoring import (
    build_all_combination_baseline,
    build_draw_structure_records,
    build_mixed_profile,
    score_mixed_record,
    structure_record,
)
from .profiles import build_profile
from .scoring import score_candidate
from .v27_validation import (
    DEFAULT_BOOTSTRAP_REPS,
    _block_bootstrap_mean_ci,
    tie_safe_percentile,
)

SCREENING_SAME_TYPE_SAMPLES = 100
CONFIRMATION_SAME_TYPE_SAMPLES = 500

NORMAL_OUTLIER_COMPONENTS = (
    "normal_structure_score",
    "outlier_survival_score",
    "historical_pattern_score",
    "type_balance_score",
    "number_dynamics_score",
)

MIXED_COMPONENTS = (
    "mixed_lift_score",
    "mixed_interaction_score",
    "normal_backbone_score",
    "controlled_extreme_score",
    "recency_consistency_score",
)

PATTERN_TYPES = ("normal", "mixed", "outlier")


def _sample_same_type_records(
    rng: random.Random,
    count: int,
    required_type: str,
) -> list[dict]:
    count = int(count)
    if count <= 0:
        raise ValueError("same-type sample count must be positive")
    if required_type not in PATTERN_TYPES:
        raise ValueError(f"unsupported pattern type: {required_type}")

    seen: set[tuple[int, ...]] = set()
    records: list[dict] = []
    max_attempts = max(1_000, count * 100)
    attempts = 0
    while len(records) < count and attempts < max_attempts:
        attempts += 1
        candidate = random_combination(rng)
        key = tuple(candidate)
        if key in seen:
            continue
        record = structure_record(candidate)
        if record["pattern_type"] != required_type:
            continue
        seen.add(key)
        records.append(record)

    if len(records) != count:
        raise RuntimeError(
            f"could not sample {count} unique {required_type} combinations "
            f"within {max_attempts} attempts"
        )
    return records


def normal_outlier_static_values(numbers: list[int], profile: dict) -> dict[str, float]:
    scored = score_candidate(numbers, profile, BASE_WEIGHTS)
    breakdown = scored["score_breakdown"]
    return {
        "current_branch_base": float(scored["prediction_score"]),
        **{
            name: float(breakdown[name])
            for name in NORMAL_OUTLIER_COMPONENTS
        },
    }


def mixed_static_values(record: dict, mixed_profile: dict) -> dict[str, float]:
    static_profile = dict(mixed_profile)
    static_profile.pop("dynamic_markov_decay", None)
    scored = score_mixed_record(record, static_profile)
    return {
        "current_branch_base": float(scored["mixed_slot_score"]),
        **{
            name: float(scored[name])
            for name in MIXED_COMPONENTS
        },
    }


def _window_values(entries: list[dict], total_draws: int, window: int | None) -> list[float]:
    selected = entries
    if window is not None:
        threshold = max(0, int(total_draws) - int(window))
        selected = [entry for entry in entries if int(entry["target_index"]) >= threshold]
    return [
        float(entry["percentile"])
        for entry in selected
        if np.isfinite(entry["percentile"])
    ]


def _mean(values: list[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _subset_summary(
    entries: list[dict],
    total_draws: int,
    bootstrap_reps: int,
    seed: int,
) -> dict:
    overall = _window_values(entries, total_draws, None)
    recent_300 = _window_values(entries, total_draws, 300)
    recent_100 = _window_values(entries, total_draws, 100)
    centered = [value - 50.0 for value in overall]
    low, high = _block_bootstrap_mean_ci(
        centered,
        reps=int(bootstrap_reps),
        seed=int(seed),
    )
    return {
        "tests": int(len(overall)),
        "mean_percentile": _mean(overall),
        "recent_300_mean_percentile": _mean(recent_300),
        "recent_100_mean_percentile": _mean(recent_100),
        "above_random_median_ratio": (
            float(np.mean([value > 50.0 for value in overall]))
            if overall else float("nan")
        ),
        "percentile_minus_50_block_bootstrap_95_ci": [low, high],
    }


def _passes_gate(summary: dict) -> bool:
    low = float(summary["percentile_minus_50_block_bootstrap_95_ci"][0])
    return bool(
        summary["tests"] > 0
        and float(summary["mean_percentile"]) > 50.0
        and low > 0.0
        and float(summary["recent_300_mean_percentile"]) >= 50.0
        and float(summary["recent_100_mean_percentile"]) >= 50.0
    )


def _component_summary(
    entries: list[dict],
    total_draws: int,
    name: str,
    bootstrap_reps: int,
    common_types: tuple[str, ...],
) -> dict:
    seed_base = 30_000 + sum((index + 1) * ord(char) for index, char in enumerate(name))
    overall = _subset_summary(entries, total_draws, bootstrap_reps, seed_base)

    by_type = {}
    for type_index, pattern in enumerate(PATTERN_TYPES):
        selected = [entry for entry in entries if entry["actual_type"] == pattern]
        by_type[pattern] = _subset_summary(
            selected,
            total_draws,
            bootstrap_reps,
            seed_base + 101 * (type_index + 1),
        )
        by_type[pattern]["screening_candidate"] = _passes_gate(by_type[pattern])

    common_type_means_ok = all(
        by_type[pattern]["tests"] > 0
        and float(by_type[pattern]["mean_percentile"]) >= 50.0
        for pattern in common_types
    )
    overall["by_actual_pattern_type"] = by_type
    overall["screening_candidate"] = bool(
        _passes_gate(overall) and common_type_means_ok
    )
    overall["type_specific_candidates"] = {
        pattern: bool(by_type[pattern]["screening_candidate"])
        for pattern in PATTERN_TYPES
        if by_type[pattern]["tests"] > 0
    }
    return overall


def run_v27_static_component_evidence_audit(
    df: pd.DataFrame,
    start_index: int = BACKTEST_START_INDEX,
    same_type_samples: int = SCREENING_SAME_TYPE_SAMPLES,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    progress_every: int = 0,
) -> dict:
    """Strict walk-forward same-pattern-type evidence audit for static components."""
    start_index = int(start_index)
    same_type_samples = int(same_type_samples)
    if len(df) <= start_index:
        raise ValueError(
            f"v2.7 static component audit requires at least {start_index + 1} draws"
        )
    if same_type_samples <= 0:
        raise ValueError("same_type_samples must be positive")

    records = build_draw_structure_records(df)
    all_combination_baseline = build_all_combination_baseline()

    benchmark_entries: list[dict] = []
    normal_outlier_entries = {
        name: [] for name in NORMAL_OUTLIER_COMPONENTS
    }
    mixed_entries = {name: [] for name in MIXED_COMPONENTS}

    total_targets = len(records) - start_index
    type_code = {"normal": 1, "mixed": 2, "outlier": 3}

    for idx in range(start_index, len(records)):
        target_record = records[idx]
        target_type = str(target_record["pattern_type"])
        round_no = int(target_record["draw_no"])
        rng = random.Random(
            round_no * 10067
            + same_type_samples * 131
            + type_code[target_type] * 1009
            + 3001
        )
        baseline_records = _sample_same_type_records(
            rng,
            same_type_samples,
            target_type,
        )

        if target_type == "mixed":
            mixed_profile = build_mixed_profile(
                records[:idx],
                all_combination_baseline,
            )
            actual_values = mixed_static_values(target_record, mixed_profile)
            baseline_values = [
                mixed_static_values(record, mixed_profile)
                for record in baseline_records
            ]
            component_names = MIXED_COMPONENTS
        else:
            profile = build_profile(df.iloc[:idx])
            actual_values = normal_outlier_static_values(
                row_numbers(df.iloc[idx]),
                profile,
            )
            baseline_values = [
                normal_outlier_static_values(list(record["numbers"]), profile)
                for record in baseline_records
            ]
            component_names = NORMAL_OUTLIER_COMPONENTS

        benchmark_entries.append({
            "target_index": idx,
            "actual_type": target_type,
            "percentile": tie_safe_percentile(
                actual_values["current_branch_base"],
                [item["current_branch_base"] for item in baseline_values],
            ),
        })

        destination = (
            mixed_entries if target_type == "mixed" else normal_outlier_entries
        )
        for name in component_names:
            destination[name].append({
                "target_index": idx,
                "actual_type": target_type,
                "percentile": tie_safe_percentile(
                    actual_values[name],
                    [item[name] for item in baseline_values],
                ),
            })

        completed = idx - start_index + 1
        if progress_every and completed % int(progress_every) == 0:
            print(
                f"v2.7 static evidence progress: {completed}/{total_targets} "
                f"targets (through draw {round_no})"
            )

    benchmark = _component_summary(
        benchmark_entries,
        len(records),
        "current_branch_base",
        int(bootstrap_reps),
        ("normal", "mixed", "outlier"),
    )
    normal_outlier = {
        name: _component_summary(
            entries,
            len(records),
            f"normal_outlier|{name}",
            int(bootstrap_reps),
            ("normal", "outlier"),
        )
        for name, entries in normal_outlier_entries.items()
    }
    mixed = {
        name: _component_summary(
            entries,
            len(records),
            f"mixed|{name}",
            int(bootstrap_reps),
            ("mixed",),
        )
        for name, entries in mixed_entries.items()
    }

    return {
        "version": "v27_static_component_evidence_screen_v1",
        "strict_walk_forward": True,
        "same_pattern_type_baseline": True,
        "start_index": start_index,
        "total_tests": int(total_targets),
        "same_type_samples_per_target": same_type_samples,
        "bootstrap_reps": int(bootstrap_reps),
        "benchmark_current_branch_base": benchmark,
        "normal_outlier_components": normal_outlier,
        "mixed_components": mixed,
        "screening_candidates": {
            "normal_outlier": [
                name
                for name, result in normal_outlier.items()
                if result["screening_candidate"]
            ],
            "mixed": [
                name
                for name, result in mixed.items()
                if result["screening_candidate"]
            ],
        },
        "notes": {
            "conditioning": (
                "Each target is compared only with fair random combinations of "
                "the same pattern type; this avoids the cross-type raw-score scale "
                "distortion documented in Phase 1/2."
            ),
            "excluded_dynamic_terms": [
                "legacy transition evidence is not reconsidered here",
                "Phase-4 transition_exact rejected",
                "Phase-4 momentum confirmation failed",
            ],
            "confirmation_same_type_samples": CONFIRMATION_SAME_TYPE_SAMPLES,
            "screening_only": (
                "A survivor earns a larger targeted same-type audit; no weight "
                "change is made from this screen."
            ),
        },
    }
