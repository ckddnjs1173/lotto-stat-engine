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
    score_mixed_record,
    structure_record,
)
from .mixed_subtypes import family_dynamic_scores, signature_family
from .profiles import build_profile
from .scoring import SCORE_WEIGHTS, score_candidate

SCREENING_BASELINE_SAMPLES = 500
CONFIRMATION_BASELINE_SAMPLES = 2_000
DEFAULT_BOOTSTRAP_REPS = 2_000
DEFAULT_BOOTSTRAP_BLOCK = 20

PRIMARY_MODELS = (
    "base_only",
    "base_plus_transition",
    "base_plus_momentum",
    "v26_full",
)

BASE_AUDIT_MODELS = (
    "base_no_type_rarity",
    "base_no_legacy_transition",
    "base_clean_no_rarity_or_legacy_transition",
)

FULL_AUDIT_MODELS = (
    "full_no_type_rarity",
    "full_no_legacy_transition",
    "full_clean_no_rarity_or_legacy_transition",
)

MODEL_NAMES = (*PRIMARY_MODELS, *BASE_AUDIT_MODELS, *FULL_AUDIT_MODELS)


def tie_safe_percentile(actual_score: float, baseline_scores: list[float]) -> float:
    """Mid-rank percentile; equal scores receive half credit."""
    if not baseline_scores:
        return float("nan")
    actual = float(actual_score)
    less = sum(float(score) < actual for score in baseline_scores)
    equal = sum(float(score) == actual for score in baseline_scores)
    return 100.0 * (less + 0.5 * equal) / len(baseline_scores)


def _renormalized_legacy_base(
    breakdown: dict[str, float],
    removed: frozenset[str],
) -> float:
    """Remove selected positive-weight legacy terms and renormalize to 100%."""
    active = {
        key: float(weight)
        for key, weight in SCORE_WEIGHTS.items()
        if float(weight) > 0.0 and key not in removed
    }
    denominator = sum(active.values())
    if denominator <= 0.0:
        raise ValueError("at least one positive legacy base component must remain")
    return float(
        sum(weight * float(breakdown[key]) for key, weight in active.items())
        / denominator
    )


def _compose(base_score: float, transition_score: float, momentum_score: float) -> dict[str, float]:
    base = float(base_score)
    transition = float(transition_score)
    momentum = float(momentum_score)
    return {
        "base_only": base,
        "base_plus_transition": 0.85 * base + 0.15 * transition,
        "base_plus_momentum": 0.85 * base + 0.15 * momentum,
        "v26_full": 0.70 * base + 0.15 * transition + 0.15 * momentum,
    }


def score_v27_audit_candidate(
    numbers: list[int],
    profile: dict,
    mixed_profile: dict,
    weights: dict[str, float] | None = None,
) -> dict:
    """Research-only mirror of v2.6 scoring plus predeclared ablations.

    Production code is not modified. Mixed candidates retain their mixed base;
    the legacy-base ablations apply only to normal/outlier candidates, which is
    exactly where type rarity and the legacy type-transition term exist.
    """
    weights = dict(weights or BASE_WEIGHTS)
    record = structure_record(numbers, profile["latest_pattern_type"])
    dynamic = mixed_profile["dynamic_markov_decay"]

    if record["pattern_type"] == "mixed":
        scored = score_mixed_record(record, mixed_profile)
        base = float(scored["base_score"])
        transition = float(scored["transition_lift_score"])
        momentum = float(scored["momentum_lift_score"])
        base_no_rarity = base
        base_no_legacy_transition = base
        base_clean = base
    else:
        scored = score_candidate(list(record["numbers"]), profile, weights)
        base = float(scored["prediction_score"])
        breakdown = {
            key: float(value) for key, value in scored["score_breakdown"].items()
        }
        base_no_rarity = _renormalized_legacy_base(
            breakdown, frozenset({"type_balance_score"})
        )
        base_no_legacy_transition = _renormalized_legacy_base(
            breakdown, frozenset({"transition_score"})
        )
        base_clean = _renormalized_legacy_base(
            breakdown, frozenset({"type_balance_score", "transition_score"})
        )
        transition, momentum = family_dynamic_scores(
            signature_family(record["subtype_signature"]), dynamic
        )

    scores = _compose(base, transition, momentum)
    scores.update({
        "base_no_type_rarity": base_no_rarity,
        "base_no_legacy_transition": base_no_legacy_transition,
        "base_clean_no_rarity_or_legacy_transition": base_clean,
        "full_no_type_rarity": 0.70 * base_no_rarity + 0.15 * transition + 0.15 * momentum,
        "full_no_legacy_transition": (
            0.70 * base_no_legacy_transition + 0.15 * transition + 0.15 * momentum
        ),
        "full_clean_no_rarity_or_legacy_transition": (
            0.70 * base_clean + 0.15 * transition + 0.15 * momentum
        ),
    })
    return {
        "numbers": list(record["numbers"]),
        "pattern_type": record["pattern_type"],
        "scores": scores,
        "components": {
            "base_score": base,
            "transition_lift_score": float(transition),
            "momentum_lift_score": float(momentum),
            "base_no_type_rarity": base_no_rarity,
            "base_no_legacy_transition": base_no_legacy_transition,
            "base_clean_no_rarity_or_legacy_transition": base_clean,
        },
    }


def _window_mean(values: list[float], window: int | None = None) -> float:
    selected = list(values)
    if window is not None:
        selected = selected[-min(int(window), len(selected)):]
    finite = [float(value) for value in selected if np.isfinite(value)]
    if not finite:
        return float("nan")
    return float(np.mean(finite))


def _block_bootstrap_mean_ci(
    values: list[float],
    reps: int = DEFAULT_BOOTSTRAP_REPS,
    block_size: int = DEFAULT_BOOTSTRAP_BLOCK,
    seed: int = 27,
) -> tuple[float, float]:
    """Circular block-bootstrap CI for a mean, preserving local dependence."""
    data = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    n = len(data)
    if n == 0:
        return float("nan"), float("nan")
    if n == 1 or int(reps) <= 0:
        value = float(data.mean())
        return value, value

    block = max(1, min(int(block_size), n))
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(reps), dtype=float)
    offsets = np.arange(block, dtype=int)
    blocks_needed = (n + block - 1) // block
    for rep in range(int(reps)):
        starts = rng.integers(0, n, size=blocks_needed)
        indices = np.concatenate([
            (int(start) + offsets) % n for start in starts
        ])[:n]
        means[rep] = float(data[indices].mean())
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def _model_seed(name: str, reference: str) -> int:
    text = f"{name}|{reference}"
    return 27_000 + sum((index + 1) * ord(char) for index, char in enumerate(text))


def _model_summary(
    percentiles: list[float],
    conditioned_percentiles: list[float],
    actual_types: list[str],
    same_type_counts: list[int],
) -> dict:
    by_type = {}
    conditioned_by_type = {}
    for pattern in ("normal", "mixed", "outlier"):
        selected = [
            value for value, actual_type in zip(percentiles, actual_types)
            if actual_type == pattern
        ]
        selected_conditioned = [
            value for value, actual_type in zip(conditioned_percentiles, actual_types)
            if actual_type == pattern
        ]
        by_type[pattern] = {
            "tests": len(selected),
            "mean_percentile": _window_mean(selected),
        }
        conditioned_by_type[pattern] = {
            "tests": sum(np.isfinite(value) for value in selected_conditioned),
            "mean_percentile": _window_mean(selected_conditioned),
        }

    finite = [value for value in percentiles if np.isfinite(value)]
    return {
        "tests": len(finite),
        "mean_percentile": _window_mean(percentiles),
        "recent_300_mean_percentile": _window_mean(percentiles, 300),
        "recent_100_mean_percentile": _window_mean(percentiles, 100),
        "above_random_median_ratio": (
            float(np.mean([value > 50.0 for value in finite])) if finite else float("nan")
        ),
        "type_conditioned_mean_percentile": _window_mean(conditioned_percentiles),
        "type_conditioned_recent_300": _window_mean(conditioned_percentiles, 300),
        "type_conditioned_recent_100": _window_mean(conditioned_percentiles, 100),
        "mean_same_type_baseline_count": (
            float(np.mean(same_type_counts)) if same_type_counts else 0.0
        ),
        "minimum_same_type_baseline_count": min(same_type_counts, default=0),
        "by_actual_pattern_type": by_type,
        "conditioned_by_actual_pattern_type": conditioned_by_type,
    }


def _paired_summary(
    left: list[float],
    right: list[float],
    name: str,
    reference: str,
    bootstrap_reps: int,
) -> dict:
    differences = [
        float(a) - float(b)
        for a, b in zip(left, right)
        if np.isfinite(a) and np.isfinite(b)
    ]
    low, high = _block_bootstrap_mean_ci(
        differences,
        reps=bootstrap_reps,
        seed=_model_seed(name, reference),
    )
    return {
        "tests": len(differences),
        "mean_percentile_delta": _window_mean(differences),
        "recent_300_delta": _window_mean(differences, 300),
        "recent_100_delta": _window_mean(differences, 100),
        "block_bootstrap_95_ci": [low, high],
    }


def run_v27_score_audit(
    df: pd.DataFrame,
    start_index: int = BACKTEST_START_INDEX,
    baseline_samples: int = SCREENING_BASELINE_SAMPLES,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    include_rows: bool = False,
    progress_every: int = 0,
) -> dict:
    """Strict walk-forward score ablation against deterministic random baselines."""
    start_index = int(start_index)
    baseline_samples = int(baseline_samples)
    if len(df) <= start_index:
        raise ValueError(f"v2.7 score audit requires at least {start_index + 1} draws")
    if baseline_samples <= 0:
        raise ValueError("baseline_samples must be positive")

    all_combination_baseline = build_all_combination_baseline()
    percentiles = {name: [] for name in MODEL_NAMES}
    conditioned_percentiles = {name: [] for name in MODEL_NAMES}
    actual_types: list[str] = []
    same_type_counts: list[int] = []
    rows: list[dict] = []

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
        rng = random.Random(round_no * 10007 + baseline_samples * 97 + 27)
        sampled: list[dict] = []
        seen: set[tuple[int, ...]] = set()
        while len(sampled) < baseline_samples:
            candidate = random_combination(rng)
            key = tuple(candidate)
            if key in seen:
                continue
            seen.add(key)
            sampled.append(
                score_v27_audit_candidate(candidate, profile, mixed_profile, BASE_WEIGHTS)
            )

        actual_type = actual["pattern_type"]
        same_type = [item for item in sampled if item["pattern_type"] == actual_type]
        actual_types.append(actual_type)
        same_type_counts.append(len(same_type))
        row_model_values = {}
        for name in MODEL_NAMES:
            all_scores = [float(item["scores"][name]) for item in sampled]
            same_scores = [float(item["scores"][name]) for item in same_type]
            percentile = tie_safe_percentile(actual["scores"][name], all_scores)
            conditioned = tie_safe_percentile(actual["scores"][name], same_scores)
            percentiles[name].append(percentile)
            conditioned_percentiles[name].append(conditioned)
            if include_rows:
                row_model_values[name] = {
                    "score": float(actual["scores"][name]),
                    "percentile": percentile,
                    "type_conditioned_percentile": conditioned,
                }

        if include_rows:
            rows.append({
                "target_round": round_no,
                "history_draws": idx,
                "actual_pattern_type": actual_type,
                "same_type_baseline_count": len(same_type),
                "components": actual["components"],
                "models": row_model_values,
            })
        if progress_every and ((idx - start_index + 1) % int(progress_every) == 0):
            print(
                f"v2.7 audit progress: {idx - start_index + 1}/{len(df) - start_index} "
                f"targets (through draw {round_no})"
            )

    summaries = {
        name: _model_summary(
            percentiles[name], conditioned_percentiles[name], actual_types, same_type_counts
        )
        for name in MODEL_NAMES
    }
    comparisons = {}
    for name in MODEL_NAMES:
        comparisons[name] = {
            reference: _paired_summary(
                percentiles[name],
                percentiles[reference],
                name,
                reference,
                bootstrap_reps,
            )
            for reference in ("base_only", "v26_full")
            if name != reference
        }

    payload = {
        "version": "v27_score_audit_v1",
        "benchmark_commit": "ecbbd02235b9ed8f6940ae47aa46e7d4c0e53499",
        "strict_walk_forward": True,
        "start_index": start_index,
        "total_tests": len(df) - start_index,
        "baseline_samples_per_target": baseline_samples,
        "bootstrap_reps": int(bootstrap_reps),
        "models": summaries,
        "paired_comparisons": comparisons,
        "notes": {
            "screening_baseline_samples": SCREENING_BASELINE_SAMPLES,
            "confirmation_baseline_samples": CONFIRMATION_BASELINE_SAMPLES,
            "percentile_definition": "mid-rank actual-vs-random sampled candidate percentile",
            "type_conditioned_definition": "actual score versus sampled candidates of the same pattern type",
        },
    }
    if include_rows:
        payload["rows"] = rows
    return payload
