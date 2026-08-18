from __future__ import annotations

import random
from itertools import combinations

import numpy as np
import pandas as pd

from .candidates import random_combination
from .config import BACKTEST_START_INDEX, NUMBER_COLUMNS, ROUND_COLUMN
from .evidence_scoring import NUMBER_PRODUCTION_SPEC, PAIR_PRODUCTION_SPEC
from .loader import row_numbers
from .number_evidence import UNIFORM_NUMBER_PROBABILITY
from .pair_evidence import PAIR_COUNT, PAIRS_PER_DRAW, PAIR_TO_INDEX, UNIFORM_PAIR_PROBABILITY
from .v27_validation import DEFAULT_BOOTSTRAP_REPS, _block_bootstrap_mean_ci, tie_safe_percentile

RANKING_SCREEN_SAMPLES = 500
RANKING_CONFIRMATION_SAMPLES = 2_000
TRACKS = ("number", "pair", "equal_family_fusion")


def _sample_fair_candidates(
    rng: random.Random,
    count: int,
    excluded: tuple[int, ...],
) -> list[tuple[int, ...]]:
    if int(count) <= 0:
        raise ValueError("ranking baseline sample count must be positive")
    seen = {tuple(excluded)}
    result: list[tuple[int, ...]] = []
    max_attempts = max(1_000, int(count) * 20)
    attempts = 0
    while len(result) < int(count) and attempts < max_attempts:
        attempts += 1
        candidate = tuple(random_combination(rng))
        if candidate in seen:
            continue
        seen.add(candidate)
        result.append(candidate)
    if len(result) != int(count):
        raise RuntimeError(
            f"could not sample {count} unique fair combinations within {max_attempts} attempts"
        )
    return result


def _number_log_lifts_from_counts(number_hits: np.ndarray, draws: int) -> np.ndarray:
    alpha = float(NUMBER_PRODUCTION_SPEC["prior_strength"])
    denominator = float(draws) + alpha
    posterior = (
        number_hits[1:] + alpha * UNIFORM_NUMBER_PROBABILITY
    ) / denominator
    lifts = np.zeros(46, dtype=float)
    lifts[1:] = np.log(np.maximum(posterior, 1e-15) / UNIFORM_NUMBER_PROBABILITY)
    return lifts


def _pair_log_lifts_from_counts(pair_hits: np.ndarray, draws: int) -> np.ndarray:
    alpha = float(PAIR_PRODUCTION_SPEC["prior_strength"])
    denominator = float(draws) + alpha
    posterior = (
        pair_hits + alpha * UNIFORM_PAIR_PROBABILITY
    ) / denominator
    return np.log(np.maximum(posterior, 1e-15) / UNIFORM_PAIR_PROBABILITY)


def _pair_indices(numbers: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(PAIR_TO_INDEX[(left, right)] for left, right in combinations(numbers, 2))


def _raw_scores(
    numbers: tuple[int, ...],
    number_lifts: np.ndarray,
    pair_lifts: np.ndarray,
) -> dict[str, float]:
    number_score = float(np.mean([number_lifts[number] for number in numbers]))
    indices = _pair_indices(numbers)
    pair_score = float(np.mean([pair_lifts[index] for index in indices]))
    return {
        "number": number_score,
        "pair": pair_score,
        # Each evidence family contributes equally after averaging within family.
        # This is predeclared and is not fit to the current target.
        "equal_family_fusion": 0.5 * number_score + 0.5 * pair_score,
    }


def _update_hits(
    numbers: tuple[int, ...],
    number_hits: np.ndarray,
    pair_hits: np.ndarray,
) -> None:
    for number in numbers:
        number_hits[number] += 1.0
    for index in _pair_indices(numbers):
        pair_hits[index] += 1.0


def _mean(values: list[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def ranking_reliability_from_summary(summary: dict) -> float:
    """Continuous rank-target reliability; diagnostic only until audit acceptance."""
    percentiles = np.asarray([
        float(summary["mean_percentile"]),
        float(summary["recent_300_mean_percentile"]),
        float(summary["recent_100_mean_percentile"]),
    ])
    excess = (percentiles - 50.0) / 50.0
    positive = np.maximum(excess, 0.0)
    positive_fraction = float(np.mean(excess > 0.0))
    return float(np.mean(positive) * positive_fraction)


def _summary(values: list[float], bootstrap_reps: int, seed: int) -> dict:
    recent_300 = values[-min(300, len(values)):]
    recent_100 = values[-min(100, len(values)):]
    centered = [float(value) - 50.0 for value in values]
    low, high = _block_bootstrap_mean_ci(
        centered,
        reps=int(bootstrap_reps),
        seed=int(seed),
    )
    result = {
        "tests": int(len(values)),
        "mean_percentile": _mean(values),
        "recent_300_mean_percentile": _mean(recent_300),
        "recent_100_mean_percentile": _mean(recent_100),
        "above_random_median_ratio": (
            float(np.mean([value > 50.0 for value in values])) if values else float("nan")
        ),
        "percentile_minus_50_block_bootstrap_95_ci": [float(low), float(high)],
    }
    result["screening_candidate"] = bool(
        result["tests"] > 0
        and float(result["mean_percentile"]) > 50.0
        and float(low) > 0.0
        and float(result["recent_300_mean_percentile"]) >= 50.0
        and float(result["recent_100_mean_percentile"]) >= 50.0
    )
    result["continuous_rank_reliability"] = ranking_reliability_from_summary(result)
    return result


def run_v271_ranking_evidence_audit(
    df: pd.DataFrame,
    start_index: int = BACKTEST_START_INDEX,
    baseline_samples: int = RANKING_SCREEN_SAMPLES,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    progress_every: int = 0,
    include_rows: bool = False,
) -> dict:
    """Strict walk-forward actual-vs-fair-combination ranking audit.

    For target index t, number and pair posteriors use only rows < t. The actual
    winning six-number combination is then ranked against deterministic unique fair
    combinations sampled uniformly from the full 6/45 candidate universe. Pattern
    type is not conditioned on or used by this audit.
    """
    start_index = int(start_index)
    baseline_samples = int(baseline_samples)
    if len(df) <= start_index:
        raise ValueError(
            f"v2.7.1 ranking audit requires at least {start_index + 1} draws"
        )
    if baseline_samples <= 0:
        raise ValueError("baseline_samples must be positive")

    number_hits = np.zeros(46, dtype=float)
    pair_hits = np.zeros(PAIR_COUNT, dtype=float)
    for idx in range(start_index):
        _update_hits(tuple(row_numbers(df.iloc[idx])), number_hits, pair_hits)

    percentiles: dict[str, list[float]] = {name: [] for name in TRACKS}
    rows: list[dict] = []
    total_targets = len(df) - start_index

    for idx in range(start_index, len(df)):
        actual = tuple(row_numbers(df.iloc[idx]))
        round_no = int(df.iloc[idx][ROUND_COLUMN])
        number_lifts = _number_log_lifts_from_counts(number_hits, idx)
        pair_lifts = _pair_log_lifts_from_counts(pair_hits, idx)
        actual_scores = _raw_scores(actual, number_lifts, pair_lifts)

        rng = random.Random(round_no * 10067 + baseline_samples * 131 + 27101)
        baseline_candidates = _sample_fair_candidates(
            rng,
            baseline_samples,
            actual,
        )
        baseline_scores = {name: [] for name in TRACKS}
        for candidate in baseline_candidates:
            scored = _raw_scores(candidate, number_lifts, pair_lifts)
            for name in TRACKS:
                baseline_scores[name].append(float(scored[name]))

        row_result = {
            "target_index": int(idx),
            "target_round": round_no,
            "history_draws": int(idx),
            "actual_numbers": list(actual),
            "tracks": {},
        }
        for name in TRACKS:
            percentile = tie_safe_percentile(
                actual_scores[name], baseline_scores[name]
            )
            percentiles[name].append(float(percentile))
            row_result["tracks"][name] = {
                "actual_raw_score": float(actual_scores[name]),
                "percentile": float(percentile),
            }

        if include_rows:
            rows.append(row_result)

        # Target t becomes history only after it has been scored.
        _update_hits(actual, number_hits, pair_hits)

        completed = idx - start_index + 1
        if progress_every and completed % int(progress_every) == 0:
            print(
                f"v2.7.1 ranking evidence progress: {completed}/{total_targets} "
                f"targets (through draw {round_no})"
            )

    summaries = {
        name: _summary(
            percentiles[name],
            int(bootstrap_reps),
            seed=71_000 + sum((i + 1) * ord(ch) for i, ch in enumerate(name)),
        )
        for name in TRACKS
    }
    payload = {
        "version": "v271_combination_ranking_evidence_audit_v1",
        "strict_walk_forward": True,
        "target": "actual_winning_combination_rank_vs_fair_random_combinations",
        "start_index": start_index,
        "total_tests": int(total_targets),
        "baseline_samples_per_target": baseline_samples,
        "bootstrap_reps": int(bootstrap_reps),
        "number_spec": dict(NUMBER_PRODUCTION_SPEC),
        "pair_spec": dict(PAIR_PRODUCTION_SPEC),
        "tracks": summaries,
        "screening_candidates": [
            name for name, summary in summaries.items() if summary["screening_candidate"]
        ],
        "notes": {
            "pattern_type": "not used; all valid combinations share one baseline",
            "fusion": "equal weight across number and pair families after within-family averaging",
            "production_change": "none; this audit must be reviewed before replacing Brier reliability",
            "confirmation_samples": RANKING_CONFIRMATION_SAMPLES,
        },
    }
    if include_rows:
        payload["rows"] = rows
    return payload
