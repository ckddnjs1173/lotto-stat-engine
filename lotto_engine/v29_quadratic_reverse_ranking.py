from __future__ import annotations

import random
from itertools import combinations

import numpy as np
import pandas as pd

from .config import BACKTEST_START_INDEX, ROUND_COLUMN
from .loader import row_numbers
from .v27_validation import (
    DEFAULT_BOOTSTRAP_REPS,
    _block_bootstrap_mean_ci,
    tie_safe_percentile,
)
from .v28_reverse_ranking import (
    CONFIRMATION_BASELINE_SAMPLES,
    FEATURE_NAMES as BASE_FEATURE_NAMES,
    MIN_META_TRAIN_TARGETS,
    RIDGE_LAMBDA,
    RMS_EPSILON,
    SCREENING_BASELINE_SAMPLES,
    TRAIN_NEGATIVES_PER_TARGET,
    _advance_history,
    _empty_history_state,
    _feature_context,
    _sample_fair_candidates,
    candidate_feature_vector,
)

MODEL_VERSION = "v29_quadratic_reverse_nested_ridge_v1"


def _quadratic_feature_names() -> tuple[str, ...]:
    names: list[str] = []
    names.extend(f"linear:{name}" for name in BASE_FEATURE_NAMES)
    names.extend(f"square:{name}" for name in BASE_FEATURE_NAMES)
    for left, right in combinations(BASE_FEATURE_NAMES, 2):
        names.append(f"interaction:{left}*{right}")
    return tuple(names)


FEATURE_NAMES = _quadratic_feature_names()

if len(FEATURE_NAMES) != 65:  # pragma: no cover - import-time invariant
    raise RuntimeError("v2.9 quadratic basis must contain exactly 65 terms")


def quadratic_basis_from_base(base_vector: np.ndarray) -> np.ndarray:
    """Expand the frozen ten v2.8 inputs into the complete degree-2 basis."""
    base = np.asarray(base_vector, dtype=float)
    if base.shape != (len(BASE_FEATURE_NAMES),):
        raise ValueError(
            f"v2.9 base vector must have shape ({len(BASE_FEATURE_NAMES)},)"
        )

    values: list[float] = [float(value) for value in base]
    values.extend(float(value * value) for value in base)
    for left, right in combinations(range(len(base)), 2):
        values.append(float(base[left] * base[right]))
    result = np.asarray(values, dtype=float)
    if result.shape != (len(FEATURE_NAMES),):
        raise RuntimeError("unexpected v2.9 quadratic basis width")
    return result


def quadratic_candidate_vector(numbers, context: dict) -> np.ndarray:
    """Compute the unchanged v2.8 base inputs and their frozen quadratic basis."""
    base = candidate_feature_vector(numbers, context)
    return quadratic_basis_from_base(base)


def _empty_training_stats() -> dict:
    width = len(FEATURE_NAMES)
    return {
        "sum_outer": np.zeros((width, width), dtype=float),
        "sum_diff": np.zeros(width, dtype=float),
        "pair_rows": 0,
        "solved_targets": 0,
    }


def _add_solved_target(
    stats: dict,
    actual_vector: np.ndarray,
    negative_vectors: list[np.ndarray],
) -> None:
    for negative in negative_vectors:
        diff = np.asarray(actual_vector - negative, dtype=float)
        stats["sum_outer"] += np.outer(diff, diff)
        stats["sum_diff"] += diff
        stats["pair_rows"] += 1
    stats["solved_targets"] += 1


def fit_quadratic_pairwise_ridge(
    stats: dict,
    ridge_lambda: float = RIDGE_LAMBDA,
) -> dict:
    """Fit the same RMS-scaled pairwise ridge objective as v2.8 on 65 terms."""
    rows = int(stats["pair_rows"])
    if rows <= 0:
        raise ValueError("v2.9 ranker requires prior solved training rows")
    ridge_lambda = float(ridge_lambda)
    if ridge_lambda <= 0.0:
        raise ValueError("ridge_lambda must be positive")

    second_moment = np.asarray(stats["sum_outer"], dtype=float) / rows
    mean_diff = np.asarray(stats["sum_diff"], dtype=float) / rows
    rms = np.sqrt(np.maximum(np.diag(second_moment), 0.0) + RMS_EPSILON)
    scaled_second = second_moment / np.outer(rms, rms)
    scaled_mean = mean_diff / rms
    system = scaled_second + ridge_lambda * np.eye(len(FEATURE_NAMES), dtype=float)
    try:
        scaled_weights = np.linalg.solve(system, scaled_mean)
    except np.linalg.LinAlgError:
        scaled_weights = np.linalg.pinv(system) @ scaled_mean
    effective_weights = scaled_weights / rms

    return {
        "ridge_lambda": ridge_lambda,
        "rms": rms,
        "scaled_weights": scaled_weights,
        "effective_weights": effective_weights,
        "pair_rows": rows,
        "solved_targets": int(stats["solved_targets"]),
    }


def score_quadratic_vector(vector: np.ndarray, model: dict) -> float:
    return float(np.dot(model["effective_weights"], np.asarray(vector, dtype=float)))


def _mean(values: list[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _summary(values: list[float], bootstrap_reps: int) -> dict:
    recent300 = values[-min(300, len(values)):]
    recent100 = values[-min(100, len(values)):]
    centered = [float(value) - 50.0 for value in values]
    low, high = _block_bootstrap_mean_ci(
        centered,
        reps=int(bootstrap_reps),
        seed=82_901,
    )
    result = {
        "tests": int(len(values)),
        "mean_percentile": _mean(values),
        "recent_300_mean_percentile": _mean(recent300),
        "recent_100_mean_percentile": _mean(recent100),
        "above_random_median_ratio": (
            float(np.mean([value > 50.0 for value in values]))
            if values else float("nan")
        ),
        "percentile_minus_50_block_bootstrap_95_ci": [float(low), float(high)],
    }
    gate_pass = bool(
        result["tests"] > 0
        and float(result["mean_percentile"]) > 50.0
        and float(low) > 0.0
        and float(result["recent_300_mean_percentile"]) >= 50.0
        and float(result["recent_100_mean_percentile"]) >= 50.0
    )
    result["historical_gate_pass"] = gate_pass
    result["exploratory_screening_candidate"] = gate_pass
    return result


def _model_diagnostics(model: dict) -> dict:
    return {
        "ridge_lambda": float(model["ridge_lambda"]),
        "pair_rows": int(model["pair_rows"]),
        "solved_targets": int(model["solved_targets"]),
        "basis_width": len(FEATURE_NAMES),
        "features": {
            name: {
                "effective_weight": float(model["effective_weights"][index]),
                "scaled_weight": float(model["scaled_weights"][index]),
                "rms": float(model["rms"][index]),
            }
            for index, name in enumerate(FEATURE_NAMES)
        },
    }


def run_v29_quadratic_reverse_ranking_audit(
    df: pd.DataFrame,
    start_index: int = BACKTEST_START_INDEX,
    min_meta_train_targets: int = MIN_META_TRAIN_TARGETS,
    train_negatives_per_target: int = TRAIN_NEGATIVES_PER_TARGET,
    baseline_samples: int = SCREENING_BASELINE_SAMPLES,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    ridge_lambda: float = RIDGE_LAMBDA,
    progress_every: int = 0,
    include_rows: bool = False,
) -> dict:
    """Strict nested walk-forward screen for the frozen quadratic v2.9 class."""
    start_index = int(start_index)
    min_meta_train_targets = int(min_meta_train_targets)
    train_negatives_per_target = int(train_negatives_per_target)
    baseline_samples = int(baseline_samples)
    if len(df) <= start_index:
        raise ValueError("v2.9 reverse ranking requires targets after start_index")
    if min_meta_train_targets <= 0:
        raise ValueError("min_meta_train_targets must be positive")
    if train_negatives_per_target <= 0 or baseline_samples <= 0:
        raise ValueError("training and evaluation candidate counts must be positive")

    state = _empty_history_state()
    for idx in range(start_index):
        _advance_history(state, tuple(row_numbers(df.iloc[idx])))

    stats = _empty_training_stats()
    percentiles: list[float] = []
    rows: list[dict] = []
    total_solved_targets = len(df) - start_index
    expected_outer_tests = max(0, total_solved_targets - min_meta_train_targets)
    outer_completed = 0

    for idx in range(start_index, len(df)):
        actual = tuple(row_numbers(df.iloc[idx]))
        round_no = int(df.iloc[idx][ROUND_COLUMN])
        context = _feature_context(state)
        actual_vector = quadratic_candidate_vector(actual, context)

        if int(stats["solved_targets"]) >= min_meta_train_targets:
            model = fit_quadratic_pairwise_ridge(stats, ridge_lambda=ridge_lambda)
            actual_score = score_quadratic_vector(actual_vector, model)

            # Deliberately reuse the v2.8 evaluation seed path so both model
            # classes see identical fair comparison candidates.
            eval_rng = random.Random(
                round_no * 100_003
                + baseline_samples * 1_009
                + min_meta_train_targets * 37
                + 28_001
            )
            fair_eval = _sample_fair_candidates(eval_rng, baseline_samples, actual)
            baseline_scores = [
                score_quadratic_vector(
                    quadratic_candidate_vector(candidate, context),
                    model,
                )
                for candidate in fair_eval
            ]
            percentile = tie_safe_percentile(actual_score, baseline_scores)
            percentiles.append(float(percentile))
            outer_completed += 1

            if include_rows:
                rows.append({
                    "target_index": int(idx),
                    "target_round": round_no,
                    "history_draws": int(state["history_draws"]),
                    "prior_solved_targets": int(stats["solved_targets"]),
                    "prior_pair_rows": int(stats["pair_rows"]),
                    "actual_numbers": list(actual),
                    "actual_score": float(actual_score),
                    "percentile": float(percentile),
                    "model": _model_diagnostics(model),
                })

            if progress_every and outer_completed % int(progress_every) == 0:
                print(
                    f"v2.9 quadratic reverse ranking progress: "
                    f"{outer_completed}/{expected_outer_tests} outer targets "
                    f"(through draw {round_no})"
                )

        # Reuse the v2.8 training seed path as well. The answer is revealed only
        # after the current target's outer score is fixed.
        train_rng = random.Random(
            round_no * 200_003
            + train_negatives_per_target * 2_009
            + 28_101
        )
        fair_train = _sample_fair_candidates(
            train_rng,
            train_negatives_per_target,
            actual,
        )
        negative_vectors = [
            quadratic_candidate_vector(candidate, context)
            for candidate in fair_train
        ]
        _add_solved_target(stats, actual_vector, negative_vectors)
        _advance_history(state, actual)

    summary = _summary(percentiles, int(bootstrap_reps))
    latest_model = fit_quadratic_pairwise_ridge(stats, ridge_lambda=ridge_lambda)
    payload = {
        "version": MODEL_VERSION,
        "strict_nested_walk_forward": True,
        "research_status": "exploratory_screen_after_v28_failure",
        "target": "actual_winning_combination_rank_vs_fair_random_combinations",
        "start_index": start_index,
        "min_meta_train_targets": min_meta_train_targets,
        "train_negatives_per_target": train_negatives_per_target,
        "baseline_samples_per_outer_target": baseline_samples,
        "bootstrap_reps": int(bootstrap_reps),
        "ridge_lambda": float(ridge_lambda),
        "base_feature_names": list(BASE_FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "basis_width": len(FEATURE_NAMES),
        "outer_summary": summary,
        "latest_model_for_next_draw": _model_diagnostics(latest_model),
        "notes": {
            "pattern_type": "not used",
            "target_reveal": (
                "target contributes training rows only after its outer score is recorded"
            ),
            "same_sampling_as_v28": True,
            "screen_confirmation_samples": CONFIRMATION_BASELINE_SAMPLES,
            "production_change": "none",
            "research_budget": (
                "final new model class on current 1235-draw dataset; failure freezes "
                "further adaptive model-class search on this dataset"
            ),
        },
    }
    if include_rows:
        payload["rows"] = rows
    return payload
