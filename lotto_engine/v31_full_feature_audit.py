from __future__ import annotations

import random
from collections import defaultdict

import numpy as np
import pandas as pd

from .config import ROUND_COLUMN
from .loader import row_numbers
from .v31_core import (
    FEATURE_NAMES,
    _add_target,
    _advance_history,
    _empty_history_state,
    _empty_stats,
    _feature_context,
    _sample_fair_candidates,
    _sample_reference,
    _training_rng,
    candidate_vector,
    fit_subset_pairwise_ridge,
)
from .v31_model_spec import FULL11_BASELINE_SPEC, V31ModelSpec

AUDIT_VERSION = "v31_full_feature_group_audit_v1"
DEFAULT_BOOTSTRAP_REPS = 2_000
DEFAULT_BOOTSTRAP_BLOCK = 20

FEATURE_GROUPS = {
    "long_run_number_pair": (
        "number_full_log_lift",
        "pair_full_log_lift",
    ),
    "recency_all": (
        "number_recent20_excess",
        "number_recent100_excess",
        "pair_recent100_excess",
    ),
    "number_marginal_all": (
        "number_full_log_lift",
        "number_recent20_excess",
        "number_recent100_excess",
    ),
    "pair_all": (
        "pair_full_log_lift",
        "pair_recent100_excess",
    ),
    "structure_all": (
        "previous_draw_overlap",
        "sum_signed_center_138",
        "high_minus_low_zone_count",
        "odd_count_signed_center_3",
        "number_range",
        "consecutive_pairs",
    ),
}


def tie_safe_percentile(actual_score: float, baseline_scores: np.ndarray) -> float:
    baseline = np.asarray(baseline_scores, dtype=float)
    if len(baseline) == 0:
        return float("nan")
    actual = float(actual_score)
    less = int(np.sum(baseline < actual))
    equal = int(np.sum(baseline == actual))
    return 100.0 * (less + 0.5 * equal) / len(baseline)


def _block_bootstrap_mean_ci(
    values: list[float],
    reps: int = DEFAULT_BOOTSTRAP_REPS,
    block_size: int = DEFAULT_BOOTSTRAP_BLOCK,
    seed: int = 31_811,
) -> tuple[float, float]:
    data = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    n = len(data)
    if n == 0:
        return float("nan"), float("nan")
    if n == 1 or int(reps) <= 0:
        value = float(np.mean(data))
        return value, value
    block = max(1, min(int(block_size), n))
    blocks_needed = (n + block - 1) // block
    offsets = np.arange(block, dtype=int)
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(reps), dtype=float)
    for rep in range(int(reps)):
        starts = rng.integers(0, n, size=blocks_needed)
        indices = np.concatenate(
            [(int(start) + offsets) % n for start in starts]
        )[:n]
        means[rep] = float(np.mean(data[indices]))
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def _summary(values: list[float], bootstrap_reps: int, seed: int) -> dict:
    recent300 = values[-min(300, len(values)):]
    recent100 = values[-min(100, len(values)):]
    centered = [float(value) - 50.0 for value in values]
    low, high = _block_bootstrap_mean_ci(
        centered,
        reps=int(bootstrap_reps),
        seed=int(seed),
    )
    return {
        "tests": len(values),
        "mean_percentile": float(np.mean(values)) if values else float("nan"),
        "recent_300_mean_percentile": float(np.mean(recent300)) if recent300 else float("nan"),
        "recent_100_mean_percentile": float(np.mean(recent100)) if recent100 else float("nan"),
        "percentile_minus_50_block_bootstrap_95_ci": [float(low), float(high)],
    }


def _paired_delta(
    full: list[float],
    without: list[float],
    bootstrap_reps: int,
    seed: int,
) -> dict:
    delta = [float(left) - float(right) for left, right in zip(full, without)]
    recent300 = delta[-min(300, len(delta)):]
    recent100 = delta[-min(100, len(delta)):]
    low, high = _block_bootstrap_mean_ci(
        delta,
        reps=int(bootstrap_reps),
        seed=int(seed),
    )
    mean = float(np.mean(delta)) if delta else float("nan")
    r300 = float(np.mean(recent300)) if recent300 else float("nan")
    r100 = float(np.mean(recent100)) if recent100 else float("nan")
    if delta and low > 0.0 and r300 >= 0.0 and r100 >= 0.0:
        direction = "retained_block_helped_full_model"
    elif delta and high < 0.0 and r300 <= 0.0 and r100 <= 0.0:
        direction = "retained_block_hurt_full_model"
    else:
        direction = "inconclusive"
    return {
        "full_minus_without_mean_percentile": mean,
        "recent_300_delta": r300,
        "recent_100_delta": r100,
        "block_bootstrap_95_ci": [float(low), float(high)],
        "direction": direction,
    }


def _active_indices_without(removed_features) -> tuple[int, ...]:
    removed = {FEATURE_NAMES.index(name) for name in removed_features}
    return tuple(index for index in range(len(FEATURE_NAMES)) if index not in removed)


def scenario_removals() -> dict[str, tuple[str, ...]]:
    result = {"full": ()}
    for feature in FEATURE_NAMES:
        result[f"without_feature:{feature}"] = (feature,)
    for group, features in FEATURE_GROUPS.items():
        result[f"without_group:{group}"] = tuple(features)
    return result


def build_scenario_weight_matrix(stats: dict, spec: V31ModelSpec) -> dict:
    removals = scenario_removals()
    names = []
    rows = []
    models = {}
    for name, removed in removals.items():
        model = fit_subset_pairwise_ridge(
            stats,
            _active_indices_without(removed),
            ridge_lambda=float(spec.ridge_lambda),
        )
        names.append(name)
        rows.append(np.asarray(model["effective_weights"], dtype=float))
        models[name] = model
    return {
        "scenario_names": tuple(names),
        "scenario_weight_matrix": np.vstack(rows),
        "models": models,
        "removals": removals,
    }


def _evaluation_rng(
    round_no: int,
    baseline_samples: int,
    minimum_meta_targets: int,
    spec: V31ModelSpec,
) -> random.Random:
    return random.Random(
        int(round_no) * 100_003
        + int(baseline_samples) * 1_009
        + int(minimum_meta_targets) * 37
        + int(spec.evaluation_seed_offset)
    )


def _weight_stability_summary(weight_rows: list[np.ndarray]) -> dict:
    if not weight_rows:
        return {}
    matrix = np.vstack(weight_rows)
    result = {}
    for index, feature in enumerate(FEATURE_NAMES):
        values = matrix[:, index]
        recent100 = values[-min(100, len(values)):]
        result[feature] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "recent_100_mean": float(np.mean(recent100)),
            "positive_fraction": float(np.mean(values > 0.0)),
            "negative_fraction": float(np.mean(values < 0.0)),
            "zero_fraction": float(np.mean(values == 0.0)),
            "sign_flip_count": int(
                np.sum(np.sign(values[1:]) != np.sign(values[:-1]))
            ),
        }
    return result


def run_full_feature_group_audit(
    df: pd.DataFrame,
    spec: V31ModelSpec = FULL11_BASELINE_SPEC,
    baseline_samples: int = 500,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    progress_every: int = 0,
) -> dict:
    """Strict walk-forward FULL11 vs all single-feature and predefined group removals."""
    start_index = int(spec.history_start_index)
    minimum_meta_targets = int(spec.minimum_meta_train_targets)
    if len(df) <= start_index + minimum_meta_targets:
        raise ValueError("not enough completed draws for full feature audit")
    baseline_samples = int(baseline_samples)
    if baseline_samples <= 0:
        raise ValueError("baseline_samples must be positive")
    progress_every = int(progress_every)
    if progress_every < 0:
        raise ValueError("progress_every must be non-negative")

    state = _empty_history_state()
    for idx in range(start_index):
        _advance_history(state, tuple(row_numbers(df.iloc[idx])), spec)

    stats = _empty_stats()
    removals = scenario_removals()
    percentiles = {name: [] for name in removals}
    full_weight_rows: list[np.ndarray] = []
    ridge_system_conditions: list[float] = []
    outer = 0
    expected = len(df) - start_index - minimum_meta_targets

    for idx in range(start_index, len(df)):
        actual = tuple(row_numbers(df.iloc[idx]))
        round_no = int(df.iloc[idx][ROUND_COLUMN])
        context = _feature_context(state, spec)
        reference = _sample_reference(
            context,
            round_no,
            int(spec.fair_reference_samples_per_target),
            spec,
        )
        actual_vector = candidate_vector(actual, context, reference)

        if int(stats["solved_targets"]) >= minimum_meta_targets:
            bundle = build_scenario_weight_matrix(stats, spec)
            weight_matrix = np.asarray(bundle["scenario_weight_matrix"], dtype=float)
            actual_scores = weight_matrix @ actual_vector

            fair_candidates = _sample_fair_candidates(
                _evaluation_rng(
                    round_no,
                    baseline_samples,
                    minimum_meta_targets,
                    spec,
                ),
                baseline_samples,
                actual,
            )
            baseline_matrix = np.empty((baseline_samples, len(bundle["scenario_names"])), dtype=float)
            for row_index, candidate in enumerate(fair_candidates):
                vector = candidate_vector(candidate, context, reference)
                baseline_matrix[row_index] = weight_matrix @ vector

            for scenario_index, name in enumerate(bundle["scenario_names"]):
                percentiles[name].append(
                    tie_safe_percentile(
                        float(actual_scores[scenario_index]),
                        baseline_matrix[:, scenario_index],
                    )
                )

            full_model = bundle["models"]["full"]
            full_weight_rows.append(
                np.asarray(full_model["effective_weights"], dtype=float).copy()
            )
            # This diagnostic is added in v31_core and tracks the unregularized
            # scaled second moment. The ridge system itself remains solved with lambda.
            condition = full_model.get("scaled_second_moment_condition_number")
            if condition is not None and np.isfinite(condition):
                ridge_system_conditions.append(float(condition))

            outer += 1
            if progress_every and outer % progress_every == 0:
                print(
                    f"v3.1 full feature audit: {outer}/{expected} targets through draw {round_no}"
                )

        negatives = _sample_fair_candidates(
            _training_rng(round_no, spec),
            int(spec.train_negatives_per_target),
            actual,
        )
        negative_vectors = [
            candidate_vector(candidate, context, reference) for candidate in negatives
        ]
        _add_target(stats, actual_vector, negative_vectors)
        _advance_history(state, actual, spec)

    summaries = {
        name: _summary(values, int(bootstrap_reps), 31_820 + index)
        for index, (name, values) in enumerate(percentiles.items())
    }
    paired_features = {
        feature: _paired_delta(
            percentiles["full"],
            percentiles[f"without_feature:{feature}"],
            int(bootstrap_reps),
            31_900 + index,
        )
        for index, feature in enumerate(FEATURE_NAMES)
    }
    paired_groups = {
        group: _paired_delta(
            percentiles["full"],
            percentiles[f"without_group:{group}"],
            int(bootstrap_reps),
            32_000 + index,
        )
        for index, group in enumerate(FEATURE_GROUPS)
    }

    condition_summary = {
        "tests": len(ridge_system_conditions),
        "mean": float(np.mean(ridge_system_conditions)) if ridge_system_conditions else float("nan"),
        "median": float(np.median(ridge_system_conditions)) if ridge_system_conditions else float("nan"),
        "max": float(np.max(ridge_system_conditions)) if ridge_system_conditions else float("nan"),
    }

    return {
        "version": AUDIT_VERSION,
        "audit_role": "diagnostic_only_no_automatic_feature_promotion_or_deletion",
        "model_version_under_audit": spec.name,
        "strict_target_isolation": True,
        "post_selection_exploratory_validation": True,
        "outer_tests": len(percentiles["full"]),
        "baseline_samples_per_target": baseline_samples,
        "feature_groups": {key: list(value) for key, value in FEATURE_GROUPS.items()},
        "scenario_summaries": summaries,
        "paired_full_minus_without_feature": paired_features,
        "paired_full_minus_without_group": paired_groups,
        "full_model_weight_stability": _weight_stability_summary(full_weight_rows),
        "scaled_second_moment_condition_summary": condition_summary,
    }
