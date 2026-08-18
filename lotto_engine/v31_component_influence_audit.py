from __future__ import annotations

import heapq
import random
from collections import Counter

import numpy as np
import pandas as pd

from .candidates import generate_candidates, iter_all_combinations, make_seed
from .config import BACKTEST_START_INDEX, DEFAULT_CANDIDATE_COUNT, ROUND_COLUMN
from .loader import row_numbers
from .v27_validation import DEFAULT_BOOTSTRAP_REPS, _block_bootstrap_mean_ci, tie_safe_percentile
from .v28_reverse_ranking import (
    MIN_META_TRAIN_TARGETS,
    RIDGE_LAMBDA,
    TRAIN_NEGATIVES_PER_TARGET,
    _advance_history,
    _empty_history_state,
    _feature_context,
    _sample_fair_candidates,
)
from .v31_final_directional_recommendation import (
    FEATURE_NAMES,
    REFERENCE_SAMPLES_PER_TARGET,
    _add_target,
    _empty_stats,
    _sample_reference,
    _tie_break,
    candidate_vector,
    fit_additive_pairwise_ridge,
)

AUDIT_VERSION = "v31_component_influence_audit_v1"
AUDIT_FEATURES = (
    "previous_draw_overlap",
    "number_full_log_lift",
    "consecutive_pairs",
    "number_range",
)
DEFAULT_LATEST_POOL_SIZE = 1_000
FEATURE_DESCRIPTIONS = {
    "previous_draw_overlap": "previous-draw repeated-number count",
    "number_full_log_lift": "mean long-run Bayesian number log-lift",
    "consecutive_pairs": "adjacent consecutive-number pair count",
    "number_range": "maximum minus minimum number",
}


def _feature_index(name: str) -> int:
    if name not in FEATURE_NAMES:
        raise ValueError(f"unknown feature: {name}")
    return FEATURE_NAMES.index(name)


def fit_subset_pairwise_ridge(stats: dict, active_indices, ridge_lambda: float = RIDGE_LAMBDA) -> dict:
    """Fit the same ridge objective after removing selected feature columns."""
    rows = int(stats["pair_rows"])
    if rows <= 0:
        raise ValueError("training rows required")
    active = tuple(int(index) for index in active_indices)
    if not active or len(set(active)) != len(active):
        raise ValueError("active feature indices must be non-empty and unique")
    if min(active) < 0 or max(active) >= len(FEATURE_NAMES):
        raise ValueError("active feature index out of range")

    second = np.asarray(stats["sum_outer"], dtype=float) / rows
    mean_diff = np.asarray(stats["sum_diff"], dtype=float) / rows
    second = second[np.ix_(active, active)]
    mean_diff = mean_diff[list(active)]
    rms = np.sqrt(np.maximum(np.diag(second), 0.0) + 1e-12)
    scaled_second = second / np.outer(rms, rms)
    scaled_mean = mean_diff / rms
    system = scaled_second + float(ridge_lambda) * np.eye(len(active), dtype=float)
    try:
        scaled_weights = np.linalg.solve(system, scaled_mean)
    except np.linalg.LinAlgError:
        scaled_weights = np.linalg.pinv(system) @ scaled_mean
    return {
        "active_indices": active,
        "effective_weights": scaled_weights / rms,
        "scaled_weights": scaled_weights,
        "rms": rms,
        "ridge_lambda": float(ridge_lambda),
        "pair_rows": rows,
        "solved_targets": int(stats["solved_targets"]),
    }


def _model_bundle(stats: dict, ridge_lambda: float) -> dict:
    full = fit_additive_pairwise_ridge(stats, ridge_lambda=ridge_lambda)
    full_weights = np.asarray(full["effective_weights"], dtype=float)
    names = ["full"]
    matrices = [full_weights]
    reduced = {}

    for feature in AUDIT_FEATURES:
        removed = _feature_index(feature)
        active = tuple(index for index in range(len(FEATURE_NAMES)) if index != removed)
        model = fit_subset_pairwise_ridge(stats, active, ridge_lambda=ridge_lambda)
        reduced[feature] = model

        direct = full_weights.copy()
        direct[removed] = 0.0
        names.append(f"direct_without:{feature}")
        matrices.append(direct)

        retrained = np.zeros(len(FEATURE_NAMES), dtype=float)
        retrained[list(active)] = model["effective_weights"]
        names.append(f"retrained_without:{feature}")
        matrices.append(retrained)

    return {
        "full": full,
        "leave_one_out": reduced,
        "scenario_names": tuple(names),
        "scenario_weight_matrix": np.vstack(matrices),
    }


def score_scenarios_for_vector(vector: np.ndarray, models: dict) -> dict[str, float]:
    scores = np.asarray(models["scenario_weight_matrix"], dtype=float) @ np.asarray(vector, dtype=float)
    return {name: float(score) for name, score in zip(models["scenario_names"], scores)}


def _advance_training(stats: dict, state: dict, actual: tuple[int, ...], round_no: int,
                      train_negatives_per_target: int, reference_samples: int) -> None:
    context = _feature_context(state)
    reference = _sample_reference(context, round_no, int(reference_samples))
    actual_vector = candidate_vector(actual, context, reference)
    rng = random.Random(round_no * 200_003 + int(train_negatives_per_target) * 2_009 + 31_101)
    negatives = _sample_fair_candidates(rng, int(train_negatives_per_target), actual)
    negative_vectors = [candidate_vector(candidate, context, reference) for candidate in negatives]
    _add_target(stats, actual_vector, negative_vectors)
    _advance_history(state, actual)


def _latest_bundle(df: pd.DataFrame, start_index: int, train_negatives_per_target: int,
                   reference_samples: int, ridge_lambda: float) -> dict:
    if len(df) <= int(start_index):
        raise ValueError("not enough completed draws")
    state = _empty_history_state()
    for idx in range(int(start_index)):
        _advance_history(state, tuple(row_numbers(df.iloc[idx])))
    stats = _empty_stats()
    for idx in range(int(start_index), len(df)):
        actual = tuple(row_numbers(df.iloc[idx]))
        round_no = int(df.iloc[idx][ROUND_COLUMN])
        _advance_training(stats, state, actual, round_no, train_negatives_per_target, reference_samples)
    latest_round = int(df.iloc[-1][ROUND_COLUMN])
    context = _feature_context(state)
    reference = _sample_reference(context, latest_round + 1, int(reference_samples))
    return {
        "models": _model_bundle(stats, float(ridge_lambda)),
        "context": context,
        "reference": reference,
        "latest_round": latest_round,
        "target_round": latest_round + 1,
    }


def _summary(values: list[float], bootstrap_reps: int, seed: int) -> dict:
    recent300 = values[-min(300, len(values)):]
    recent100 = values[-min(100, len(values)):]
    centered = [float(value) - 50.0 for value in values]
    low, high = _block_bootstrap_mean_ci(centered, reps=int(bootstrap_reps), seed=int(seed))
    return {
        "tests": len(values),
        "mean_percentile": float(np.mean(values)) if values else float("nan"),
        "recent_300_mean_percentile": float(np.mean(recent300)) if recent300 else float("nan"),
        "recent_100_mean_percentile": float(np.mean(recent100)) if recent100 else float("nan"),
        "percentile_minus_50_block_bootstrap_95_ci": [float(low), float(high)],
    }


def _paired_delta(full: list[float], without: list[float], bootstrap_reps: int, seed: int) -> dict:
    delta = [float(left) - float(right) for left, right in zip(full, without)]
    recent300 = delta[-min(300, len(delta)):]
    recent100 = delta[-min(100, len(delta)):]
    low, high = _block_bootstrap_mean_ci(delta, reps=int(bootstrap_reps), seed=int(seed))
    mean_delta = float(np.mean(delta)) if delta else float("nan")
    r300 = float(np.mean(recent300)) if recent300 else float("nan")
    r100 = float(np.mean(recent100)) if recent100 else float("nan")
    if delta and low > 0.0 and r300 >= 0.0 and r100 >= 0.0:
        direction = "feature_helped_full_model"
    elif delta and high < 0.0 and r300 <= 0.0 and r100 <= 0.0:
        direction = "feature_hurt_full_model"
    else:
        direction = "inconclusive"
    return {
        "full_minus_without_mean_percentile": mean_delta,
        "recent_300_delta": r300,
        "recent_100_delta": r100,
        "block_bootstrap_95_ci": [float(low), float(high)],
        "direction": direction,
    }


def run_historical_component_audit(
    df: pd.DataFrame,
    start_index: int = BACKTEST_START_INDEX,
    min_meta_train_targets: int = MIN_META_TRAIN_TARGETS,
    train_negatives_per_target: int = TRAIN_NEGATIVES_PER_TARGET,
    baseline_samples: int = 500,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    ridge_lambda: float = RIDGE_LAMBDA,
    reference_samples: int = REFERENCE_SAMPLES_PER_TARGET,
    progress_every: int = 0,
) -> dict:
    """Strict walk-forward full-vs-retrained-without-feature comparison."""
    state = _empty_history_state()
    for idx in range(int(start_index)):
        _advance_history(state, tuple(row_numbers(df.iloc[idx])))
    stats = _empty_stats()
    scenario_names = ["full"] + [f"retrained_without:{feature}" for feature in AUDIT_FEATURES]
    percentiles = {name: [] for name in scenario_names}
    outer = 0
    expected = max(0, len(df) - int(start_index) - int(min_meta_train_targets))

    for idx in range(int(start_index), len(df)):
        actual = tuple(row_numbers(df.iloc[idx]))
        round_no = int(df.iloc[idx][ROUND_COLUMN])
        context = _feature_context(state)
        reference = _sample_reference(context, round_no, int(reference_samples))
        actual_vector = candidate_vector(actual, context, reference)

        if int(stats["solved_targets"]) >= int(min_meta_train_targets):
            models = _model_bundle(stats, float(ridge_lambda))
            actual_scores = score_scenarios_for_vector(actual_vector, models)
            rng = random.Random(
                round_no * 100_003 + int(baseline_samples) * 1_009
                + int(min_meta_train_targets) * 37 + 28_001
            )
            fair_eval = _sample_fair_candidates(rng, int(baseline_samples), actual)
            baselines = {name: [] for name in scenario_names}
            for candidate in fair_eval:
                scores = score_scenarios_for_vector(candidate_vector(candidate, context, reference), models)
                for name in scenario_names:
                    baselines[name].append(scores[name])
            for name in scenario_names:
                percentiles[name].append(tie_safe_percentile(actual_scores[name], baselines[name]))
            outer += 1
            if progress_every and outer % int(progress_every) == 0:
                print(f"v3.1 component historical audit: {outer}/{expected} targets through draw {round_no}")

        rng = random.Random(round_no * 200_003 + int(train_negatives_per_target) * 2_009 + 31_101)
        fair_train = _sample_fair_candidates(rng, int(train_negatives_per_target), actual)
        negative_vectors = [candidate_vector(candidate, context, reference) for candidate in fair_train]
        _add_target(stats, actual_vector, negative_vectors)
        _advance_history(state, actual)

    summaries = {
        name: _summary(values, int(bootstrap_reps), 31_500 + index)
        for index, (name, values) in enumerate(percentiles.items())
    }
    paired = {
        feature: _paired_delta(
            percentiles["full"],
            percentiles[f"retrained_without:{feature}"],
            int(bootstrap_reps),
            31_700 + index,
        )
        for index, feature in enumerate(AUDIT_FEATURES)
    }
    return {
        "strict_nested_walk_forward": True,
        "outer_tests": len(percentiles["full"]),
        "baseline_samples_per_target": int(baseline_samples),
        "summaries": summaries,
        "paired_feature_value": paired,
    }


def _push(heap, entry, keep: int) -> None:
    if len(heap) < keep:
        heapq.heappush(heap, entry)
    elif entry[:2] > heap[0][:2]:
        heapq.heapreplace(heap, entry)


def _pool_summary(ranked, previous_draw: frozenset[int], top_k: int = 10) -> dict:
    combos = [entry[2] for entry in ranked]
    size = len(combos)
    counts = Counter(number for combo in combos for number in combo)
    overlaps = [len(set(combo) & previous_draw) for combo in combos]
    consecutive = [sum(right - left == 1 for left, right in zip(combo, combo[1:])) for combo in combos]
    ranges = [combo[-1] - combo[0] for combo in combos]
    rates = {str(number): (counts[number] / size if size else 0.0) for number in range(1, 46)}
    dominant = sorted(rates.items(), key=lambda item: (-item[1], int(item[0])))[:10]
    return {
        "pool_size": size,
        "previous_draw_numbers": sorted(previous_draw),
        "number_inclusion_rates": rates,
        "dominant_numbers": [{"number": int(n), "inclusion_rate": float(rate)} for n, rate in dominant],
        "max_number_inclusion_rate": max(rates.values(), default=0.0),
        "mean_previous_draw_overlap": float(np.mean(overlaps)) if overlaps else float("nan"),
        "any_previous_draw_overlap_rate": float(np.mean([v > 0 for v in overlaps])) if overlaps else float("nan"),
        "mean_consecutive_pairs": float(np.mean(consecutive)) if consecutive else float("nan"),
        "any_consecutive_pair_rate": float(np.mean([v > 0 for v in consecutive])) if consecutive else float("nan"),
        "mean_number_range": float(np.mean(ranges)) if ranges else float("nan"),
        "top_combinations": [
            {"rank": rank, "numbers": list(entry[2]), "score": float(entry[0])}
            for rank, entry in enumerate(ranked[: int(top_k)], 1)
        ],
    }


def _compare(full: dict, other: dict, full_ranked, other_ranked) -> dict:
    left = {entry[2] for entry in full_ranked}
    right = {entry[2] for entry in other_ranked}
    common = len(left & right)
    union = len(left | right)
    number_delta = {
        number: float(other["number_inclusion_rates"][number]) - float(full["number_inclusion_rates"][number])
        for number in full["number_inclusion_rates"]
    }
    largest = sorted(number_delta.items(), key=lambda item: abs(item[1]), reverse=True)[:10]
    return {
        "top_pool_overlap_count_with_full": common,
        "top_pool_jaccard_with_full": common / union if union else 1.0,
        "max_number_inclusion_rate_delta": other["max_number_inclusion_rate"] - full["max_number_inclusion_rate"],
        "mean_previous_draw_overlap_delta": other["mean_previous_draw_overlap"] - full["mean_previous_draw_overlap"],
        "mean_consecutive_pairs_delta": other["mean_consecutive_pairs"] - full["mean_consecutive_pairs"],
        "mean_number_range_delta": other["mean_number_range"] - full["mean_number_range"],
        "largest_number_inclusion_rate_changes": [
            {"number": int(number), "delta": float(delta)} for number, delta in largest
        ],
    }


def run_latest_component_influence_audit(
    df: pd.DataFrame,
    exhaustive: bool = True,
    candidate_count: int = DEFAULT_CANDIDATE_COUNT,
    pool_size: int = DEFAULT_LATEST_POOL_SIZE,
    top_k: int = 10,
    seed_offset: int = 0,
    progress_every: int = 0,
) -> dict:
    """One candidate pass, nine rankings: full + direct/retrained ablations."""
    bundle = _latest_bundle(
        df,
        BACKTEST_START_INDEX,
        TRAIN_NEGATIVES_PER_TARGET,
        REFERENCE_SAMPLES_PER_TARGET,
        RIDGE_LAMBDA,
    )
    models = bundle["models"]
    seed = make_seed(int(bundle["latest_round"]), int(seed_offset))
    if exhaustive:
        candidates = iter_all_combinations()
        available = 8_145_060
        mode = "exhaustive_all_8,145,060"
    else:
        candidates = generate_candidates(int(candidate_count), seed)
        available = int(candidate_count)
        mode = "sampled_candidates"
    keep = min(available, max(1, int(pool_size)))
    heaps = {name: [] for name in models["scenario_names"]}

    evaluated = 0
    for evaluated, candidate in enumerate(candidates, 1):
        combo = tuple(sorted(int(number) for number in candidate))
        vector = candidate_vector(combo, bundle["context"], bundle["reference"])
        scores = score_scenarios_for_vector(vector, models)
        tie = _tie_break(combo, seed)
        for name in models["scenario_names"]:
            _push(heaps[name], (scores[name], tie, combo), keep)
        if progress_every and evaluated % int(progress_every) == 0:
            print(f"v3.1 component latest audit: {evaluated:,} candidates evaluated")

    ranked = {
        name: sorted(heap, key=lambda entry: (entry[0], entry[1]), reverse=True)
        for name, heap in heaps.items()
    }
    previous_draw = frozenset(bundle["context"]["previous_draw"])
    summaries = {name: _pool_summary(entries, previous_draw, top_k) for name, entries in ranked.items()}
    comparisons = {
        name: _compare(summaries["full"], summaries[name], ranked["full"], entries)
        for name, entries in ranked.items() if name != "full"
    }
    full_weights = np.asarray(models["full"]["effective_weights"], dtype=float)
    focused = {}
    for feature in AUDIT_FEATURES:
        weight = float(full_weights[_feature_index(feature)])
        focused[feature] = {
            "description": FEATURE_DESCRIPTIONS[feature],
            "effective_weight": weight,
            "learned_direction": (
                "higher_than_fair_null_is_favored" if weight > 0.0
                else "lower_than_fair_null_is_favored" if weight < 0.0
                else "neutral"
            ),
        }
    return {
        "latest_draw": int(bundle["latest_round"]),
        "target_draw": int(bundle["target_round"]),
        "evaluation_mode": mode,
        "evaluated_count": int(evaluated),
        "pool_size": keep,
        "formula_unchanged": True,
        "focused_features": focused,
        "scenario_summaries": summaries,
        "scenario_comparisons_vs_full": comparisons,
    }


def run_component_influence_audit(
    df: pd.DataFrame,
    include_historical: bool = True,
    include_latest: bool = True,
    historical_baseline_samples: int = 500,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    latest_exhaustive: bool = True,
    latest_candidate_count: int = DEFAULT_CANDIDATE_COUNT,
    latest_pool_size: int = DEFAULT_LATEST_POOL_SIZE,
    top_k: int = 10,
    historical_progress_every: int = 0,
    latest_progress_every: int = 0,
) -> dict:
    if not include_historical and not include_latest:
        raise ValueError("at least one audit section must be enabled")
    payload = {
        "version": AUDIT_VERSION,
        "model_version_under_audit": "v31_directional_fair_null_additive_reverse_ridge_v1",
        "scoring_formula_modified": False,
        "audit_features": list(AUDIT_FEATURES),
    }
    if include_historical:
        payload["historical_oos"] = run_historical_component_audit(
            df,
            baseline_samples=int(historical_baseline_samples),
            bootstrap_reps=int(bootstrap_reps),
            progress_every=int(historical_progress_every),
        )
    if include_latest:
        payload["latest_ranking_influence"] = run_latest_component_influence_audit(
            df,
            exhaustive=bool(latest_exhaustive),
            candidate_count=int(latest_candidate_count),
            pool_size=int(latest_pool_size),
            top_k=int(top_k),
            progress_every=int(latest_progress_every),
        )
    return payload
