from __future__ import annotations

import random
from collections import Counter

import numpy as np

from .candidates import generate_candidates, make_seed, random_combination
from .loader import load_lotto_data
from .v31_core import (
    FEATURE_NAMES,
    build_latest_model,
    directional_raw_features,
    fair_null_transform,
)
from .v31_model_spec import V31ModelSpec
from .v31_scenario_recommendation import scenario_spec

DEFAULT_REFERENCE_COUNTS = (1_024, 4_096, 16_384)
DEFAULT_SEED_DELTAS = (1, 2, 3)


def _reference_rows_from_candidates(candidates, context: dict) -> np.ndarray:
    return np.sort(
        np.asarray(
            [directional_raw_features(candidate, context) for candidate in candidates],
            dtype=float,
        ),
        axis=0,
    )


def sample_reference_with_seed_delta(
    context: dict,
    round_no: int,
    count: int,
    spec: V31ModelSpec,
    seed_delta: int,
) -> np.ndarray:
    """Current sampler shape with an explicit seed perturbation for sensitivity checks."""
    count = int(count)
    if count <= 0:
        raise ValueError("reference count must be positive")
    rng = random.Random(
        int(round_no) * 310_007
        + count * 3_101
        + int(spec.reference_seed_offset)
        + int(seed_delta) * 1_000_003
    )
    seen = set()
    candidates = []
    while len(candidates) < count:
        candidate = tuple(random_combination(rng))
        if candidate in seen:
            continue
        seen.add(candidate)
        candidates.append(candidate)
    return _reference_rows_from_candidates(candidates, context)


def sample_nested_reference_candidates(
    round_no: int,
    count: int,
    spec: V31ModelSpec,
    stream_delta: int = 0,
) -> list[tuple[int, ...]]:
    """Count-independent deterministic stream so larger references contain smaller prefixes."""
    count = int(count)
    if count <= 0:
        raise ValueError("reference count must be positive")
    rng = random.Random(
        int(round_no) * 310_007
        + int(spec.reference_seed_offset)
        + 77_003
        + int(stream_delta) * 1_000_003
    )
    seen = set()
    candidates = []
    while len(candidates) < count:
        candidate = tuple(random_combination(rng))
        if candidate in seen:
            continue
        seen.add(candidate)
        candidates.append(candidate)
    return candidates


def sample_nested_reference(
    context: dict,
    round_no: int,
    count: int,
    spec: V31ModelSpec,
    stream_delta: int = 0,
) -> np.ndarray:
    return _reference_rows_from_candidates(
        sample_nested_reference_candidates(round_no, count, spec, stream_delta),
        context,
    )


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        average = 0.5 * (start + end - 1)
        ranks[order[start:end]] = average
        start = end
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if len(left) != len(right) or len(left) == 0:
        return float("nan")
    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return 1.0 if np.array_equal(left, right) else 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _top_indices(scores: np.ndarray, count: int) -> np.ndarray:
    count = min(int(count), len(scores))
    if count <= 0:
        return np.asarray([], dtype=int)
    # Stable fixed index order is used after score ordering for deterministic diagnostics.
    return np.lexsort((np.arange(len(scores)), -np.asarray(scores, dtype=float)))[:count]


def _jaccard(left: np.ndarray, right: np.ndarray) -> float:
    a = set(int(value) for value in left)
    b = set(int(value) for value in right)
    union = len(a | b)
    return len(a & b) / union if union else 1.0


def _scenario_summary(
    scores: np.ndarray,
    saturation_counts: np.ndarray,
    candidate_count: int,
) -> dict:
    rates = saturation_counts / max(1, int(candidate_count))
    return {
        "score_min": float(np.min(scores)),
        "score_mean": float(np.mean(scores)),
        "score_max": float(np.max(scores)),
        "exact_score_unique_count": int(len(np.unique(scores))),
        "exact_score_duplicate_count": int(len(scores) - len(np.unique(scores))),
        "feature_saturation_rates": {
            name: float(rates[index]) for index, name in enumerate(FEATURE_NAMES)
        },
    }


def _compare_scores(base: np.ndarray, other: np.ndarray) -> dict:
    base = np.asarray(base, dtype=float)
    other = np.asarray(other, dtype=float)
    delta = other - base
    result = {
        "pearson_score_correlation": _correlation(base, other),
        "spearman_rank_correlation": _correlation(
            _average_ranks(base), _average_ranks(other)
        ),
        "mean_absolute_score_delta": float(np.mean(np.abs(delta))),
        "max_absolute_score_delta": float(np.max(np.abs(delta))),
        "top_jaccard": {},
    }
    for top_n in (10, 100, 1_000):
        if len(base) >= top_n:
            result["top_jaccard"][str(top_n)] = _jaccard(
                _top_indices(base, top_n), _top_indices(other, top_n)
            )
    return result


def run_reference_stability_audit(
    scenario: str = "full11",
    candidate_count: int = 20_000,
    candidate_seed_offset: int = 901,
    reference_counts: tuple[int, ...] = DEFAULT_REFERENCE_COUNTS,
    seed_deltas: tuple[int, ...] = DEFAULT_SEED_DELTAS,
    progress_every: int = 0,
) -> dict:
    """Audit latest-reference numerical stability with model weights held fixed.

    This intentionally does not choose a better-performing reference. It measures
    how much the same fitted model ranking moves when the empirical fair-null
    reference is perturbed or enlarged.
    """
    candidate_count = int(candidate_count)
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    progress_every = int(progress_every)
    if progress_every < 0:
        raise ValueError("progress_every must be non-negative")

    spec = scenario_spec(scenario)
    df = load_lotto_data()
    fitted = build_latest_model(df, spec)
    target_round = int(fitted["target_round"])
    context = fitted["context"]
    weights = np.asarray(fitted["model"]["effective_weights"], dtype=float)

    references: dict[str, np.ndarray] = {"current_reference": fitted["reference"]}
    baseline_count = int(spec.fair_reference_samples_per_target)
    for delta in seed_deltas:
        references[f"current_policy_seed_delta:{int(delta)}"] = sample_reference_with_seed_delta(
            context, target_round, baseline_count, spec, int(delta)
        )

    normalized_counts = tuple(sorted(set(int(count) for count in reference_counts)))
    for count in normalized_counts:
        if count <= 0:
            raise ValueError("reference counts must be positive")
        references[f"nested_count:{count}"] = sample_nested_reference(
            context, target_round, count, spec, stream_delta=0
        )

    candidate_seed = make_seed(int(fitted["latest_round"]), int(candidate_seed_offset))
    candidates = [tuple(candidate) for candidate in generate_candidates(candidate_count, candidate_seed)]
    names = tuple(references)
    score_matrix = np.empty((candidate_count, len(names)), dtype=float)
    saturation = np.zeros((len(names), len(FEATURE_NAMES)), dtype=int)
    coordinate_abs_delta = np.zeros((len(names), len(FEATURE_NAMES)), dtype=float)

    for row_index, candidate in enumerate(candidates):
        raw = directional_raw_features(candidate, context)
        baseline_bounded = fair_null_transform(raw, references["current_reference"])
        for scenario_index, name in enumerate(names):
            bounded = fair_null_transform(raw, references[name])
            score_matrix[row_index, scenario_index] = float(np.dot(weights, bounded))
            saturation[scenario_index] += (np.abs(bounded) == 1.0).astype(int)
            coordinate_abs_delta[scenario_index] += np.abs(bounded - baseline_bounded)
        if progress_every and (row_index + 1) % progress_every == 0:
            print(
                f"v3.1 reference stability: {row_index + 1:,}/{candidate_count:,} candidates"
            )

    summaries = {
        name: {
            **_scenario_summary(score_matrix[:, index], saturation[index], candidate_count),
            "mean_absolute_coordinate_delta_vs_current": {
                feature: float(coordinate_abs_delta[index, feature_index] / candidate_count)
                for feature_index, feature in enumerate(FEATURE_NAMES)
            },
        }
        for index, name in enumerate(names)
    }

    comparisons_vs_current = {
        name: _compare_scores(score_matrix[:, 0], score_matrix[:, index])
        for index, name in enumerate(names)
        if index != 0
    }

    nested_comparisons = {}
    nested_labels = [f"nested_count:{count}" for count in normalized_counts]
    for left, right in zip(nested_labels, nested_labels[1:]):
        left_index = names.index(left)
        right_index = names.index(right)
        nested_comparisons[f"{left}->{right}"] = _compare_scores(
            score_matrix[:, left_index], score_matrix[:, right_index]
        )

    return {
        "version": "v31_reference_stability_audit_v1",
        "audit_role": "numerical_stability_only_not_parameter_tuning",
        "scenario": str(scenario).lower(),
        "model_version": spec.name,
        "model_status": spec.status,
        "latest_draw": int(fitted["latest_round"]),
        "target_draw": target_round,
        "candidate_count": candidate_count,
        "candidate_seed": int(candidate_seed),
        "weights_held_fixed": True,
        "training_reference_policy_unchanged": True,
        "reference_counts": list(normalized_counts),
        "seed_deltas": [int(value) for value in seed_deltas],
        "scenario_summaries": summaries,
        "comparisons_vs_current_reference": comparisons_vs_current,
        "nested_reference_convergence": nested_comparisons,
    }
