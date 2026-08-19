from __future__ import annotations

import heapq
import math
from dataclasses import replace
from typing import Iterable

import numpy as np
import pandas as pd

from .candidates import (
    TOTAL_COMBINATION_COUNT,
    generate_candidates,
    iter_all_combinations,
    make_seed,
)
from .config import BACKTEST_START_INDEX, DEFAULT_CANDIDATE_COUNT, TOP_K_RECOMMENDATIONS
from .loader import load_lotto_data, row_numbers
from .mixed_scoring import structure_record
from .v31_core import (
    FEATURE_NAMES,
    _tie_break,
    build_latest_model as _core_build_latest_model,
    candidate_vector,
    directional_raw_features,
    fair_null_transform,
    fit_subset_pairwise_ridge,
    model_score,
)
from .v31_model_spec import (
    CLEAN3_CANDIDATE_SPEC,
    CLEAN3_REMOVED_FEATURES,
    spec_metadata,
)

MODEL_VERSION = CLEAN3_CANDIDATE_SPEC.name
AUDIT_POOL_SIZE = 1000
PORTFOLIO_MAX_SHARED_NUMBERS = 2
FAIR_RANDOM_OVERLAP_GE3_RATE = 0.023834078570323606

# Compatibility aliases. CLEAN3 is an experimental candidate, not a promoted production model.
DISABLED_PRODUCTION_FEATURES = CLEAN3_REMOVED_FEATURES
ACTIVE_PRODUCTION_FEATURES = CLEAN3_CANDIDATE_SPEC.active_features
ACTIVE_PRODUCTION_INDICES = tuple(
    FEATURE_NAMES.index(name) for name in ACTIVE_PRODUCTION_FEATURES
)

FAIR_EXPECTED_SUM = 138.0
FAIR_EXPECTED_ODD_COUNT = 6.0 * 23.0 / 45.0
FAIR_EXPECTED_NUMBER_RANGE = 5.0 * 46.0 / 7.0
FAIR_EXPECTED_CONSECUTIVE_PAIRS = 2.0 / 3.0
FAIR_EXPECTED_PREVIOUS_DRAW_OVERLAP = 36.0 / 45.0
FAIR_ANY_PREVIOUS_DRAW_OVERLAP_RATE = 1.0 - math.comb(39, 6) / math.comb(45, 6)


def fit_clean3_pairwise_ridge(
    stats: dict,
    ridge_lambda: float = CLEAN3_CANDIDATE_SPEC.ridge_lambda,
) -> dict:
    """Refit the same ridge objective on the CLEAN3 candidate subset."""
    model = fit_subset_pairwise_ridge(
        stats,
        ACTIVE_PRODUCTION_INDICES,
        ridge_lambda=float(ridge_lambda),
    )
    return {
        **model,
        "active_features": ACTIVE_PRODUCTION_FEATURES,
        "disabled_features": DISABLED_PRODUCTION_FEATURES,
    }


def build_latest_model(
    df: pd.DataFrame,
    start_index: int = BACKTEST_START_INDEX,
    train_negatives_per_target: int = CLEAN3_CANDIDATE_SPEC.train_negatives_per_target,
    reference_samples: int = CLEAN3_CANDIDATE_SPEC.fair_reference_samples_per_target,
    ridge_lambda: float = CLEAN3_CANDIDATE_SPEC.ridge_lambda,
) -> dict:
    spec = replace(
        CLEAN3_CANDIDATE_SPEC,
        history_start_index=int(start_index),
        train_negatives_per_target=int(train_negatives_per_target),
        fair_reference_samples_per_target=int(reference_samples),
        ridge_lambda=float(ridge_lambda),
    )
    fitted = _core_build_latest_model(df, spec)
    fitted["model"]["active_features"] = ACTIVE_PRODUCTION_FEATURES
    fitted["model"]["disabled_features"] = DISABLED_PRODUCTION_FEATURES
    return fitted


def explain_candidate(numbers: Iterable[int], fitted: dict) -> dict:
    nums = tuple(sorted(int(n) for n in numbers))
    raw = directional_raw_features(nums, fitted["context"])
    bounded = fair_null_transform(raw, fitted["reference"])
    weights = np.asarray(fitted["model"]["effective_weights"], dtype=float)
    contributions = weights * bounded
    order = np.argsort(np.abs(contributions))[::-1]
    active = set(ACTIVE_PRODUCTION_FEATURES)
    return {
        "numbers": list(nums),
        "model_score": float(np.sum(contributions)),
        "raw_features": {name: float(raw[i]) for i, name in enumerate(FEATURE_NAMES)},
        "bounded_features": {name: float(bounded[i]) for i, name in enumerate(FEATURE_NAMES)},
        "feature_contributions": [
            {
                "feature": FEATURE_NAMES[int(i)],
                "active": FEATURE_NAMES[int(i)] in active,
                "bounded_value": float(bounded[int(i)]),
                "weight": float(weights[int(i)]),
                "contribution": float(contributions[int(i)]),
            }
            for i in order
        ],
    }


def attach_structure(item: dict, latest_pattern_type: str | None) -> None:
    record = structure_record(item["numbers"], latest_pattern_type)
    item["pattern_type"] = str(record["pattern_type"])
    item["structure_metadata"] = {
        "sum": int(record["sum"]),
        "odd_count": int(record["odd_count"]),
        "number_range": int(record["number_range"]),
        "consecutive_pair_count": int(record["consecutive_pair_count"]),
        "section_distribution": list(record["section_distribution"]),
    }


def bias_audit(
    entries: list[tuple[float, int, tuple[int, ...]]],
    previous_draw: Iterable[int] | None = None,
) -> dict:
    """Distribution diagnostics only; distance from fair is not a deletion criterion."""
    combos = [tuple(entry[2]) for entry in entries]
    if not combos:
        return {}

    previous = frozenset(int(n) for n in (previous_draw or ()))
    counts = {str(n): 0 for n in range(1, 46)}
    sums: list[int] = []
    odds: list[int] = []
    ranges: list[int] = []
    consecutive: list[int] = []
    overlaps: list[int] = []
    low = mid = high = 0

    for combo in combos:
        sums.append(sum(combo))
        odds.append(sum(number % 2 for number in combo))
        ranges.append(combo[-1] - combo[0])
        consecutive.append(
            sum(right - left == 1 for left, right in zip(combo, combo[1:]))
        )
        if previous:
            overlaps.append(len(set(combo) & previous))
        for number in combo:
            counts[str(number)] += 1
            low += int(number <= 15)
            mid += int(16 <= number <= 30)
            high += int(number >= 31)

    total_slots = 6 * len(combos)
    payload = {
        "diagnostic_role": "ranking_distribution_only_not_model_selection_gate",
        "pool_size": len(combos),
        "fair_expected_number_inclusion_rate": 6.0 / 45.0,
        "number_inclusion_rates": {
            key: value / len(combos) for key, value in counts.items()
        },
        "mean_sum": float(np.mean(sums)),
        "min_sum": int(min(sums)),
        "max_sum": int(max(sums)),
        "mean_odd_count": float(np.mean(odds)),
        "mean_number_range": float(np.mean(ranges)),
        "mean_consecutive_pairs": float(np.mean(consecutive)),
        "any_consecutive_pair_rate": float(np.mean([value > 0 for value in consecutive])),
        "zone_slot_rates": {
            "1_15": low / total_slots,
            "16_30": mid / total_slots,
            "31_45": high / total_slots,
        },
        "fair_references": {
            "expected_sum": FAIR_EXPECTED_SUM,
            "expected_odd_count": FAIR_EXPECTED_ODD_COUNT,
            "expected_number_range": FAIR_EXPECTED_NUMBER_RANGE,
            "expected_consecutive_pairs": FAIR_EXPECTED_CONSECUTIVE_PAIRS,
            "expected_previous_draw_overlap": FAIR_EXPECTED_PREVIOUS_DRAW_OVERLAP,
            "any_previous_draw_overlap_rate": FAIR_ANY_PREVIOUS_DRAW_OVERLAP_RATE,
        },
    }
    if previous:
        payload.update(
            {
                "previous_draw_numbers": sorted(previous),
                "mean_previous_draw_overlap": float(np.mean(overlaps)),
                "any_previous_draw_overlap_rate": float(np.mean([value > 0 for value in overlaps])),
            }
        )
    return payload


def generate_raw_recommendations(
    exhaustive: bool = True,
    candidate_count: int = DEFAULT_CANDIDATE_COUNT,
    top_k: int = TOP_K_RECOMMENDATIONS,
    seed_offset: int = 0,
    progress_every: int = 0,
) -> dict:
    """Experimental CLEAN3 ranking. This function does not imply model promotion."""
    top_k = int(top_k)
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    progress_every = int(progress_every)
    if progress_every < 0:
        raise ValueError("progress_every must be non-negative")

    df = load_lotto_data()
    fitted = build_latest_model(df)
    seed = make_seed(int(fitted["latest_round"]), int(seed_offset))

    if exhaustive:
        candidates = iter_all_combinations()
        available = TOTAL_COMBINATION_COUNT
        mode = "exhaustive_all_8,145,060"
    else:
        candidates = generate_candidates(int(candidate_count), seed)
        available = len(candidates)
        mode = "sampled_candidates"
    if top_k > available:
        raise ValueError("top_k cannot exceed available candidate count")

    keep = min(available, max(top_k, AUDIT_POOL_SIZE))
    heap: list[tuple[float, int, tuple[int, ...]]] = []
    evaluated = 0
    for evaluated, candidate in enumerate(candidates, 1):
        nums = tuple(sorted(int(n) for n in candidate))
        entry = (model_score(nums, fitted), _tie_break(nums), nums)
        if len(heap) < keep:
            heapq.heappush(heap, entry)
        elif entry[:2] > heap[0][:2]:
            heapq.heapreplace(heap, entry)
        if progress_every and evaluated % progress_every == 0:
            print(f"v3.1 CLEAN3 candidate ranking progress: {evaluated:,} candidates evaluated")

    ranked = sorted(heap, key=lambda item: (item[0], item[1]), reverse=True)
    latest_pattern_type = str(structure_record(row_numbers(df.iloc[-1]))["pattern_type"])
    recommendations = [explain_candidate(entry[2], fitted) for entry in ranked[:top_k]]
    for rank, item in enumerate(recommendations, 1):
        item["rank"] = rank
        attach_structure(item, latest_pattern_type)

    return {
        "meta": {
            "model_version": MODEL_VERSION,
            "model_status": CLEAN3_CANDIDATE_SPEC.status,
            "model_spec": spec_metadata(CLEAN3_CANDIDATE_SPEC),
            "latest_draw": int(fitted["latest_round"]),
            "target_draw": int(fitted["target_round"]),
            "evaluation_mode": mode,
            "evaluated_count": int(evaluated),
            "formula": "score(c)=w_A^T z_A(c), z_j=2*F_mid,j(x_j)-1",
            "raw_feature_names": list(FEATURE_NAMES),
            "active_feature_names": list(ACTIVE_PRODUCTION_FEATURES),
            "disabled_feature_names": list(DISABLED_PRODUCTION_FEATURES),
            "candidate_policy": "all valid 6-of-45 combinations; no hard filters or pattern quotas",
            "pattern_type_role": "metadata_only_not_used_in_ranking",
            "tie_policy": CLEAN3_CANDIDATE_SPEC.tie_policy,
            "data": {
                "rows": int(fitted["dataset_rows"]),
                "sha256": str(fitted["dataset_sha256"]),
            },
            "training": {
                "history_draws": int(fitted["history_draws"]),
                "solved_targets": int(fitted["model"]["solved_targets"]),
                "pair_rows": int(fitted["model"]["pair_rows"]),
                "reference_samples_per_target": int(fitted["reference_samples"]),
                "train_negatives_per_target": int(fitted["train_negatives_per_target"]),
                "ridge_lambda": float(fitted["model"]["ridge_lambda"]),
                "scaled_second_moment_condition_number": float(
                    fitted["model"]["scaled_second_moment_condition_number"]
                ),
            },
        },
        "ranking_distribution_diagnostics_top_pool": bias_audit(
            ranked, fitted["context"]["previous_draw"]
        ),
        "bias_audit_top_pool": bias_audit(ranked, fitted["context"]["previous_draw"]),
        "recommendations": recommendations,
    }
