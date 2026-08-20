from __future__ import annotations

import heapq
import json
from dataclasses import replace
from pathlib import Path
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
    _add_target,
    _empty_stats,
    _sample_reference,
    _tie_break,
    build_latest_model as _core_build_latest_model,
    candidate_vector,
    directional_raw_features,
    fair_null_transform,
    fit_additive_pairwise_ridge as _core_fit_additive_pairwise_ridge,
    model_score,
)
from .v31_model_spec import FULL11_BASELINE_SPEC, spec_metadata

MODEL_VERSION = FULL11_BASELINE_SPEC.name
REFERENCE_SAMPLES_PER_TARGET = FULL11_BASELINE_SPEC.fair_reference_samples_per_target
REFERENCE_SEED_OFFSET = FULL11_BASELINE_SPEC.reference_seed_offset
SELECTION_STRATEGY = "v31_full11_experimental_baseline_score_top_k"
PORTFOLIO_STRATEGY = "highest_raw_score_subject_to_pairwise_shared_numbers_le_2"
AUDIT_POOL_SIZE = 1000
PORTFOLIO_MAX_SHARED_NUMBERS = 2
FAIR_RANDOM_OVERLAP_GE3_RATE = 0.023834078570323606


def fit_additive_pairwise_ridge(
    stats: dict,
    ridge_lambda: float = FULL11_BASELINE_SPEC.ridge_lambda,
) -> dict:
    return _core_fit_additive_pairwise_ridge(stats, ridge_lambda=float(ridge_lambda))


def build_latest_model(
    df: pd.DataFrame,
    start_index: int = BACKTEST_START_INDEX,
    train_negatives_per_target: int = FULL11_BASELINE_SPEC.train_negatives_per_target,
    reference_samples: int = REFERENCE_SAMPLES_PER_TARGET,
    ridge_lambda: float = FULL11_BASELINE_SPEC.ridge_lambda,
) -> dict:
    spec = replace(
        FULL11_BASELINE_SPEC,
        history_start_index=int(start_index),
        train_negatives_per_target=int(train_negatives_per_target),
        fair_reference_samples_per_target=int(reference_samples),
        ridge_lambda=float(ridge_lambda),
    )
    return _core_build_latest_model(df, spec)


def explain_candidate(numbers: Iterable[int], fitted: dict) -> dict:
    nums = tuple(sorted(int(n) for n in numbers))
    raw = directional_raw_features(nums, fitted["context"])
    bounded = fair_null_transform(raw, fitted["reference"])
    weights = np.asarray(fitted["model"]["effective_weights"], dtype=float)
    contributions = weights * bounded
    order = np.argsort(np.abs(contributions))[::-1]
    return {
        "numbers": list(nums),
        "model_score": float(np.sum(contributions)),
        "raw_features": {name: float(raw[i]) for i, name in enumerate(FEATURE_NAMES)},
        "bounded_features": {name: float(bounded[i]) for i, name in enumerate(FEATURE_NAMES)},
        "feature_contributions": [
            {
                "feature": FEATURE_NAMES[int(i)],
                "bounded_value": float(bounded[int(i)]),
                "weight": float(weights[int(i)]),
                "contribution": float(contributions[int(i)]),
            }
            for i in order
        ],
    }


def _bias_audit(entries: list[tuple[float, int, tuple[int, ...]]]) -> dict:
    numbers = [entry[2] for entry in entries]
    if not numbers:
        return {}
    counts = {str(n): 0 for n in range(1, 46)}
    sums = []
    one_digit = 0
    low = mid = high = 0
    for combo in numbers:
        sums.append(sum(combo))
        one_digit += int(any(n <= 9 for n in combo))
        for n in combo:
            counts[str(n)] += 1
            low += int(n <= 15)
            mid += int(16 <= n <= 30)
            high += int(n >= 31)
    total_slots = 6 * len(numbers)
    return {
        "pool_size": len(numbers),
        "fair_expected_number_inclusion_rate": 6.0 / 45.0,
        "number_inclusion_rates": {k: v / len(numbers) for k, v in counts.items()},
        "one_digit_present_rate": one_digit / len(numbers),
        "mean_sum": float(np.mean(sums)),
        "min_sum": int(min(sums)),
        "max_sum": int(max(sums)),
        "zone_slot_rates": {
            "1_15": low / total_slots,
            "16_30": mid / total_slots,
            "31_45": high / total_slots,
        },
    }


def _select_portfolio_entries(
    ranked: list[tuple[float, int, tuple[int, ...]]],
    top_k: int,
    max_shared_numbers: int = PORTFOLIO_MAX_SHARED_NUMBERS,
) -> tuple[list[tuple[float, int, tuple[int, ...]]], bool]:
    """Historical baseline selector retained for audit/test compatibility only."""
    top_k = int(top_k)
    max_shared_numbers = int(max_shared_numbers)
    selected: list[tuple[float, int, tuple[int, ...]]] = []
    selected_sets: list[set[int]] = []
    for entry in ranked:
        combo_set = set(entry[2])
        if all(len(combo_set & prior) <= max_shared_numbers for prior in selected_sets):
            selected.append(entry)
            selected_sets.append(combo_set)
            if len(selected) >= top_k:
                return selected, False

    chosen = {entry[2] for entry in selected}
    for entry in ranked:
        if entry[2] in chosen:
            continue
        selected.append(entry)
        chosen.add(entry[2])
        if len(selected) >= top_k:
            break
    return selected, True


def _attach_structure(item: dict, latest_pattern_type: str) -> None:
    record = structure_record(item["numbers"], latest_pattern_type)
    item["pattern_type"] = str(record["pattern_type"])
    item["structure_metadata"] = {
        "sum": int(record["sum"]),
        "odd_count": int(record["odd_count"]),
        "number_range": int(record["number_range"]),
        "section_distribution": list(record["section_distribution"]),
    }


def generate_recommendations(
    exhaustive: bool = True,
    candidate_count: int = DEFAULT_CANDIDATE_COUNT,
    top_k: int = TOP_K_RECOMMENDATIONS,
    seed_offset: int = 0,
    progress_every: int = 0,
) -> dict:
    """Historical full-11 baseline. It is not the promoted current model."""
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

    keep = max(top_k, min(AUDIT_POOL_SIZE, available))
    heap: list[tuple[float, int, tuple[int, ...]]] = []
    evaluated = 0
    for evaluated, candidate in enumerate(candidates, 1):
        nums = tuple(sorted(int(n) for n in candidate))
        score = model_score(nums, fitted)
        entry = (score, _tie_break(nums), nums)
        if len(heap) < keep:
            heapq.heappush(heap, entry)
        elif entry[:2] > heap[0][:2]:
            heapq.heapreplace(heap, entry)
        if progress_every and evaluated % progress_every == 0:
            print(f"v3.1 full11 baseline ranking progress: {evaluated:,} candidates evaluated")

    ranked = sorted(heap, key=lambda item: (item[0], item[1]), reverse=True)
    latest_pattern_type = str(structure_record(row_numbers(df.iloc[-1]))["pattern_type"])
    raw_rank_lookup = {entry[2]: rank for rank, entry in enumerate(ranked, 1)}

    selected = [explain_candidate(entry[2], fitted) for entry in ranked[:top_k]]
    for rank, item in enumerate(selected, 1):
        item["rank"] = rank
        _attach_structure(item, latest_pattern_type)

    portfolio_entries, portfolio_relaxed = _select_portfolio_entries(ranked, top_k)
    portfolio = [explain_candidate(entry[2], fitted) for entry in portfolio_entries]
    for rank, item in enumerate(portfolio, 1):
        item["portfolio_rank"] = rank
        item["raw_rank_within_retained_pool"] = int(raw_rank_lookup[tuple(item["numbers"])])
        _attach_structure(item, latest_pattern_type)

    return {
        "meta": {
            "model_version": MODEL_VERSION,
            "model_status": FULL11_BASELINE_SPEC.status,
            "model_spec": spec_metadata(FULL11_BASELINE_SPEC),
            "latest_draw": int(fitted["latest_round"]),
            "target_draw": int(fitted["target_round"]),
            "evaluation_mode": mode,
            "evaluated_count": int(evaluated),
            "selection_strategy": SELECTION_STRATEGY,
            "formula": "score(c)=w^T z(c), z_j=2*F_mid,j(x_j)-1",
            "feature_names": list(FEATURE_NAMES),
            "candidate_policy": "all valid 6-of-45 combinations; no hard filters or pattern quotas",
            "pattern_type_role": "metadata_only_not_used_in_ranking",
            "tie_policy": FULL11_BASELINE_SPEC.tie_policy,
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
        "bias_audit_top_pool": _bias_audit(ranked),
        "portfolio_bias_audit": _bias_audit(portfolio_entries),
        "recommendations": selected,
        "portfolio_recommendations": portfolio,
    }


def write_json(payload: dict, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path.resolve()
