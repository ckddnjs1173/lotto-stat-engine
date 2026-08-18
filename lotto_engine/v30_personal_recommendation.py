from __future__ import annotations

import hashlib
import heapq
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .candidates import generate_candidates, iter_all_combinations, make_seed
from .config import (
    BACKTEST_START_INDEX,
    DEFAULT_CANDIDATE_COUNT,
    DEFAULT_EXHAUSTIVE_RECOMMENDATION,
    ROUND_COLUMN,
    TOP_K_RECOMMENDATIONS,
)
from .loader import load_lotto_data, row_numbers
from .mixed_scoring import structure_record
from .profiles import build_profile
from .v28_reverse_ranking import (
    TRAIN_NEGATIVES_PER_TARGET,
    _advance_history,
    _empty_history_state,
    _feature_context,
    _sample_fair_candidates,
    candidate_feature_vector,
)
from .v29_quadratic_reverse_ranking import (
    RIDGE_LAMBDA,
    _add_solved_target,
    _empty_training_stats,
    fit_quadratic_pairwise_ridge,
    score_quadratic_vector,
)
from .v30_null_bounded_reverse_ranking import (
    BOUNDED_BASE_FEATURE_NAMES,
    FEATURE_NAMES,
    MODEL_VERSION as V30_SOURCE_MODEL_VERSION,
    REFERENCE_SAMPLES_PER_TARGET,
    bounded_quadratic_basis,
    build_null_reference_matrix,
    null_midrank_transform,
)

PERSONAL_MODEL_VERSION = "v30_null_bounded_personal_v1"
SELECTION_STRATEGY = "v30_bounded_model_score_top_k"
DEFAULT_VALIDATION_JSON = Path("data/cache/v30_null_bounded_screen.json")
VALIDATION_ROLE = "diagnostic_only_not_used_in_ranking"


def _seeded_tiebreak(numbers: Iterable[int], seed: int) -> int:
    encoded = f"{int(seed)}:" + ",".join(str(int(number)) for number in numbers)
    digest = hashlib.blake2b(encoded.encode("ascii"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


def build_latest_v30_model(
    df: pd.DataFrame,
    start_index: int = BACKTEST_START_INDEX,
    train_negatives_per_target: int = TRAIN_NEGATIVES_PER_TARGET,
    ridge_lambda: float = RIDGE_LAMBDA,
    reference_samples: int = REFERENCE_SAMPLES_PER_TARGET,
) -> dict:
    """Fit the bounded v3.0 model using every completed draw, without outer backtesting."""
    start_index = int(start_index)
    train_negatives_per_target = int(train_negatives_per_target)
    reference_samples = int(reference_samples)
    if len(df) <= start_index:
        raise ValueError("v3.0 personal recommendation requires draws after start_index")
    if train_negatives_per_target <= 0 or reference_samples <= 0:
        raise ValueError("training negatives and reference samples must be positive")

    state = _empty_history_state()
    for idx in range(start_index):
        _advance_history(state, tuple(row_numbers(df.iloc[idx])))

    stats = _empty_training_stats()
    for idx in range(start_index, len(df)):
        actual = tuple(row_numbers(df.iloc[idx]))
        round_no = int(df.iloc[idx][ROUND_COLUMN])
        context = _feature_context(state)
        reference = build_null_reference_matrix(
            context, round_no, reference_samples=reference_samples
        )
        actual_raw = candidate_feature_vector(actual, context)
        actual_bounded = null_midrank_transform(actual_raw, reference)
        actual_vector = bounded_quadratic_basis(actual_bounded)

        train_rng = random.Random(
            round_no * 200_003
            + train_negatives_per_target * 2_009
            + 28_101
        )
        fair_train = _sample_fair_candidates(
            train_rng, train_negatives_per_target, actual
        )
        negative_vectors = []
        for candidate in fair_train:
            raw = candidate_feature_vector(candidate, context)
            bounded = null_midrank_transform(raw, reference)
            negative_vectors.append(bounded_quadratic_basis(bounded))

        _add_solved_target(stats, actual_vector, negative_vectors)
        _advance_history(state, actual)

    model = fit_quadratic_pairwise_ridge(stats, ridge_lambda=float(ridge_lambda))
    next_context = _feature_context(state)
    latest_round = int(df.iloc[-1][ROUND_COLUMN])
    target_round = latest_round + 1
    next_reference = build_null_reference_matrix(
        next_context,
        target_round,
        reference_samples=reference_samples,
    )
    return {
        "model": model,
        "context": next_context,
        "reference": next_reference,
        "latest_round": latest_round,
        "target_round": target_round,
        "history_draws": int(state["history_draws"]),
        "solved_targets": int(stats["solved_targets"]),
        "pair_rows": int(stats["pair_rows"]),
        "start_index": start_index,
        "train_negatives_per_target": train_negatives_per_target,
        "ridge_lambda": float(ridge_lambda),
        "reference_samples": reference_samples,
    }


def v30_candidate_components(
    numbers: Iterable[int], model: dict, context: dict, reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    raw = candidate_feature_vector(numbers, context)
    bounded = null_midrank_transform(raw, reference)
    basis = bounded_quadratic_basis(bounded)
    score = score_quadratic_vector(basis, model)
    return raw, bounded, basis, float(score)


def v30_personal_score(
    numbers: Iterable[int], model: dict, context: dict, reference: np.ndarray
) -> float:
    return v30_candidate_components(numbers, model, context, reference)[3]


def explain_v30_candidate(
    numbers: Iterable[int], model: dict, context: dict, reference: np.ndarray
) -> dict:
    nums = tuple(sorted(int(number) for number in numbers))
    raw, bounded, basis, score = v30_candidate_components(
        nums, model, context, reference
    )
    weights = np.asarray(model["effective_weights"], dtype=float)
    contributions = weights * basis
    order = np.argsort(np.abs(contributions))[::-1][:10]
    return {
        "numbers": list(nums),
        "model_score": float(score),
        "ranking_score": float(score),
        "score_origin": PERSONAL_MODEL_VERSION,
        "raw_base_features": [float(value) for value in raw],
        "bounded_base_features": {
            name: float(bounded[index])
            for index, name in enumerate(BOUNDED_BASE_FEATURE_NAMES)
        },
        "strongest_terms": [
            {
                "term": str(FEATURE_NAMES[int(index)]),
                "value": float(basis[int(index)]),
                "weight": float(weights[int(index)]),
                "contribution": float(contributions[int(index)]),
            }
            for index in order
        ],
    }


def top_v30_stream(
    candidates: Iterable[Iterable[int]],
    model: dict,
    context: dict,
    reference: np.ndarray,
    top_k: int,
    seed: int,
    progress_every: int = 0,
) -> tuple[list[dict], int]:
    top_k = int(top_k)
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    heap: list[tuple[float, int, tuple[int, ...]]] = []
    evaluated = 0
    for evaluated, candidate in enumerate(candidates, 1):
        nums = tuple(sorted(int(number) for number in candidate))
        score = v30_personal_score(nums, model, context, reference)
        entry = (score, _seeded_tiebreak(nums, seed), nums)
        if len(heap) < top_k:
            heapq.heappush(heap, entry)
        elif entry[:2] > heap[0][:2]:
            heapq.heapreplace(heap, entry)

        if progress_every and evaluated % int(progress_every) == 0:
            print(f"v3.0 personal ranking progress: {evaluated:,} candidates evaluated")

    selected_entries = sorted(heap, key=lambda value: (value[0], value[1]), reverse=True)
    selected = [
        explain_v30_candidate(entry[2], model, context, reference)
        for entry in selected_entries
    ]
    for rank, item in enumerate(selected, 1):
        item["rank"] = rank
        item["strategy"] = "V3.0 FAIR-NULL BOUNDED PERSONAL RANKING"
    return selected, evaluated


def load_validation_diagnostics(path: str | Path | None) -> dict:
    if path is None:
        return {"status": "not_loaded", "role": VALIDATION_ROLE, "ranking_influenced": False}
    validation_path = Path(path)
    if not validation_path.exists():
        return {
            "status": "file_not_found",
            "path": str(validation_path),
            "role": VALIDATION_ROLE,
            "ranking_influenced": False,
        }
    with validation_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    summary = dict(payload.get("outer_summary", {}))
    gate = summary.get("historical_gate_pass")
    if gate is None:
        gate = summary.get("exploratory_screening_candidate")
    return {
        "status": "loaded",
        "path": str(validation_path),
        "version": payload.get("version"),
        "outer_summary": summary,
        "historical_gate_pass": bool(gate) if gate is not None else None,
        "role": VALIDATION_ROLE,
        "ranking_influenced": False,
    }


def _attach_structure_metadata(item: dict, latest_pattern_type: str) -> None:
    record = structure_record(item["numbers"], latest_pattern_type)
    item["pattern_type"] = str(record["pattern_type"])
    item["structure_metadata"] = {
        "sum": int(record["sum"]),
        "odd_count": int(record["odd_count"]),
        "number_range": int(record["number_range"]),
        "max_gap": int(record["max_gap"]),
        "min_gap": int(record["min_gap"]),
        "section_distribution": list(record["section_distribution"]),
    }


def generate_v30_personal_recommendations(
    seed_offset: int = 0,
    candidate_count: int = DEFAULT_CANDIDATE_COUNT,
    exhaustive: bool = DEFAULT_EXHAUSTIVE_RECOMMENDATION,
    top_k: int = TOP_K_RECOMMENDATIONS,
    progress_every: int = 0,
    validation_json: str | Path | None = DEFAULT_VALIDATION_JSON,
    reference_samples: int = REFERENCE_SAMPLES_PER_TARGET,
) -> dict:
    df = load_lotto_data()
    profile = build_profile(df)
    latest_draw = int(profile["latest_round"])
    target_draw = latest_draw + 1
    seed = make_seed(latest_draw, int(seed_offset))

    fitted = build_latest_v30_model(df, reference_samples=int(reference_samples))
    model = fitted["model"]
    context = fitted["context"]
    reference = fitted["reference"]

    if exhaustive:
        candidates = iter_all_combinations()
        mode = "exhaustive_all_8,145,060"
    else:
        candidates = generate_candidates(int(candidate_count), seed)
        mode = "sampled_candidates"

    recommendations, evaluated_count = top_v30_stream(
        candidates,
        model,
        context,
        reference,
        int(top_k),
        seed,
        progress_every=int(progress_every),
    )
    for item in recommendations:
        _attach_structure_metadata(item, str(profile["latest_pattern_type"]))

    return {
        "meta": {
            "model_version": PERSONAL_MODEL_VERSION,
            "source_model_version": V30_SOURCE_MODEL_VERSION,
            "latest_draw": latest_draw,
            "target_draw": target_draw,
            "historical_draw_count": int(len(df)),
            "recommendation_mode": mode,
            "evaluated_count": int(evaluated_count),
            "top_k": int(top_k),
            "seed": int(seed),
            "selection_strategy": SELECTION_STRATEGY,
            "score_name": "model_score",
            "score_scale": "bounded_fair_null_quadratic_linear_predictor",
            "candidate_policy": "all valid 6-of-45 combinations; no hard filters",
            "pattern_type_role": "metadata_only_not_used_for_ranking",
            "validation_role": VALIDATION_ROLE,
            "ranking_influenced_by_validation": False,
            "training": {
                "start_index": int(fitted["start_index"]),
                "history_draws": int(fitted["history_draws"]),
                "solved_targets": int(fitted["solved_targets"]),
                "pair_rows": int(fitted["pair_rows"]),
                "train_negatives_per_target": int(fitted["train_negatives_per_target"]),
                "ridge_lambda": float(fitted["ridge_lambda"]),
                "reference_samples_per_target": int(fitted["reference_samples"]),
                "basis_width": len(FEATURE_NAMES),
            },
            "historical_validation": load_validation_diagnostics(validation_json),
            "score_disclaimer": (
                "model_score는 최신 완료 회차까지 학습한 v3.0 bounded reverse-ranking 식의 원 계산값입니다. "
                "실제 당첨확률이 아니며 역사 검증 결과는 순위 계산에 곱하지 않습니다."
            ),
        },
        "recommendations": recommendations,
    }


def public_v30_payload(payload: dict) -> dict:
    meta = payload["meta"]
    return {
        "model_version": meta["model_version"],
        "source_model_version": meta["source_model_version"],
        "latest_draw": meta["latest_draw"],
        "target_draw": meta["target_draw"],
        "recommendation_mode": meta["recommendation_mode"],
        "evaluated_count": meta["evaluated_count"],
        "selection_strategy": meta["selection_strategy"],
        "score_name": meta["score_name"],
        "score_scale": meta["score_scale"],
        "candidate_policy": meta["candidate_policy"],
        "pattern_type_role": meta["pattern_type_role"],
        "validation_role": meta["validation_role"],
        "ranking_influenced_by_validation": meta["ranking_influenced_by_validation"],
        "training": dict(meta["training"]),
        "historical_validation": dict(meta["historical_validation"]),
        "score_disclaimer": meta["score_disclaimer"],
        "recommendations": [
            {
                "rank": int(item["rank"]),
                "numbers": list(item["numbers"]),
                "score": float(item["model_score"]),
                "model_score": float(item["model_score"]),
                "pattern_type": item.get("pattern_type", "unknown"),
                "bounded_base_features": dict(item["bounded_base_features"]),
                "strongest_terms": list(item["strongest_terms"]),
                "structure_metadata": dict(item.get("structure_metadata", {})),
            }
            for item in payload["recommendations"]
        ],
    }


def write_v30_json(payload: dict, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(public_v30_payload(payload), handle, ensure_ascii=False, indent=2)
    return path.resolve()


def print_v30_recommendations(payload: dict) -> None:
    meta = payload["meta"]
    print("=" * 96)
    print("LOTTO STAT ENGINE v3.0 - FINAL PERSONAL BOUNDED REVERSE RANKING")
    print("=" * 96)
    print(f"latest reflected draw: {meta['latest_draw']}")
    print(f"target draw: {meta['target_draw']}")
    print(f"evaluation mode: {meta['recommendation_mode']}")
    print(f"evaluated combinations: {meta['evaluated_count']:,}")
    print(f"selection: {meta['selection_strategy']}")
    print("validation: diagnostic only; it never changes the model ranking")
    print(meta["score_disclaimer"])

    validation = meta.get("historical_validation", {})
    summary = validation.get("outer_summary", {})
    if summary:
        ci = summary.get("percentile_minus_50_block_bootstrap_95_ci", [float("nan"), float("nan")])
        print(
            "historical validation: "
            f"mean={float(summary.get('mean_percentile', float('nan'))):.4f} "
            f"recent300={float(summary.get('recent_300_mean_percentile', float('nan'))):.4f} "
            f"recent100={float(summary.get('recent_100_mean_percentile', float('nan'))):.4f} "
            f"CI95=[{float(ci[0]):.4f}, {float(ci[1]):.4f}] "
            f"gate={validation.get('historical_gate_pass')}"
        )

    for item in payload["recommendations"]:
        numbers = " ".join(str(number) for number in item["numbers"])
        print()
        print(f"#{item['rank']}  {numbers}")
        print(f"model_score={item['model_score']:.12f} type={item.get('pattern_type', 'unknown')}")
        for term in item["strongest_terms"][:5]:
            print(
                f"  {term['term']}: contribution={term['contribution']:.12f} "
                f"weight={term['weight']:.12f} value={term['value']:.12f}"
            )
