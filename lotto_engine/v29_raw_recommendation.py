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
    FEATURE_NAMES as BASE_FEATURE_NAMES,
    TRAIN_NEGATIVES_PER_TARGET,
    _advance_history,
    _empty_history_state,
    _feature_context,
    _sample_fair_candidates,
    candidate_feature_vector,
)
from .v29_quadratic_reverse_ranking import (
    FEATURE_NAMES as QUADRATIC_FEATURE_NAMES,
    MODEL_VERSION as V29_MODEL_VERSION,
    RIDGE_LAMBDA,
    _add_solved_target,
    _empty_training_stats,
    fit_quadratic_pairwise_ridge,
    quadratic_basis_from_base,
    score_quadratic_vector,
)

PERSONAL_MODEL_VERSION = "v29_quadratic_raw_personal_v1"
SELECTION_STRATEGY = "v29_raw_model_score_top_k"
DEFAULT_VALIDATION_JSON = Path("data/cache/v29_quadratic_reverse_ranking_screen.json")
VALIDATION_ROLE = "diagnostic_only_not_used_in_ranking"


def _seeded_fair_tiebreak(numbers: Iterable[int], seed: int) -> int:
    encoded = f"{int(seed)}:" + ",".join(str(int(number)) for number in numbers)
    digest = hashlib.blake2b(encoded.encode("ascii"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


def build_latest_v29_model(
    df: pd.DataFrame,
    start_index: int = BACKTEST_START_INDEX,
    train_negatives_per_target: int = TRAIN_NEGATIVES_PER_TARGET,
    ridge_lambda: float = RIDGE_LAMBDA,
) -> dict:
    """Fit the frozen v2.9 ranker for the next unknown draw.

    The training sequence is identical to the v2.9 audit: each historical target is
    converted into training evidence only after its feature context has been built
    from strictly earlier draws. For the next unknown draw, all completed draws are
    therefore valid solved training evidence.
    """
    start_index = int(start_index)
    train_negatives_per_target = int(train_negatives_per_target)
    if len(df) <= start_index:
        raise ValueError("v2.9 raw recommendation requires draws after start_index")
    if train_negatives_per_target <= 0:
        raise ValueError("train_negatives_per_target must be positive")

    state = _empty_history_state()
    for idx in range(start_index):
        _advance_history(state, tuple(row_numbers(df.iloc[idx])))

    stats = _empty_training_stats()
    for idx in range(start_index, len(df)):
        actual = tuple(row_numbers(df.iloc[idx]))
        round_no = int(df.iloc[idx][ROUND_COLUMN])
        context = _feature_context(state)
        actual_vector = quadratic_basis_from_base(candidate_feature_vector(actual, context))

        # Exact frozen v2.9 training-negative seed path.
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
            quadratic_basis_from_base(candidate_feature_vector(candidate, context))
            for candidate in fair_train
        ]
        _add_solved_target(stats, actual_vector, negative_vectors)
        _advance_history(state, actual)

    model = fit_quadratic_pairwise_ridge(stats, ridge_lambda=float(ridge_lambda))
    next_context = _feature_context(state)
    return {
        "model": model,
        "context": next_context,
        "history_draws": int(state["history_draws"]),
        "solved_targets": int(stats["solved_targets"]),
        "pair_rows": int(stats["pair_rows"]),
        "start_index": start_index,
        "train_negatives_per_target": train_negatives_per_target,
        "ridge_lambda": float(ridge_lambda),
    }


def raw_v29_score(numbers: Iterable[int], model: dict, context: dict) -> float:
    base = candidate_feature_vector(numbers, context)
    quadratic = quadratic_basis_from_base(base)
    return score_quadratic_vector(quadratic, model)


def explain_v29_candidate(numbers: Iterable[int], model: dict, context: dict) -> dict:
    nums = tuple(sorted(int(number) for number in numbers))
    base = candidate_feature_vector(nums, context)
    quadratic = quadratic_basis_from_base(base)
    weights = np.asarray(model["effective_weights"], dtype=float)
    contributions = weights * quadratic
    raw_score = float(np.sum(contributions))

    order = np.argsort(np.abs(contributions))[::-1][:10]
    strongest_terms = [
        {
            "term": str(QUADRATIC_FEATURE_NAMES[int(index)]),
            "value": float(quadratic[int(index)]),
            "weight": float(weights[int(index)]),
            "contribution": float(contributions[int(index)]),
        }
        for index in order
    ]
    return {
        "numbers": list(nums),
        "raw_model_score": raw_score,
        "ranking_score": raw_score,
        "score_origin": PERSONAL_MODEL_VERSION,
        "base_features": {
            name: float(base[index])
            for index, name in enumerate(BASE_FEATURE_NAMES)
        },
        "strongest_quadratic_terms": strongest_terms,
    }


def top_raw_v29_stream(
    candidates: Iterable[Iterable[int]],
    model: dict,
    context: dict,
    top_k: int,
    seed: int,
    progress_every: int = 0,
) -> tuple[list[dict], int]:
    """Rank candidates only by the raw frozen v2.9 model score."""
    top_k = int(top_k)
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    heap: list[tuple[float, int, tuple[int, ...]]] = []
    evaluated = 0
    for evaluated, candidate in enumerate(candidates, 1):
        nums = tuple(sorted(int(number) for number in candidate))
        score = raw_v29_score(nums, model, context)
        tie_break = _seeded_fair_tiebreak(nums, seed)
        entry = (score, tie_break, nums)
        if len(heap) < top_k:
            heapq.heappush(heap, entry)
        elif entry[:2] > heap[0][:2]:
            heapq.heapreplace(heap, entry)

        if progress_every and evaluated % int(progress_every) == 0:
            print(f"v2.9 raw ranking progress: {evaluated:,} candidates evaluated")

    selected_entries = sorted(heap, key=lambda value: (value[0], value[1]), reverse=True)
    selected = [explain_v29_candidate(entry[2], model, context) for entry in selected_entries]
    for rank, item in enumerate(selected, 1):
        item["rank"] = rank
        item["strategy"] = "V2.9 RAW QUADRATIC REVERSE RANKING"
    return selected, evaluated


def load_validation_diagnostics(path: str | Path | None) -> dict:
    """Load historical validation as metadata only; never alter candidate scores."""
    if path is None:
        return {
            "status": "not_loaded",
            "role": VALIDATION_ROLE,
            "ranking_influenced": False,
        }

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


def generate_v29_raw_recommendations(
    seed_offset: int = 0,
    candidate_count: int = DEFAULT_CANDIDATE_COUNT,
    exhaustive: bool = DEFAULT_EXHAUSTIVE_RECOMMENDATION,
    top_k: int = TOP_K_RECOMMENDATIONS,
    progress_every: int = 0,
    validation_json: str | Path | None = DEFAULT_VALIDATION_JSON,
) -> dict:
    """Personal-use next-draw ranking that preserves raw v2.9 model output."""
    df = load_lotto_data()
    profile = build_profile(df)
    latest_draw = int(profile["latest_round"])
    target_draw = latest_draw + 1
    seed = make_seed(latest_draw, int(seed_offset))

    fitted = build_latest_v29_model(df)
    model = fitted["model"]
    context = fitted["context"]

    if exhaustive:
        candidates = iter_all_combinations()
        mode = "exhaustive_all_8,145,060"
    else:
        candidates = generate_candidates(int(candidate_count), seed)
        mode = "sampled_candidates"

    recommendations, evaluated_count = top_raw_v29_stream(
        candidates,
        model,
        context,
        int(top_k),
        seed,
        progress_every=int(progress_every),
    )
    for item in recommendations:
        _attach_structure_metadata(item, str(profile["latest_pattern_type"]))

    validation = load_validation_diagnostics(validation_json)
    return {
        "meta": {
            "model_version": PERSONAL_MODEL_VERSION,
            "source_model_version": V29_MODEL_VERSION,
            "latest_draw": latest_draw,
            "target_draw": target_draw,
            "historical_draw_count": int(len(df)),
            "recommendation_mode": mode,
            "evaluated_count": int(evaluated_count),
            "top_k": int(top_k),
            "seed": int(seed),
            "seed_offset": int(seed_offset),
            "candidate_count": int(candidate_count),
            "selection_strategy": SELECTION_STRATEGY,
            "score_name": "raw_model_score",
            "score_scale": "unbounded_raw_linear_predictor_w_dot_phi_x",
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
                "basis_width": len(QUADRATIC_FEATURE_NAMES),
            },
            "historical_validation": validation,
            "score_disclaimer": (
                "raw_model_score는 v2.9가 계산한 w^T phi(x) 원값이며 실제 당첨확률이 아닙니다. "
                "역사 검증 결과는 별도 진단 정보로만 표시되고 순위 점수에는 곱하거나 50점으로 "
                "중립화하지 않습니다."
            ),
        },
        "recommendations": recommendations,
    }


def public_v29_raw_payload(payload: dict) -> dict:
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
                "score": float(item["raw_model_score"]),
                "raw_model_score": float(item["raw_model_score"]),
                "pattern_type": item.get("pattern_type", "unknown"),
                "base_features": dict(item["base_features"]),
                "strongest_quadratic_terms": list(item["strongest_quadratic_terms"]),
                "structure_metadata": dict(item.get("structure_metadata", {})),
            }
            for item in payload["recommendations"]
        ],
    }


def write_v29_raw_json(payload: dict, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(public_v29_raw_payload(payload), handle, ensure_ascii=False, indent=2)
    return path.resolve()


def print_v29_raw_recommendations(payload: dict) -> None:
    meta = payload["meta"]
    print("=" * 92)
    print("LOTTO STAT ENGINE v2.9 - RAW QUADRATIC REVERSE RANKING (PERSONAL RESEARCH)")
    print("=" * 92)
    print(f"latest reflected draw: {meta['latest_draw']}")
    print(f"target draw: {meta['target_draw']}")
    print(f"evaluation mode: {meta['recommendation_mode']}")
    print(f"evaluated combinations: {meta['evaluated_count']:,}")
    print(f"selection: {meta['selection_strategy']}")
    print("validation: diagnostic only; raw model ranking is never neutralized")
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
        print(
            f"raw_model_score={item['raw_model_score']:.12f} "
            f"type={item.get('pattern_type', 'unknown')}"
        )
        for term in item["strongest_quadratic_terms"][:5]:
            print(
                f"  {term['term']}: contribution={term['contribution']:.12f} "
                f"weight={term['weight']:.12f} value={term['value']:.12f}"
            )
