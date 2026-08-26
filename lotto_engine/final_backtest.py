from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import random

import numpy as np
import pandas as pd

from .backtest import run_walk_forward_backtest
from .candidates import generate_candidates
from .config import BASE_WEIGHTS
from .loader import NUMBER_COLUMNS, ROUND_COLUMN, row_numbers
from .mixed_scoring import (
    build_all_combination_baseline,
    build_draw_structure_records,
    build_mixed_profile,
    score_mixed_record,
    structure_record,
)
from .mixed_subtypes import (
    build_exact_mixed_subtype_baseline,
    build_historical_subtype_records,
    mixed_subtype_fit_components,
    signature_family,
    signature_family_diagnostics,
    subtype_information_diagnostics,
    suggest_mixed_signature_allocation,
)
from .profiles import build_profile
from .recommender import (
    PATTERN_TYPES,
    _portfolio_allocation,
    _select_diverse_mixed,
    _select_diverse_static,
)
from .scoring import score_static_candidate


@dataclass(frozen=True)
class WalkForwardConfig:
    start_index: int = 300
    candidate_count: int = 2000
    top_k: int = 10
    max_targets: int | None = None
    calibration_index: int = 300


def _largest_remainder(probabilities: dict[str, float], total: int) -> dict[str, int]:
    raw = {key: total * float(probabilities.get(key, 0.0)) for key in PATTERN_TYPES}
    allocation = {key: int(value) for key, value in raw.items()}
    missing = total - sum(allocation.values())
    if missing > 0:
        order = sorted(PATTERN_TYPES, key=lambda key: raw[key] - allocation[key], reverse=True)
        for key in order[:missing]:
            allocation[key] += 1
    return allocation


def _score_candidate_record(record: dict, profile: dict, mixed_profile: dict, weights: dict[str, float]) -> dict:
    if record["pattern_type"] == "mixed":
        scored = score_mixed_record(record, mixed_profile)
        static_score = float(scored.get("base_score", scored["mixed_slot_score"]))
        dynamic_score = float(scored["mixed_slot_score"])
    else:
        scored = score_static_candidate(list(record["numbers"]), profile, weights)
        static_score = float(scored["static_score"])
        dynamic_score = static_score

    return {
        **scored,
        "numbers": list(record["numbers"]),
        "pattern_type": record["pattern_type"],
        "structure_record": record,
        "subtype_signature": record["subtype_signature"],
        "subtype_tags": list(record["subtype_tags"]),
        "primary_subtype": record["primary_subtype"],
        "static_rank_score": static_score,
        "dynamic_rank_score": dynamic_score,
        "static_score": static_score,
        "mixed_slot_score": dynamic_score,
    }


def _top_by_budget(items: list[dict], allocation: dict[str, int], score_key: str) -> list[dict]:
    selected: list[dict] = []
    for candidate_type in PATTERN_TYPES:
        pool = [item for item in items if item["pattern_type"] == candidate_type]
        pool.sort(key=lambda item: float(item[score_key]), reverse=True)
        selected.extend(pool[: allocation.get(candidate_type, 0)])
    return selected


def _family_guided_mixed(
    mixed_pool: list[dict],
    quota: int,
    family_allocation: dict[str, int],
    subtype_diagnostics: dict[str, dict],
    family_diagnostics: dict[str, dict],
) -> list[dict]:
    remaining = list(mixed_pool)
    selected: list[dict] = []
    selected_families: Counter = Counter()
    while remaining and len(selected) < quota:
        choices = []
        for index, item in enumerate(remaining):
            fit = mixed_subtype_fit_components(
                item, subtype_diagnostics, family_diagnostics,
                family_allocation, selected_families,
            )
            family = fit["selected_family"]
            overfill = 3.0 if selected_families[family] >= family_allocation.get(family, 0) else 0.0
            utility = float(item["dynamic_rank_score"]) + 6.0 * fit["mixed_subtype_fit_score"] / 100.0 - overfill
            choices.append((utility, float(item["dynamic_rank_score"]), -index, index, fit))
        _, _, _, best_index, fit = max(choices)
        item = remaining.pop(best_index)
        item = dict(item)
        item.update(fit)
        selected.append(item)
        selected_families[fit["selected_family"]] += 1
    return selected


def _family_variant(
    items: list[dict],
    allocation: dict[str, int],
    family_allocation: dict[str, int],
    subtype_diagnostics: dict[str, dict],
    family_diagnostics: dict[str, dict],
) -> list[dict]:
    selected: list[dict] = []
    for candidate_type in ("normal", "outlier"):
        pool = [item for item in items if item["pattern_type"] == candidate_type]
        pool.sort(key=lambda item: float(item["dynamic_rank_score"]), reverse=True)
        selected.extend(pool[: allocation.get(candidate_type, 0)])
    mixed_pool = [item for item in items if item["pattern_type"] == "mixed"]
    selected.extend(_family_guided_mixed(
        mixed_pool,
        allocation.get("mixed", 0),
        family_allocation,
        subtype_diagnostics,
        family_diagnostics,
    ))
    return selected


def _final_variant(
    items: list[dict],
    allocation: dict[str, int],
    family_allocation: dict[str, int],
    subtype_diagnostics: dict[str, dict],
    family_diagnostics: dict[str, dict],
) -> list[dict]:
    normal_pool = [dict(item) for item in items if item["pattern_type"] == "normal"]
    outlier_pool = [dict(item) for item in items if item["pattern_type"] == "outlier"]
    mixed_pool = [dict(item) for item in items if item["pattern_type"] == "mixed"]
    return [
        *_select_diverse_static(normal_pool, allocation.get("normal", 0)),
        *_select_diverse_mixed(
            mixed_pool,
            allocation.get("mixed", 0),
            family_allocation,
            subtype_diagnostics,
            family_diagnostics,
        ),
        *_select_diverse_static(outlier_pool, allocation.get("outlier", 0)),
    ]


def _portfolio_metrics(portfolio: list[dict], actual_numbers: list[int]) -> dict:
    actual = set(actual_numbers)
    hits = [len(actual & set(item["numbers"])) for item in portfolio]
    union = set(number for item in portfolio for number in item["numbers"])
    return {
        "best_hit": max(hits, default=0),
        "mean_hit": float(np.mean(hits)) if hits else 0.0,
        "coverage": len(actual & union) / 6.0 if actual else 0.0,
        "hit3": int(any(hit >= 3 for hit in hits)),
        "hit4": int(any(hit >= 4 for hit in hits)),
        "hit5": int(any(hit >= 5 for hit in hits)),
        "hit6": int(any(hit >= 6 for hit in hits)),
    }


def _summary(rows: list[dict]) -> dict:
    if not rows:
        return {
            "tests": 0,
            "mean_best_hit": 0.0,
            "mean_ticket_hit": 0.0,
            "mean_coverage": 0.0,
            "hit3_rate": 0.0,
            "hit4_rate": 0.0,
            "hit5_rate": 0.0,
            "hit6_rate": 0.0,
        }
    return {
        "tests": len(rows),
        "mean_best_hit": float(np.mean([row["best_hit"] for row in rows])),
        "mean_ticket_hit": float(np.mean([row["mean_hit"] for row in rows])),
        "mean_coverage": float(np.mean([row["coverage"] for row in rows])),
        "hit3_rate": float(np.mean([row["hit3"] for row in rows])),
        "hit4_rate": float(np.mean([row["hit4"] for row in rows])),
        "hit5_rate": float(np.mean([row["hit5"] for row in rows])),
        "hit6_rate": float(np.mean([row["hit6"] for row in rows])),
    }


def _model_summary(rows: list[dict], model: str) -> dict:
    model_rows = [row[model] for row in rows]
    return {
        "overall": _summary(model_rows),
        "recent_300": _summary(model_rows[-300:]),
        "recent_100": _summary(model_rows[-100:]),
    }


def _verdict(rows: list[dict]) -> dict:
    better = worse = equal = 0
    for row in rows:
        final_hit = row["final"]["best_hit"]
        static_hit = row["static"]["best_hit"]
        if final_hit > static_hit:
            better += 1
        elif final_hit < static_hit:
            worse += 1
        else:
            equal += 1
    final_summary = _summary([row["final"] for row in rows])
    static_summary = _summary([row["static"] for row in rows])
    if better > worse and final_summary["mean_best_hit"] >= static_summary["mean_best_hit"]:
        label = "IMPROVED"
    elif worse > better and final_summary["mean_best_hit"] <= static_summary["mean_best_hit"]:
        label = "DEGRADED"
    else:
        label = "NEUTRAL/MIXED"
    return {
        "label": label,
        "final_better_targets": better,
        "final_worse_targets": worse,
        "equal_targets": equal,
    }


def run_final_walk_forward_backtest(
    df: pd.DataFrame,
    config: WalkForwardConfig | None = None,
) -> dict:
    config = config or WalkForwardConfig()
    start_index = max(config.start_index, config.calibration_index)
    if len(df) <= start_index:
        raise ValueError(f"Final backtest requires more than {start_index} draws.")

    # Strict anti-leakage policy: feature weights are calibrated only on the prefix
    # before the evaluated walk-forward region and then frozen for every target.
    calibration_df = df.iloc[: config.calibration_index].copy()
    try:
        weights = run_walk_forward_backtest(calibration_df)["final_weights"]
        weight_source = f"prefix_calibrated_first_{config.calibration_index}"
    except Exception:
        weights = dict(BASE_WEIGHTS)
        weight_source = "fixed_BASE_WEIGHTS_fallback"

    all_distribution = build_all_combination_baseline()
    subtype_baseline = build_exact_mixed_subtype_baseline()
    all_records = build_draw_structure_records(df)
    all_subtype_records = build_historical_subtype_records(df)

    target_indices = list(range(start_index, len(df)))
    if config.max_targets is not None:
        target_indices = target_indices[-config.max_targets :]

    rows: list[dict] = []
    type_budget_actual_slots: Counter = Counter()
    type_budget_total_slots: Counter = Counter()
    mixed_family_hit = 0
    mixed_family_tests = 0

    for idx in target_indices:
        history_df = df.iloc[:idx].copy()
        history_records = all_records[:idx]
        history_subtypes = all_subtype_records[:idx]
        profile = build_profile(history_df)
        mixed_profile = build_mixed_profile(history_records, all_distribution)
        subtype_diagnostics = subtype_information_diagnostics(history_subtypes, subtype_baseline)
        family_diagnostics = signature_family_diagnostics(history_subtypes, subtype_baseline)

        dynamic_allocation = _portfolio_allocation(profile, config.top_k)
        static_allocation = _largest_remainder(profile["pattern_type_probs"], config.top_k)
        family_allocation = suggest_mixed_signature_allocation(
            history_subtypes,
            slots=dynamic_allocation.get("mixed", 0),
            baseline=subtype_baseline,
        )

        target_round = int(df.iloc[idx][ROUND_COLUMN])
        seed = target_round * 1_000_003 + config.candidate_count
        candidate_lists = generate_candidates(config.candidate_count, seed)
        items: list[dict] = []
        previous_type = profile["latest_pattern_type"]
        for candidate in candidate_lists:
            record = structure_record(candidate, previous_type)
            items.append(_score_candidate_record(record, profile, mixed_profile, weights))

        static_portfolio = _top_by_budget(items, static_allocation, "static_rank_score")
        dynamic_portfolio = _top_by_budget(items, dynamic_allocation, "dynamic_rank_score")
        family_portfolio = _family_variant(
            items, dynamic_allocation, family_allocation,
            subtype_diagnostics, family_diagnostics,
        )
        final_portfolio = _final_variant(
            items, dynamic_allocation, family_allocation,
            subtype_diagnostics, family_diagnostics,
        )

        actual_numbers = row_numbers(df.iloc[idx])
        target_record = all_records[idx]
        actual_type = target_record["pattern_type"]
        type_budget_actual_slots[actual_type] += dynamic_allocation.get(actual_type, 0)
        type_budget_total_slots[actual_type] += config.top_k

        if actual_type == "mixed":
            mixed_family_tests += 1
            actual_family = signature_family(target_record["subtype_signature"])
            selected_families = {
                item.get("selected_family", signature_family(item["subtype_signature"]))
                for item in final_portfolio
                if item["pattern_type"] == "mixed"
            }
            if actual_family in selected_families:
                mixed_family_hit += 1

        rows.append({
            "round": target_round,
            "actual_type": actual_type,
            "static": _portfolio_metrics(static_portfolio, actual_numbers),
            "dynamic": _portfolio_metrics(dynamic_portfolio, actual_numbers),
            "dynamic_family": _portfolio_metrics(family_portfolio, actual_numbers),
            "final": _portfolio_metrics(final_portfolio, actual_numbers),
        })

    models = ("static", "dynamic", "dynamic_family", "final")
    return {
        "version": "FINAL_TYPE_SEPARATED_WALK_FORWARD",
        "candidate_count": config.candidate_count,
        "top_k": config.top_k,
        "start_index": start_index,
        "tested_targets": len(rows),
        "weight_source": weight_source,
        "models": {model: _model_summary(rows, model) for model in models},
        "verdict": _verdict(rows),
        "type_budget": {
            candidate_type: {
                "mean_slots_given_actual_type": (
                    type_budget_actual_slots[candidate_type]
                    / max(1, sum(row["actual_type"] == candidate_type for row in rows))
                )
            }
            for candidate_type in PATTERN_TYPES
        },
        "mixed_family": {
            "tests": mixed_family_tests,
            "actual_family_present_rate": mixed_family_hit / mixed_family_tests if mixed_family_tests else 0.0,
        },
        "rows": rows,
    }
