from __future__ import annotations

import json
import math
from collections import Counter
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd

from .config import CACHE_DIR, ROUND_COLUMN
from .features import pattern_type
from .loader import row_numbers
from .mixed_subtypes import (
    build_dynamic_family_model,
    family_dynamic_scores,
    signature_family,
    subtype_record,
)

MIXED_MODEL_VERSION = "v2.3.1"
BASELINE_CACHE_VERSION = "v2.3"
ALPHA = 1.0
MIN_INTERACTION_SUPPORT = 3
BASELINE_CACHE_PATH = CACHE_DIR / "mixed_all_combination_baseline_v23.json"

SINGLE_FEATURES = (
    "sum_bin",
    "odd_even",
    "range_bin",
    "max_gap_bin",
    "min_gap_bin",
    "consecutive_pair_count",
    "ending_duplicate_count",
    "empty_decade_section_count",
    "section_distribution",
    "low_mid_high_distribution",
    "extreme_count",
    "extreme_signature",
)
INTERACTIONS = (
    ("odd_even", "sum_bin"),
    ("range_bin", "max_gap_bin"),
    ("ending_duplicate_count", "consecutive_pair_count"),
    ("empty_decade_section_count", "sum_bin"),
    ("extreme_count", "normal_backbone_bin"),
    ("odd_even", "range_bin"),
)
BACKBONE_FEATURES = (
    "sum",
    "odd_count",
    "number_range",
    "max_gap",
    "min_gap",
    "consecutive_pair_count",
    "ending_duplicate_count",
    "empty_decade_section_count",
    "section_spread",
    "low_mid_high_spread",
)
MIXED_SCORE_WEIGHTS = {
    "mixed_lift_score": 0.35,
    "mixed_interaction_score": 0.25,
    "normal_backbone_score": 0.20,
    "controlled_extreme_score": 0.15,
    "recency_consistency_score": 0.05,
}


def _bin(value: int, boundaries: tuple[int, ...]) -> str:
    for upper in boundaries:
        if value <= upper:
            return f"le_{upper}"
    return f"gt_{boundaries[-1]}"


def _key(value) -> str:
    if isinstance(value, tuple):
        return "|".join(str(item) for item in value)
    return str(value)


def structure_record(
    numbers: list[int] | tuple[int, ...],
    previous_pattern_type: str | None = None,
) -> dict:
    nums = sorted(int(number) for number in numbers)
    gaps = [b - a for a, b in zip(nums, nums[1:])]
    endings = Counter(number % 10 for number in nums)
    odd_count = sum(number % 2 for number in nums)
    total = sum(nums)
    number_range = nums[-1] - nums[0]
    max_gap = max(gaps)
    min_gap = min(gaps)
    consecutive_pairs = sum(gap == 1 for gap in gaps)
    ending_duplicates = sum(count - 1 for count in endings.values() if count > 1)
    sections = tuple(sum(lo <= number <= hi for number in nums) for lo, hi in (
        (1, 10), (11, 20), (21, 30), (31, 40), (41, 45)
    ))
    lmh = tuple(sum(lo <= number <= hi for number in nums) for lo, hi in (
        (1, 15), (16, 30), (31, 45)
    ))
    empty_sections = sum(count == 0 for count in sections)

    flags = {
        "odd_even_extreme": odd_count in {0, 1, 5, 6},
        "sum_low": total <= 105,
        "sum_high": total >= 170,
        "sum_extreme": total <= 90 or total >= 185,
        "range_narrow": number_range <= 20,
        "range_wide": number_range >= 41,
        "max_gap_extreme": max_gap >= 20,
        "has_consecutive_pairs": consecutive_pairs > 0,
        "ending_duplicates": ending_duplicates >= 2,
        "empty_sections": empty_sections >= 2,
    }
    signature = tuple(key for key, active in flags.items() if active) or ("none",)
    feature_proxy = {
        "odd_even_extreme": int(flags["odd_even_extreme"]),
        "sum_extreme": int(flags["sum_extreme"]),
        "range_narrow": int(flags["range_narrow"]),
        "range_wide": int(flags["range_wide"]),
        "max_gap_extreme": int(flags["max_gap_extreme"]),
        "has_consecutive_pairs": int(flags["has_consecutive_pairs"]),
        "empty_section_count": empty_sections,
    }
    # A neutral coverage descriptor used only as an interaction category. It
    # does not encode an ideal sum, parity, or range.
    occupied_sections = 5 - empty_sections
    occupied_lmh = sum(count > 0 for count in lmh)
    normal_backbone_bin = f"coverage_{occupied_sections}_{occupied_lmh}"

    record = {
        "numbers": tuple(nums),
        "sum": total,
        "sum_bin": _bin(total, (89, 104, 119, 134, 149, 164, 179)),
        "odd_count": odd_count,
        "odd_even": f"{odd_count}:{6 - odd_count}",
        "number_range": number_range,
        "range_bin": _bin(number_range, (20, 25, 30, 35, 40)),
        "max_gap": max_gap,
        "max_gap_bin": _bin(max_gap, (5, 9, 14, 19)),
        "min_gap": min_gap,
        "min_gap_bin": _bin(min_gap, (1, 2, 4)),
        "consecutive_pair_count": consecutive_pairs,
        "ending_duplicate_count": ending_duplicates,
        "empty_decade_section_count": empty_sections,
        "section_distribution": sections,
        "low_mid_high_distribution": lmh,
        "section_spread": max(sections) - min(sections),
        "low_mid_high_spread": max(lmh) - min(lmh),
        "extreme_count": sum(flags.values()),
        "extreme_signature": signature,
        "normal_backbone_bin": normal_backbone_bin,
        "pattern_type": pattern_type(feature_proxy),
        "previous_pattern_type": previous_pattern_type,
    }
    subtype = subtype_record(nums)
    record.update({key: subtype[key] for key in (
        "subtype_tags", "primary_subtype", "subtype_signature", "cluster_shape", "gap_shape"
    )})
    return record


def build_draw_structure_records(df: pd.DataFrame) -> list[dict]:
    records: list[dict] = []
    previous_type = None
    for _, row in df.iterrows():
        record = structure_record(row_numbers(row), previous_type)
        record["draw_no"] = int(row[ROUND_COLUMN])
        records.append(record)
        previous_type = record["pattern_type"]
    return records


def _empty_distribution() -> dict:
    return {
        "total": 0,
        "features": {feature: Counter() for feature in SINGLE_FEATURES},
        "interactions": {
            f"{left}__{right}": Counter() for left, right in INTERACTIONS
        },
    }


def _accumulate(distribution: dict, record: dict) -> None:
    distribution["total"] += 1
    for feature in SINGLE_FEATURES:
        distribution["features"][feature][_key(record[feature])] += 1
    for left, right in INTERACTIONS:
        name = f"{left}__{right}"
        distribution["interactions"][name][
            f"{_key(record[left])}::{_key(record[right])}"
        ] += 1


def _serialize_distribution(distribution: dict) -> dict:
    return {
        "version": BASELINE_CACHE_VERSION,
        "total": distribution["total"],
        "features": {
            key: dict(counter) for key, counter in distribution["features"].items()
        },
        "interactions": {
            key: dict(counter) for key, counter in distribution["interactions"].items()
        },
    }


def _deserialize_distribution(payload: dict) -> dict:
    return {
        "total": int(payload["total"]),
        "features": {
            key: Counter(values) for key, values in payload["features"].items()
        },
        "interactions": {
            key: Counter(values) for key, values in payload["interactions"].items()
        },
    }


@lru_cache(maxsize=1)
def build_all_combination_baseline(
    cache_path: Path = BASELINE_CACHE_PATH,
) -> dict:
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if (
            payload.get("version") == BASELINE_CACHE_VERSION
            and int(payload.get("total", 0)) == math.comb(45, 6)
        ):
            return _deserialize_distribution(payload)

    distribution = _empty_distribution()
    for combo in combinations(range(1, 46), 6):
        _accumulate(distribution, structure_record(combo))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(_serialize_distribution(distribution), handle, ensure_ascii=False)
    return distribution


def _robust_stats(records: list[dict]) -> dict:
    stats = {}
    for feature in BACKBONE_FEATURES:
        values = [float(record[feature]) for record in records]
        center = float(median(values))
        mad = float(median(abs(value - center) for value in values))
        stats[feature] = {"median": center, "mad": mad}
    return stats


def _record_distribution(records: list[dict]) -> dict:
    distribution = _empty_distribution()
    for record in records:
        _accumulate(distribution, record)
    return distribution


def build_mixed_profile(
    records: list[dict],
    all_distribution: dict | None = None,
) -> dict:
    baseline = all_distribution or build_all_combination_baseline()
    mixed = [record for record in records if record["pattern_type"] == "mixed"]
    if not mixed:
        raise ValueError("At least one historical mixed draw is required.")
    full = _record_distribution(mixed)
    recent_300_records = [
        record for record in records[-min(300, len(records)):]
        if record["pattern_type"] == "mixed"
    ]
    recent_100_records = [
        record for record in records[-min(100, len(records)):]
        if record["pattern_type"] == "mixed"
    ]
    profile = {
        "all": baseline,
        "full": full,
        "recent_300": _record_distribution(recent_300_records),
        "recent_100": _record_distribution(recent_100_records),
        "backbone": _robust_stats(mixed),
        "mixed_total": len(mixed),
    }
    # Draw-aware callers get one dynamic context for the entire candidate
    # stream. Legacy/synthetic callers without draw numbers retain v2.3
    # scoring behavior and the existing public function signatures.
    if records and all("draw_no" in record for record in records):
        profile["dynamic_markov_decay"] = build_dynamic_family_model(records)
    return profile


def _log_lift(
    value: str,
    feature: str,
    mixed_distribution: dict,
    all_distribution: dict,
    interaction: bool = False,
) -> float:
    group = "interactions" if interaction else "features"
    mixed_counter = mixed_distribution[group][feature]
    all_counter = all_distribution[group][feature]
    bins = max(1, len(set(mixed_counter) | set(all_counter)))
    mixed_probability = (
        mixed_counter.get(value, 0) + ALPHA
    ) / (mixed_distribution["total"] + ALPHA * bins)
    all_probability = (
        all_counter.get(value, 0) + ALPHA
    ) / (all_distribution["total"] + ALPHA * bins)
    return math.log(mixed_probability / all_probability)


def _lift_to_score(raw_lift: float) -> float:
    return float(100.0 / (1.0 + math.exp(-1.5 * max(-20.0, min(20.0, raw_lift)))))


def _single_lifts(record: dict, profile: dict, distribution_name: str = "full") -> list[float]:
    distribution = profile[distribution_name]
    if distribution["total"] <= 0:
        return []
    return [
        _log_lift(
            _key(record[feature]), feature, distribution, profile["all"]
        )
        for feature in SINGLE_FEATURES
    ]


def mixed_lift_score(record: dict, profile: dict) -> float:
    lifts = _single_lifts(record, profile)
    return _lift_to_score(float(np.mean(lifts)) if lifts else 0.0)


def mixed_interaction_score(record: dict, profile: dict) -> float:
    lifts = []
    for left, right in INTERACTIONS:
        name = f"{left}__{right}"
        value = f"{_key(record[left])}::{_key(record[right])}"
        if profile["full"]["interactions"][name].get(value, 0) < MIN_INTERACTION_SUPPORT:
            continue
        lifts.append(_log_lift(
            value, name, profile["full"], profile["all"], interaction=True
        ))
    return _lift_to_score(float(np.mean(lifts)) if lifts else 0.0)


def normal_backbone_score(record: dict, profile: dict) -> float:
    dimension_scores = []
    for feature in BACKBONE_FEATURES:
        stats = profile["backbone"][feature]
        scale = max(1.0, 1.4826 * stats["mad"])
        robust_z = min(4.0, abs(float(record[feature]) - stats["median"]) / scale)
        # Continuous clipped robust-z scaling preserves ordering throughout the
        # historical backbone instead of saturating every in-band value at 100.
        dimension_scores.append(100.0 * math.exp(-0.22 * robust_z))
    return float(np.mean(dimension_scores))


def controlled_extreme_score(record: dict, profile: dict) -> float:
    lifts = [
        _log_lift(_key(record["extreme_count"]), "extreme_count", profile["full"], profile["all"]),
        _log_lift(
            _key(record["extreme_signature"]),
            "extreme_signature",
            profile["full"],
            profile["all"],
        ),
    ]
    empirical = _lift_to_score(float(np.mean(lifts)))
    # Signature lift alone is identical for every candidate with that category.
    # Add continuous, clipped robust-z severity so candidates with the same
    # signature retain empirical candidate-level discrimination.
    severity_dimensions = ("sum", "number_range", "max_gap", "empty_decade_section_count")
    severities = []
    for feature in severity_dimensions:
        stats = profile["backbone"][feature]
        scale = max(1.0, 1.4826 * stats["mad"])
        severities.append(min(4.0, abs(float(record[feature]) - stats["median"]) / scale))
    severity_score = 100.0 * float(np.mean(severities)) / 4.0
    return 0.75 * empirical + 0.25 * severity_score


def recency_consistency_score(record: dict, profile: dict) -> float:
    full = _single_lifts(record, profile, "full")
    if not full:
        return 50.0
    comparisons = []
    for name, weight in (("recent_300", 0.60), ("recent_100", 0.40)):
        recent = _single_lifts(record, profile, name)
        if not recent:
            continue
        agreement = float(np.mean([
            math.exp(-abs(current - historical))
            * (1.0 if current * historical >= 0 else 0.5)
            for current, historical in zip(recent, full)
        ]))
        comparisons.append((agreement, weight))
    if not comparisons:
        return 50.0
    total_weight = sum(weight for _, weight in comparisons)
    return 100.0 * sum(value * weight for value, weight in comparisons) / total_weight


def score_mixed_candidate(
    numbers: list[int] | tuple[int, ...],
    profile: dict,
    previous_pattern_type: str | None = None,
) -> dict:
    record = structure_record(numbers, previous_pattern_type)
    return score_mixed_record(record, profile)


def score_mixed_record(record: dict, profile: dict) -> dict:
    components = {
        "mixed_lift_score": mixed_lift_score(record, profile),
        "mixed_interaction_score": mixed_interaction_score(record, profile),
        "normal_backbone_score": normal_backbone_score(record, profile),
        "controlled_extreme_score": controlled_extreme_score(record, profile),
        "recency_consistency_score": recency_consistency_score(record, profile),
    }
    base_score = sum(
        MIXED_SCORE_WEIGHTS[key] * value for key, value in components.items()
    )
    slot_score = base_score
    dynamic = profile.get("dynamic_markov_decay")
    dynamic_components = {}
    if dynamic:
        cand_family = signature_family(record.get("subtype_signature", "none"))
        transition_score, momentum_score = family_dynamic_scores(cand_family, dynamic)
        slot_score = (
            0.70 * base_score
            + 0.15 * transition_score
            + 0.15 * momentum_score
        )
        dynamic_components = {
            "base_score": round(float(base_score), 4),
            "candidate_family": cand_family,
            "transition_lift_score": round(float(transition_score), 4),
            "momentum_lift_score": round(float(momentum_score), 4),
        }
    return {
        "numbers": list(record["numbers"]),
        "mixed_slot_score": round(float(slot_score), 4),
        **dynamic_components,
        **{key: round(float(value), 4) for key, value in components.items()},
        "extreme_count": record["extreme_count"],
        "extreme_signature": list(record["extreme_signature"]),
        "subtype_tags": list(record["subtype_tags"]),
        "primary_subtype": record["primary_subtype"],
        "subtype_signature": record["subtype_signature"],
        "cluster_shape": record["cluster_shape"],
        "gap_shape": record["gap_shape"],
        "pattern_type": record["pattern_type"],
        "structure_record": record,
    }
