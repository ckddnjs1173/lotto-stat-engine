from __future__ import annotations

import json
import math
from collections import Counter
from functools import lru_cache
from itertools import combinations
from pathlib import Path

from .config import CACHE_DIR, ROUND_COLUMN
from .features import pattern_type
from .loader import row_numbers

SUBTYPE_DEFINITION_VERSION = "v2.4.0"
SUBTYPE_BASELINE_PATH = CACHE_DIR / "mixed_subtype_baseline_v240.json"
SUBTYPE_ORDER = (
    "parity_extreme",
    "parity_skew",
    "gap_bridge",
    "section_hole",
    "sum_edge",
    "range_edge",
    "ending_duplicate",
    "consecutive_anchor",
    "low_high_split",
    "high_cluster",
    "low_cluster",
    "compound_mixed",
)
BACKGROUND_SUBTYPES = frozenset({"section_hole", "consecutive_anchor", "compound_mixed"})


def latest_target_draw(df) -> tuple[int, int]:
    latest = int(df[ROUND_COLUMN].max())
    return latest, latest + 1


def subtype_record(numbers) -> dict:
    nums = tuple(sorted(int(number) for number in numbers))
    if len(nums) != 6 or len(set(nums)) != 6 or nums[0] < 1 or nums[-1] > 45:
        raise ValueError("A subtype record requires six unique numbers from 1 through 45.")
    gaps = tuple(b - a for a, b in zip(nums, nums[1:]))
    odd_count = sum(number % 2 for number in nums)
    total = sum(nums)
    number_range = nums[-1] - nums[0]
    max_gap = max(gaps)
    sections = tuple(sum(lo <= number <= hi for number in nums) for lo, hi in (
        (1, 10), (11, 20), (21, 30), (31, 40), (41, 45)
    ))
    low, mid, high = (
        sum(1 <= number <= 15 for number in nums),
        sum(16 <= number <= 30 for number in nums),
        sum(31 <= number <= 45 for number in nums),
    )
    endings = Counter(number % 10 for number in nums)
    ending_duplicates = sum(count - 1 for count in endings.values() if count > 1)
    consecutive_pairs = sum(gap == 1 for gap in gaps)
    feature_proxy = {
        "odd_even_extreme": int(odd_count in {0, 1, 5, 6}),
        "sum_extreme": int(total <= 90 or total >= 185),
        "range_narrow": int(number_range <= 20),
        "range_wide": int(number_range >= 41),
        "max_gap_extreme": int(max_gap >= 20),
        "has_consecutive_pairs": int(consecutive_pairs > 0),
        "empty_section_count": sum(count == 0 for count in sections),
    }
    tags = []
    checks = {
        "parity_skew": odd_count in {1, 5},
        "parity_extreme": odd_count in {0, 6},
        "gap_bridge": max_gap >= 20,
        "section_hole": any(count == 0 for count in sections),
        "sum_edge": total <= 105 or total >= 170,
        "range_edge": number_range <= 20 or number_range >= 42,
        "ending_duplicate": ending_duplicates >= 2,
        "consecutive_anchor": consecutive_pairs >= 1,
        "low_high_split": low >= 1 and high >= 1 and mid <= 1,
        "high_cluster": high >= 3,
        "low_cluster": low >= 3,
    }
    tags.extend(tag for tag in SUBTYPE_ORDER if tag != "compound_mixed" and checks[tag])
    if len(tags) >= 3:
        tags.append("compound_mixed")
    ordered_tags = tuple(tag for tag in SUBTYPE_ORDER if tag in tags)
    primary = next((tag for tag in SUBTYPE_ORDER if tag != "compound_mixed" and tag in ordered_tags), "untyped_mixed")
    if max_gap >= 20:
        gap_shape = "bridge"
    elif consecutive_pairs >= 2:
        gap_shape = "anchored_multi"
    elif consecutive_pairs == 1:
        gap_shape = "anchored_single"
    else:
        gap_shape = "distributed"
    if high >= 3 and low >= 3:
        cluster_shape = "low_high_dual"
    elif high >= 3:
        cluster_shape = "high_cluster"
    elif low >= 3:
        cluster_shape = "low_cluster"
    elif mid <= 1 and low and high:
        cluster_shape = "low_high_split"
    else:
        cluster_shape = "distributed"
    return {
        "numbers": nums,
        "pattern_type": pattern_type(feature_proxy),
        "subtype_tags": ordered_tags,
        "primary_subtype": primary,
        "subtype_signature": "+".join(ordered_tags) if ordered_tags else "none",
        "cluster_shape": cluster_shape,
        "gap_shape": gap_shape,
    }


def build_historical_subtype_records(df) -> list[dict]:
    return [
        {**subtype_record(row_numbers(row)), "draw_no": int(row[ROUND_COLUMN])}
        for _, row in df.iterrows()
    ]


def _summarize(records: list[dict]) -> dict:
    mixed = [record for record in records if record["pattern_type"] == "mixed"]
    tags = Counter(tag for record in mixed for tag in record["subtype_tags"])
    primary = Counter(record["primary_subtype"] for record in mixed)
    signatures = Counter(record["subtype_signature"] for record in mixed)
    cooccurrence = Counter()
    for record in mixed:
        for left, right in combinations(record["subtype_tags"], 2):
            cooccurrence[f"{left}|{right}"] += 1
    return {
        "mixed_total": len(mixed),
        "tag_counts": dict(tags),
        "primary_counts": dict(primary),
        "signature_counts": dict(signatures),
        "cooccurrence_counts": dict(cooccurrence),
    }


@lru_cache(maxsize=1)
def build_exact_mixed_subtype_baseline(cache_path: Path = SUBTYPE_BASELINE_PATH) -> dict:
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("definition_version") == SUBTYPE_DEFINITION_VERSION and payload.get("combination_total") == math.comb(45, 6):
            return payload
    mixed_total = 0
    tags: Counter = Counter()
    primary: Counter = Counter()
    signatures: Counter = Counter()
    cooccurrence: Counter = Counter()
    for combo in combinations(range(1, 46), 6):
        record = subtype_record(combo)
        if record["pattern_type"] != "mixed":
            continue
        mixed_total += 1
        tags.update(record["subtype_tags"])
        primary[record["primary_subtype"]] += 1
        signatures[record["subtype_signature"]] += 1
        cooccurrence.update(f"{left}|{right}" for left, right in combinations(record["subtype_tags"], 2))
    summary = {
        "mixed_total": mixed_total,
        "tag_counts": dict(tags),
        "primary_counts": dict(primary),
        "signature_counts": dict(signatures),
        "cooccurrence_counts": dict(cooccurrence),
    }
    payload = {
        "definition_version": SUBTYPE_DEFINITION_VERSION,
        "combination_total": math.comb(45, 6),
        **summary,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
    return payload


def _allocate_scores(scores: dict[str, float], slots: int, max_per_item: int = 2) -> dict[str, int]:
    allocated = {key: 0 for key in scores}
    for _ in range(slots):
        eligible = [key for key in scores if allocated[key] < max_per_item]
        if not eligible:
            break
        chosen = max(eligible, key=lambda key: (scores[key] / (allocated[key] + 1), scores[key], key))
        allocated[chosen] += 1
    return {key: value for key, value in allocated.items() if value}


def subtype_information_diagnostics(records: list[dict], baseline: dict | None = None) -> dict[str, dict]:
    mixed = [record for record in records if record["pattern_type"] == "mixed"]
    recent_300 = [record for record in records[-300:] if record["pattern_type"] == "mixed"]
    recent_100 = [record for record in records[-100:] if record["pattern_type"] == "mixed"]
    baseline = baseline or build_exact_mixed_subtype_baseline()
    full_counts = Counter(tag for record in mixed for tag in record["subtype_tags"])
    counts_300 = Counter(tag for record in recent_300 for tag in record["subtype_tags"])
    counts_100 = Counter(tag for record in recent_100 for tag in record["subtype_tags"])
    result = {}
    for tag in SUBTYPE_ORDER:
        support = full_counts[tag]
        historical_ratio = support / max(1, len(mixed))
        baseline_ratio = baseline["tag_counts"].get(tag, 0) / max(1, baseline["mixed_total"])
        lift = historical_ratio / baseline_ratio if baseline_ratio else 0.0
        ratio_300 = counts_300[tag] / max(1, len(recent_300))
        ratio_100 = counts_100[tag] / max(1, len(recent_100))
        recent_ratio = 0.4 * ratio_300 + 0.6 * ratio_100
        recent_trend = recent_ratio / historical_ratio if historical_ratio else 0.0
        cooccurrence_support = (
            sum(sum(other in record["subtype_tags"] for other in SUBTYPE_ORDER if other not in {tag, "compound_mixed"}) > 0
                for record in mixed if tag in record["subtype_tags"]) / max(1, support)
        )
        signature_counts = Counter(record["subtype_signature"] for record in mixed if tag in record["subtype_tags"])
        signature_support = max(signature_counts.values(), default=0) / max(1, support)
        reliability = min(1.0, math.sqrt(support / 25.0))
        information_score = historical_ratio * abs(math.log2(max(lift, 1e-12))) * reliability
        result[tag] = {
            "historical_ratio": historical_ratio,
            "baseline_ratio": baseline_ratio,
            "lift": lift,
            "information_score": information_score,
            "support_count": support,
            "recent_300_ratio": ratio_300,
            "recent_100_ratio": ratio_100,
            "recent_trend": recent_trend,
            "cooccurrence_support": cooccurrence_support,
            "signature_support": signature_support,
        }
    return result


def suggest_mixed_subtype_allocation(records: list[dict], slots: int = 6, baseline: dict | None = None) -> dict[str, int]:
    if slots <= 0:
        return {}
    diagnostics = subtype_information_diagnostics(records, baseline)
    scores = {}
    for tag, values in diagnostics.items():
        positive_lift = max(0.0, math.log2(max(values["lift"], 1e-12)))
        trend = min(2.0, values["recent_trend"])
        score = (
            0.40 * values["information_score"]
            + 0.25 * values["historical_ratio"] * positive_lift
            + 0.15 * values["historical_ratio"] * trend
            + 0.10 * values["historical_ratio"] * values["cooccurrence_support"]
            + 0.10 * values["historical_ratio"] * values["signature_support"]
        )
        if tag in BACKGROUND_SUBTYPES:
            score *= 0.20
        if values["support_count"]:
            scores[tag] = score
    return _allocate_scores(scores, slots)


def suggest_mixed_signature_allocation(records: list[dict], slots: int = 6, baseline: dict | None = None) -> dict[str, int]:
    if slots <= 0:
        return {}
    baseline = baseline or build_exact_mixed_subtype_baseline()
    mixed = [record for record in records if record["pattern_type"] == "mixed"]
    recent_300 = [record for record in records[-300:] if record["pattern_type"] == "mixed"]
    recent_100 = [record for record in records[-100:] if record["pattern_type"] == "mixed"]

    def family(signature: str) -> str:
        return "+".join(tag for tag in signature.split("+") if tag != "compound_mixed")

    full = Counter(family(record["subtype_signature"]) for record in mixed)
    count_300 = Counter(family(record["subtype_signature"]) for record in recent_300)
    count_100 = Counter(family(record["subtype_signature"]) for record in recent_100)
    baseline_counts = Counter()
    for signature, count in baseline["signature_counts"].items():
        baseline_counts[family(signature)] += count
    minimum_support = max(3, math.ceil(len(mixed) * 0.005))
    scores = {}
    for name, support in full.items():
        if name == "none" or support < minimum_support:
            continue
        historical_ratio = support / len(mixed)
        baseline_ratio = baseline_counts[name] / max(1, baseline["mixed_total"])
        lift = historical_ratio / baseline_ratio if baseline_ratio else 0.0
        recent_ratio = (
            0.4 * count_300[name] / max(1, len(recent_300))
            + 0.6 * count_100[name] / max(1, len(recent_100))
        )
        trend = min(2.0, recent_ratio / historical_ratio) if historical_ratio else 0.0
        family_tags = name.split("+")
        informative_tags = sum(tag not in BACKGROUND_SUBTYPES for tag in family_tags)
        if informative_tags == 0 and len(family_tags) < 2:
            continue
        information = historical_ratio * abs(math.log2(max(lift, 1e-12)))
        support_reliability = min(1.0, math.sqrt(support / 20.0))
        combination_value = 1.0 + 0.20 * informative_tags + 0.05 * (len(family_tags) - 1)
        scores[name] = support_reliability * combination_value * (
            0.45 * information
            + 0.25 * historical_ratio * max(0.0, math.log2(max(lift, 1e-12)))
            + 0.20 * historical_ratio * trend
            + 0.10 * historical_ratio
        )
    return _allocate_scores(scores, slots)


def signature_family(signature: str) -> str:
    return "+".join(tag for tag in signature.split("+") if tag != "compound_mixed")


def signature_family_diagnostics(records: list[dict], baseline: dict | None = None) -> dict[str, dict]:
    baseline = baseline or build_exact_mixed_subtype_baseline()
    mixed = [record for record in records if record["pattern_type"] == "mixed"]
    full = Counter(signature_family(record["subtype_signature"]) for record in mixed)
    baseline_counts = Counter()
    for signature, count in baseline["signature_counts"].items():
        baseline_counts[signature_family(signature)] += count
    return {
        family: {
            "support_count": support,
            "historical_ratio": support / max(1, len(mixed)),
            "baseline_ratio": baseline_counts[family] / max(1, baseline["mixed_total"]),
        }
        for family, support in full.items()
    }


def mixed_subtype_fit_components(
    record: dict,
    subtype_diagnostics: dict[str, dict],
    family_diagnostics: dict[str, dict],
    family_allocation: dict[str, int],
    selected_family_counts: Counter | None = None,
) -> dict:
    selected_family_counts = selected_family_counts or Counter()
    candidate_tags = set(record["subtype_tags"]) - {"compound_mixed"}
    best_family = "unallocated"
    best_match = 0.0
    for family in family_allocation:
        family_tags = set(family.split("+"))
        union = candidate_tags | family_tags
        match = len(candidate_tags & family_tags) / len(union) if union else 0.0
        if (match, family) > (best_match, best_family):
            best_match, best_family = match, family
    exact_family = signature_family(record["subtype_signature"])
    if exact_family in family_allocation:
        best_family, best_match = exact_family, 1.0

    max_information = max((values["information_score"] for values in subtype_diagnostics.values()), default=1.0)
    information_values = [
        subtype_diagnostics[tag]["information_score"] / max(max_information, 1e-12)
        for tag in record["subtype_tags"] if tag in subtype_diagnostics and tag != "compound_mixed"
    ]
    information_score = 100.0 * sum(information_values) / max(1, len(information_values))
    max_support = max((values["support_count"] for values in family_diagnostics.values()), default=1)
    signature_support = 100.0 * family_diagnostics.get(exact_family, {}).get("support_count", 0) / max_support
    target = family_allocation.get(best_family, 0)
    selected = selected_family_counts[best_family]
    if target and selected < target:
        allocation_fit = 100.0 * (target - selected) / target
    elif target:
        allocation_fit = max(0.0, 20.0 - 10.0 * (selected - target))
    else:
        allocation_fit = 0.0
    family_match = 100.0 * best_match
    mixed_fit = (
        0.35 * family_match
        + 0.25 * information_score
        + 0.15 * signature_support
        + 0.25 * allocation_fit
    )
    return {
        "selected_family": best_family,
        "family_match_score": round(family_match, 4),
        "subtype_information_score": round(information_score, 4),
        "signature_support_score": round(signature_support, 4),
        "family_allocation_fit_score": round(allocation_fit, 4),
        "mixed_subtype_fit_score": round(mixed_fit, 4),
    }


def _coarse_family(record: dict) -> str:
    return str(record.get("primary_subtype") or record.get("pattern_type") or "none")


def lift_to_score(lift: float, clip: float = 2.0) -> float:
    log_lift = max(-clip, min(clip, math.log(max(float(lift), 1e-12))))
    return 100.0 / (1.0 + math.exp(-1.5 * log_lift))


def _smoothed_distribution(counts: Counter, vocabulary: list[str], alpha: float) -> dict[str, float]:
    denominator = sum(counts.values()) + alpha * len(vocabulary)
    if denominator <= 0:
        return {key: 1.0 / max(1, len(vocabulary)) for key in vocabulary}
    return {key: (counts[key] + alpha) / denominator for key in vocabulary}


def _shrink_probability(
    counts: Counter,
    family: str,
    parent_probability: float,
    prior_strength: float,
) -> float:
    support = float(sum(counts.values()))
    strength = max(0.0, float(prior_strength))
    if support <= 0.0:
        return float(parent_probability)
    return float(
        (counts.get(family, 0) + strength * parent_probability)
        / (support + strength)
    )


def build_family_transition_matrix(records: list[dict], alpha: float = 0.1) -> dict:
    ordered = sorted(records, key=lambda item: item["draw_no"])
    families = sorted({signature_family(item["subtype_signature"]) for item in ordered})
    rows: dict[str, Counter] = {family: Counter() for family in families}
    for source_record, target_record in zip(ordered, ordered[1:]):
        source = signature_family(source_record["subtype_signature"])
        target = signature_family(target_record["subtype_signature"])
        rows[source][target] += 1
    return {
        source: _smoothed_distribution(counts, families, alpha)
        for source, counts in rows.items()
    }


def calculate_decay_momentum(
    records: list[dict], decay_rate: float = 0.05, alpha: float = 0.1,
    latest_draw: int | None = None,
) -> dict:
    """Legacy mixed-only momentum diagnostic retained for regression comparison."""
    if not records:
        return {}
    global_latest = int(latest_draw if latest_draw is not None else max(r["draw_no"] for r in records))
    eligible = [record for record in records if record["pattern_type"] == "mixed"]
    families = sorted({signature_family(record["subtype_signature"]) for record in eligible})
    if not families:
        return {}
    historical = Counter(signature_family(record["subtype_signature"]) for record in eligible)
    weighted = Counter()
    total_weight = 0.0
    for record in eligible:
        weight = math.exp(-decay_rate * (global_latest - int(record["draw_no"])))
        weighted[signature_family(record["subtype_signature"])] += weight
        total_weight += weight
    historical_total = len(eligible)
    return {
        family: ((weighted[family] + alpha) / (total_weight + alpha * len(families)))
        / ((historical[family] + alpha) / (historical_total + alpha * len(families)))
        for family in families
    }


def _calculate_all_family_decay_momentum(
    records: list[dict],
    decay_rate: float = 0.05,
    alpha: float = 0.1,
    prior_strength: float = 12.0,
    latest_draw: int | None = None,
) -> dict[str, float]:
    """All-draw family momentum, shrunk toward the long-run family distribution."""
    if not records:
        return {}
    ordered = sorted(records, key=lambda item: item["draw_no"])
    global_latest = int(latest_draw if latest_draw is not None else ordered[-1]["draw_no"])
    families = sorted({signature_family(record["subtype_signature"]) for record in ordered})
    historical = Counter(signature_family(record["subtype_signature"]) for record in ordered)
    historical_probs = _smoothed_distribution(historical, families, alpha)
    weighted = Counter()
    total_weight = 0.0
    for record in ordered:
        weight = math.exp(-decay_rate * (global_latest - int(record["draw_no"])))
        weighted[signature_family(record["subtype_signature"])] += weight
        total_weight += weight
    strength = max(0.0, float(prior_strength))
    denominator = total_weight + strength
    if denominator <= 0.0:
        return {family: 1.0 for family in families}
    lifts = {}
    for family in families:
        historical_probability = historical_probs[family]
        recent_probability = (
            weighted[family] + strength * historical_probability
        ) / denominator
        lifts[family] = recent_probability / max(historical_probability, 1e-12)
    return lifts


def build_dynamic_family_model(
    records: list[dict],
    alpha: float = 0.1,
    decay_rate: float = 0.05,
    transition_prior_strength: float = 12.0,
    momentum_prior_strength: float = 12.0,
) -> dict:
    """Precompute adjacent transitions and all-draw shrinkage momentum."""
    ordered = sorted(records, key=lambda item: item["draw_no"])
    exact_vocab = sorted({signature_family(item["subtype_signature"]) for item in ordered})
    coarse_vocab = sorted({_coarse_family(item) for item in ordered})
    type_vocab = ("normal", "mixed", "outlier")
    exact_rows: dict[str, Counter] = {key: Counter() for key in exact_vocab}
    coarse_rows: dict[str, Counter] = {key: Counter() for key in coarse_vocab}
    type_family_rows: dict[str, Counter] = {key: Counter() for key in type_vocab}
    for source, target in zip(ordered, ordered[1:]):
        target_family = signature_family(target["subtype_signature"])
        exact_rows[signature_family(source["subtype_signature"])][target_family] += 1
        coarse_rows[_coarse_family(source)][target_family] += 1
        type_family_rows[source["pattern_type"]][target_family] += 1
    priors = _smoothed_distribution(
        Counter(signature_family(item["subtype_signature"]) for item in ordered), exact_vocab, alpha
    )
    latest = ordered[-1]
    momentum_lifts = _calculate_all_family_decay_momentum(
        ordered,
        decay_rate=decay_rate,
        alpha=alpha,
        prior_strength=momentum_prior_strength,
        latest_draw=latest["draw_no"],
    )
    return {
        "alpha": alpha,
        "transition_prior_strength": float(transition_prior_strength),
        "momentum_prior_strength": float(momentum_prior_strength),
        "families": exact_vocab,
        "priors": priors,
        "exact_counts": exact_rows,
        "coarse_counts": coarse_rows,
        "type_counts": type_family_rows,
        "latest_family": signature_family(latest["subtype_signature"]),
        "latest_coarse_family": _coarse_family(latest),
        "latest_pattern_type": latest["pattern_type"],
        "latest_draw": int(latest["draw_no"]),
        "momentum_lifts": momentum_lifts,
        "momentum_scores": {family: lift_to_score(lift) for family, lift in momentum_lifts.items()},
    }


def family_dynamic_scores(family: str, model: dict) -> tuple[float, float]:
    prior = float(model.get("priors", {}).get(family, model.get("alpha", 0.1) /
                  max(1.0, 1.0 + model.get("alpha", 0.1) * len(model.get("families", ())))))
    strength = float(model.get("transition_prior_strength", 12.0))

    type_counts = model.get("type_counts", {}).get(
        model.get("latest_pattern_type"), Counter()
    )
    type_probability = _shrink_probability(
        type_counts, family, prior, strength
    )

    coarse_counts = model.get("coarse_counts", {}).get(
        model.get("latest_coarse_family"), Counter()
    )
    coarse_probability = _shrink_probability(
        coarse_counts, family, type_probability, strength
    )

    exact_counts = model.get("exact_counts", {}).get(
        model.get("latest_family"), Counter()
    )
    transition_probability = _shrink_probability(
        exact_counts, family, coarse_probability, strength
    )

    transition_score = lift_to_score(transition_probability / max(prior, 1e-12))
    momentum_score = float(model.get("momentum_scores", {}).get(family, 50.0))
    return transition_score, momentum_score
