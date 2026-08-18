from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pandas as pd

from .number_evidence import (
    UNIFORM_NUMBER_PROBABILITY,
    number_posterior_probabilities,
    run_number_evidence_backtest,
)
from .pair_evidence import (
    UNIFORM_PAIR_PROBABILITY,
    pair_posterior_probabilities,
    run_pair_evidence_backtest,
)

EVIDENCE_PROFILE_VERSION = "unified_bayesian_evidence_v1"
NUMBER_COUNT = 45
DRAW_SIZE = 6
PAIR_MATRIX_WIDTH = NUMBER_COUNT + 1

# Production-facing research specs are deliberately fixed before the target draw.
# Phase 3A found no durable recent-shift/regime evidence, so the unified scorer uses
# full-history posteriors rather than selecting a decay horizon after seeing results.
# Stronger priors keep deviations from the fair 6/45 null conservative.
NUMBER_PRODUCTION_SPEC = {
    "name": "full_prior_120",
    "prior_strength": 120.0,
    "half_life": None,
}
PAIR_PRODUCTION_SPEC = {
    "name": "full_prior_330",
    "prior_strength": 330.0,
    "half_life": None,
}


def _skill_triplet(model_summary: dict) -> dict[str, float]:
    """Return overall/recent Brier skill against the fair uniform null."""
    windows = model_summary.get("windows", {})
    return {
        "overall": float(model_summary.get("brier_skill_vs_uniform", 0.0)),
        "recent300": float(windows.get("300", {}).get("brier_skill_vs_uniform", 0.0)),
        "recent100": float(windows.get("100", {}).get("brier_skill_vs_uniform", 0.0)),
    }


def reliability_from_brier_skill(model_summary: dict) -> float:
    """Convert OOS Brier skill into a conservative continuous reliability weight.

    This is intentionally not a pass/fail promotion gate. Positive OOS skill is
    retained in proportion to both its magnitude and its stability across the
    overall, recent-300, and recent-100 windows. Negative skill is not inverted;
    being worse than the null does not prove that reversing the forecast predicts
    the future.
    """
    skills = _skill_triplet(model_summary)
    values = np.asarray(list(skills.values()), dtype=float)
    positive = np.maximum(values, 0.0)
    positive_mean = float(positive.mean())
    positive_fraction = float(np.mean(values > 0.0))
    return float(positive_mean * positive_fraction)


def _number_log_lifts(probabilities: dict[int, float]) -> tuple[float, ...]:
    lifts = [0.0] * (NUMBER_COUNT + 1)
    for number in range(1, NUMBER_COUNT + 1):
        probability = max(float(probabilities[number]), 1e-15)
        lifts[number] = math.log(probability / UNIFORM_NUMBER_PROBABILITY)
    return tuple(lifts)


def _pair_log_lifts(probabilities: dict[tuple[int, int], float]) -> tuple[float, ...]:
    lifts = [0.0] * (PAIR_MATRIX_WIDTH * PAIR_MATRIX_WIDTH)
    for (left, right), probability in probabilities.items():
        value = math.log(max(float(probability), 1e-15) / UNIFORM_PAIR_PROBABILITY)
        lifts[left * PAIR_MATRIX_WIDTH + right] = value
        lifts[right * PAIR_MATRIX_WIDTH + left] = value
    return tuple(lifts)


def build_unified_evidence_profile(df: pd.DataFrame) -> dict:
    """Build the production-facing evidence profile from the same local draw data.

    Number and pair reliability are recomputed by strict walk-forward validation on
    every recommendation run. The posterior used to score the next draw is then fit
    to all completed draws using exactly the same predeclared specification.
    """
    number_backtest = run_number_evidence_backtest(
        df,
        specs=(NUMBER_PRODUCTION_SPEC,),
        include_rows=False,
    )
    pair_backtest = run_pair_evidence_backtest(
        df,
        specs=(PAIR_PRODUCTION_SPEC,),
        include_rows=False,
    )

    number_name = str(NUMBER_PRODUCTION_SPEC["name"])
    pair_name = str(PAIR_PRODUCTION_SPEC["name"])
    number_summary = number_backtest["models"][number_name]
    pair_summary = pair_backtest["models"][pair_name]
    number_reliability = reliability_from_brier_skill(number_summary)
    pair_reliability = reliability_from_brier_skill(pair_summary)

    number_probabilities = number_posterior_probabilities(
        df,
        prior_strength=float(NUMBER_PRODUCTION_SPEC["prior_strength"]),
        half_life=NUMBER_PRODUCTION_SPEC["half_life"],
    )
    pair_probabilities = pair_posterior_probabilities(
        df,
        prior_strength=float(PAIR_PRODUCTION_SPEC["prior_strength"]),
        half_life=PAIR_PRODUCTION_SPEC["half_life"],
    )

    return {
        "version": EVIDENCE_PROFILE_VERSION,
        "strict_walk_forward_reliability": True,
        "number_spec": dict(NUMBER_PRODUCTION_SPEC),
        "pair_spec": dict(PAIR_PRODUCTION_SPEC),
        "number_reliability": float(number_reliability),
        "pair_reliability": float(pair_reliability),
        "total_active_reliability": float(number_reliability + pair_reliability),
        "number_skill": _skill_triplet(number_summary),
        "pair_skill": _skill_triplet(pair_summary),
        "number_log_lifts": _number_log_lifts(number_probabilities),
        "pair_log_lifts": _pair_log_lifts(pair_probabilities),
    }


def score_unified_evidence(numbers, evidence_profile: dict) -> dict:
    """Score one 6-number candidate on a single common Bayesian evidence scale."""
    nums = tuple(sorted(int(number) for number in numbers))
    if len(nums) != DRAW_SIZE or len(set(nums)) != DRAW_SIZE:
        raise ValueError("unified evidence scoring requires six unique numbers")
    if nums[0] < 1 or nums[-1] > NUMBER_COUNT:
        raise ValueError("unified evidence scoring requires numbers in 1..45")

    number_lifts = evidence_profile["number_log_lifts"]
    pair_lifts = evidence_profile["pair_log_lifts"]
    number_log_lift = sum(float(number_lifts[number]) for number in nums) / DRAW_SIZE

    pair_total = 0.0
    pair_count = 0
    for left, right in combinations(nums, 2):
        pair_total += float(pair_lifts[left * PAIR_MATRIX_WIDTH + right])
        pair_count += 1
    pair_log_lift = pair_total / pair_count

    number_reliability = float(evidence_profile.get("number_reliability", 0.0))
    pair_reliability = float(evidence_profile.get("pair_reliability", 0.0))
    number_contribution = number_reliability * number_log_lift
    pair_contribution = pair_reliability * pair_log_lift
    weighted_evidence = number_contribution + pair_contribution

    # Monotone display transform only. Ranking uses weighted_evidence directly.
    prediction_score = 50.0 + 50.0 * math.tanh(weighted_evidence)
    return {
        "numbers": list(nums),
        "ranking_score": float(weighted_evidence),
        "prediction_score": float(prediction_score),
        "score_origin": EVIDENCE_PROFILE_VERSION,
        "score_breakdown": {
            "number_log_lift": float(number_log_lift),
            "pair_log_lift": float(pair_log_lift),
            "number_reliability": number_reliability,
            "pair_reliability": pair_reliability,
            "number_contribution": float(number_contribution),
            "pair_contribution": float(pair_contribution),
        },
    }


def public_evidence_diagnostics(evidence_profile: dict) -> dict:
    """Return JSON-safe diagnostics without the large precomputed lift arrays."""
    return {
        "version": str(evidence_profile["version"]),
        "strict_walk_forward_reliability": bool(
            evidence_profile.get("strict_walk_forward_reliability", False)
        ),
        "number": {
            "spec": dict(evidence_profile["number_spec"]),
            "reliability": float(evidence_profile["number_reliability"]),
            "brier_skill_vs_uniform": {
                key: float(value) for key, value in evidence_profile["number_skill"].items()
            },
        },
        "pair": {
            "spec": dict(evidence_profile["pair_spec"]),
            "reliability": float(evidence_profile["pair_reliability"]),
            "brier_skill_vs_uniform": {
                key: float(value) for key, value in evidence_profile["pair_skill"].items()
            },
        },
        "total_active_reliability": float(evidence_profile["total_active_reliability"]),
    }
