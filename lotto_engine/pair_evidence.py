from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pandas as pd

from .config import BACKTEST_START_INDEX, NUMBER_COLUMNS, ROUND_COLUMN

NUMBER_COUNT = 45
DRAW_SIZE = 6
PAIR_COUNT = math.comb(NUMBER_COUNT, 2)
PAIRS_PER_DRAW = math.comb(DRAW_SIZE, 2)
UNIFORM_PAIR_PROBABILITY = PAIRS_PER_DRAW / PAIR_COUNT  # 1 / 66
PROBABILITY_EPSILON = 1e-12

PAIR_VOCABULARY = tuple(combinations(range(1, NUMBER_COUNT + 1), 2))
PAIR_TO_INDEX = {pair: index for index, pair in enumerate(PAIR_VOCABULARY)}

# Research-only candidates. Prior strengths are measured in draw-equivalents.
# Under the fair null, 66 draws imply one expected occurrence per pair and
# 330 draws imply five. They are deliberately coarse, predeclared values.
DEFAULT_PAIR_BAYES_SPECS = (
    {"name": "full_prior_66", "prior_strength": 66.0, "half_life": None},
    {"name": "full_prior_330", "prior_strength": 330.0, "half_life": None},
    {"name": "decay_h25_prior66", "prior_strength": 66.0, "half_life": 25.0},
    {"name": "decay_h75_prior66", "prior_strength": 66.0, "half_life": 75.0},
    {"name": "decay_h200_prior66", "prior_strength": 66.0, "half_life": 200.0},
)


def _draw_numbers(row) -> tuple[int, ...]:
    return tuple(sorted(int(row[column]) for column in NUMBER_COLUMNS))


def _pair_indicator_matrix(df: pd.DataFrame) -> np.ndarray:
    matrix = np.zeros((len(df), PAIR_COUNT), dtype=float)
    for row_index, (_, row) in enumerate(df.iterrows()):
        for pair in combinations(_draw_numbers(row), 2):
            matrix[row_index, PAIR_TO_INDEX[pair]] = 1.0
    return matrix


def _history_weights(rounds: np.ndarray, half_life: float | None) -> np.ndarray:
    if len(rounds) == 0:
        return np.asarray([], dtype=float)
    if half_life is None:
        return np.ones(len(rounds), dtype=float)
    half_life = float(half_life)
    if half_life <= 0.0:
        raise ValueError("half_life must be positive or None")
    decay_rate = math.log(2.0) / half_life
    latest_round = float(rounds[-1])
    return np.exp(-decay_rate * (latest_round - rounds.astype(float)))


def _posterior_from_state(
    weighted_hits: np.ndarray,
    effective_draws: float,
    prior_strength: float,
) -> np.ndarray:
    prior_strength = float(prior_strength)
    if prior_strength < 0.0:
        raise ValueError("prior_strength must be non-negative")
    denominator = float(effective_draws) + prior_strength
    if denominator <= 0.0:
        return np.full(PAIR_COUNT, UNIFORM_PAIR_PROBABILITY, dtype=float)
    return (
        weighted_hits + prior_strength * UNIFORM_PAIR_PROBABILITY
    ) / denominator


def _posterior_from_history(
    history_matrix: np.ndarray,
    history_rounds: np.ndarray,
    prior_strength: float,
    half_life: float | None,
) -> np.ndarray:
    if len(history_matrix) == 0:
        return np.full(PAIR_COUNT, UNIFORM_PAIR_PROBABILITY, dtype=float)
    weights = _history_weights(history_rounds, half_life)
    return _posterior_from_state(
        weights @ history_matrix,
        float(weights.sum()),
        prior_strength,
    ).astype(float)


def _advance_state(state: dict, actual: np.ndarray, round_no: float) -> None:
    half_life = state["half_life"]
    latest_round = state["latest_round"]
    if half_life is not None and latest_round is not None:
        decay_rate = math.log(2.0) / float(half_life)
        factor = math.exp(-decay_rate * (float(round_no) - float(latest_round)))
        state["weighted_hits"] *= factor
        state["effective_draws"] *= factor
    state["weighted_hits"] += actual
    state["effective_draws"] += 1.0
    state["latest_round"] = float(round_no)


def pair_posterior_probabilities(
    df: pd.DataFrame,
    prior_strength: float = 66.0,
    half_life: float | None = None,
) -> dict[tuple[int, int], float]:
    """Posterior marginal inclusion probability for all 990 unordered pairs.

    A common fair-null prior plus exactly 15 observed pairs per draw guarantees
    that the 990 posterior means sum to 15 (up to floating-point error).
    """
    matrix = _pair_indicator_matrix(df)
    rounds = df[ROUND_COLUMN].to_numpy(dtype=float)
    probabilities = _posterior_from_history(
        matrix, rounds, prior_strength=prior_strength, half_life=half_life
    )
    return {
        pair: float(probabilities[index])
        for index, pair in enumerate(PAIR_VOCABULARY)
    }


def _brier_score(probabilities: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean((probabilities - actual) ** 2))


def _binary_log_loss(probabilities: np.ndarray, actual: np.ndarray) -> float:
    p = np.clip(probabilities, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    return float(-np.mean(actual * np.log(p) + (1.0 - actual) * np.log(1.0 - p)))


def _metric_summary(values: list[float], window: int | None = None) -> float:
    if not values:
        return 0.0
    selected = values if window is None else values[-min(window, len(values)):]
    return float(np.mean(selected))


def _skill_score(model_loss: float, baseline_loss: float) -> float:
    if baseline_loss <= 0.0:
        return 0.0
    return float(1.0 - model_loss / baseline_loss)


def _spec_summary(metrics: dict[str, list[float]], baseline: dict[str, list[float]], spec: dict) -> dict:
    overall_brier = _metric_summary(metrics["brier"])
    baseline_brier = _metric_summary(baseline["brier"])
    overall_log = _metric_summary(metrics["log_loss"])
    baseline_log = _metric_summary(baseline["log_loss"])
    windows = {}
    for window in (100, 300):
        model_brier = _metric_summary(metrics["brier"], window)
        null_brier = _metric_summary(baseline["brier"], window)
        model_log = _metric_summary(metrics["log_loss"], window)
        null_log = _metric_summary(baseline["log_loss"], window)
        windows[str(window)] = {
            "tests": min(window, len(metrics["brier"])),
            "mean_brier": model_brier,
            "brier_skill_vs_uniform": _skill_score(model_brier, null_brier),
            "mean_log_loss": model_log,
            "log_loss_improvement_vs_uniform": null_log - model_log,
            "mean_winner_pair_probability": _metric_summary(metrics["winner_pair_probability"], window),
            "mean_top15_pair_matches": _metric_summary(metrics["top15_pair_matches"], window),
        }
    return {
        "prior_strength": float(spec["prior_strength"]),
        "half_life": spec.get("half_life"),
        "mean_brier": overall_brier,
        "brier_skill_vs_uniform": _skill_score(overall_brier, baseline_brier),
        "mean_log_loss": overall_log,
        "log_loss_improvement_vs_uniform": baseline_log - overall_log,
        "mean_winner_pair_probability": _metric_summary(metrics["winner_pair_probability"]),
        "mean_top15_pair_matches": _metric_summary(metrics["top15_pair_matches"]),
        "windows": windows,
    }


def run_pair_evidence_backtest(
    df: pd.DataFrame,
    start_index: int = BACKTEST_START_INDEX,
    specs: tuple[dict, ...] | list[dict] | None = None,
    include_rows: bool = False,
) -> dict:
    """Strict walk-forward test of pair-frequency Bayesian forecasts.

    For target row idx, every pair probability is estimated from df.iloc[:idx]
    only. The fair 1/66 pair probability is the null. Positive Brier skill means
    lower out-of-sample loss than the fair-null forecast. Exponentially decayed
    histories are updated recursively so each past draw is processed once.
    """
    if len(df) <= start_index:
        raise ValueError(f"pair evidence backtest requires at least {start_index + 1} draws")
    specs = tuple(specs or DEFAULT_PAIR_BAYES_SPECS)
    names = [str(spec["name"]) for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("pair evidence spec names must be unique")
    for spec in specs:
        if float(spec["prior_strength"]) < 0.0:
            raise ValueError("prior_strength must be non-negative")
        half_life = spec.get("half_life")
        if half_life is not None and float(half_life) <= 0.0:
            raise ValueError("half_life must be positive or None")

    matrix = _pair_indicator_matrix(df)
    rounds = df[ROUND_COLUMN].to_numpy(dtype=float)
    uniform = np.full(PAIR_COUNT, UNIFORM_PAIR_PROBABILITY, dtype=float)
    baseline = {"brier": [], "log_loss": []}
    metrics = {
        name: {
            "brier": [],
            "log_loss": [],
            "winner_pair_probability": [],
            "top15_pair_matches": [],
        }
        for name in names
    }
    states = {
        str(spec["name"]): {
            "weighted_hits": np.zeros(PAIR_COUNT, dtype=float),
            "effective_draws": 0.0,
            "latest_round": None,
            "half_life": spec.get("half_life"),
            "prior_strength": float(spec["prior_strength"]),
        }
        for spec in specs
    }
    detail_rows: list[dict] = []

    for idx in range(len(df)):
        actual = matrix[idx]
        round_no = float(rounds[idx])

        # States contain rows strictly before idx at this point.
        if idx >= int(start_index):
            baseline_brier = _brier_score(uniform, actual)
            baseline_log = _binary_log_loss(uniform, actual)
            baseline["brier"].append(baseline_brier)
            baseline["log_loss"].append(baseline_log)
            winner_mask = actual.astype(bool)
            row_detail = {
                "target_round": int(round_no),
                "history_draws": int(idx),
                "uniform_brier": baseline_brier,
                "models": {},
            }

            for spec in specs:
                name = str(spec["name"])
                state = states[name]
                probabilities = _posterior_from_state(
                    state["weighted_hits"],
                    state["effective_draws"],
                    state["prior_strength"],
                )
                brier = _brier_score(probabilities, actual)
                log_loss = _binary_log_loss(probabilities, actual)
                winner_probability = float(np.mean(probabilities[winner_mask]))
                top15 = np.argpartition(probabilities, -PAIRS_PER_DRAW)[-PAIRS_PER_DRAW:]
                top15_matches = float(actual[top15].sum())
                metrics[name]["brier"].append(brier)
                metrics[name]["log_loss"].append(log_loss)
                metrics[name]["winner_pair_probability"].append(winner_probability)
                metrics[name]["top15_pair_matches"].append(top15_matches)
                if include_rows:
                    row_detail["models"][name] = {
                        "brier": brier,
                        "log_loss": log_loss,
                        "winner_pair_probability": winner_probability,
                        "top15_pair_matches": top15_matches,
                    }
            if include_rows:
                detail_rows.append(row_detail)

        # Only after scoring target idx does it become available to the next
        # target, preserving strict rolling-origin / no-future-leakage logic.
        for state in states.values():
            _advance_state(state, actual, round_no)

    baseline_brier = _metric_summary(baseline["brier"])
    baseline_log = _metric_summary(baseline["log_loss"])
    payload = {
        "version": "pair_bayes_evidence_v1",
        "strict_walk_forward": True,
        "start_index": int(start_index),
        "total_tests": len(df) - int(start_index),
        "pair_count": PAIR_COUNT,
        "pairs_per_draw": PAIRS_PER_DRAW,
        "uniform_probability": UNIFORM_PAIR_PROBABILITY,
        "random_top15_expected_matches": PAIRS_PER_DRAW * PAIRS_PER_DRAW / PAIR_COUNT,
        "baseline": {
            "mean_brier": baseline_brier,
            "mean_log_loss": baseline_log,
            "windows": {
                str(window): {
                    "tests": min(window, len(baseline["brier"])),
                    "mean_brier": _metric_summary(baseline["brier"], window),
                    "mean_log_loss": _metric_summary(baseline["log_loss"], window),
                }
                for window in (100, 300)
            },
        },
        "models": {
            str(spec["name"]): _spec_summary(metrics[str(spec["name"])], baseline, spec)
            for spec in specs
        },
    }
    if include_rows:
        payload["rows"] = detail_rows
    return payload
