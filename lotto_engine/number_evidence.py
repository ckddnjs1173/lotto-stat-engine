from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import BACKTEST_START_INDEX, NUMBER_COLUMNS, ROUND_COLUMN

NUMBER_COUNT = 45
DRAW_SIZE = 6
UNIFORM_NUMBER_PROBABILITY = DRAW_SIZE / NUMBER_COUNT
PROBABILITY_EPSILON = 1e-12

# Research candidates only. These are intentionally coarse, predeclared
# horizons rather than parameters optimized against the final target draw.
DEFAULT_NUMBER_BAYES_SPECS = (
    {"name": "full_prior_24", "prior_strength": 24.0, "half_life": None},
    {"name": "full_prior_120", "prior_strength": 120.0, "half_life": None},
    {"name": "decay_h25_prior24", "prior_strength": 24.0, "half_life": 25.0},
    {"name": "decay_h75_prior24", "prior_strength": 24.0, "half_life": 75.0},
    {"name": "decay_h200_prior24", "prior_strength": 24.0, "half_life": 200.0},
)


def _indicator_matrix(df: pd.DataFrame) -> np.ndarray:
    matrix = np.zeros((len(df), NUMBER_COUNT), dtype=float)
    row_index = np.arange(len(df))
    for column in NUMBER_COLUMNS:
        numbers = df[column].to_numpy(dtype=int) - 1
        matrix[row_index, numbers] = 1.0
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


def _posterior_from_history(
    history_matrix: np.ndarray,
    history_rounds: np.ndarray,
    prior_strength: float,
    half_life: float | None,
) -> np.ndarray:
    prior_strength = float(prior_strength)
    if prior_strength < 0.0:
        raise ValueError("prior_strength must be non-negative")
    if len(history_matrix) == 0:
        return np.full(NUMBER_COUNT, UNIFORM_NUMBER_PROBABILITY, dtype=float)

    weights = _history_weights(history_rounds, half_life)
    effective_draws = float(weights.sum())
    weighted_hits = weights @ history_matrix
    denominator = effective_draws + prior_strength
    if denominator <= 0.0:
        return np.full(NUMBER_COUNT, UNIFORM_NUMBER_PROBABILITY, dtype=float)

    probabilities = (
        weighted_hits + prior_strength * UNIFORM_NUMBER_PROBABILITY
    ) / denominator
    return probabilities.astype(float)


def number_posterior_probabilities(
    df: pd.DataFrame,
    prior_strength: float = 24.0,
    half_life: float | None = None,
) -> dict[int, float]:
    """Posterior marginal inclusion probability for each number 1..45.

    The shared 6/45 prior and six observed hits per draw guarantee that the
    45 posterior means sum to six (up to floating-point error).
    """
    matrix = _indicator_matrix(df)
    rounds = df[ROUND_COLUMN].to_numpy(dtype=float)
    probabilities = _posterior_from_history(
        matrix, rounds, prior_strength=prior_strength, half_life=half_life
    )
    return {number: float(probabilities[number - 1]) for number in range(1, 46)}


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
            "mean_winner_probability": _metric_summary(metrics["winner_probability"], window),
            "mean_top6_matches": _metric_summary(metrics["top6_matches"], window),
        }
    return {
        "prior_strength": float(spec["prior_strength"]),
        "half_life": spec.get("half_life"),
        "mean_brier": overall_brier,
        "brier_skill_vs_uniform": _skill_score(overall_brier, baseline_brier),
        "mean_log_loss": overall_log,
        "log_loss_improvement_vs_uniform": baseline_log - overall_log,
        "mean_winner_probability": _metric_summary(metrics["winner_probability"]),
        "mean_top6_matches": _metric_summary(metrics["top6_matches"]),
        "windows": windows,
    }


def run_number_evidence_backtest(
    df: pd.DataFrame,
    start_index: int = BACKTEST_START_INDEX,
    specs: tuple[dict, ...] | list[dict] | None = None,
    include_rows: bool = False,
) -> dict:
    """Strict walk-forward test of number-level Bayesian marginal forecasts.

    For target row idx, every probability is built from df.iloc[:idx] only.
    The uniform 6/45 forecast is the null model. Positive Brier skill means
    lower out-of-sample Brier loss than that null.
    """
    if len(df) <= start_index:
        raise ValueError(f"number evidence backtest requires at least {start_index + 1} draws")
    specs = tuple(specs or DEFAULT_NUMBER_BAYES_SPECS)
    names = [str(spec["name"]) for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("number evidence spec names must be unique")

    matrix = _indicator_matrix(df)
    rounds = df[ROUND_COLUMN].to_numpy(dtype=float)
    uniform = np.full(NUMBER_COUNT, UNIFORM_NUMBER_PROBABILITY, dtype=float)
    baseline = {"brier": [], "log_loss": []}
    metrics = {
        name: {"brier": [], "log_loss": [], "winner_probability": [], "top6_matches": []}
        for name in names
    }
    detail_rows: list[dict] = []

    for idx in range(int(start_index), len(df)):
        actual = matrix[idx]
        baseline_brier = _brier_score(uniform, actual)
        baseline_log = _binary_log_loss(uniform, actual)
        baseline["brier"].append(baseline_brier)
        baseline["log_loss"].append(baseline_log)
        row_detail = {
            "target_round": int(rounds[idx]),
            "history_draws": int(idx),
            "uniform_brier": baseline_brier,
            "models": {},
        }

        history_matrix = matrix[:idx]
        history_rounds = rounds[:idx]
        winner_mask = actual.astype(bool)
        for spec in specs:
            name = str(spec["name"])
            probabilities = _posterior_from_history(
                history_matrix,
                history_rounds,
                prior_strength=float(spec["prior_strength"]),
                half_life=spec.get("half_life"),
            )
            brier = _brier_score(probabilities, actual)
            log_loss = _binary_log_loss(probabilities, actual)
            winner_probability = float(np.mean(probabilities[winner_mask]))
            top6 = np.argpartition(probabilities, -DRAW_SIZE)[-DRAW_SIZE:]
            top6_matches = float(actual[top6].sum())
            metrics[name]["brier"].append(brier)
            metrics[name]["log_loss"].append(log_loss)
            metrics[name]["winner_probability"].append(winner_probability)
            metrics[name]["top6_matches"].append(top6_matches)
            if include_rows:
                row_detail["models"][name] = {
                    "brier": brier,
                    "log_loss": log_loss,
                    "winner_probability": winner_probability,
                    "top6_matches": top6_matches,
                }
        if include_rows:
            detail_rows.append(row_detail)

    baseline_brier = _metric_summary(baseline["brier"])
    baseline_log = _metric_summary(baseline["log_loss"])
    payload = {
        "version": "number_bayes_evidence_v1",
        "strict_walk_forward": True,
        "start_index": int(start_index),
        "total_tests": len(df) - int(start_index),
        "uniform_probability": UNIFORM_NUMBER_PROBABILITY,
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
