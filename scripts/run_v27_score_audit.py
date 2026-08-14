from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lotto_engine.loader import load_lotto_data
from lotto_engine.v27_validation import (
    DEFAULT_BOOTSTRAP_REPS,
    SCREENING_BASELINE_SAMPLES,
    run_v27_score_audit,
)


def _fmt(value: float) -> str:
    return "nan" if value != value else f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict walk-forward v2.7 score ablation audit"
    )
    parser.add_argument(
        "--baseline-samples",
        type=int,
        default=SCREENING_BASELINE_SAMPLES,
        help="random candidate baseline per target (screening default: 500)",
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPS,
        help="block-bootstrap repetitions for paired percentile deltas",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="print progress every N targets; 0 disables progress output",
    )
    args = parser.parse_args()

    df = load_lotto_data()
    payload = run_v27_score_audit(
        df,
        baseline_samples=args.baseline_samples,
        bootstrap_reps=args.bootstrap_reps,
        progress_every=args.progress_every,
    )

    print("=" * 88)
    print("V2.7 SCORE AUDIT - STRICT WALK FORWARD")
    print("=" * 88)
    print(f"benchmark commit: {payload['benchmark_commit'][:8]}")
    print(f"tests: {payload['total_tests']}")
    print(f"random baseline samples / target: {payload['baseline_samples_per_target']}")
    print(f"block bootstrap reps: {payload['bootstrap_reps']}")
    print()

    for name, result in payload["models"].items():
        print(name)
        print(
            "  global percentile: "
            f"mean={_fmt(result['mean_percentile'])} "
            f"recent300={_fmt(result['recent_300_mean_percentile'])} "
            f"recent100={_fmt(result['recent_100_mean_percentile'])} "
            f"above50={result['above_random_median_ratio']:.4f}"
        )
        print(
            "  type-conditioned: "
            f"mean={_fmt(result['type_conditioned_mean_percentile'])} "
            f"recent300={_fmt(result['type_conditioned_recent_300'])} "
            f"recent100={_fmt(result['type_conditioned_recent_100'])}"
        )
        print(
            "  same-type baseline candidates: "
            f"mean={result['mean_same_type_baseline_count']:.2f} "
            f"min={result['minimum_same_type_baseline_count']}"
        )
        type_text = ", ".join(
            f"{pattern}={_fmt(values['mean_percentile'])} (n={values['tests']})"
            for pattern, values in result["by_actual_pattern_type"].items()
        )
        print(f"  by actual type: {type_text}")

        comparisons = payload["paired_comparisons"].get(name, {})
        for reference, values in comparisons.items():
            low, high = values["block_bootstrap_95_ci"]
            print(
                f"  delta vs {reference}: "
                f"overall={_fmt(values['mean_percentile_delta'])} "
                f"recent300={_fmt(values['recent_300_delta'])} "
                f"recent100={_fmt(values['recent_100_delta'])} "
                f"CI95=[{_fmt(low)}, {_fmt(high)}]"
            )
        print()

    print("Interpretation gate:")
    print("  - positive point estimates alone are NOT promotion evidence")
    print("  - promotion candidate requires paired CI > 0 and recent-window stability")
    print("  - screening winners must be rerun with --baseline-samples 2000")


if __name__ == "__main__":
    main()
