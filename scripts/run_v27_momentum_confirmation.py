from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lotto_engine.config import BACKTEST_START_INDEX
from lotto_engine.loader import load_lotto_data
from lotto_engine.v27_audit_artifact import current_git_commit, dataframe_fingerprint, write_json_artifact
from lotto_engine.v27_momentum_confirmation import CONFIRMATION_BASELINE_SAMPLES, run_v27_momentum_confirmation
from lotto_engine.v27_validation import DEFAULT_BOOTSTRAP_BLOCK, DEFAULT_BOOTSTRAP_REPS


def _fmt(value: float) -> str:
    return "nan" if value != value else f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Targeted v2.7 momentum confirmation")
    parser.add_argument("--start-index", type=int, default=BACKTEST_START_INDEX)
    parser.add_argument("--baseline-samples", type=int, default=CONFIRMATION_BASELINE_SAMPLES)
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    df = load_lotto_data()
    payload = run_v27_momentum_confirmation(
        df,
        start_index=args.start_index,
        baseline_samples=args.baseline_samples,
        bootstrap_reps=args.bootstrap_reps,
        progress_every=args.progress_every,
    )
    payload["run_metadata"] = {
        "runner_commit": current_git_commit(PROJECT_ROOT),
        "bootstrap_block_size": DEFAULT_BOOTSTRAP_BLOCK,
        "data_fingerprint": dataframe_fingerprint(df),
    }

    print("=" * 92)
    print("V2.7 MOMENTUM CONFIRMATION - TARGETED PHASE 4")
    print("=" * 92)
    print(f"runner commit: {payload['run_metadata']['runner_commit'] or 'unknown'}")
    fp = payload["run_metadata"]["data_fingerprint"]
    print(f"data: rows={fp['row_count']} latest={fp['latest_draw']} sha256={fp['sha256'][:12]}...")
    print(f"start index: {payload['start_index']}")
    print(f"tests: {payload['total_tests']}")
    print(f"fair baseline samples / target: {payload['baseline_samples_per_target']}")
    print(f"block bootstrap reps: {payload['bootstrap_reps']}")
    print(f"block bootstrap size: {payload['run_metadata']['bootstrap_block_size']}")
    print()

    full = payload["full_fair_baseline"]
    low, high = full["percentile_minus_50_block_bootstrap_95_ci"]
    print("Full fair-baseline momentum:")
    print(f"  percentile: mean={_fmt(full['mean_percentile'])} recent300={_fmt(full['recent_300_mean_percentile'])} recent100={_fmt(full['recent_100_mean_percentile'])} above50={full['above_random_median_ratio']:.4f}")
    print(f"  percentile-50 CI95: [{_fmt(low)}, {_fmt(high)}]")
    print("  by actual type: " + ", ".join(
        f"{pattern}={_fmt(values['mean_percentile'])} (n={values['tests']})"
        for pattern, values in full["by_actual_pattern_type"].items()
    ))
    print()

    seen = payload["seen_family_robustness"]
    seen_low, seen_high = seen["percentile_minus_50_block_bootstrap_95_ci"]
    print("Seen-family robustness:")
    print(f"  tests={seen['tests']} mean={_fmt(seen['mean_percentile'])} recent300-target-window={_fmt(seen['recent_300_target_window_mean_percentile'])} recent100-target-window={_fmt(seen['recent_100_target_window_mean_percentile'])}")
    print(f"  percentile-50 CI95: [{_fmt(seen_low)}, {_fmt(seen_high)}]")
    print()

    coverage = payload["family_coverage"]
    print("Family coverage:")
    print(f"  actual seen={coverage['actual_family_seen_ratio']:.4f} recent300={coverage['actual_family_seen_recent_300']:.4f} recent100={coverage['actual_family_seen_recent_100']:.4f}")
    print(f"  baseline seen mean={coverage['mean_baseline_family_seen_ratio']:.4f} mean-count={coverage['mean_baseline_family_seen_count']:.2f} min-count={coverage['minimum_baseline_family_seen_count']}")
    print()

    lift = payload["actual_lift"]
    print(f"Actual momentum log-lift: mean={_fmt(lift['mean_actual_log_lift'])} recent300={_fmt(lift['recent_300_actual_log_lift'])} recent100={_fmt(lift['recent_100_actual_log_lift'])}")
    print()
    print(f"CONFIRMED: {payload['confirmed']}")
    print("Gate:")
    print("  - full fair-baseline CI > 0 and recent300/recent100 >= 50")
    print("  - seen-family conditional CI > 0")
    print("  - failure ends momentum confirmation; no retuning rescue")

    if args.output_json is not None:
        saved = write_json_artifact(payload, args.output_json)
        print(f"audit JSON saved: {saved}")


if __name__ == "__main__":
    main()
