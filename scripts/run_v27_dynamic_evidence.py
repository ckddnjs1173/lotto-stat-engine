from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lotto_engine.config import BACKTEST_START_INDEX
from lotto_engine.loader import load_lotto_data
from lotto_engine.v27_audit_artifact import (
    current_git_commit,
    dataframe_fingerprint,
    write_json_artifact,
)
from lotto_engine.v27_dynamic_evidence import (
    SCREENING_BASELINE_SAMPLES,
    run_v27_dynamic_evidence_audit,
)
from lotto_engine.v27_validation import DEFAULT_BOOTSTRAP_BLOCK, DEFAULT_BOOTSTRAP_REPS


def _fmt(value: float) -> str:
    return "nan" if value != value else f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict walk-forward v2.7 raw dynamic evidence audit"
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=BACKTEST_START_INDEX,
        help="first zero-based target index; use a late index only for cheap smoke runs",
    )
    parser.add_argument(
        "--baseline-samples",
        type=int,
        default=SCREENING_BASELINE_SAMPLES,
        help="fair random valid combinations per historical target",
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPS,
        help="circular block-bootstrap repetitions",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="print progress every N targets; 0 disables progress output",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="optional strict-JSON result path",
    )
    args = parser.parse_args()

    df = load_lotto_data()
    payload = run_v27_dynamic_evidence_audit(
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
    print("V2.7 DIRECT DYNAMIC EVIDENCE AUDIT - PHASE 4 SCREENING")
    print("=" * 92)
    print(f"runner commit: {payload['run_metadata']['runner_commit'] or 'unknown'}")
    fingerprint = payload["run_metadata"]["data_fingerprint"]
    print(
        f"data: rows={fingerprint['row_count']} latest={fingerprint['latest_draw']} "
        f"sha256={fingerprint['sha256'][:12]}..."
    )
    print(f"start index: {payload['start_index']}")
    print(f"tests: {payload['total_tests']}")
    print(f"fair baseline samples / target: {payload['baseline_samples_per_target']}")
    print(f"block bootstrap reps: {payload['bootstrap_reps']}")
    print(f"block bootstrap size: {payload['run_metadata']['bootstrap_block_size']}")
    print()

    for name, result in payload["components"].items():
        low, high = result["percentile_minus_50_block_bootstrap_95_ci"]
        print(name)
        print(
            "  percentile: "
            f"mean={_fmt(result['mean_percentile'])} "
            f"recent300={_fmt(result['recent_300_mean_percentile'])} "
            f"recent100={_fmt(result['recent_100_mean_percentile'])} "
            f"above50={result['above_random_median_ratio']:.4f}"
        )
        print(
            "  actual log-lift: "
            f"mean={_fmt(result['mean_actual_log_lift'])} "
            f"recent300={_fmt(result['recent_300_actual_log_lift'])} "
            f"recent100={_fmt(result['recent_100_actual_log_lift'])}"
        )
        print(
            "  percentile-50 CI95: "
            f"[{_fmt(low)}, {_fmt(high)}] "
            f"screening_candidate={result['screening_candidate']}"
        )
        type_text = ", ".join(
            f"{pattern}={_fmt(values['mean_percentile'])} (n={values['tests']})"
            for pattern, values in result["by_actual_pattern_type"].items()
        )
        print(f"  by actual type: {type_text}")
        print()

    print("Transition hierarchy diagnostics:")
    for name, result in payload["transition_level_comparisons"].items():
        low, high = result["block_bootstrap_95_ci"]
        print(
            f"  {name}: overall={_fmt(result['mean_percentile_delta'])} "
            f"recent300={_fmt(result['recent_300_delta'])} "
            f"recent100={_fmt(result['recent_100_delta'])} "
            f"CI95=[{_fmt(low)}, {_fmt(high)}]"
        )

    print()
    print("Primary screening candidates:")
    for name, passed in payload["screening_candidates"].items():
        print(f"  {name}: {passed}")
    print()
    print("Gate:")
    print("  - failed primary component => no 1000-sample confirmation")
    print("  - survivor => confirm only that component at the larger baseline budget")
    print("  - no production weight changes are made by this screening script")

    if args.output_json is not None:
        saved = write_json_artifact(payload, args.output_json)
        print(f"audit JSON saved: {saved}")


if __name__ == "__main__":
    main()
