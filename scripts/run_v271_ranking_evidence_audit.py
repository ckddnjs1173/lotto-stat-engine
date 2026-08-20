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
from lotto_engine.v27_validation import DEFAULT_BOOTSTRAP_BLOCK, DEFAULT_BOOTSTRAP_REPS
from lotto_engine.v271_ranking_evidence_audit import (
    RANKING_SCREEN_SAMPLES,
    run_v271_ranking_evidence_audit,
)


def _fmt(value: float) -> str:
    return "nan" if value != value else f"{value:.4f}"


def _print_track(name: str, result: dict) -> None:
    low, high = result["percentile_minus_50_block_bootstrap_95_ci"]
    print(name)
    print(
        "  percentile: "
        f"mean={_fmt(result['mean_percentile'])} "
        f"recent300={_fmt(result['recent_300_mean_percentile'])} "
        f"recent100={_fmt(result['recent_100_mean_percentile'])} "
        f"above50={_fmt(result['above_random_median_ratio'])}"
    )
    print(
        "  percentile-50 CI95: "
        f"[{_fmt(low)}, {_fmt(high)}] "
        f"screening_candidate={result['screening_candidate']}"
    )
    print(
        "  continuous rank reliability (diagnostic only): "
        f"{result['continuous_rank_reliability']:.8f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Strict walk-forward v2.7.1 actual-winning-combination ranking audit "
            "for raw number/pair Bayesian evidence"
        )
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=BACKTEST_START_INDEX,
        help="first zero-based target index",
    )
    parser.add_argument(
        "--baseline-samples",
        type=int,
        default=RANKING_SCREEN_SAMPLES,
        help="unique fair random combinations per historical target",
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
        default=50,
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
    payload = run_v271_ranking_evidence_audit(
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

    print("=" * 96)
    print("V2.7.1 BAYESIAN EVIDENCE - COMBINATION RANKING AUDIT")
    print("=" * 96)
    fingerprint = payload["run_metadata"]["data_fingerprint"]
    print(f"runner commit: {payload['run_metadata']['runner_commit'] or 'unknown'}")
    print(
        f"data: rows={fingerprint['row_count']} latest={fingerprint['latest_draw']} "
        f"sha256={fingerprint['sha256'][:12]}..."
    )
    print(f"start index: {payload['start_index']}")
    print(f"tests: {payload['total_tests']}")
    print(f"fair baseline samples / target: {payload['baseline_samples_per_target']}")
    print(f"block bootstrap reps: {payload['bootstrap_reps']}")
    print(f"block bootstrap size: {payload['run_metadata']['bootstrap_block_size']}")
    print("pattern type: not used")
    print()

    for name, result in payload["tracks"].items():
        _print_track(name, result)
        print()

    print(f"Screening candidates: {payload['screening_candidates']}")
    print()
    print("Interpretation:")
    print("  - this tests ranking utility directly, not probability calibration")
    print("  - no production reliability is changed by this audit")
    print("  - survivor => rerun only the survivor with 2,000 fair samples/target")
    print("  - no survivor => current number/pair posterior family remains neutral in production")

    if args.output_json is not None:
        saved = write_json_artifact(payload, args.output_json)
        print(f"audit JSON saved: {saved}")


if __name__ == "__main__":
    main()
