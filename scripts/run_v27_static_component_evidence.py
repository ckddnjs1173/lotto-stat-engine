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
from lotto_engine.v27_static_component_evidence import (
    SCREENING_SAME_TYPE_SAMPLES,
    run_v27_static_component_evidence_audit,
)
from lotto_engine.v27_validation import DEFAULT_BOOTSTRAP_BLOCK, DEFAULT_BOOTSTRAP_REPS


def _fmt(value: float) -> str:
    return "nan" if value != value else f"{value:.4f}"


def _print_result(name: str, result: dict) -> None:
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
    type_text = []
    for pattern, values in result["by_actual_pattern_type"].items():
        if values["tests"] <= 0:
            continue
        t_low, t_high = values["percentile_minus_50_block_bootstrap_95_ci"]
        type_text.append(
            f"{pattern}={_fmt(values['mean_percentile'])} "
            f"r300={_fmt(values['recent_300_mean_percentile'])} "
            f"r100={_fmt(values['recent_100_mean_percentile'])} "
            f"CI=[{_fmt(t_low)},{_fmt(t_high)}] "
            f"cand={values['screening_candidate']} "
            f"(n={values['tests']})"
        )
    print("  by actual type: " + "; ".join(type_text))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict walk-forward v2.7 same-type static component evidence audit"
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=BACKTEST_START_INDEX,
        help="first zero-based target index; use a late index only for smoke runs",
    )
    parser.add_argument(
        "--same-type-samples",
        type=int,
        default=SCREENING_SAME_TYPE_SAMPLES,
        help="fair random combinations of the actual target pattern type per target",
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
    payload = run_v27_static_component_evidence_audit(
        df,
        start_index=args.start_index,
        same_type_samples=args.same_type_samples,
        bootstrap_reps=args.bootstrap_reps,
        progress_every=args.progress_every,
    )
    payload["run_metadata"] = {
        "runner_commit": current_git_commit(PROJECT_ROOT),
        "bootstrap_block_size": DEFAULT_BOOTSTRAP_BLOCK,
        "data_fingerprint": dataframe_fingerprint(df),
    }

    print("=" * 96)
    print("V2.7 STATIC COMPONENT EVIDENCE AUDIT - PHASE 5A SCREENING")
    print("=" * 96)
    fingerprint = payload["run_metadata"]["data_fingerprint"]
    print(f"runner commit: {payload['run_metadata']['runner_commit'] or 'unknown'}")
    print(
        f"data: rows={fingerprint['row_count']} latest={fingerprint['latest_draw']} "
        f"sha256={fingerprint['sha256'][:12]}..."
    )
    print(f"start index: {payload['start_index']}")
    print(f"tests: {payload['total_tests']}")
    print(f"same-type fair samples / target: {payload['same_type_samples_per_target']}")
    print(f"block bootstrap reps: {payload['bootstrap_reps']}")
    print(f"block bootstrap size: {payload['run_metadata']['bootstrap_block_size']}")
    print()

    print("Current branch-base benchmark (same-type ranking):")
    _print_result("current_branch_base", payload["benchmark_current_branch_base"])
    print()

    print("Normal / outlier static components:")
    for name, result in payload["normal_outlier_components"].items():
        _print_result(name, result)
        print()

    print("Mixed static components:")
    for name, result in payload["mixed_components"].items():
        _print_result(name, result)
        print()

    print("Screening candidates:")
    print(
        "  normal/outlier common: "
        f"{payload['screening_candidates']['normal_outlier']}"
    )
    print(f"  mixed: {payload['screening_candidates']['mixed']}")
    print()
    print("Gate:")
    print("  - same-type percentile only; no cross-type scale inference")
    print("  - survivor => targeted 500-same-type confirmation only")
    print("  - no survivor => do not retune current static weights")
    print("  - dynamic transition/momentum remain closed")

    if args.output_json is not None:
        saved = write_json_artifact(payload, args.output_json)
        print(f"audit JSON saved: {saved}")


if __name__ == "__main__":
    main()
