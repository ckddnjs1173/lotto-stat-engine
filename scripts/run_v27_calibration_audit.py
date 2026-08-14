from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lotto_engine.loader import load_lotto_data
from lotto_engine.v27_audit_artifact import (
    current_git_commit,
    dataframe_fingerprint,
    write_json_artifact,
)
from lotto_engine.v27_calibration import (
    DEFAULT_BOOTSTRAP_REPS,
    SCREENING_CALIBRATION_SAMPLES,
    SCREENING_EVALUATION_SAMPLES,
    run_v27_calibration_audit,
)
from lotto_engine.v27_validation import DEFAULT_BOOTSTRAP_BLOCK


def _fmt(value: float) -> str:
    return "nan" if value != value else f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict walk-forward v2.7 type-score calibration audit"
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=SCREENING_CALIBRATION_SAMPLES,
        help="independent same-type calibration candidates per target",
    )
    parser.add_argument(
        "--evaluation-samples",
        type=int,
        default=SCREENING_EVALUATION_SAMPLES,
        help="independent random evaluation candidates per target",
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPS,
        help="block-bootstrap repetitions for calibrated-minus-raw deltas",
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
        help="optional path for a strict-JSON reproducibility artifact",
    )
    args = parser.parse_args()

    df = load_lotto_data()
    payload = run_v27_calibration_audit(
        df,
        calibration_samples=args.calibration_samples,
        evaluation_samples=args.evaluation_samples,
        bootstrap_reps=args.bootstrap_reps,
        progress_every=args.progress_every,
    )
    payload["run_metadata"] = {
        "runner_commit": current_git_commit(PROJECT_ROOT),
        "bootstrap_block_size": DEFAULT_BOOTSTRAP_BLOCK,
        "data_fingerprint": dataframe_fingerprint(df),
    }

    print("=" * 92)
    print("V2.7 TYPE-SCORE CALIBRATION AUDIT - STRICT WALK FORWARD")
    print("=" * 92)
    print(f"benchmark commit: {payload['benchmark_commit'][:8]}")
    print(f"runner commit: {payload['run_metadata']['runner_commit'] or 'unknown'}")
    fingerprint = payload["run_metadata"]["data_fingerprint"]
    print(
        f"data: rows={fingerprint['row_count']} latest={fingerprint['latest_draw']} "
        f"sha256={fingerprint['sha256'][:12]}..."
    )
    print(f"tests: {payload['total_tests']}")
    print(f"calibration samples / target: {payload['calibration_samples_per_target']}")
    print(f"evaluation samples / target: {payload['evaluation_samples_per_target']}")
    print(f"block bootstrap reps: {payload['bootstrap_reps']}")
    print(f"block bootstrap size: {payload['run_metadata']['bootstrap_block_size']}")
    print()

    for name, result in payload["models"].items():
        delta = result["calibrated_minus_raw"]
        low, high = delta["block_bootstrap_95_ci"]
        print(name)
        print(
            "  raw: "
            f"mean={_fmt(result['raw_mean_percentile'])} "
            f"recent300={_fmt(result['raw_recent_300'])} "
            f"recent100={_fmt(result['raw_recent_100'])}"
        )
        print(
            "  calibrated: "
            f"mean={_fmt(result['calibrated_mean_percentile'])} "
            f"recent300={_fmt(result['calibrated_recent_300'])} "
            f"recent100={_fmt(result['calibrated_recent_100'])} "
            f"above50={result['calibrated_above_random_median_ratio']:.4f}"
        )
        print(
            "  calibrated - raw: "
            f"overall={_fmt(delta['overall'])} "
            f"recent300={_fmt(delta['recent_300'])} "
            f"recent100={_fmt(delta['recent_100'])} "
            f"CI95=[{_fmt(low)}, {_fmt(high)}]"
        )
        print(
            "  calibration type coverage: "
            f"mean-min={result['mean_minimum_calibration_type_count']:.2f} "
            f"global-min={result['minimum_calibration_type_count']}"
        )
        type_text = ", ".join(
            (
                f"{pattern}=raw {_fmt(values['raw_mean_percentile'])}"
                f" -> cal {_fmt(values['calibrated_mean_percentile'])}"
                f" (delta {_fmt(values['mean_delta'])}, n={values['tests']})"
            )
            for pattern, values in result["by_actual_pattern_type"].items()
        )
        print(f"  by actual type: {type_text}")
        print()

    print("Interpretation gate:")
    print("  - calibration must improve global ranking, not merely equalize score scales")
    print("  - promotion requires CI > 0 plus non-negative recent300/recent100 deltas")
    print("  - production scoring remains unchanged during this audit")

    if args.output_json is not None:
        saved = write_json_artifact(payload, args.output_json)
        print(f"audit JSON saved: {saved}")


if __name__ == "__main__":
    main()
