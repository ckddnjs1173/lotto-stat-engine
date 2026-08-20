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
from lotto_engine.v29_quadratic_reverse_ranking import (
    CONFIRMATION_BASELINE_SAMPLES,
    MIN_META_TRAIN_TARGETS,
    RIDGE_LAMBDA,
    SCREENING_BASELINE_SAMPLES,
    TRAIN_NEGATIVES_PER_TARGET,
    run_v29_quadratic_reverse_ranking_audit,
)


def _fmt(value: float, digits: int = 4) -> str:
    return "nan" if value != value else f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="v2.9 strict nested quadratic reverse-learning ranking audit"
    )
    parser.add_argument("--start-index", type=int, default=BACKTEST_START_INDEX)
    parser.add_argument(
        "--min-meta-train-targets", type=int, default=MIN_META_TRAIN_TARGETS
    )
    parser.add_argument(
        "--train-negatives-per-target", type=int, default=TRAIN_NEGATIVES_PER_TARGET
    )
    parser.add_argument(
        "--baseline-samples", type=int, default=SCREENING_BASELINE_SAMPLES
    )
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument("--ridge-lambda", type=float, default=RIDGE_LAMBDA)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    df = load_lotto_data()
    payload = run_v29_quadratic_reverse_ranking_audit(
        df,
        start_index=args.start_index,
        min_meta_train_targets=args.min_meta_train_targets,
        train_negatives_per_target=args.train_negatives_per_target,
        baseline_samples=args.baseline_samples,
        bootstrap_reps=args.bootstrap_reps,
        ridge_lambda=args.ridge_lambda,
        progress_every=args.progress_every,
        include_rows=False,
    )
    payload["run_metadata"] = {
        "runner_commit": current_git_commit(PROJECT_ROOT),
        "bootstrap_block_size": DEFAULT_BOOTSTRAP_BLOCK,
        "data_fingerprint": dataframe_fingerprint(df),
    }

    print("=" * 108)
    print("V2.9 QUADRATIC REVERSE-LEARNING NESTED RANKING AUDIT")
    print("=" * 108)
    fingerprint = payload["run_metadata"]["data_fingerprint"]
    print(f"runner commit: {payload['run_metadata']['runner_commit'] or 'unknown'}")
    print(
        f"data: rows={fingerprint['row_count']} latest={fingerprint['latest_draw']} "
        f"sha256={fingerprint['sha256'][:12]}..."
    )
    print(f"history start index: {payload['start_index']}")
    print(f"minimum solved meta-training targets: {payload['min_meta_train_targets']}")
    print(f"training negatives / solved target: {payload['train_negatives_per_target']}")
    print(f"fair evaluation samples / outer target: {payload['baseline_samples_per_outer_target']}")
    print(f"ridge lambda: {payload['ridge_lambda']}")
    print(f"quadratic basis width: {payload['basis_width']}")
    print(f"block bootstrap reps: {payload['bootstrap_reps']}")
    print(f"block bootstrap size: {payload['run_metadata']['bootstrap_block_size']}")
    print("pattern type: not used")
    print("same fair sample paths as v2.8: yes")
    print()

    result = payload["outer_summary"]
    low, high = result["percentile_minus_50_block_bootstrap_95_ci"]
    print("outer ranking result")
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
        f"historical_gate_pass={result['historical_gate_pass']} "
        f"exploratory_candidate={result['exploratory_screening_candidate']}"
    )
    print(f"  outer tests: {result['tests']}")
    print()

    latest = payload["latest_model_for_next_draw"]
    print("latest quadratic model (diagnostic only; top 15 |scaled weight|)")
    print(
        f"  solved_targets={latest['solved_targets']} pair_rows={latest['pair_rows']} "
        f"ridge_lambda={latest['ridge_lambda']} basis_width={latest['basis_width']}"
    )
    ranked = sorted(
        latest["features"].items(),
        key=lambda item: abs(float(item[1]["scaled_weight"])),
        reverse=True,
    )[:15]
    for name, values in ranked:
        print(
            f"  {name}: effective_weight={values['effective_weight']:.10f} "
            f"scaled_weight={values['scaled_weight']:.10f} "
            f"rms={values['rms']:.10f}"
        )

    print()
    print("Gate and research-budget rule:")
    print("  - overall mean percentile > 50")
    print("  - percentile-minus-50 CI95 lower bound > 0")
    print("  - recent300 >= 50 and recent100 >= 50")
    print(
        f"  - survivor => frozen {CONFIRMATION_BASELINE_SAMPLES:,}-candidate confirmation; "
        "not direct production approval"
    )
    print("  - failure => freeze new model-class search on the current 1,235-draw dataset")
    print("  - no lambda/term/degree/horizon retuning from this result")
    print("  - production recommendation path remains unchanged")

    if args.output_json is not None:
        saved = write_json_artifact(payload, args.output_json)
        print(f"audit JSON saved: {saved}")


if __name__ == "__main__":
    main()
