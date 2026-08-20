from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lotto_engine.loader import load_lotto_data
from lotto_engine.v31_joint_ablation_audit import run_joint_ablation_audit


def _print_scenario(name: str, summary: dict, fair_distance: dict | None = None) -> None:
    dominant = ", ".join(
        f"{item['number']}:{item['inclusion_rate']:.3f}"
        for item in summary["dominant_numbers"][:6]
    )
    print(
        f"{name}: dominant={dominant} "
        f"prev_overlap={summary['mean_previous_draw_overlap']:.3f} "
        f"any_prev={summary['any_previous_draw_overlap_rate']:.3f} "
        f"consecutive={summary['mean_consecutive_pairs']:.3f} "
        f"range={summary['mean_number_range']:.3f}"
    )
    if fair_distance is not None:
        print(
            "  distance from fair: "
            f"prev={fair_distance['previous_draw_overlap_minus_fair']:+.3f} "
            f"any_prev={fair_distance['any_previous_draw_overlap_rate_minus_fair']:+.3f} "
            f"consecutive={fair_distance['consecutive_pairs_minus_fair']:+.3f} "
            f"range={fair_distance['number_range_minus_fair']:+.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Joint v3.1 ablation audit for structural concentration terms"
    )
    parser.add_argument("--historical-baseline-samples", type=int, default=500)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--sampled-latest", action="store_true")
    parser.add_argument("--latest-candidate-count", type=int, default=20000)
    parser.add_argument("--latest-pool-size", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--historical-progress-every", type=int, default=50)
    parser.add_argument("--latest-progress-every", type=int, default=1000000)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/cache/v31_joint_ablation_audit.json"),
    )
    args = parser.parse_args()

    df = load_lotto_data()
    payload = run_joint_ablation_audit(
        df,
        historical_baseline_samples=args.historical_baseline_samples,
        bootstrap_reps=args.bootstrap_reps,
        latest_exhaustive=not args.sampled_latest,
        latest_candidate_count=args.latest_candidate_count,
        latest_pool_size=args.latest_pool_size,
        top_k=args.top_k,
        historical_progress_every=args.historical_progress_every,
        latest_progress_every=args.latest_progress_every,
    )

    print("=" * 100)
    print("LOTTO STAT ENGINE v3.1 - JOINT STRUCTURAL BIAS ABLATION")
    print("prediction formula is NOT modified by this audit")
    print("=" * 100)

    historical = payload["historical_oos"]
    print("HISTORICAL OOS")
    print("-" * 100)
    for name in ("full", "retrained_without:clean3", "retrained_without:clean4"):
        summary = historical["summaries"][name]
        print(
            f"{name}: mean={summary['mean_percentile']:.4f} "
            f"recent300={summary['recent_300_mean_percentile']:.4f} "
            f"recent100={summary['recent_100_mean_percentile']:.4f} "
            f"CI50={summary['percentile_minus_50_block_bootstrap_95_ci']}"
        )
    for label in ("clean3", "clean4"):
        delta = historical["paired_full_minus_without"][label]
        print(
            f"full-minus-{label}: mean={delta['full_minus_without_mean_percentile']:+.4f} "
            f"recent300={delta['recent_300_delta']:+.4f} "
            f"recent100={delta['recent_100_delta']:+.4f} "
            f"CI={delta['block_bootstrap_95_ci']} => {delta['direction']}"
        )
    print()

    latest = payload["latest_ranking_influence"]
    print("LATEST RAW TOP-POOL")
    print("-" * 100)
    print(
        f"latest={latest['latest_draw']} target={latest['target_draw']} "
        f"mode={latest['evaluation_mode']} evaluated={latest['evaluated_count']:,} "
        f"pool={latest['pool_size']}"
    )
    refs = latest["fair_references"]
    print(
        "fair references: "
        f"prev_overlap={refs['expected_previous_draw_overlap']:.3f}, "
        f"any_prev={refs['any_previous_draw_overlap_rate']:.3f}, "
        f"consecutive={refs['expected_consecutive_pairs']:.3f}, "
        f"range={refs['expected_number_range']:.3f}"
    )
    for name in (
        "full",
        "direct_without:clean3",
        "retrained_without:clean3",
        "direct_without:clean4",
        "retrained_without:clean4",
    ):
        _print_scenario(
            name,
            latest["scenario_summaries"][name],
            latest["scenario_fair_distance"][name],
        )
        if name != "full":
            comparison = latest["scenario_comparisons_vs_full"][name]
            print(
                f"  pool overlap={comparison['top_pool_overlap_count_with_full']}/{latest['pool_size']} "
                f"jaccard={comparison['top_pool_jaccard_with_full']:.3f}"
            )
    print()

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"audit JSON saved: {args.output_json.resolve()}")


if __name__ == "__main__":
    main()
