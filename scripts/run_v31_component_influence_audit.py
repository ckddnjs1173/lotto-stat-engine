from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lotto_engine.loader import load_lotto_data
from lotto_engine.v31_component_influence_audit import AUDIT_FEATURES, run_component_influence_audit
from lotto_engine.v31_final_directional_recommendation import write_json


def _dominant(summary: dict, limit: int = 6) -> str:
    return ", ".join(
        f"{item['number']}:{item['inclusion_rate']:.3f}"
        for item in summary["dominant_numbers"][:limit]
    )


def _print_historical(section: dict) -> None:
    print("HISTORICAL OOS RETRAINED LEAVE-ONE-FEATURE-OUT")
    print("-" * 100)
    full = section["summaries"]["full"]
    print(
        f"full mean={full['mean_percentile']:.4f} "
        f"recent300={full['recent_300_mean_percentile']:.4f} "
        f"recent100={full['recent_100_mean_percentile']:.4f} "
        f"CI50={full['percentile_minus_50_block_bootstrap_95_ci']}"
    )
    for feature in AUDIT_FEATURES:
        without = section["summaries"][f"retrained_without:{feature}"]
        paired = section["paired_feature_value"][feature]
        print(
            f"{feature}: without_mean={without['mean_percentile']:.4f} "
            f"full-minus-without={paired['full_minus_without_mean_percentile']:+.4f} "
            f"recent300_delta={paired['recent_300_delta']:+.4f} "
            f"recent100_delta={paired['recent_100_delta']:+.4f} "
            f"CI={paired['block_bootstrap_95_ci']} => {paired['direction']}"
        )
    print()


def _print_latest(section: dict) -> None:
    print("LATEST RAW TOP-POOL COMPONENT INFLUENCE")
    print("-" * 100)
    print(
        f"latest={section['latest_draw']} target={section['target_draw']} "
        f"mode={section['evaluation_mode']} evaluated={section['evaluated_count']:,} "
        f"pool={section['pool_size']}"
    )
    for feature in AUDIT_FEATURES:
        info = section["focused_features"][feature]
        print(
            f"weight {feature}={info['effective_weight']:+.12f} "
            f"({info['learned_direction']})"
        )
    print()

    full = section["scenario_summaries"]["full"]
    print(
        "FULL dominant=" + _dominant(full)
        + f" | prev_overlap={full['mean_previous_draw_overlap']:.3f}"
        + f" any_prev={full['any_previous_draw_overlap_rate']:.3f}"
        + f" consecutive={full['mean_consecutive_pairs']:.3f}"
        + f" any_consecutive={full['any_consecutive_pair_rate']:.3f}"
        + f" range={full['mean_number_range']:.3f}"
    )
    print()

    for feature in AUDIT_FEATURES:
        print(f"[{feature}]")
        for prefix in ("direct_without", "retrained_without"):
            name = f"{prefix}:{feature}"
            summary = section["scenario_summaries"][name]
            delta = section["scenario_comparisons_vs_full"][name]
            print(
                f"  {prefix}: dominant={_dominant(summary)} "
                f"pool_overlap={delta['top_pool_overlap_count_with_full']}/{section['pool_size']} "
                f"jaccard={delta['top_pool_jaccard_with_full']:.3f} "
                f"prev_delta={delta['mean_previous_draw_overlap_delta']:+.3f} "
                f"consecutive_delta={delta['mean_consecutive_pairs_delta']:+.3f} "
                f"range_delta={delta['mean_number_range_delta']:+.3f}"
            )
            changes = delta["largest_number_inclusion_rate_changes"][:5]
            print(
                "    largest number-rate changes: "
                + ", ".join(f"{x['number']}:{x['delta']:+.3f}" for x in changes)
            )
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit v3.1 RAW concentration without modifying the prediction formula"
    )
    parser.add_argument("--skip-historical", action="store_true")
    parser.add_argument("--skip-latest", action="store_true")
    parser.add_argument("--historical-baseline-samples", type=int, default=500)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--sampled-latest", action="store_true")
    parser.add_argument("--latest-candidate-count", type=int, default=20_000)
    parser.add_argument("--latest-pool-size", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--historical-progress-every", type=int, default=50)
    parser.add_argument("--latest-progress-every", type=int, default=1_000_000)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/cache/v31_component_influence_audit.json"),
    )
    args = parser.parse_args()

    payload = run_component_influence_audit(
        load_lotto_data(),
        include_historical=not args.skip_historical,
        include_latest=not args.skip_latest,
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
    print("LOTTO STAT ENGINE v3.1 - COMPONENT INFLUENCE AUDIT")
    print("prediction formula is NOT modified by this audit")
    print("=" * 100)
    if "historical_oos" in payload:
        _print_historical(payload["historical_oos"])
    if "latest_ranking_influence" in payload:
        _print_latest(payload["latest_ranking_influence"])

    saved = write_json(payload, args.output_json)
    print(f"audit JSON saved: {saved}")


if __name__ == "__main__":
    main()
