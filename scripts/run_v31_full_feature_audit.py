from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lotto_engine.loader import LottoDataError, load_lotto_data
from lotto_engine.v31_full_feature_audit import run_full_feature_group_audit


def _print_delta(label: str, item: dict) -> None:
    print(
        f"{label}: mean={item['full_minus_without_mean_percentile']:+.4f} "
        f"recent300={item['recent_300_delta']:+.4f} "
        f"recent100={item['recent_100_delta']:+.4f} "
        f"CI={item['block_bootstrap_95_ci']} => {item['direction']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict v3.1 FULL11 all-feature and correlated-group walk-forward audit"
    )
    parser.add_argument("--baseline-samples", type=int, default=500)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/cache/v31_full_feature_group_audit.json"),
    )
    args = parser.parse_args()

    try:
        df = load_lotto_data()
        payload = run_full_feature_group_audit(
            df,
            baseline_samples=args.baseline_samples,
            bootstrap_reps=args.bootstrap_reps,
            progress_every=args.progress_every,
        )
    except (LottoDataError, ValueError, RuntimeError) as exc:
        print(f"full feature audit failed: {exc}")
        raise SystemExit(1) from exc

    print("=" * 100)
    print("LOTTO STAT ENGINE v3.1 - FULL FEATURE / GROUP AUDIT")
    print("DIAGNOSTIC ONLY; NO AUTOMATIC FEATURE PROMOTION OR DELETION")
    print("=" * 100)
    full = payload["scenario_summaries"]["full"]
    print(
        f"FULL11 mean={full['mean_percentile']:.4f} "
        f"recent300={full['recent_300_mean_percentile']:.4f} "
        f"recent100={full['recent_100_mean_percentile']:.4f} "
        f"CI50={full['percentile_minus_50_block_bootstrap_95_ci']}"
    )
    print()

    print("SINGLE FEATURE LEAVE-ONE-OUT")
    print("-" * 100)
    for feature, item in payload["paired_full_minus_without_feature"].items():
        _print_delta(feature, item)
    print()

    print("CORRELATED GROUP ABLATION")
    print("-" * 100)
    for group, item in payload["paired_full_minus_without_group"].items():
        _print_delta(group, item)
    print()

    print("WEIGHT SIGN STABILITY")
    print("-" * 100)
    for feature, item in payload["full_model_weight_stability"].items():
        print(
            f"{feature}: mean={item['mean']:+.8f} std={item['std']:.8f} "
            f"positive={item['positive_fraction']:.3f} negative={item['negative_fraction']:.3f} "
            f"sign_flips={item['sign_flip_count']} recent100={item['recent_100_mean']:+.8f}"
        )
    print()

    condition = payload["scaled_second_moment_condition_summary"]
    print(
        "scaled second moment condition: "
        f"tests={condition['tests']} mean={condition['mean']:.4f} "
        f"median={condition['median']:.4f} max={condition['max']:.4f}"
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print()
    print(f"audit JSON saved: {args.output_json.resolve()}")


if __name__ == "__main__":
    main()
