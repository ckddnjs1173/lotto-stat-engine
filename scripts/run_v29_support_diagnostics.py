from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lotto_engine.v29_support_diagnostics import (
    DEFAULT_REFERENCE_SAMPLES,
    DEFAULT_REFERENCE_SEED,
    DEFAULT_TOP_JSON,
    run_v29_support_diagnostics,
    write_v29_support_diagnostics,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose v2.9 raw TOP-K extrapolation against a fair next-draw reference sample"
    )
    parser.add_argument("--top-json", type=Path, default=DEFAULT_TOP_JSON)
    parser.add_argument("--reference-samples", type=int, default=DEFAULT_REFERENCE_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_REFERENCE_SEED)
    parser.add_argument("--progress-every", type=int, default=25_000)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/cache/v29_support_diagnostic.json"),
    )
    args = parser.parse_args()

    payload = run_v29_support_diagnostics(
        top_json=args.top_json,
        reference_samples=args.reference_samples,
        seed=args.seed,
        progress_every=args.progress_every,
    )
    saved = write_v29_support_diagnostics(payload, args.output_json)

    print("=" * 92)
    print("LOTTO STAT ENGINE v2.9 - RAW MODEL SUPPORT / EXTRAPOLATION DIAGNOSTIC")
    print("=" * 92)
    print(f"reference samples: {payload['reference_samples']:,}")
    score_ref = payload["raw_score_reference"]
    print(
        "fair raw score reference: "
        f"mean={score_ref['mean']:.8f} std={score_ref['std']:.8f} "
        f"min={score_ref['min']:.8f} max={score_ref['max']:.8f}"
    )
    print()

    for rank, item in enumerate(payload["top_candidate_diagnostics"], 1):
        numbers = " ".join(str(number) for number in item["numbers"])
        print(
            f"#{rank} {numbers} score={item['raw_model_score']:.12f} "
            f"fair_percentile={item['raw_score_percentile_vs_fair_reference']:.6f} "
            f"outside99={item['outside_central_99_feature_count']} "
            f"outside99.9={item['outside_central_99_9_feature_count']}"
        )
        for feature_name in ("sum_abs_deviation_138", "consecutive_pairs"):
            detail = item["feature_support"][feature_name]
            print(
                f"  {feature_name}: value={detail['value']:.8f} "
                f"pct={detail['percentile_vs_fair_reference']:.6f} "
                f"robust_z={detail['robust_z']:.4f}"
            )
        concentration = item["contribution_concentration"]
        print(
            "  contribution concentration: "
            f"top5={concentration['top5_abs_share']:.4f} "
            f"top10={concentration['top10_abs_share']:.4f}"
        )

    print()
    print("sum_abs_deviation_138 bucket score shape:")
    for row in payload["score_shape"]["sum_abs_deviation_buckets"]:
        print(
            f"  {row['bucket']}: n={row['count']:,} "
            f"mean={row['mean_raw_score']:.8f} max={row['max_raw_score']:.8f}"
        )

    print()
    print("consecutive_pairs bucket score shape:")
    for row in payload["score_shape"]["consecutive_pair_buckets"]:
        print(
            f"  {row['consecutive_pairs']}: n={row['count']:,} "
            f"mean={row['mean_raw_score']:.8f} max={row['max_raw_score']:.8f}"
        )

    print()
    print("diagnostic only: no score clipping, rescaling, or reranking was applied")
    print(f"diagnostic JSON saved: {saved}")


if __name__ == "__main__":
    main()
