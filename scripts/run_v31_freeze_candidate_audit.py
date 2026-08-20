from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lotto_engine.loader import LottoDataError, load_lotto_data
from lotto_engine.v31_audit_utils import write_json
from lotto_engine.v31_freeze_candidate_audit import run_freeze_candidate_audit


def _print_delta(label: str, item: dict) -> None:
    print(
        f"{label}: mean={item['mean_percentile_delta']:+.4f} "
        f"recent300={item['recent_300_delta']:+.4f} "
        f"recent100={item['recent_100_delta']:+.4f} "
        f"CI={item['block_bootstrap_95_ci']} => {item['direction']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Final v3.1 pre-freeze shortlist audit: exact structural null, 4096 nested "
            "reference, nested feature removals, and 64/128/256 training-negative stability"
        )
    )
    parser.add_argument("--reference-count", type=int, default=4096)
    parser.add_argument("--baseline-samples", type=int, default=500)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--latest-candidate-count", type=int, default=20000)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/cache/v31_freeze_candidate_audit.json"),
    )
    args = parser.parse_args()

    try:
        df = load_lotto_data()
        payload = run_freeze_candidate_audit(
            df,
            reference_count=args.reference_count,
            baseline_samples=args.baseline_samples,
            bootstrap_reps=args.bootstrap_reps,
            latest_candidate_count=args.latest_candidate_count,
            progress_every=args.progress_every,
        )
    except (LottoDataError, ValueError, RuntimeError) as exc:
        print(f"freeze candidate audit failed: {exc}")
        raise SystemExit(1) from exc

    print("=" * 100)
    print("LOTTO STAT ENGINE v3.1 - FINAL PRE-FREEZE CANDIDATE AUDIT")
    print("NO MODEL IS AUTOMATICALLY PROMOTED")
    print("=" * 100)
    print(
        f"representation: exact structural + nested reference "
        f"{payload['representation']['reference_count']}"
    )
    print()

    print("HISTORICAL SCENARIOS")
    print("-" * 100)
    for name, item in payload["historical"]["scenario_summaries"].items():
        print(
            f"{name}: mean={item['mean_percentile']:.4f} "
            f"recent300={item['recent_300_mean_percentile']:.4f} "
            f"recent100={item['recent_100_mean_percentile']:.4f} "
            f"CI50={item['percentile_minus_50_block_bootstrap_95_ci']}"
        )
    print()

    print("CANDIDATE MINUS EXACT FULL11")
    print("-" * 100)
    for name, item in payload["historical"]["paired_candidate_minus_exact_full11"].items():
        _print_delta(name, item)
    print()

    print("INCREMENTAL NESTED FEATURE REMOVAL")
    print("-" * 100)
    for name, item in payload["historical"]["incremental_nested_deltas"].items():
        _print_delta(name, item)
    print()

    print("TRAINING-NEGATIVE STABILITY")
    print("-" * 100)
    for finalist, item in payload["negative_count_stability"].items():
        print(finalist)
        for transition, comparison in item["nested_convergence"].items():
            weights = comparison["weights"]
            ranking = comparison["ranking"]
            print(
                f"  {transition}: weight_corr={weights['pearson_weight_correlation']:.8f} "
                f"weight_rel_delta={weights['l2_relative_delta']:.6f} "
                f"sign_disagree={weights['sign_disagreement_count']} "
                f"rank_spearman={ranking['spearman_rank_correlation']:.8f} "
                f"top10_jaccard={ranking['top_jaccard'].get('10')} "
                f"top100_jaccard={ranking['top_jaccard'].get('100')}"
            )
        print()

    saved = write_json(payload, args.output_json)
    print(f"freeze candidate JSON saved: {saved}")


if __name__ == "__main__":
    main()
