from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lotto_engine.loader import LottoDataError
from lotto_engine.v31_reference_stability_audit import run_reference_stability_audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit v3.1 latest fair-reference numerical stability with fitted weights held fixed. "
            "This is not a parameter-tuning runner."
        )
    )
    parser.add_argument("--scenario", choices=("full11", "clean3"), default="full11")
    parser.add_argument("--candidate-count", type=int, default=20_000)
    parser.add_argument("--candidate-seed-offset", type=int, default=901)
    parser.add_argument(
        "--reference-counts",
        type=int,
        nargs="+",
        default=[1_024, 4_096, 16_384],
    )
    parser.add_argument(
        "--seed-deltas",
        type=int,
        nargs="+",
        default=[1, 2, 3],
    )
    parser.add_argument("--progress-every", type=int, default=5_000)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/cache/v31_reference_stability_audit.json"),
    )
    args = parser.parse_args()

    try:
        payload = run_reference_stability_audit(
            scenario=args.scenario,
            candidate_count=args.candidate_count,
            candidate_seed_offset=args.candidate_seed_offset,
            reference_counts=tuple(args.reference_counts),
            seed_deltas=tuple(args.seed_deltas),
            progress_every=args.progress_every,
        )
    except (LottoDataError, ValueError, RuntimeError) as exc:
        print(f"reference stability audit failed: {exc}")
        raise SystemExit(1) from exc

    print("=" * 100)
    print("LOTTO STAT ENGINE v3.1 - REFERENCE STABILITY AUDIT")
    print("WEIGHTS HELD FIXED; THIS RUN DOES NOT PICK A BETTER REFERENCE")
    print("=" * 100)
    print(
        f"scenario={payload['scenario']} latest={payload['latest_draw']} "
        f"target={payload['target_draw']} candidates={payload['candidate_count']:,}"
    )
    print()

    print("CURRENT POLICY SEED SENSITIVITY")
    print("-" * 100)
    for name, result in payload["comparisons_vs_current_reference"].items():
        if not name.startswith("current_policy_seed_delta:"):
            continue
        top = result["top_jaccard"]
        print(
            f"{name}: spearman={result['spearman_rank_correlation']:.6f} "
            f"mean_abs_score_delta={result['mean_absolute_score_delta']:.8f} "
            f"J10={top.get('10', float('nan')):.4f} "
            f"J100={top.get('100', float('nan')):.4f} "
            f"J1000={top.get('1000', float('nan')):.4f}"
        )
    print()

    print("NESTED COUNT CONVERGENCE")
    print("-" * 100)
    for name, result in payload["nested_reference_convergence"].items():
        top = result["top_jaccard"]
        print(
            f"{name}: spearman={result['spearman_rank_correlation']:.6f} "
            f"mean_abs_score_delta={result['mean_absolute_score_delta']:.8f} "
            f"J10={top.get('10', float('nan')):.4f} "
            f"J100={top.get('100', float('nan')):.4f} "
            f"J1000={top.get('1000', float('nan')):.4f}"
        )
    print()

    print("SATURATION RATES")
    print("-" * 100)
    for name, summary in payload["scenario_summaries"].items():
        highest = sorted(
            summary["feature_saturation_rates"].items(),
            key=lambda item: item[1],
            reverse=True,
        )[:5]
        print(
            f"{name}: score_unique={summary['exact_score_unique_count']:,}/"
            f"{payload['candidate_count']:,}; "
            + ", ".join(f"{feature}={rate:.4f}" for feature, rate in highest)
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print()
    print(f"audit JSON saved: {args.output_json.resolve()}")


if __name__ == "__main__":
    main()
