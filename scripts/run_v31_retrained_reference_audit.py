from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lotto_engine.loader import LottoDataError, load_lotto_data
from lotto_engine.v31_audit_utils import write_json
from lotto_engine.v31_retrained_reference_audit import (
    run_retrained_reference_stability_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="v3.1 full retraining stability audit under alternative reference streams"
    )
    parser.add_argument("--reference-counts", nargs="+", type=int, default=[1024, 4096])
    parser.add_argument("--stream-deltas", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--latest-candidate-count", type=int, default=20000)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/cache/v31_retrained_reference_audit.json"),
    )
    args = parser.parse_args()

    try:
        payload = run_retrained_reference_stability_audit(
            load_lotto_data(),
            reference_counts=tuple(args.reference_counts),
            stream_deltas=tuple(args.stream_deltas),
            latest_candidate_count=args.latest_candidate_count,
            progress_every=args.progress_every,
        )
    except (LottoDataError, ValueError, RuntimeError) as exc:
        print(f"retrained reference audit failed: {exc}")
        raise SystemExit(1) from exc

    print("=" * 100)
    print("V3.1 RETRAINED REFERENCE STABILITY AUDIT - NUMERICAL STABILITY ONLY")
    print("=" * 100)
    print("baseline:", payload["baseline_variant"])
    for label, item in payload["comparisons_vs_current_policy"].items():
        weights = item["weights"]
        ranking = item["latest_sampled_ranking"]
        print(
            f"{label}: weight_corr={weights['pearson_weight_correlation']:.6f} "
            f"weight_rel_l2={weights['l2_relative_delta']:.6f} "
            f"sign_disagree={weights['sign_disagreement_count']} "
            f"rank_spearman={ranking['spearman_rank_correlation']:.6f} "
            f"topJ={ranking['top_jaccard']}"
        )
    saved = write_json(payload, args.output_json)
    print(f"audit JSON saved: {saved}")


if __name__ == "__main__":
    main()
