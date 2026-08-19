from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lotto_engine.loader import LottoDataError, load_lotto_data
from lotto_engine.v31_audit_utils import write_json
from lotto_engine.v31_exact_null_audit import run_exact_structural_null_audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="v3.1 sampled-vs-exact structural fair-null walk-forward audit"
    )
    parser.add_argument("--baseline-samples", type=int, default=500)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--latest-candidate-count", type=int, default=20000)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/cache/v31_exact_structural_null_audit.json"),
    )
    args = parser.parse_args()

    try:
        payload = run_exact_structural_null_audit(
            load_lotto_data(),
            baseline_samples=args.baseline_samples,
            bootstrap_reps=args.bootstrap_reps,
            latest_candidate_count=args.latest_candidate_count,
            progress_every=args.progress_every,
        )
    except (LottoDataError, ValueError, RuntimeError) as exc:
        print(f"exact-null audit failed: {exc}")
        raise SystemExit(1) from exc

    print("=" * 100)
    print("V3.1 EXACT STRUCTURAL NULL AUDIT - DIAGNOSTIC ONLY")
    print("=" * 100)
    for name, item in payload["scenario_summaries"].items():
        print(
            f"{name}: mean={item['mean_percentile']:.4f} "
            f"recent300={item['recent_300_mean_percentile']:.4f} "
            f"recent100={item['recent_100_mean_percentile']:.4f}"
        )
    delta = payload["exact_minus_sampled"]
    print(
        f"exact-sampled: mean={delta['mean_percentile_delta']:+.4f} "
        f"recent300={delta['recent_300_delta']:+.4f} "
        f"recent100={delta['recent_100_delta']:+.4f} "
        f"CI={delta['block_bootstrap_95_ci']} => {delta['direction']}"
    )
    print("latest ranking comparison:", payload["latest_sampled_ranking_comparison"])
    saved = write_json(payload, args.output_json)
    print(f"audit JSON saved: {saved}")


if __name__ == "__main__":
    main()
