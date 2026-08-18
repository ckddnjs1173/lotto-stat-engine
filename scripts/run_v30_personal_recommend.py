from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lotto_engine.config import DEFAULT_CANDIDATE_COUNT, TOP_K_RECOMMENDATIONS
from lotto_engine.loader import LottoDataError
from lotto_engine.v30_null_bounded_reverse_ranking import REFERENCE_SAMPLES_PER_TARGET
from lotto_engine.v30_personal_recommendation import (
    generate_v30_personal_recommendations,
    print_v30_recommendations,
    write_v30_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate final personal v3.0 bounded reverse-ranking recommendations"
    )
    parser.add_argument(
        "--sampled",
        action="store_true",
        help="use sampled candidates instead of all 8,145,060 combinations",
    )
    parser.add_argument("--candidate-count", type=int, default=DEFAULT_CANDIDATE_COUNT)
    parser.add_argument("--top-k", type=int, default=TOP_K_RECOMMENDATIONS)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=1_000_000)
    parser.add_argument(
        "--reference-samples",
        type=int,
        default=REFERENCE_SAMPLES_PER_TARGET,
        help="frozen fair-null reference size used by the v3.0 model",
    )
    parser.add_argument(
        "--validation-json",
        type=Path,
        default=Path("data/cache/v30_null_bounded_screen.json"),
        help="optional historical validation JSON; diagnostics only",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/cache/v30_personal.json"),
    )
    args = parser.parse_args()

    try:
        payload = generate_v30_personal_recommendations(
            seed_offset=args.seed_offset,
            candidate_count=args.candidate_count,
            exhaustive=not args.sampled,
            top_k=args.top_k,
            progress_every=args.progress_every,
            validation_json=args.validation_json,
            reference_samples=args.reference_samples,
        )
    except (LottoDataError, ValueError) as exc:
        print(f"추천 실패: {exc}")
        raise SystemExit(1)

    print_v30_recommendations(payload)
    saved = write_v30_json(payload, args.output_json)
    print()
    print(f"personal JSON saved: {saved}")


if __name__ == "__main__":
    main()
