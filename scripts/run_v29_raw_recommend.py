from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lotto_engine.config import DEFAULT_CANDIDATE_COUNT, TOP_K_RECOMMENDATIONS
from lotto_engine.loader import LottoDataError
from lotto_engine.v29_raw_recommendation import (
    DEFAULT_VALIDATION_JSON,
    generate_v29_raw_recommendations,
    print_v29_raw_recommendations,
    write_v29_raw_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate personal-use Lotto v2.9 recommendations from the raw quadratic model score"
    )
    parser.add_argument(
        "--sampled",
        action="store_true",
        help="use sampled candidates instead of all 8,145,060 combinations",
    )
    parser.add_argument("--candidate-count", type=int, default=DEFAULT_CANDIDATE_COUNT)
    parser.add_argument("--top-k", type=int, default=TOP_K_RECOMMENDATIONS)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1_000_000,
        help="print progress every N evaluated combinations; 0 disables progress",
    )
    parser.add_argument(
        "--validation-json",
        type=Path,
        default=DEFAULT_VALIDATION_JSON,
        help="historical v2.9 audit JSON; diagnostics only, never changes ranking",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="optional personal-site JSON output path",
    )
    args = parser.parse_args()

    try:
        payload = generate_v29_raw_recommendations(
            seed_offset=args.seed_offset,
            candidate_count=args.candidate_count,
            exhaustive=not args.sampled,
            top_k=args.top_k,
            progress_every=args.progress_every,
            validation_json=args.validation_json,
        )
    except (LottoDataError, ValueError) as exc:
        print(f"추천 실패: {exc}")
        raise SystemExit(1)

    print_v29_raw_recommendations(payload)
    if args.output_json is not None:
        saved = write_v29_raw_json(payload, args.output_json)
        print()
        print(f"personal JSON saved: {saved}")


if __name__ == "__main__":
    main()
