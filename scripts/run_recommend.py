from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lotto_engine.config import DEFAULT_CANDIDATE_COUNT, TOP_K_RECOMMENDATIONS
from lotto_engine.loader import LottoDataError
from lotto_engine.v27_release import (
    generate_release_recommendations,
    print_release_recommendations,
    write_public_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Lotto Stat Engine v2.7.1 evidence recommendations")
    parser.add_argument("--sampled", action="store_true", help="use sampled candidates instead of exhaustive research mode")
    parser.add_argument("--candidate-count", type=int, default=DEFAULT_CANDIDATE_COUNT)
    parser.add_argument("--top-k", type=int, default=TOP_K_RECOMMENDATIONS)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=None, help="optional site/API JSON output path")
    args = parser.parse_args()

    try:
        payload = generate_release_recommendations(
            seed_offset=args.seed_offset,
            candidate_count=args.candidate_count,
            exhaustive=not args.sampled,
            top_k=args.top_k,
        )
    except (LottoDataError, ValueError) as exc:
        print(f"추천 실패: {exc}")
        raise SystemExit(1)

    print_release_recommendations(payload)
    if args.output_json is not None:
        saved = write_public_json(payload, args.output_json)
        print()
        print(f"public JSON saved: {saved}")


if __name__ == "__main__":
    main()
