from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lotto_engine.loader import LottoDataError, load_lotto_data
from lotto_engine.v31_audit_utils import write_json
from lotto_engine.v31_dependency_audit import run_dependency_audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="v3.1 ridge conditioning and feature dependency audit"
    )
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/cache/v31_dependency_audit.json"),
    )
    args = parser.parse_args()

    try:
        payload = run_dependency_audit(
            load_lotto_data(),
            progress_every=args.progress_every,
        )
    except (LottoDataError, ValueError, RuntimeError) as exc:
        print(f"dependency audit failed: {exc}")
        raise SystemExit(1) from exc

    print("=" * 100)
    print("V3.1 RIDGE / FEATURE DEPENDENCY AUDIT - DIAGNOSTIC ONLY")
    print("=" * 100)
    conditioning = payload["historical_conditioning"]
    for name, item in conditioning.items():
        print(
            f"{name}: mean={item['mean']:.4f} median={item['median']:.4f} "
            f"max={item['max']:.4f} recent100={item['recent_100_mean']:.4f}"
        )
    print("predeclared dependency pairs")
    for item in payload["predeclared_dependency_pairs"]:
        print(
            f"  {item['requested_left']} <-> {item['requested_right']}: "
            f"{item['second_moment_cosine']:+.6f}"
        )
    print("strongest dependencies")
    for item in payload["strongest_dependency_pairs"][:10]:
        print(
            f"  {item['left']} <-> {item['right']}: "
            f"{item['second_moment_cosine']:+.6f}"
        )
    saved = write_json(payload, args.output_json)
    print(f"audit JSON saved: {saved}")


if __name__ == "__main__":
    main()
