import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lotto_engine.loader import load_lotto_data
from lotto_engine.profiles import build_profile


def main() -> None:
    profile = build_profile(load_lotto_data())
    dynamics = profile["number_dynamics"]
    print("NUMBER DYNAMICS (descriptive only; no popularity avoidance)")
    print("number historical_rate recent_rate draws_since_seen")
    for number in range(1, 46):
        item = dynamics[number]
        print(
            f"{number:2d} {item['historical_rate']:.5f} "
            f"{item['recent_rate']:.5f} {item['draws_since_seen']:3d}"
        )


if __name__ == "__main__":
    main()
