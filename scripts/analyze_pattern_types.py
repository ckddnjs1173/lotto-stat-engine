import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lotto_engine.loader import load_lotto_data
from lotto_engine.profiles import PATTERN_TYPES, build_profile


def main() -> None:
    profile = build_profile(load_lotto_data())
    frame = profile["feature_frame"]
    print("PATTERN TYPE ANALYSIS")
    print(f"draws: {len(frame)}, latest round: {profile['latest_round']}")
    for pattern in PATTERN_TYPES:
        count = int((frame["pattern_type"] == pattern).sum())
        print(f"{pattern}: {count} ({count / len(frame) * 100:.2f}%)")
    print("\nPattern flag frequencies:")
    for key, value in profile["pattern_frequencies"].items():
        print(f"{key}: {value * 100:.2f}%")
    print("\nRecent-window type distribution:")
    for window, distribution in profile["recent_distributions"].items():
        values = ", ".join(
            f"{key}={distribution['pattern_type_probs'][key] * 100:.2f}%"
            for key in PATTERN_TYPES
        )
        print(f"{window}: {values}")


if __name__ == "__main__":
    main()
