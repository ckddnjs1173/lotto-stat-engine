import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lotto_engine.loader import load_lotto_data
from lotto_engine.profiles import PATTERN_TYPES, build_profile


def main() -> None:
    profile = build_profile(load_lotto_data())
    print("DRAW PATTERN TRANSITIONS")
    print("source -> target: count, probability")
    for source in PATTERN_TYPES:
        for target in PATTERN_TYPES:
            count = profile["transition_counts"][source][target]
            probability = profile["transition_probs"][source][target]
            print(f"{source:7s} -> {target:7s}: {count:4d}, {probability * 100:6.2f}%")
    print(f"\nLatest type: {profile['latest_pattern_type']}")


if __name__ == "__main__":
    main()
