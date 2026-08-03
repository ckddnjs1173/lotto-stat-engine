import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lotto_engine.backtest import run_walk_forward_backtest
from lotto_engine.loader import load_lotto_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose low-percentile v2.2 prediction components.")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--baseline-samples", type=int, default=500)
    args = parser.parse_args()
    df = load_lotto_data()
    start_index = max(100, len(df) - max(1, args.rounds))
    result = run_walk_forward_backtest(
        df, start_index=start_index, baseline_samples=max(1, args.baseline_samples)
    )
    print(result["score_disclaimer"])
    print(f"prediction_score percentile: {result['prediction_percentile']:.2f}%")
    print("\nComponents requiring review (< 50th percentile):")
    failures = [
        (key, value) for key, value in result["component_percentiles"].items() if value < 50.0
    ]
    if not failures:
        print("none")
    for key, value in sorted(failures, key=lambda item: item[1]):
        stability = result["component_stability"][key]
        print(f"{key}: percentile={value:.2f}%, stability={stability:.3f}")


if __name__ == "__main__":
    main()
