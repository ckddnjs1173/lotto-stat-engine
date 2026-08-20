import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lotto_engine.backtest import run_walk_forward_backtest
from lotto_engine.loader import LottoDataError, load_lotto_data


def main() -> None:
    print("v2.2 WALK-FORWARD ACTUAL-VS-RANDOM BACKTEST")
    try:
        payload = run_walk_forward_backtest(load_lotto_data())
    except (LottoDataError, ValueError) as exc:
        print(f"백테스트 실패: {exc}")
        raise SystemExit(1)
    print(payload["score_disclaimer"])
    print(f"테스트 수: {payload['total_tests']}")
    print(f"prediction_score percentile: {payload['prediction_percentile']:.4f}%")
    print("\nComponent percentiles:")
    for key, value in payload["component_percentiles"].items():
        print(f"{key}: {value:.4f}%")
    for window, values in payload["stability_windows"].items():
        print(f"\nRecent {window} stability:")
        for key, summary in values.items():
            print(
                f"{key}: mean={summary['mean_percentile']:.4f}%, "
                f"std={summary['percentile_std']:.4f}, stability={summary['stability']:.4f}"
            )


if __name__ == "__main__":
    main()
