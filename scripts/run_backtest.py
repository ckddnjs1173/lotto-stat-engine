import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lotto_engine.backtest import run_walk_forward_backtest
from lotto_engine.loader import LottoDataError, load_lotto_data


def main() -> None:
    print("WALK-FORWARD ACTUAL-VS-RANDOM BACKTEST")
    try:
        df = load_lotto_data()
        payload = run_walk_forward_backtest(df)
    except (LottoDataError, ValueError) as exc:
        print(f"백테스트 실패: {exc}")
        raise SystemExit(1)

    print(f"테스트 수: {payload['total_tests']}")
    print(f"회차별 랜덤 기준선 샘플 수: {payload['baseline_samples']}")
    print()
    print("Feature 평균 구조 점수:")
    for key, value in payload["feature_scores"].items():
        print(f"{key}: {value:.4f}")
    print()
    print("Actual-vs-random percentile:")
    for key, value in payload["percentile_scores"].items():
        print(f"{key}: {value:.4f}%")
    print()
    print("Stability:")
    for key, value in payload["stability_scores"].items():
        print(f"{key}: {value:.4f}")
    print()
    print("Final weights:")
    for key, value in payload["final_weights"].items():
        print(f"{key}: {value:.6f}")
    print()
    print("가중치 저장 완료: data/cache/feature_weights.json")


if __name__ == "__main__":
    main()
