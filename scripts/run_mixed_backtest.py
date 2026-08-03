import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lotto_engine.loader import LottoDataError, load_lotto_data
from lotto_engine.mixed_backtest import run_mixed_walk_forward_backtest


def main() -> None:
    try:
        payload = run_mixed_walk_forward_backtest(load_lotto_data())
    except (LottoDataError, ValueError) as exc:
        print(f"Mixed backtest failed: {exc}")
        raise SystemExit(1)
    print("LOTTO STAT ENGINE v2.3.1 - PORTFOLIO MIXED-SLOT WALK-FORWARD BACKTEST")
    print(payload["score_disclaimer"])
    print(f"mixed target tests: {payload['mixed_tests']}")
    print(f"baseline samples per target: {payload['baseline_samples']}")
    print(f"mean tie-safe percentile: {payload['mean_percentile']:.4f}%")
    print(f"median tie-safe percentile: {payload['median_percentile']:.4f}%")
    print(f"percentile std: {payload['percentile_std']:.4f}")
    print(f"above-random ratio: {payload['above_random_ratio']:.4f}")


if __name__ == "__main__":
    main()
