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
    print("LOTTO STAT ENGINE v2.5 - MIXED SUBTYPE ALLOCATION BACKTEST")
    print(payload["score_disclaimer"])
    print(f"mixed target tests: {payload['mixed_tests']}")
    print(f"baseline samples per target: {payload['baseline_samples']}")
    for label in ("v231", "v25"):
        values = payload[label]
        print(f"{label}: mean={values['mean_percentile']:.4f}%, median={values['median_percentile']:.4f}%, above_random={values['above_random_ratio']:.4f}, std={values['percentile_std']:.4f}")
    print("RECENT 100/300 STABILITY")
    for label, values in payload["recent_stability"].items():
        print(f"{label}: mean={values['mean_percentile']:.4f}%, median={values['median_percentile']:.4f}%, above_random={values['above_random_ratio']:.4f}")
    print("FAMILY COVERAGE")
    for family, values in payload["family_coverage"].items():
        print(f"{family}: target_slots={values['target_slots']}, actual_matches={values['actual_matches']}")


if __name__ == "__main__":
    main()
