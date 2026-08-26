import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lotto_engine.final_backtest import WalkForwardConfig, run_final_walk_forward_backtest
from lotto_engine.loader import LottoDataError, load_lotto_data


def _print_summary(label: str, values: dict) -> None:
    print(
        f"{label}: tests={values['tests']}, "
        f"mean_best_hit={values['mean_best_hit']:.4f}, "
        f"mean_ticket_hit={values['mean_ticket_hit']:.4f}, "
        f"coverage={values['mean_coverage']:.4f}, "
        f"3+={values['hit3_rate']:.4f}, "
        f"4+={values['hit4_rate']:.4f}, "
        f"5+={values['hit5_rate']:.4f}, "
        f"6={values['hit6_rate']:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-count", type=int, default=2000)
    parser.add_argument("--start-index", type=int, default=300)
    parser.add_argument("--calibration-index", type=int, default=300)
    parser.add_argument("--max-targets", type=int, default=None)
    args = parser.parse_args()

    config = WalkForwardConfig(
        start_index=args.start_index,
        candidate_count=args.candidate_count,
        max_targets=args.max_targets,
        calibration_index=args.calibration_index,
    )
    try:
        payload = run_final_walk_forward_backtest(load_lotto_data(), config)
    except (LottoDataError, ValueError) as exc:
        print(f"Final walk-forward backtest failed: {exc}")
        raise SystemExit(1)

    print("LOTTO STAT ENGINE FINAL - WALK-FORWARD VALIDATION")
    print(f"tested targets: {payload['tested_targets']}")
    print(f"candidate sample per target: {payload['candidate_count']}")
    print(f"weight source: {payload['weight_source']}")
    print()

    for model in ("static", "dynamic", "dynamic_family", "final"):
        print(model.upper())
        for window in ("overall", "recent_300", "recent_100"):
            _print_summary(window, payload["models"][model][window])
        print()

    print("TYPE BUDGET")
    for pattern_type, values in payload["type_budget"].items():
        print(
            f"{pattern_type}: mean_slots_given_actual_type="
            f"{values['mean_slots_given_actual_type']:.4f}"
        )

    mixed = payload["mixed_family"]
    print("MIXED FAMILY")
    print(
        f"tests={mixed['tests']}, "
        f"actual_family_present_rate={mixed['actual_family_present_rate']:.4f}"
    )

    verdict = payload["verdict"]
    print("FINAL VERDICT")
    print(
        f"{verdict['label']} | better={verdict['final_better_targets']} "
        f"worse={verdict['final_worse_targets']} equal={verdict['equal_targets']}"
    )


if __name__ == "__main__":
    main()
