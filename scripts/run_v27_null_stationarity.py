from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lotto_engine.loader import load_lotto_data
from lotto_engine.v27_audit_artifact import (
    current_git_commit,
    dataframe_fingerprint,
    write_json_artifact,
)
from lotto_engine.v27_null_stationarity import (
    DEFAULT_MAX_LAG,
    DEFAULT_RECENT_WINDOW,
    SCREENING_FAIR_SAMPLES,
    SCREENING_PERMUTATION_REPS,
    run_v27_null_stationarity_screen,
)


def _fmt(value: float) -> str:
    return "nan" if value != value else f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cheap v2.7 fair-null and stationarity screening lab"
    )
    parser.add_argument("--fair-samples", type=int, default=SCREENING_FAIR_SAMPLES)
    parser.add_argument(
        "--permutation-reps", type=int, default=SCREENING_PERMUTATION_REPS
    )
    parser.add_argument("--max-lag", type=int, default=DEFAULT_MAX_LAG)
    parser.add_argument("--recent-window", type=int, default=DEFAULT_RECENT_WINDOW)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    df = load_lotto_data()
    payload = run_v27_null_stationarity_screen(
        df,
        fair_samples=args.fair_samples,
        permutation_reps=args.permutation_reps,
        max_lag=args.max_lag,
        recent_window=args.recent_window,
    )
    payload["run_metadata"] = {
        "runner_commit": current_git_commit(PROJECT_ROOT),
        "data_fingerprint": dataframe_fingerprint(df),
    }

    print("=" * 92)
    print("V2.7 NULL / STATIONARITY LAB - PHASE 3A SCREENING")
    print("=" * 92)
    print(f"runner commit: {payload['run_metadata']['runner_commit'] or 'unknown'}")
    fingerprint = payload["run_metadata"]["data_fingerprint"]
    print(
        f"data: rows={fingerprint['row_count']} latest={fingerprint['latest_draw']} "
        f"sha256={fingerprint['sha256'][:12]}..."
    )
    print(
        f"fair samples={payload['fair_samples']} "
        f"permutation reps={payload['permutation_reps']} "
        f"max lag={payload['max_lag']} recent window={payload['recent_window']}"
    )
    print()

    print("Scalar marginal vs fair 6/45:")
    for feature, result in payload["scalar_marginal_vs_fair"].items():
        print(
            f"  {feature}: hist={_fmt(result['historical_mean'])} "
            f"fair={_fmt(result['fair_mean'])} "
            f"mean-z={_fmt(result['historical_mean_z_vs_fair'])}"
        )
    print()

    print("Time-order screening (Holm-adjusted):")
    for feature, result in payload["series_screen"].items():
        print(
            f"  {feature}: max|acf|={_fmt(result['max_abs_acf'])} "
            f"p_acf={_fmt(result['holm_p_max_abs_acf'])} "
            f"p_recent={_fmt(result['holm_p_recent_shift'])} "
            f"p_cusum={_fmt(result['holm_p_cusum'])}"
        )
    print()

    transition = payload["pattern_transition"]
    print(
        "Pattern transition: "
        f"MI={_fmt(transition['mutual_information'])} "
        f"permutation-p={_fmt(transition['p_value'])}"
    )
    print("Pattern proportions:")
    for pattern, result in payload["pattern_type_marginal_vs_fair"].items():
        print(
            f"  {pattern}: hist={_fmt(result['historical'])} "
            f"fair={_fmt(result['fair'])} diff={_fmt(result['difference'])}"
        )

    candidates = payload["screening_candidates"]
    print()
    print("Screening candidates (confirmation required):")
    print(f"  serial dependence: {candidates['serial_dependence']}")
    print(f"  recent shift: {candidates['recent_distribution_shift']}")
    print(f"  change point: {candidates['change_point']}")
    print(
        "  pattern transition dependence: "
        f"{candidates['pattern_transition_dependence']}"
    )
    print()
    print("Gate:")
    print("  - no candidate => do not spend compute on AR/ARIMA or regime models")
    print("  - screening candidate => confirm only that statistic with larger reps")
    print("  - transition screening is not production Markov evidence")

    if args.output_json is not None:
        saved = write_json_artifact(payload, args.output_json)
        print(f"audit JSON saved: {saved}")


if __name__ == "__main__":
    main()
