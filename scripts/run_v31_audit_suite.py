from __future__ import annotations

import argparse
import traceback
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lotto_engine.loader import LottoDataError, dataset_fingerprint, load_lotto_data
from lotto_engine.v31_audit_utils import runtime_identity, write_json
from lotto_engine.v31_component_influence_audit import run_component_influence_audit
from lotto_engine.v31_dependency_audit import run_dependency_audit
from lotto_engine.v31_exact_null_audit import run_exact_structural_null_audit
from lotto_engine.v31_full_feature_audit import run_full_feature_group_audit
from lotto_engine.v31_joint_ablation_audit import run_joint_ablation_audit
from lotto_engine.v31_model_spec import FULL11_BASELINE_SPEC, spec_metadata
from lotto_engine.v31_pair_residual_audit import run_pair_residualization_audit
from lotto_engine.v31_reference_stability_audit import run_reference_stability_audit
from lotto_engine.v31_retrained_reference_audit import (
    run_retrained_reference_stability_audit,
)

AUDIT_ORDER = (
    "reference_fixed_weights",
    "reference_retrained",
    "focused_component",
    "joint_clean3_clean4",
    "full_feature_group",
    "pair_residualization",
    "exact_structural_null",
    "ridge_dependency",
)


def _mode_settings(mode: str) -> dict:
    if mode == "quick":
        return {
            "baseline_samples": 100,
            "bootstrap_reps": 500,
            "latest_candidate_count": 10_000,
            "focused_latest_candidate_count": 20_000,
            "reference_counts": (1_024,),
            "stream_deltas": (0,),
        }
    if mode == "full":
        return {
            "baseline_samples": 500,
            "bootstrap_reps": 2_000,
            "latest_candidate_count": 20_000,
            "focused_latest_candidate_count": 200_000,
            "reference_counts": (1_024, 4_096),
            "stream_deltas": (0, 1),
        }
    raise ValueError("mode must be quick or full")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Complete v3.1 audit suite. Runs numerical/reference, feature, pair, exact-null, "
            "and conditioning diagnostics without promoting a model."
        )
    )
    parser.add_argument("--mode", choices=("quick", "full"), default="full")
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/cache/v31_complete_audit_suite.json"),
    )
    args = parser.parse_args()
    if args.progress_every < 0:
        raise SystemExit("--progress-every must be non-negative")

    settings = _mode_settings(args.mode)
    try:
        df = load_lotto_data()
    except LottoDataError as exc:
        print(f"audit suite failed before start: {exc}")
        raise SystemExit(1) from exc

    payload = {
        "version": "v31_complete_audit_suite_v1",
        "status": "in_progress",
        "mode": args.mode,
        "audit_role": "diagnostic_only_no_automatic_model_promotion",
        "audit_order": list(AUDIT_ORDER),
        "model_spec": spec_metadata(FULL11_BASELINE_SPEC),
        "data": {"rows": int(len(df)), "sha256": dataset_fingerprint(df)},
        "runtime": runtime_identity(),
        "settings": settings,
        "results": {},
    }
    write_json(payload, args.output_json)

    def run_step(name: str, function):
        print()
        print("=" * 100)
        print(f"AUDIT STEP: {name}")
        print("=" * 100)
        try:
            payload["results"][name] = function()
            write_json(payload, args.output_json)
        except Exception as exc:
            payload["status"] = "failed"
            payload["failed_step"] = name
            payload["failure"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            write_json(payload, args.output_json)
            raise

    try:
        run_step(
            "reference_fixed_weights",
            lambda: run_reference_stability_audit(
                scenario="full11",
                candidate_count=settings["latest_candidate_count"],
                reference_counts=(1_024, 4_096, 16_384),
                seed_deltas=(1, 2, 3),
                progress_every=max(0, args.progress_every * 200),
            ),
        )
        run_step(
            "reference_retrained",
            lambda: run_retrained_reference_stability_audit(
                df,
                reference_counts=settings["reference_counts"],
                stream_deltas=settings["stream_deltas"],
                latest_candidate_count=settings["latest_candidate_count"],
                progress_every=args.progress_every,
            ),
        )
        run_step(
            "focused_component",
            lambda: run_component_influence_audit(
                df,
                include_historical=True,
                include_latest=True,
                historical_baseline_samples=settings["baseline_samples"],
                bootstrap_reps=settings["bootstrap_reps"],
                latest_exhaustive=False,
                latest_candidate_count=settings["focused_latest_candidate_count"],
                latest_pool_size=1_000,
                top_k=10,
                historical_progress_every=args.progress_every,
                latest_progress_every=max(0, args.progress_every * 2_000),
            ),
        )
        run_step(
            "joint_clean3_clean4",
            lambda: run_joint_ablation_audit(
                df,
                historical_baseline_samples=settings["baseline_samples"],
                bootstrap_reps=settings["bootstrap_reps"],
                latest_exhaustive=False,
                latest_candidate_count=settings["focused_latest_candidate_count"],
                latest_pool_size=1_000,
                top_k=10,
                historical_progress_every=args.progress_every,
                latest_progress_every=max(0, args.progress_every * 2_000),
            ),
        )
        run_step(
            "full_feature_group",
            lambda: run_full_feature_group_audit(
                df,
                baseline_samples=settings["baseline_samples"],
                bootstrap_reps=settings["bootstrap_reps"],
                progress_every=args.progress_every,
            ),
        )
        run_step(
            "pair_residualization",
            lambda: run_pair_residualization_audit(
                df,
                baseline_samples=settings["baseline_samples"],
                bootstrap_reps=settings["bootstrap_reps"],
                latest_candidate_count=settings["latest_candidate_count"],
                progress_every=args.progress_every,
            ),
        )
        run_step(
            "exact_structural_null",
            lambda: run_exact_structural_null_audit(
                df,
                baseline_samples=settings["baseline_samples"],
                bootstrap_reps=settings["bootstrap_reps"],
                latest_candidate_count=settings["latest_candidate_count"],
                progress_every=args.progress_every,
            ),
        )
        run_step(
            "ridge_dependency",
            lambda: run_dependency_audit(
                df,
                progress_every=args.progress_every,
            ),
        )
    except Exception as exc:
        print(f"audit suite failed: {type(exc).__name__}: {exc}")
        print(f"partial diagnostic JSON saved: {args.output_json.resolve()}")
        raise SystemExit(1) from exc

    payload["status"] = "completed"
    payload["completed_steps"] = list(AUDIT_ORDER)
    saved = write_json(payload, args.output_json)
    print()
    print("=" * 100)
    print("V3.1 COMPLETE AUDIT SUITE FINISHED")
    print("NO MODEL WAS AUTOMATICALLY PROMOTED")
    print("=" * 100)
    print(f"audit JSON saved: {saved}")


if __name__ == "__main__":
    main()
