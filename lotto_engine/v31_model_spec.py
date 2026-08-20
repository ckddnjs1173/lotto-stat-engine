from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


BASE_FEATURE_NAMES = (
    "number_full_log_lift",
    "pair_full_log_lift",
    "number_recent20_excess",
    "number_recent100_excess",
    "pair_recent100_excess",
    "previous_draw_overlap",
    "sum_signed_center_138",
    "high_minus_low_zone_count",
    "odd_count_signed_center_3",
    "number_range",
    "consecutive_pairs",
)

CLEAN3_REMOVED_FEATURES = (
    "previous_draw_overlap",
    "number_range",
    "consecutive_pairs",
)

FROZEN7_REMOVED_FEATURES = (
    "pair_full_log_lift",
    "pair_recent100_excess",
    "sum_signed_center_138",
    "odd_count_signed_center_3",
)


@dataclass(frozen=True)
class V31ModelSpec:
    name: str
    status: str
    feature_names: tuple[str, ...]
    active_features: tuple[str, ...]
    history_start_index: int = 100
    minimum_meta_train_targets: int = 100
    number_prior_strength: float = 120.0
    pair_prior_strength: float = 330.0
    recent_number_short_window: int = 20
    recent_window: int = 100
    train_negatives_per_target: int = 64
    fair_reference_samples_per_target: int = 1024
    ridge_lambda: float = 2.0
    reference_seed_offset: int = 31_301
    training_seed_offset: int = 31_101
    evaluation_seed_offset: int = 28_001
    tie_policy: str = "fixed_combination_lexicographic_no_rng"
    reference_policy: str = "deterministic_unique_fair_sample_empirical_midrank"
    training_objective: str = "pairwise_actual_minus_fair_negative_ridge"

    def payload(self) -> dict:
        return asdict(self)

    def sha256(self) -> str:
        encoded = json.dumps(
            self.payload(), ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class V31FrozenModelSpec(V31ModelSpec):
    structural_null_policy: str = "exact_whole_universe_midrank_for_structural_features"
    reference_stream_delta: int = 0
    negative_sampling_policy: str = "nested_count_independent_deterministic_unique_fair_stream"
    negative_stream_delta: int = 0
    validation_status: str = "post_selection_exploratory_not_proven_edge"


FULL11_BASELINE_SPEC = V31ModelSpec(
    name="v31_full11_experimental_baseline_v2",
    status="experimental_baseline_not_promoted",
    feature_names=BASE_FEATURE_NAMES,
    active_features=BASE_FEATURE_NAMES,
)

CLEAN3_CANDIDATE_SPEC = V31ModelSpec(
    name="v31_clean3_experimental_candidate_v3",
    status="experimental_candidate_requires_joint_and_stability_audit",
    feature_names=BASE_FEATURE_NAMES,
    active_features=tuple(
        name for name in BASE_FEATURE_NAMES if name not in CLEAN3_REMOVED_FEATURES
    ),
)

FROZEN7_SPEC = V31FrozenModelSpec(
    name="v31_frozen7_exact4096_neg256_v1",
    status="frozen_personal_model_promoted_after_v31_audits",
    feature_names=BASE_FEATURE_NAMES,
    active_features=tuple(
        name for name in BASE_FEATURE_NAMES if name not in FROZEN7_REMOVED_FEATURES
    ),
    train_negatives_per_target=256,
    fair_reference_samples_per_target=4096,
    reference_policy="nested_deterministic_fair_sample_empirical_midrank_stream0",
    training_objective="pairwise_actual_minus_nested_fair_negative_ridge",
)


def spec_metadata(spec: V31ModelSpec) -> dict:
    return {
        "name": spec.name,
        "status": spec.status,
        "sha256": spec.sha256(),
        "payload": spec.payload(),
    }
