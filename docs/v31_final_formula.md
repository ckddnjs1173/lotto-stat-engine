# v3.1 formula audit state

This file used to describe CLEAN3 as the current personal prediction formula. That promotion is withdrawn.

No v3.1 feature subset is currently promoted. FULL11 is the historical baseline and CLEAN3 is an experimental joint-ablation candidate that still requires joint historical and numerical-stability audits.

## Shared v3.1 model family

All `C(45,6) = 8,145,060` valid combinations remain eligible. There are no aesthetic hard filters, type quotas, or validation-confidence multipliers.

The historical training chronology remains:

1. build state using only draws before target `t`;
2. build target `t` feature context;
3. build the pre-target fair-null reference;
4. transform the target and fair negatives;
5. score outer target using only prior solved targets when performing walk-forward validation;
6. after target `t` is solved, add its actual-vs-fair differences to training sufficient statistics;
7. advance history to `t+1`.

This blocks within-target leakage. It does not make the full model-development history independent OOS evidence because model classes and feature subsets were selected after inspecting the same historical period.

## Raw feature representation

The reproducible v3.1 raw representation contains 11 columns:

- `number_full_log_lift`
- `pair_full_log_lift`
- `number_recent20_excess`
- `number_recent100_excess`
- `pair_recent100_excess`
- `previous_draw_overlap`
- `sum_signed_center_138`
- `high_minus_low_zone_count`
- `odd_count_signed_center_3`
- `number_range`
- `consecutive_pairs`

The historical baseline fair-null transform is:

```text
z_j = 2 * F_mid,j(x_j) - 1
```

where `F_mid,j` is estimated from 1,024 deterministic fair combinations for that target/context.

The additive score is:

```text
score(c) = w_A^T z_A(c)
```

`A` is the active feature set for the explicitly selected scenario.

## Explicit scenarios

### FULL11

```text
status = experimental_baseline_not_promoted
active = all 11 features
```

This scenario exists to reproduce the v3.1 baseline used by component audits.

### CLEAN3

```text
status = experimental_candidate_requires_joint_and_stability_audit
removed =
  previous_draw_overlap
  number_range
  consecutive_pairs
```

CLEAN3 solves the reduced ridge system on the remaining eight features. It is not a cosmetic zeroing of final contributions.

The earlier provisional CLEAN3 promotion was withdrawn because all focused single-feature bootstrap intervals included zero and the three-feature joint model had not yet been validated before promotion.

## Distribution diagnostics are not model-selection gates

The old full-11 TOP-1000 showed strong displacement from fair expectations in previous-draw overlap, consecutive pairs, and range. Those results show strong model influence but do **not** by themselves prove that a feature is wrong.

A predictive top tail is allowed to look unlike a random draw. Therefore distance from the fair-null mean is descriptive only. Feature removal must be supported by target-isolated historical winner-ranking evidence and stability analysis, not aesthetics.

## Current numerical audit concerns

### 1. 1,024-reference resolution

The 1,024-sample empirical CDF can saturate at `-1/+1` beyond sampled support. Because the final universe contains 8,145,060 combinations, top-tail plateaus can be much larger than the reference itself. Reference seed/size stability must therefore be measured before interpreting small TOP score differences as stable ordering.

### 2. Tie policy

The old ranking used a seed-dependent BLAKE2 hash to break exact score ties. The shared v3.1 core now uses a fixed RNG-free combination order. Exhaustive ranking therefore does not change tie order when `seed_offset` changes.

### 3. Exact structural nulls

Six structural features have exact whole-universe distributions and do not mathematically require Monte Carlo reference sampling:

- previous-draw overlap: hypergeometric
- odd count: exact combinatorics
- high-minus-low zone count: exact multinomial-combination sum
- range: `(45-r) * C(r-1,4)`
- consecutive adjacent pairs: exact run combinatorics
- sum: exact dynamic programming over 6-of-45 subsets

These distributions are implemented in `lotto_engine/v31_exact_structural_null.py`. They are intentionally not wired into FULL11/CLEAN3 yet; they define a separately audited representation change.

## Reproducibility

`lotto_engine/v31_model_spec.py` contains frozen dataclass specifications for the two v3.1 scenarios. Each run records a canonical SHA-256 model-spec hash plus the normalized draw-history SHA-256.

The model spec fixes:

- feature set
- history start index
- number/pair priors
- recent windows
- training negatives
- fair-reference count
- ridge lambda
- seed offsets
- tie policy

## Current commands

Scenario selection is mandatory:

```powershell
python scripts\run_recommend.py --scenario full11
python scripts\run_recommend.py --scenario clean3
```

For wiring checks:

```powershell
python scripts\run_recommend.py --scenario full11 --sampled --candidate-count 20000
```

Do not describe either scenario as the final model until reference stability, full feature/group ablation, and coefficient stability audits are complete.
