# v3.1 formula audit state

This file used to describe CLEAN3 as the current personal prediction formula. That promotion is withdrawn.

No v3.1 feature subset or fair-null representation is currently promoted. FULL11 is the historical baseline; CLEAN3 and the representation variants below are experimental audit scenarios only.

## Shared v3.1 model family

All `C(45,6) = 8,145,060` valid combinations remain eligible. There are no aesthetic hard filters, type quotas, or validation-confidence multipliers.

Historical chronology:

1. build state using only draws before target `t`;
2. build target `t` feature context/reference;
3. transform target and fair negatives;
4. when validating, score target using only previously solved targets;
5. after `t` is solved, add its actual-minus-fair rows to the pairwise-ridge sufficient statistics;
6. advance history to `t+1`.

This blocks within-target leakage. Because model classes/features were selected after repeatedly inspecting the same historical period, the resulting historical validation remains **post-selection exploratory walk-forward**, not independent final OOS proof.

## Reproducible 11-feature raw representation

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

Historical sampled-null transform:

```text
z_j = 2 * F_mid,j(x_j) - 1
```

with 1,024 deterministic fair combinations per target/context.

Additive score:

```text
score(c) = w_A^T z_A(c)
```

`A` is the explicit scenario's active feature set.

Pairwise ridge is fit on actual-minus-fair difference rows. The common `v31_core.py` holds the mathematical primitives; `v31_model_spec.py` freezes priors, windows, reference/negative counts, ridge lambda, seeds, and tie policy.

## Explicit feature-set scenarios

### FULL11

```text
status = experimental_baseline_not_promoted
active = all 11 features
```

### CLEAN3

```text
status = experimental_candidate_requires_joint_and_stability_audit
removed =
  previous_draw_overlap
  number_range
  consecutive_pairs
```

CLEAN3 solves the reduced ridge system on the remaining eight features. It is not cosmetic zeroing. Its earlier provisional promotion was withdrawn because the focused single-feature intervals all included zero and the joint model had not yet been reviewed before promotion.

## Fair-distance interpretation

A TOP pool is allowed to differ strongly from the fair 6/45 population. Therefore previous-draw overlap, range, consecutive-pair count, sum, or number inclusion rates are **descriptive influence diagnostics**, not automatic feature-deletion gates.

## Numerical/reference audit representations

### Fixed-weight reference stability

`v31_reference_stability_audit.py` keeps fitted weights fixed and changes only the latest empirical reference. It measures score/rank correlation, TOP Jaccard, saturation, and exact-score duplication across seed/count changes.

### Retrained reference stability

`v31_retrained_reference_audit.py` reruns the complete training history under deterministic nested reference streams/counts. It compares final coefficient correlation/L2/signs and latest ranking against the historical current-reference policy. The purpose is numerical convergence, not choosing the reference with the best historical winner score.

## Pair residual representation

Pair counts structurally contain individual-number marginal effects. For each 990-edge vector, the audit fits the complete-graph vertex-main-effect model:

```text
y_ij = a_i + a_j + residual_ij
```

The design has 45 vertex columns and two ones per pair edge. `v31_pair_residual_audit.py` compares:

```text
original pair features
vs
pair features built from residual_ij
```

using the same historical target, reference-candidate, training-negative, and outer-evaluation streams, then retrains each representation independently.

This directly tests whether apparent pair signal is mostly duplicated individual-number hot/cold information.

## Exact structural fair-null representation

Six structural features have exact whole-universe null distributions:

- previous-draw overlap: hypergeometric
- odd count: exact combinations
- high-minus-low zone count: exact 15/15/15 allocation counts
- range: `(45-r) * C(r-1,4)`
- consecutive adjacent pairs: exact selected-run combinatorics
- sum: exact six-number subset dynamic programming

Implementation: `v31_exact_structural_null.py`.

The exact values are **not silently substituted into FULL11/CLEAN3**. `v31_exact_null_audit.py` separately retrains:

```text
sampled structural midrank representation
vs
exact structural midrank representation
```

while leaving the five history-dependent evidence coordinates on the historical sampled reference. This makes the representation change measurable before adoption.

## Ridge / dependency diagnostics

`v31_dependency_audit.py` distinguishes:

```text
unregularized scaled second-moment condition
actual ridge-system condition of (scaled_second + lambda I)
```

and reports coefficient sign/dispersion stability plus pairwise second-moment cosine on training difference rows.

Predeclared dependencies include:

- number full ↔ pair full
- recent20 number ↔ recent100 number
- recent100 number ↔ recent100 pair
- full ↔ recent number/pair
- sum ↔ high-minus-low
- range ↔ consecutive

The reported dependency metric is not a centered Pearson correlation; it describes the geometry of the pairwise-ridge normal equation.

## Tie policy

The historical seed-dependent BLAKE2 tie-break is no longer used by the shared v3.1 ranking path.

```text
primary   = model_score
secondary = fixed combination order
```

`--seed-offset` can change sampled candidate generation but cannot change exhaustive exact-score tie ordering.

## Reproducibility identity

Scenario output records:

- canonical model-spec SHA-256
- normalized draw-history SHA-256
- Python version
- NumPy version
- platform
- fixed tie policy
- active features
- training counts/lambda

The complete audit suite also checkpoints partial results after every audit step and records a traceback if any step fails.

## Decision workflow

Run:

```powershell
python scripts/run_v31_audit_suite.py --mode full --progress-every 50
```

The suite covers:

1. fixed-weight reference stability
2. retrained reference stability
3. focused component audit
4. CLEAN3/CLEAN4 joint audit
5. all 11 single-feature and predefined group ablations
6. pair residualization
7. exact structural-null comparison
8. ridge/dependency stability

No code path automatically promotes the scenario with the highest historical metric.

Only after these results are reviewed should one immutable feature/reference representation be marked frozen/promoted. After the freeze, run all 8,145,060 combinations for the final RAW ranking and treat subsequent future draws as the new independent validation period.
