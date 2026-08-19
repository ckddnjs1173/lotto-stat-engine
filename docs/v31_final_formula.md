# v3.1 CLEAN3 personal prediction formula

This document describes the **current** personal-use calculation path.

## Purpose

The engine preserves the raw value calculated by the model even when historical validation does not establish predictive edge. Validation is diagnostic metadata, not a multiplier that neutralizes the model score.

All `C(45,6) = 8,145,060` valid combinations remain eligible. There are no pattern quotas or aesthetic hard filters.

## Historical training sequence

Starting after the first 100 completed draws, each solved target is processed as follows:

1. build state using only earlier draws;
2. calculate the candidate raw features for the target;
3. sample 1,024 deterministic fair 6/45 reference combinations using only the pre-target context;
4. map each raw feature to a tie-safe fair-null coordinate in `[-1,1]`;
5. sample 64 fair negative combinations for that solved target;
6. accumulate pairwise actual-minus-negative sufficient statistics;
7. only then advance history with the solved target.

The latest model is fit after all completed draws are processed. The target draw is always `latest + 1`.

## Raw representation

The stable raw representation contains 11 columns so previous v3.1 audit results remain reproducible:

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

## Production CLEAN3 subset

The current production fit excludes:

- `previous_draw_overlap`
- `number_range`
- `consecutive_pairs`

The remaining eight columns are fit jointly by the same pairwise-ridge objective. Disabled columns receive zero weight only **after** the reduced ridge system has been solved; the active weights are therefore re-estimated in the reduced model, not copied from the old 11-feature model.

Active set:

```text
A = {
  number_full_log_lift,
  pair_full_log_lift,
  number_recent20_excess,
  number_recent100_excess,
  pair_recent100_excess,
  sum_signed_center_138,
  high_minus_low_zone_count,
  odd_count_signed_center_3
}
```

## Fair-null transform

For raw feature `x_j` and its pre-target fair reference column:

```text
z_j = 2 * F_mid,j(x_j) - 1
```

where exact ties receive half credit. Every transformed coordinate is bounded to `[-1, 1]`.

## Score

```text
score(c) = w_A^T z_A(c)
```

`w_A` is learned by pairwise ridge from solved historical winners versus fair negative combinations.

There are no square terms, feature interactions, Normal/Mixed/Outlier branch, validation-confidence multiplier, or type/section quota.

The signed sum/zone/odd features do not encode “closest to the center is best.” Their direction and magnitude are learned from historical actual-vs-fair comparisons.

## Why CLEAN3

The component audit on data through draw 1237 found the legacy full-11 TOP-1000 had:

```text
mean previous-draw overlap = 2.007
any previous overlap       = 1.000
mean consecutive pairs     = 0.000
mean number range          = 21.904
```

Fair 6/45 references are approximately:

```text
expected previous overlap  = 0.800
any previous overlap       = 0.5994
expected consecutive pairs = 0.6667
expected number range      = 32.8571
```

The three corresponding features produced large current-pool structural displacement while their individual historical leave-one-out value was inconclusive/near-neutral. They are therefore excluded from the personal calculation.

`number_full_log_lift` is retained because its leave-one-out historical delta was the strongest of the four audited terms and was positive overall, recent-300, and recent-100, even though its bootstrap interval still included zero.

See `docs/v31_component_influence_result.md`.

## Pattern metadata

`structure_record()` is used after ranking to label candidates as Normal/Mixed/Outlier and expose structural diagnostics. Pattern type does **not** enter current ranking or portfolio allocation.

## Portfolio layer

Portfolio construction is downstream of the score. It does not change `score(c)`.

For requested TOP-10:

- pairwise shared numbers <= 2;
- each number may appear on at most 4 tickets;
- no fallback relaxation.

The engine scans the retained top-50,000 raw candidates in descending model-score order and takes the first candidates satisfying those coverage constraints.

## Reproducibility

Every result includes latest reflected draw, target draw, evaluated count, active/disabled feature names, training settings, and normalized data SHA-256 fingerprint.

## Current command

```powershell
python scripts\run_recommend.py --top-k 10 --progress-every 1000000 --output-json data\cache\v31_final_personal.json
```

`python main.py` delegates to the same current runner.
