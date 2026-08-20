# v3.0 Fair-Null Bounded Quadratic Reverse Ranking

Status: exploratory boundary-correction experiment after the v2.9 exhaustive support diagnostic.

## Why this exists

The v2.9 implementation was verified correct, but exhaustive ranking exposed a numerical/model-shape problem rather than a type-allocation bug.

The exact 8,145,060-combination TOP-10 was dominated by low-sum, highly consecutive outliers such as `5 6 7 8 9 10`. A 200,000-combination fair-reference diagnostic then showed:

- TOP candidates were at essentially the 100th raw-score percentile of the reference sample;
- `sum_abs_deviation_138` values were mostly above the 99.8th fair percentile;
- consecutive-pair counts of 4 or 5 were at roughly the 99.97th to 99.999th fair percentile;
- mean raw score increased as absolute sum deviation moved from the central region toward the extreme tail;
- large positive score contributions repeatedly came from raw-magnitude quadratic terms such as `sum_abs_deviation_138^2` and interactions containing the same large coordinate.

This means the v2.9 equation can reward a candidate because a raw feature value is numerically huge, not necessarily because the learned relationship remains supported at that boundary.

The purpose of v3.0 is narrowly defined:

> Preserve the ordering information of every v2.9 input feature while preventing polynomial magnitude explosion outside the typical fair-null support.

This is not a return to hand-written balancing rules.

## What does not change

The following remain frozen from v2.8/v2.9:

- same ten base inputs;
- same reverse-learning chronology;
- same first 100 solved targets before outer scoring;
- same 64 fair training negatives per solved target;
- same 500 fair evaluation alternatives per outer target for screening;
- same ridge lambda `2.0`;
- same linear + square + pairwise-interaction basis width of 65;
- same v2.8/v2.9 training/evaluation RNG paths;
- no Normal/Mixed/Outlier branch;
- no hard candidate filters;
- no manual preference for odd/even balance, section balance, central sums, or low consecutiveness.

## New representation: fair-null midrank coordinates

For target `t`, build the normal historical feature context using only draws before `t`.

Before the target answer is revealed, draw a deterministic set of 1,024 unique fair 6-of-45 combinations. This reference sampler does not receive, exclude, or otherwise depend on the target answer.

For raw feature `x_j`, let its empirical tie-safe percentile in that reference distribution be `F_mid,j(x_j)` in `[0,1]`.

Transform:

```text
z_j = 2 * F_mid,j(x_j) - 1
```

Therefore every base coordinate is bounded:

```text
-1 <= z_j <= 1
```

The mapping is monotone. If a larger raw feature value was larger before the transform, it remains larger after the transform. No direction is declared favorable in advance.

Values beyond sampled support saturate at `-1` or `+1` instead of continuing to grow numerically.

## Quadratic basis

The same complete degree-2 form is applied to the bounded coordinates:

```text
phi(z) = [
  z1..z10,
  z1^2..z10^2,
  z1*z2 .. z9*z10
]
```

This still contains exactly 65 terms.

Because every `z_j` is in `[-1,1]`, every linear, squared, and interaction basis value is also bounded to `[-1,1]`.

Thus an extreme combination can still rank first if learned coefficients genuinely favor its null-relative location, but `105^2` can no longer overwhelm the model merely because 105 is numerically much larger than a typical feature value.

## Strict target isolation

At historical target `t`:

1. build history/context from draws `< t` only;
2. build the deterministic fair reference from that context, without target-answer input;
3. transform and outer-score the actual target using a model trained only on prior solved targets;
4. compare it with the same fair evaluation candidates used by v2.8/v2.9;
5. record the percentile;
6. only then add target `t` and its training-negative comparisons to the solved training statistics;
7. advance history to `t+1`.

Appending a future draw must not alter any earlier outer target result.

## Screening parameters

```text
meta-training solved targets       = 100
training negatives / solved target = 64
fair reference / target            = 1,024
fair evaluation / outer target     = 500
ridge lambda                       = 2.0
bootstrap reps                     = 2,000
quadratic basis width              = 65
```

The existing historical screening summary is retained:

```text
overall mean percentile > 50
CI95 lower bound of percentile-50 > 0
recent300 >= 50
recent100 >= 50
```

Because this model class was designed after diagnosing v2.9, any pass is exploratory evidence, not independent proof.

## Decision rule

If v3.0 does not improve the historical ranking behavior, do not tune the reference size, ridge lambda, feature list, or percentile transform from the observed result.

If v3.0 materially improves the screen, rerun the exact frozen representation with 2,000 fair evaluation candidates per outer target before considering a new personal raw recommendation path.

The existing v2.9 personal raw output remains preserved until this experiment is evaluated.
