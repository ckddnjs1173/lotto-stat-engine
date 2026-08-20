# v2.8 Reverse-Learning Nested Ranking Specification

Status: predeclared research specification. No production score change is authorized by this document.

## Objective

Implement the original reverse-statistics idea without leaking the answer of the target being predicted.

The historical winning combination is allowed to teach the model only **after** that historical target has been scored. Therefore each future target is evaluated with coefficients learned exclusively from older solved targets.

## Timeline

Let zero-based target index `100` correspond to draw 101 when draw numbering starts at 1.

For each historical target `t`:

1. construct all candidate features using only draws `< t`;
2. if at least `100` earlier solved targets exist, fit the ranking model using only those earlier targets and score target `t`;
3. compare the actual target combination with fair random valid 6/45 combinations;
4. only after the score is recorded, reveal target `t` to the learner by adding actual-vs-fair training differences for use on `t+1` and later.

The first 100 solved targets are meta-training only. With `start_index=100` and `min_meta_train_targets=100`, the first outer evaluation is target index 200.

## Candidate universe

- every valid `C(45,6)=8,145,060` combination remains eligible;
- fair training and evaluation negatives are sampled uniformly from the full valid universe;
- the actual winning combination is excluded from its negative sample;
- pattern type is not used for sampling, branching, weighting, or filtering.

## Predeclared feature vector

The first reverse learner is deliberately small and interpretable. Ten features are computed for every candidate from the information available immediately before its target draw:

1. `number_full_log_lift`
   - full-history Bayesian number posterior log-lift versus `6/45`;
2. `pair_full_log_lift`
   - full-history Bayesian pair posterior log-lift versus `1/66`;
3. `number_recent20_excess`
   - candidate-number mean inclusion rate over the most recent 20 draws minus `6/45`;
4. `number_recent100_excess`
   - candidate-number mean inclusion rate over the most recent 100 draws minus `6/45`;
5. `pair_recent100_excess`
   - candidate-pair mean inclusion rate over the most recent 100 draws minus `1/66`;
6. `previous_draw_overlap`
   - count of candidate numbers present in the immediately previous draw;
7. `sum_abs_deviation_138`
   - `abs(sum(candidate)-138)`;
8. `odd_imbalance`
   - `abs(odd_count-3)`;
9. `number_range`
   - maximum number minus minimum number;
10. `consecutive_pairs`
   - count of adjacent number pairs separated by exactly one.

The last four structural descriptors are not given a preferred sign. Because each target is learned against fair random combinations, the model is free to learn central, extreme, positive, negative, or zero influence. No aesthetic rule is hard-coded.

## Training examples

For a solved historical target `j`, let:

```text
x+ = feature vector of the actual winning combination using history < j
x- = feature vector of a fair random valid combination using history < j
d  = x+ - x-
```

Each solved target contributes exactly `64` deterministic fair negative comparisons.

Training and evaluation random streams use separate deterministic seeds. A target is never added to the training sufficient statistics before its own outer evaluation.

## Model

The model is a linear pairwise ridge ranker implemented with NumPy only.

For accumulated historical difference rows `d_k`, define:

```text
S = sum(d_k d_k^T)
b = sum(d_k)
n = number of pairwise rows
```

Per-feature RMS scaling is derived only from prior training rows:

```text
rms_i = sqrt(S_ii / n + 1e-12)
```

Then solve the predeclared ridge system:

```text
A = D^-1 (S/n) D^-1 + lambda * I
z = D^-1 (b/n)
w = solve(A, z)
```

where `D = diag(rms)` and the fixed regularization strength is:

```text
lambda = 2.0
```

Candidate ranking score is:

```text
score(x) = w^T D^-1 x
```

Only score differences matter. There is no fitted intercept and no target-specific tuning.

## Screening protocol

Frozen defaults:

```text
history start index       = 100
minimum solved targets    = 100
training negatives/target = 64
evaluation negatives      = 500
bootstrap reps            = 2000
bootstrap block size      = 20
ridge lambda              = 2.0
```

For each outer target, rank the actual winner against 500 deterministic fair alternatives with tie-safe percentile.

Screening gate:

- overall mean percentile > 50;
- 95% circular block-bootstrap CI for `percentile - 50` has lower bound > 0;
- recent-300 mean percentile >= 50;
- recent-100 mean percentile >= 50.

A screen survivor is confirmed with 2,000 fair alternatives per outer target using a different deterministic evaluation seed. No coefficient, feature, lambda, or horizon changes are allowed between screen and confirmation.

## Diagnostics

The audit records:

- outer target count;
- percentile summary and confidence interval;
- latest learned coefficient for each feature;
- corresponding RMS scale;
- number of solved meta-training targets and pairwise training rows;
- optional per-target rows for leakage tests and deeper analysis.

Coefficient signs are descriptive only. A visually interesting coefficient is not evidence unless the complete outer ranking gate passes.

## Failure rule

If the nested learner does not survive screening:

- do not tune lambda from the failed result;
- do not delete poorly signed features and rerun as if predeclared;
- do not select a recent window post hoc;
- archive the result and define a materially different next hypothesis before another test.

## Production rule

This module is research-only. `scripts/run_recommend.py` remains neutral while the current production-facing number/pair reliability is zero. A v2.8 production ranking path can be created only after independent confirmation of the frozen nested learner.
