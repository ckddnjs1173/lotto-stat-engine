# v2.9 Quadratic Reverse-Learning Ranking Specification

Status: predeclared exploratory screen; production path remains unchanged.

## Objective

Test one specific follow-up hypothesis after the v2.8 linear ranker failed:

> The same ten reverse-statistics inputs may contain ranking information only through
> stable second-order nonlinearities or interactions that a linear score cannot
> represent.

This version does **not** change the historical target isolation rule, the base input
features, recent horizons, training-negative count, evaluation baseline size, ridge
lambda, or promotion gate. Only the candidate representation changes from a 10-term
linear basis to its complete degree-2 polynomial basis.

## Research-budget rule

v2.9 is the final new model class to be screened on the current 1,235-draw dataset
without new draws. Repeatedly inventing additional classes after seeing each aggregate
historical result would create research-process overfitting even when each individual
run is strict walk-forward.

If v2.9 fails, model development on this fixed dataset is frozen and future model
claims require new external/future draw evidence or a separately declared research
program.

If v2.9 passes, it earns only a frozen confirmation. Historical passage alone is not
an independent production validation because v2.9 was selected after observing prior
model-family results.

## Base candidate inputs

The exact v2.8 ten-feature vector is retained unchanged:

1. `number_full_log_lift`
2. `pair_full_log_lift`
3. `number_recent20_excess`
4. `number_recent100_excess`
5. `pair_recent100_excess`
6. `previous_draw_overlap`
7. `sum_abs_deviation_138`
8. `odd_imbalance`
9. `number_range`
10. `consecutive_pairs`

No feature is selected, removed, sign-flipped, or redefined from the v2.8 result.
Pattern type remains absent from the ranking equation.

## Frozen quadratic basis

Let the base feature vector be:

```text
x = [x1, ..., x10]
```

The v2.9 candidate representation is the complete degree-2 basis without an
intercept:

```text
phi(x) = [
  x1, ..., x10,
  x1^2, ..., x10^2,
  x1*x2, x1*x3, ..., x9*x10
]
```

This produces exactly:

```text
10 linear
+ 10 squared
+ 45 pairwise interaction
= 65 features
```

An intercept is unnecessary because training is based on actual-minus-negative
candidate differences, where a common intercept cancels.

The expansion is deterministic and contains every second-order term. No individual
interaction is chosen from observed v2.8 coefficients.

## Pairwise training objective

For historical solved target `t`, with actual candidate feature basis `phi(a_t)` and
a fair training negative `phi(r_tj)`:

```text
d_tj = phi(a_t) - phi(r_tj)
```

As in v2.8, sufficient statistics accumulate only after target `t` has already been
outer-scored:

```text
M = mean(d d^T)
m = mean(d)
```

Each basis coordinate is RMS-scaled from prior solved training differences. The
frozen ridge solution is:

```text
w_scaled = (M_scaled + 2.0 I)^-1 m_scaled
w = w_scaled / rms
```

Candidate ranking score:

```text
score(c) = w^T phi(x(c))
```

The ridge lambda remains exactly `2.0` so that the tested change is the nonlinear
basis, not a simultaneous regularization retune.

## Strict nested walk-forward

The sequence remains identical to v2.8:

```text
history draws 1..100

solve targets 101..200 as meta-training examples only
- features for each target use only earlier draws
- target answer is added to model-training statistics only after that target is
  revealed

outer target 201
- fit coefficients from solved targets 101..200 only
- construct target-201 context from draws 1..200 only
- score actual 201 winner against fair alternatives
- record percentile
- only then add solved target 201 to later training

repeat forward
```

With 1,235 draw rows and the frozen 100-target meta-training period, the expected
outer test count is `1,035`.

Appending a future draw must not alter any already-recorded outer target score.

## Frozen sampling

To isolate model-class effects, v2.9 deliberately reuses the same deterministic
sampling seed formulas as v2.8.

```text
training negatives per solved target = 64
screen fair candidates per outer target = 500
confirmation fair candidates per outer target = 2,000
```

Therefore v2.8 and v2.9 see the same sampled fair alternatives for corresponding
historical targets.

## Screening summary

For each outer target, compute the tie-safe percentile of the actual winning
combination against 500 fair valid combinations.

Report:

```text
overall mean percentile
recent-300 mean percentile
recent-100 mean percentile
above-random-median ratio
circular block-bootstrap 95% CI of percentile - 50
```

The historical screening flag remains the existing frozen gate:

```text
overall mean percentile > 50
CI95 lower bound > 0
recent300 >= 50
recent100 >= 50
```

Because v2.9 is adaptively chosen after earlier model classes were observed, this
flag is labeled `exploratory_screening_candidate`; it is not a production approval.

## Confirmation and production rule

If the screen fails:

- no lambda tuning;
- no term selection;
- no interaction pruning;
- no alternate polynomial degree;
- no new recent horizon on the current data;
- no production change.

If the screen passes:

- freeze the exact 65-term basis and lambda;
- rerun only with 2,000 fair evaluation candidates per historical outer target;
- keep training samples and all target-isolation rules unchanged;
- still treat the result as research evidence pending a future-data validation rule.

The site/recommendation path remains neutral until a separately documented promotion
decision is made.