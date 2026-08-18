# v2.7.1 Unified Evidence Integration Specification

## Objective

Repair the missing research-to-recommendation connection without reintroducing the
legacy aesthetic/static structure bias exposed by Phase 5A.

The ranking target is a single comparable evidence value for every valid 6-of-45
combination. Pattern type is not allowed to choose a different scoring equation.

## Frozen principles

1. All `C(45,6) = 8,145,060` combinations remain eligible.
2. No hard structural filters.
3. Target draw `t` may use only draws `< t` during walk-forward reliability testing.
4. Transition and momentum remain disabled after v2.7 validation.
5. Phase 5A legacy static components do not drive the new ranking.
6. Reliability is continuous shrinkage, not post-hoc weight tuning.
7. A model worse than the fair null is not automatically reversed into a predictor.
8. The exact same local draw data is used for reliability estimation and for fitting
   the next-draw posterior after the historical OOS evaluation is complete.

## Fixed number model

The fair marginal inclusion probability for each number is:

```text
p0 = 6 / 45
```

The fixed next-draw research model is `full_prior_120`:

```text
p_i = (hits_i + 120 * p0) / (N + 120)
L_i = log(p_i / p0)
```

For candidate `c`:

```text
E_number(c) = mean(L_i for i in c)
```

No decay horizon is selected after seeing the target. The full-history specification
is consistent with the Phase 3A result that did not establish a recent regime/shift.

## Fixed pair model

A six-number draw contains 15 unordered pairs out of 990 possible pairs, so the fair
pair inclusion probability is:

```text
q0 = 15 / 990 = 1 / 66
```

The fixed next-draw research model is `full_prior_330`:

```text
q_ij = (pair_hits_ij + 330 * q0) / (N + 330)
L_ij = log(q_ij / q0)
```

For candidate `c`:

```text
E_pair(c) = mean(L_ij for the 15 unordered pairs in c)
```

## Strict walk-forward reliability

Each fixed model is evaluated against its fair uniform null using the existing
strict walk-forward implementation. Brier skill is collected for:

```text
s_all
s_300
s_100
```

where positive skill means lower OOS Brier loss than the corresponding fair null.

Define:

```text
positive_mean
  = (max(s_all,0) + max(s_300,0) + max(s_100,0)) / 3

positive_fraction
  = count(skill > 0 among all/300/100) / 3

reliability
  = positive_mean * positive_fraction
```

This rule has two purposes:

- retain small positive evidence instead of imposing an all-or-nothing promotion gate;
- shrink evidence sharply when its recent-window sign is unstable.

If all three Brier skills are non-positive, reliability is zero. Negative Brier skill
is not used with a negative weight because poor probability calibration is not proof
that reversing the forecast produces predictive information.

## Common candidate score

For every candidate, regardless of Normal / Mixed / Outlier classification:

```text
E(c)
  = r_number * E_number(c)
  + r_pair   * E_pair(c)
```

`E(c)` is the ranking key. The displayed bounded score is:

```text
score(c) = 50 + 50 * tanh(E(c))
```

The transform is strictly monotone and therefore does not change candidate ordering.
Its purpose is only to keep a familiar bounded display scale and make weak evidence
visibly stay close to neutral 50.

## Exact ties

If two candidates have exactly equal `E(c)`, selection uses a deterministic BLAKE2b
hash of `(seed, numbers)` as a tie-break. This tie-break contains no structural
features and exists only to avoid lexicographic number order becoming an accidental
preference when evidence is neutral or tied.

## Pattern metadata

Only after the TOP-K evidence ranking is determined does the engine attach:

- Normal / Mixed / Outlier;
- sum;
- odd count;
- range;
- min/max gap;
- section distribution.

These fields explain the selected ticket; they do not alter the score.

## Release status

v2.7.1 is a `research_candidate`, not a production claim of increased mathematical
lottery probability. The first required execution after pulling the implementation is
therefore a sampled smoke run that prints the actual number and pair Brier skills and
reliabilities from the user's current `data/lotto.xlsx`.
