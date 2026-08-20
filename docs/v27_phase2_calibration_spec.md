# v2.7 Phase 2 Type-Score Calibration Protocol

This protocol is predeclared before observing Phase-2 results.

## Question

Do raw score-scale differences between `normal`, `mixed`, and `outlier` distort
global candidate ranking, and does removing that distortion improve strict
walk-forward ranking?

Calibration is a score mapping, not a new predictive signal.

## Frozen production constraints

- v2.6 production scoring remains unchanged.
- Candidate eligibility remains all valid 8,145,060 combinations.
- No hard filters, quota enforcement, or portfolio diversity terms enter this test.
- No Phase-1 component weight is retuned.
- Target draw `t` may use only draws `< t`.

## Models under test

Phase 2 tests two predeclared score sources:

1. `base_only`
2. `v26_full`

Testing both separates a base-scale problem from any scale problem that remains
after transition and momentum are composed.

## Calibration method

For each historical target:

1. Build the scoring state from prior draws only.
2. Draw an independent deterministic calibration candidate sample.
3. Score calibration candidates and split them by pattern type.
4. For each type and model, build a monotonic empirical mid-rank CDF.
5. Score the real target and map its raw score through the CDF for its own type.
6. Draw a separate deterministic evaluation candidate sample, disjoint from the
   calibration sample.
7. Map every evaluation candidate through its own type's calibration CDF.
8. Compare the real target's calibrated score against all calibrated evaluation
   candidates globally.

The target result is never used to estimate the calibration CDF.

## Screening sample sizes

- calibration candidates per target: `1000`
- evaluation candidates per target: `500`
- walk-forward start index: `100`
- block-bootstrap repetitions: `2000`
- block size: `20`

The script records the minimum calibration sample count across all three pattern
types so insufficient type coverage cannot pass silently.

## Required output

For each model:

- raw global percentile: overall / recent 300 / recent 100
- calibrated global percentile: overall / recent 300 / recent 100
- calibrated fraction above random median
- calibrated-minus-raw paired delta
- deterministic block-bootstrap 95% interval for that delta
- raw and calibrated percentile by actual pattern type
- minimum same-type calibration coverage

## Decision gate

Calibration is a **promotion candidate** only if:

- calibrated-minus-raw overall delta is positive,
- its 95% block-bootstrap interval is entirely above zero,
- recent-300 delta is non-negative,
- recent-100 delta is non-negative,
- and no pattern type shows a material collapse that explains the aggregate gain.

Calibration is **rejected** if the overall effect is non-positive and recent
windows do not improve.

Calibration is **unresolved** if the interval crosses zero or recent windows
reverse.

A screening promotion candidate must be rerun with:

- calibration candidates per target: `3000`
- evaluation candidates per target: `2000`

before any production implementation is designed.
