# v2.7 Phase 3A Null & Stationarity Screening Protocol

This protocol is predeclared after Phase 2 rejected type-score calibration and
before observing any Phase-3 result.

## Purpose

Before adding AR/ARIMA, regime weighting, change-point logic, or stronger dynamic
weights, test whether Lotto structural features contain detectable time-order
dependence at all.

This is a cheap screening stage. It is intentionally much less expensive than the
Phase-1 and Phase-2 walk-forward candidate audits.

## Frozen constraints

- v2.6 production scoring remains unchanged.
- No score weight is tuned.
- No candidate is filtered.
- No AR/ARIMA model is fit during screening.
- No screening p-value is treated as production evidence.
- Only statistics that survive screening may receive a larger confirmation run.

## Screening budget

- fair 6/45 candidate draws: `20,000`
- permutation repetitions: `500`
- maximum ACF lag: `10`
- recent-window shift: `300` draws
- family-wise correction: Holm correction at `0.05`

If a statistic does not survive this stage, no larger run is performed for it.

## Scalar structural features

The first screen covers the existing structural features already used by the
engine:

- sum
- range
- odd count
- maximum gap
- minimum gap
- consecutive-pair count
- duplicate-ending count
- empty decade-section count
- extreme-flag count

### Fair-draw marginal reference

Generate independent exact 6-of-45 draws. Selection is without replacement inside
each draw. Compare the historical feature mean with the fair-draw reference and
report a standardized mean displacement.

This marginal comparison is descriptive. A marginal anomaly by itself does not
establish predictability.

## Time-order tests

For each scalar feature, preserve the historical marginal values and permute their
order.

Three predeclared statistics are screened:

1. maximum absolute autocorrelation across lags 1 through 10;
2. recent-300 versus prior-history standardized mean shift;
3. maximum standardized CUSUM displacement.

Permutation p-values are computed from the same statistic under shuffled order.
Holm adjustment is applied separately to the ACF, recent-shift, and CUSUM feature
families.

A feature becomes a confirmation candidate only when the relevant adjusted
p-value is at most `0.05`.

## Pattern-type transition test

For the historical `normal / mixed / outlier` sequence, compute mutual information
between adjacent pattern types.

The null preserves the pattern-type counts but randomly permutes their order.

A permutation p-value at most `0.05` creates a Phase-4 transition confirmation
candidate. It does not validate the current Markov score or its production weight.

## Pattern-type marginal reference

Historical normal/mixed/outlier proportions are also compared with proportions in
the exact fair-draw Monte Carlo sample.

This is descriptive and must not be confused with transition dependence.

## Decision routing

- No serial-dependence candidate:
  - AR/ARIMA remains out of scope.
- No recent-shift or CUSUM candidate:
  - do not add regime/change-point weighting.
- Transition permutation test fails:
  - current Markov component has no Phase-3 support and must rely on later direct
    predictive evidence to survive.
- Any candidate passes:
  - rerun only that statistic with `5,000` permutations; increase the fair-draw
    sample to `100,000` only if the marginal reference itself matters to the
    surviving question.

This selective confirmation policy is designed to avoid repeating expensive
full-engine backtests when the prerequisite statistical signal is absent.
