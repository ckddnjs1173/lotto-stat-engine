# v2.7 Validation & Rebuild Protocol

Benchmark frozen from commit `ecbbd02235b9ed8f6940ae47aa46e7d4c0e53499`.
The v2.6 production formula remains untouched while v2.7 research is evaluated.

## Goal

Rank all valid Lotto 6/45 combinations using only signals that demonstrate
repeatable out-of-sample value under strict rolling-origin evaluation.
No component is promoted because it is intuitive, visually plausible, or fits
the latest draw.

## Non-negotiable rules

- Target draw `t` may use only draws `< t`.
- All 8,145,060 valid combinations remain eligible in production.
- No structural hard filters or aesthetic-number rules.
- Research parameters are predeclared before seeing the target results.
- Production and portfolio construction are evaluated separately.
- A failed signal is assigned zero production weight instead of being rescued
  by cosmetic retuning.
- v2.6 remains the benchmark until a v2.7 candidate passes the full protocol.

## Current evidence status

### Keep as benchmark / research candidate

- structural base models
- real-adjacent hierarchical Markov transition
- all-draw decay momentum
- exhaustive candidate evaluation

### Rejected as standalone predictive signals

- number hot/cold frequency (`number_bayes_evidence_v1`)
- raw unordered pair frequency (`pair_bayes_evidence_v1`)

Both rejected families were worse than the fair null in overall, recent-300,
and recent-100 proper-score evaluation.

## Phase 1 — End-to-end score ablation

For every historical target from draw index 100 onward, build all profiles from
prior draws only. Score the real winning combination and a deterministic random
baseline sample from the same historical state.

Primary v2.6 score variants:

1. `base_only`
   - `B`
2. `base_plus_transition`
   - `0.85 B + 0.15 T`
3. `base_plus_momentum`
   - `0.85 B + 0.15 M`
4. `v26_full`
   - `0.70 B + 0.15 T + 0.15 M`

Normal/outlier base audit variants (mixed base remains unchanged so the effect
of the legacy normal/outlier component can be isolated):

5. `base_no_type_rarity`
   - remove `type_balance_score`, renormalize the remaining legacy base weights
6. `base_no_legacy_transition`
   - remove the legacy type-level `transition_score`, renormalize
7. `base_clean_no_rarity_or_legacy_transition`
   - remove both suspected terms, renormalize

For each variant report:

- mean actual-vs-random percentile
- recent 300 / recent 100 mean percentile
- fraction of targets above the random median
- mean percentile by actual pattern type
- paired difference versus `v26_full`
- paired difference versus `base_only`
- deterministic block-bootstrap 95% interval for paired differences

Also report type-conditioned percentiles: the actual target is compared only
with sampled candidates of the same pattern type. This is the first audit of
whether raw score scale differences between normal/mixed/outlier distort global
ranking.

### Phase-1 decision policy

A component is not promoted merely because its point estimate is positive.

- **Promote candidate:** positive overall paired improvement with 95% block-
  bootstrap interval entirely above zero, and no material reversal in recent
  300 and recent 100 windows.
- **Reject candidate:** negative overall paired effect and non-positive recent
  windows.
- **Research / unresolved:** mixed signs, interval crossing zero, or strong
  regime dependence.

No 70/15/15 weight tuning is allowed during Phase 1.

## Phase 2 — Base-score calibration

If Phase 1 confirms predictive value in the underlying base models, map each
pattern type's raw base score to a comparable out-of-sample scale. Candidate
methods are restricted to monotonic empirical percentile / quantile calibration
estimated from historical candidate distributions only.

Required comparison:

- raw global base ranking
- type-conditioned calibrated base ranking

Calibration is adopted only if walk-forward ranking improves. Calibration may
change scale, never candidate eligibility.

## Phase 3 — Null & stationarity laboratory

Before adding AR/ARIMA or regime weighting, test whether the data support their
assumptions.

Fair 6/45 Monte Carlo / permutation null diagnostics:

- single-number frequency deviation
- pair-frequency deviation
- sum / range / parity / gap distributions
- pattern-type proportions
- transition-strength statistics

Time-series diagnostics on structural features:

- lag autocorrelation
- Ljung-Box or permutation equivalent
- rolling distribution shift
- change-point / regime evidence

AR/ARIMA research begins only for features with stable out-of-sample evidence of
serial dependence. Deep recurrent models remain out of scope until simpler
models demonstrate reproducible signal.

## Phase 4 — Dynamic component significance

Test Markov and momentum independently of score-mapping cosmetics.

Transition null:

`P(next_state | current_state) = P(next_state)`

Evaluate type, coarse-family, and exact-family levels with shrinkage, permutation
or likelihood comparisons, and strict walk-forward ranking.

Momentum null:

recent/long-run lift contains no predictive information for the next draw.

Evaluate `base` versus `base + momentum`; if momentum fails, its production
weight becomes zero rather than being manually reduced.

The current `lift_to_score` saturation (large lifts mapping to the same upper
score) is calibrated only after raw lift evidence is established.

## Phase 5 — Portfolio A/B

Prediction ranking and ticket portfolio construction are separate layers.
Compare:

- pure prediction TOP-10
- soft-allocation / diversity portfolio TOP-10

Portfolio diversity is retained only if it improves historical ticket-level
outcomes (max match count and 3/4/5/6-match frequency) without degrading the
primary ranking evidence.

## Production promotion gate

A v2.7 formula replaces v2.6 only after:

1. strict walk-forward component evidence,
2. common-scale calibration evidence,
3. recent-window stability checks,
4. leakage/regression tests,
5. pure-vs-portfolio A/B,
6. final exhaustive 8,145,060 smoke/production validation.
