# v2.7 Phase 5A Static Component Evidence Protocol

This protocol is fixed after Phase 4 rejected transition and momentum for production
promotion and before observing any Phase-5 static-component result.

## Purpose

Determine which existing static/base score components contain strict walk-forward
ranking information after removing the cross-pattern-type score-scale distortion
identified in Phase 1 and rejected as a calibration target in Phase 2.

This is an evidence screen, not a weight-tuning exercise.

## Frozen constraints

- v2.6 production scoring remains unchanged.
- No component weight is tuned.
- No candidate is hard-filtered.
- Every target draw uses only earlier draws to build its profile.
- Dynamic transition and momentum are excluded from this phase.
- A component that fails is not rescued by threshold, window, or weight tuning.

## Why the baseline is conditioned on pattern type

Phase 1 showed large raw score-scale separation between normal, mixed, and outlier
candidates. Phase 2 showed that equalizing those scales did not improve predictive
ranking.

Therefore Phase 5A does not compare a normal target against mixed/outlier raw scores
or vice versa. For each historical target, the baseline consists only of fair
random valid Lotto 6/45 combinations having the **same pattern type as that target**.

Rejection sampling from uniform valid combinations gives a fair conditional sample
within the requested pattern type. Candidate eligibility in production remains all
8,145,060 combinations; this conditioning exists only inside the diagnostic audit.

## Screening budget

- walk-forward start index: `100`
- same-pattern-type fair candidates per target: `100`
- circular block-bootstrap repetitions: `2000`
- block size: `20`
- confirmation budget for a survivor: `500` same-type candidates per target

The 100-candidate screen is intentionally small. Only a predeclared survivor can
justify a larger run.

## Current branch-base benchmark

For every target, also compute a same-type percentile for the branch score currently
used before the final v2.6 dynamic composition:

- normal/outlier: current `prediction_score`
- mixed: current static `base_score` / `mixed_slot_score` with dynamic context removed

Because every target percentile is already conditioned on its own pattern type,
these percentiles can be pooled as a diagnostic benchmark.

## Normal / outlier components

The shared normal/outlier scoring branch is decomposed into:

1. `normal_structure_score`
2. `outlier_survival_score`
3. `historical_pattern_score`
4. `type_balance_score`
5. `number_dynamics_score`

`transition_score` is intentionally excluded. Pattern transition dependence failed
Phase 3A and the direct transition quantity failed Phase 4.

`historical_pattern_score` and `number_dynamics_score` currently have zero
production weight, but they remain in the evidence map so a potentially useful
static signal is not ignored merely because an older version assigned it zero.

## Mixed components

The static Mixed branch is decomposed into:

1. `mixed_lift_score`
2. `mixed_interaction_score`
3. `normal_backbone_score`
4. `controlled_extreme_score`
5. `recency_consistency_score`

The dynamic Markov/momentum context is removed before these values are scored.

## Walk-forward test

For each historical target draw `t`:

1. determine the actual target pattern type;
2. build the relevant static profile from draws `< t` only;
3. score the actual target component values;
4. rejection-sample 100 unique fair combinations having the same pattern type;
5. score the same component values for those baseline combinations;
6. calculate a tie-safe actual-vs-baseline percentile.

Recent-300 and recent-100 summaries refer to the last 300 and 100 **global target
draw positions**, not the last 300/100 observations of a particular type.

## Screening gate

A component becomes a common-branch screening candidate only if:

- overall mean percentile is above `50`;
- the 95% block-bootstrap interval for `(percentile - 50)` is entirely above zero;
- recent-300 mean percentile is at least `50`;
- recent-100 mean percentile is at least `50`;
- and every pattern type to which that component is currently applied has a mean
  percentile of at least `50`.

For normal/outlier components, type-specific summaries and type-specific candidate
flags are also reported. A type-specific flag is diagnostic only. It does not
authorize branch-specific production weights without a separate predeclared
confirmation.

For Mixed components, the applicable type is Mixed only.

## Decision routing

- No component survives:
  - do not retune current static weights;
  - move to architecture simplification / benchmark cleanup rather than inventing
    new formulas.
- One or more components survive:
  - rerun only those components with `500` same-type fair candidates per target;
  - use a different deterministic random seed family;
  - production integration still requires a later composition-level audit.
- The current branch-base benchmark is descriptive and does not itself create a
  promotion candidate.

This phase is designed to map evidence efficiently while preserving all valid Lotto
combinations for eventual production ranking.
