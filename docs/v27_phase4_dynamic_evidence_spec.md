# v2.7 Phase 4 Direct Dynamic-Evidence Protocol

This protocol is predeclared after Phase 3A produced no serial-dependence,
recent-shift, change-point, or pattern-transition confirmation candidate.

## Question

Do the current dynamic components rank the *next winning combination* above fair
random valid Lotto 6/45 combinations when their raw lifts are evaluated strictly
out of sample?

Phase 4 deliberately removes base-score mapping and the 70/15/15 composition from
the test. It evaluates the dynamic information itself.

## Frozen constraints

- v2.6 production scoring remains unchanged.
- No dynamic weight is tuned.
- No candidate is filtered.
- Every target draw uses only prior draws to build the dynamic model.
- Screening failure does not trigger a larger confirmation run.
- Screening success is not production evidence; it only earns a larger targeted
  confirmation.

## Screening budget

- walk-forward start index: `100`
- fair random valid combinations per target: `200`
- block-bootstrap repetitions: `2000`
- block size: inherited from the v2.7 validation utility (`20`)

This is intentionally much cheaper than Phase 1/2 because only structural family
classification and dynamic lift lookup are performed for random candidates.

## Dynamic quantities

For an exact target family, reproduce the current production hierarchy before the
nonlinear `lift_to_score` mapping:

1. `transition_type`
   - target-family probability after conditioning on the latest pattern type,
     divided by the historical target-family prior.
2. `transition_coarse`
   - target-family probability after additionally conditioning on the latest
     coarse family, divided by the same prior.
3. `transition_exact`
   - final target-family probability after additionally conditioning on the
     latest exact family, divided by the same prior.
4. `momentum`
   - current all-draw decayed recent probability divided by the historical
     target-family probability.

The existing shrinkage strengths, alpha, decay, and family definitions are reused
without retuning.

`transition_type` and `transition_coarse` are diagnostic localization levels.
The two primary screening hypotheses are the production-relevant final
`transition_exact` and `momentum` quantities.

## Walk-forward ranking test

For each historical target draw `t`:

1. build the dynamic family model from draws `< t` only;
2. compute the raw lifts for the actual winning combination;
3. generate 200 deterministic unique fair 6-of-45 combinations;
4. compute the same raw lifts for those combinations under the identical historical
   model state;
5. calculate a tie-safe actual-vs-random percentile for every dynamic quantity.

Because actual and random combinations are scored under the same historical model,
this comparison remains valid when exact-family vocabulary grows through time.
No future family vocabulary is injected into a past model.

## Required output

For each dynamic quantity:

- mean percentile
- recent-300 mean percentile
- recent-100 mean percentile
- fraction above the random median
- mean raw log-lift of actual targets
- mean percentile by actual pattern type
- circular block-bootstrap 95% interval for `(percentile - 50)`

Also report paired percentile differences:

- `transition_coarse - transition_type`
- `transition_exact - transition_coarse`
- `transition_exact - transition_type`

These incremental comparisons are diagnostic and do not create extra production
hypotheses.

## Screening decision gate

A primary component (`transition_exact` or `momentum`) becomes a confirmation
candidate only if all conditions hold:

- overall mean percentile is above 50;
- the 95% block-bootstrap interval for `(percentile - 50)` is entirely above zero;
- recent-300 mean percentile is at least 50;
- recent-100 mean percentile is at least 50.

Otherwise the component does not earn a larger run.

A screening survivor is confirmed only with a targeted rerun using `1000` fair
random combinations per target. No unrelated component is recomputed at a larger
budget merely because another component survives.

If both primary components fail, Phase 4 ends without confirmation and v2.7 moves
to base-component evidence rather than inventing new dynamic weights.
