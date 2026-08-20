# v2.7 Phase 4 Momentum Confirmation Protocol

This protocol is fixed after the 200-candidate Phase-4 screen selected momentum as
the only dynamic survivor and before the 1,000-candidate confirmation result is
observed.

## Question

Does the current production momentum quantity reproducibly rank the next winning
combination above independent fair Lotto 6/45 combinations, and does that advantage
remain when exact-family coverage is controlled?

## Frozen model

The confirmation reuses the production momentum definition without retuning:

- exact structural family: current `signature_family`
- decay rate: `0.05`
- smoothing alpha: `0.1`
- momentum prior strength: `12.0`
- unseen exact family: neutral lift `1.0`

The implementation calls the same `_calculate_all_family_decay_momentum` routine
used by the production dynamic model. Transition probabilities are not computed.

## Confirmation budget

- walk-forward start index: `100`
- independent fair random combinations per target: `1000`
- block-bootstrap repetitions: `2000`
- block size: `20`

The confirmation candidate seed family is deliberately different from the
200-candidate screening seed family.

## Primary full-baseline test

For every target draw, use only earlier draws to calculate momentum lifts. Compare
the actual target lift with 1,000 independent fair valid combinations using the
tie-safe percentile.

Confirmation requires all of:

- overall mean percentile above `50`;
- 95% circular block-bootstrap interval for `(percentile - 50)` entirely above zero;
- recent-300 mean percentile at least `50`;
- recent-100 mean percentile at least `50`.

## Exact-family coverage robustness

An exact family absent from prior history receives the neutral production fallback
lift `1.0`. This can mix two effects:

1. whether a family has been observed before;
2. whether the current decay momentum ranks observed families correctly.

Therefore the confirmation also records:

- actual-target family-seen rate;
- fair-baseline family-seen rate;
- baseline seen-family counts;
- a conditional percentile for targets whose exact family is already observed,
  comparing the actual lift only against fair candidates whose families are also
  observed.

The seen-family conditional mean must be above `50` and its 95% block-bootstrap
interval for `(percentile - 50)` must be entirely above zero.

This robustness gate is intentionally stricter than screening. It prevents a
production promotion whose apparent advantage is explained only by known-versus-
unseen family coverage.

## Decision

- Full baseline passes and seen-family robustness passes:
  - momentum is statistically confirmed as a v2.7 component candidate;
  - production integration still requires a later composition/backtest gate.
- Either gate fails:
  - momentum confirmation fails;
  - do not retune decay rate, shrinkage, or weight to rescue it.
- Transition is not recomputed at 1,000 samples because it already failed screening.
