# v2.7.1 Brier Reliability Result

Status: implementation verified; number/pair Brier reliability neutral; superseded by direct ranking rejection.

## Run identity

- branch: `dev/v2.7-validation-rebuild`
- pulled head: `6abdd4c`
- local draw rows: `1235`
- latest reflected draw: `1235`
- target draw: `1236`
- unit tests: `78`
- failures: `0`
- errors: `0`
- recommendation smoke candidates: `2,000`

## Number evidence

Production research spec:

- model: `full_prior_120`
- prior strength: `120`
- half-life: none
- fair marginal null: `6 / 45`

Strict walk-forward Brier skill versus uniform:

- overall: `-0.0013900204`
- recent 300: `-0.0009973398`
- recent 100: `-0.0011491355`
- derived reliability: `0.0000000000`

All three windows were worse than the fair uniform marginal forecast, so the continuous Brier reliability rule shrank number evidence to zero.

## Pair evidence

Production research spec:

- model: `full_prior_330`
- prior strength: `330`
- half-life: none
- fair pair null: `1 / 66`

Strict walk-forward Brier skill versus uniform:

- overall: `-0.0006724232`
- recent 300: `-0.0004667738`
- recent 100: `-0.0004752381`
- derived reliability: `0.0000000000`

All three windows were worse than the fair pair marginal forecast, so pair evidence was also shrunk to zero.

## Smoke ranking consequence

Because both reliabilities were exactly zero, every candidate had neutral weighted evidence and the display transform returned `prediction_score == 50`.

The smoke TOP-10 therefore was not predictive ordering; it only exercised deterministic exact-tie handling. Raw number/pair log-lifts were correctly computed and reached the recommendation path, but their validated contribution was zero.

## Follow-up ranking audit

A separate strict walk-forward audit then tested the application target directly: rank the actual historical winning combination against fair random valid 6/45 alternatives.

That audit also produced no survivor:

- number overall percentile: `48.6213`;
- pair overall percentile: `49.7819`;
- frozen 50:50 fusion overall percentile: `49.3711`;
- all 95% block-bootstrap intervals crossed or remained below zero relative to percentile 50;
- no 2,000-sample confirmation was authorized.

See `docs/v271_ranking_evidence_result.md`.

## Final interpretation

The v2.7.1 wiring worked correctly, but the tested full-history number/pair posterior family did not establish predictive ranking evidence.

The active research hypothesis has therefore moved to the original reverse-learning concept expressed as a strict nested rolling ranker. See `docs/v28_reverse_ranking_spec.md`.
