# v2.7.1 Brier Reliability Result

Status: implementation verified; current number/pair Brier reliability is neutral.

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

All three windows are worse than the fair uniform marginal forecast, so the
continuous Brier reliability rule correctly shrinks number evidence to zero.

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

All three windows are worse than the fair pair marginal forecast, so pair evidence
is also shrunk to zero under the Brier reliability rule.

## Smoke ranking consequence

Because both reliabilities are exactly zero, every candidate has
`ranking_evidence == 0` up to signed floating-point zero and the display transform
returns `prediction_score == 50`.

The smoke TOP-10 therefore must **not** be interpreted as predictive ordering. It is
only the deterministic exact-tie selection path being exercised successfully.

The raw number and pair log-lifts are still computed and visible in the breakdown,
but they make no contribution while reliability is zero.

## Interpretation

This result demonstrates that the research-to-production wiring now works: raw
Bayesian evidence reaches the recommendation path and validation controls its
influence. It does **not** establish that the raw evidence family has no ranking
information.

Brier score evaluates probability calibration over all 45 marginal numbers or all
990 marginal pairs. The application target is different: rank the actual six-number
winning combination above fair alternative combinations.

Therefore the next frozen audit is a direct strict walk-forward combination-ranking
test:

- actual winning combination versus fair random valid 6/45 combinations;
- number raw log-lift track;
- pair raw log-lift track;
- predeclared equal-family fusion track;
- no pattern-type conditioning;
- no production weight change from the screening result alone.

See `scripts/run_v271_ranking_evidence_audit.py` and
`lotto_engine/v271_ranking_evidence_audit.py`.
