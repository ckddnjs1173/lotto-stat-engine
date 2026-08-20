# v2.7 Phase 1 Screening Result

Status: screening complete; no production promotion.

## Run identity

- benchmark commit: `ecbbd02235b9ed8f6940ae47aa46e7d4c0e53499`
- runner commit: `da9481ac9845dd70f3e2841a5303e91b68db3953`
- data rows: `1235`
- latest reflected draw: `1235`
- data fingerprint prefix printed by the run: `fef021c1efdc...`
- walk-forward targets: `1135`
- random baseline candidates per target: `500`
- block bootstrap repetitions: `2000`
- block size: `20`
- regression tests before the run: `48 / 48 OK`

The score audit itself completed all 1,135 targets. JSON persistence failed only after
the statistics were printed because a NumPy `int64` was not converted to a JSON
primitive. That artifact-only bug was fixed separately in commit `09ba2d18`.

## Primary model screening

| Model | Overall percentile | Recent 300 | Recent 100 | Delta vs base | 95% block-bootstrap CI vs base |
|---|---:|---:|---:|---:|---:|
| `base_only` | 50.2330 | 52.1633 | 51.3540 | — | — |
| `base_plus_transition` | 50.3762 | 52.6233 | 53.4580 | +0.1433 | [-0.8207, 1.1332] |
| `base_plus_momentum` | 50.8814 | 51.7200 | 49.4280 | +0.6485 | [-0.2189, 1.5426] |
| `v26_full` | 51.1244 | 52.4807 | 52.2200 | +0.8915 | [-0.4273, 2.2438] |

No primary model passes the predeclared Phase-1 promotion gate because every paired
95% interval versus `base_only` includes zero. Momentum also reverses direction in
recent windows.

## Score-scale distortion finding

The strongest Phase-1 result is not a winning component. It is a comparability
problem in the raw score scale.

For `base_only`, mean global actual-vs-random percentile by actual pattern type was:

- normal: `71.7544`
- mixed: `28.2606`
- outlier: `85.1764`

Yet the same model's overall type-conditioned percentile was only `49.4593`.
This is consistent with large raw-score scale differences between
normal/mixed/outlier rather than strong within-type predictive discrimination.

For `v26_full`, the corresponding global by-type values were:

- normal: `71.1436`
- mixed: `34.7103`
- outlier: `73.4882`

while the overall type-conditioned percentile was `51.2447`.

This finding promotes score calibration to the next research phase. It does not
promote any scoring component to production.

## Legacy-component ablation

Removing type rarity, legacy type transition, or both produced near-zero changes.
All relevant bootstrap intervals crossed zero. Phase 1 therefore does not support
either retaining or removing those terms on predictive evidence alone.

## Decision

1. Freeze production formula and candidate eligibility.
2. Do not tune 70/15/15.
3. Do not promote transition or momentum.
4. Run a strict independent-sample type calibration audit.
5. Only after calibration is resolved, continue to null/stationarity and dynamic
   component significance tests.
