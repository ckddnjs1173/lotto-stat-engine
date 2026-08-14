# v2.7 Phase 4 Momentum Confirmation Result

Status: confirmation failed; momentum is not promoted.

## Run identity

- runner commit: `a2affe4bdde1a254f3433c363f01bed101f91e48`
- data rows: `1235`
- latest reflected draw: `1235`
- data fingerprint prefix: `fef021c1efdc...`
- walk-forward start index: `100`
- targets: `1135`
- fair random combinations per target: `1000`
- block-bootstrap repetitions: `2000`
- block size: `20`
- targeted confirmation tests before run: `3 / 3 OK`

## Full fair-baseline result

- overall percentile: `52.0493`
- recent 300: `51.9145`
- recent 100: `49.5720`
- fraction above random median: `0.5383`
- 95% CI for `(percentile - 50)`: `[0.5126, 3.5500]`
- by actual type:
  - normal: `54.3664`
  - mixed: `52.6819`
  - outlier: `48.2569`

The long-run effect is positive and the confidence interval excludes zero, but the
predeclared confirmation rule also requires the recent-100 percentile to be at
least 50. The observed value, `49.5720`, fails that condition.

## Seen-family robustness

- targets with a finite seen-family conditional percentile: `943`
- mean conditional percentile: `51.9941`
- recent 300 target-window mean: `52.0342`
- recent 100 target-window mean: `50.4833`
- 95% CI for `(conditional percentile - 50)`: `[0.2661, 3.6713]`

The coverage-controlled result is positive. Therefore the screening effect is not
explained solely by known-versus-unseen exact-family coverage.

## Family coverage

- actual target family seen ratio: `0.8308`
- recent-300 actual seen ratio: `0.9100`
- recent-100 actual seen ratio: `0.9400`
- fair baseline mean family-seen ratio: `0.8460`
- fair baseline mean seen count per 1000: `845.97`
- minimum seen count: `546`

## Interpretation

Momentum shows a reproducible positive long-run ranking displacement under both the
full fair baseline and the seen-family-controlled comparison. However, the effect
is not sufficiently stable in the most recent 100 targets under the gate fixed
before confirmation.

This result is therefore classified as **long-run positive but not confirmed for
production promotion**, rather than as evidence of no effect whatsoever.

## Decision

1. Momentum confirmation fails because `recent100 = 49.5720 < 50`.
2. Do not retune decay rate, prior strength, or production weight to rescue it.
3. Transition remains rejected from the Phase-4 screen.
4. Dynamic components are closed for v2.7 production promotion.
5. Keep v2.6 production scoring frozen.
6. Move to a same-pattern-type audit of the static/base component evidence so raw
   cross-type score-scale distortion cannot create false component wins.
