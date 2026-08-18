# v2.8 Reverse-Learning Nested Ranking Result

Status: screening failed; linear reverse-learning ranker is not promoted.

## Run identity

- branch: `dev/v2.7-validation-rebuild`
- runner commit: `b17fb9480b2c5b0e580a57cdbf36e24b45f72389`
- local draw rows: `1235`
- latest reflected draw: `1235`
- data fingerprint prefix: `fef021c1efdc`
- unit tests: `86`
- failures: `0`
- errors: `0`

## Frozen screen configuration

- history start index: `100`
- minimum solved meta-training targets: `100`
- training negatives per solved target: `64`
- fair evaluation samples per outer target: `500`
- ridge lambda: `2.0`
- block bootstrap repetitions: `2000`
- block size: `20`
- pattern type: not used
- outer tests: `1035`

## Outer ranking result

```text
mean percentile      48.9486
recent 300           48.2933
recent 100           47.3860
above random median   0.4667
percentile-50 CI95   [-2.9488, 0.8963]
screening candidate  false
```

The model fails every directional promotion requirement. The overall mean is below
50, both recent windows are below 50, and the bootstrap interval crosses zero.

## Latest learned coefficients

The final diagnostic model fitted after all historical targets produced the following
effective weights:

```text
number_full_log_lift       -0.1715823574
pair_full_log_lift         -0.0238747351
number_recent20_excess      0.0078774964
number_recent100_excess     0.1672304088
pair_recent100_excess      -0.6179855287
previous_draw_overlap       0.0059877174
sum_abs_deviation_138       0.0002973766
odd_imbalance              -0.0008404926
number_range               -0.0007427873
consecutive_pairs          -0.0021879002
```

These values are diagnostic only. They are not evidence that reversing any one
coefficient is predictive, and they must not be reused as hand-tuned production
weights.

## Decision

- Do not run the 2,000-candidate confirmation for v2.8.
- Do not retune `ridge_lambda` from this result.
- Do not delete or select individual v2.8 features from this result.
- Do not choose a different recent horizon from this result.
- Keep the production recommendation path neutral.

The tested hypothesis was specifically that a **linear** pairwise ridge function of
the ten frozen reverse-statistics features could transfer from older solved targets
to unseen later targets. That hypothesis was not supported.

## Next research class

One final predeclared nonlinear class is permitted on the current 1,235-draw dataset:
a second-order polynomial expansion of the exact same ten base features, with all
other nested-walk-forward mechanics held fixed. This is v2.9 and is documented in
`docs/v29_quadratic_reverse_ranking_spec.md`.

Because model-class selection has already used aggregate results from the same data,
v2.9 is an exploratory screen. Even a v2.9 historical survivor does not by itself
constitute an independent production validation; a separate frozen confirmation and
future-data validation policy is still required.