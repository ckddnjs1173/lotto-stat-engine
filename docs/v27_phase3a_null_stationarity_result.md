# v2.7 Phase 3A Null & Stationarity Screening Result

Status: screening complete; no confirmation candidates.

## Run identity

- runner commit: `34122284b3360a330a27f8bc163db664a7ddc09b`
- data rows: `1235`
- latest reflected draw: `1235`
- data fingerprint prefix: `fef021c1efdc...`
- fair 6/45 samples: `20000`
- permutation repetitions: `500`
- maximum ACF lag: `10`
- recent-window size: `300`
- targeted regression tests before run: `5 / 5 OK`

## Fair-draw marginal comparison

All nine structural-feature mean displacements were small after including both
historical sampling error and fair-reference Monte Carlo error:

| Feature | Mean z vs fair |
|---|---:|
| sum | 0.1266 |
| number range | -0.6487 |
| odd count | 0.4885 |
| max gap | -1.2267 |
| min gap | 1.1533 |
| consecutive-pair count | 0.1249 |
| duplicate-ending count | 0.2441 |
| empty decade-section count | 0.5025 |
| extreme count | 0.3638 |

The marginal comparison is descriptive and supplies no predictive evidence.

## Time-order screening

After Holm correction across the nine structural features:

- serial-dependence candidates: none
- recent-distribution-shift candidates: none
- CUSUM/change-point candidates: none

The smallest adjusted recent-shift p-value was `0.3054` for `max_gap`.
Adjusted ACF p-values were at least `0.8623`, and all adjusted CUSUM p-values
were `1.0000`.

Therefore the predeclared prerequisite for AR/ARIMA, recent-regime weighting, or
change-point logic was not met.

## Pattern-type transition screening

- adjacent pattern-type mutual information: `0.0020`
- permutation p-value: `0.2954`

This does not support detectable first-order dependence in the observed
normal/mixed/outlier sequence. It therefore supplies no Phase-3 support for the
current Markov component.

## Pattern-type marginal reference

Historical vs exact-fair Monte Carlo proportions:

| Pattern | Historical | Fair | Difference |
|---|---:|---:|---:|
| normal | 0.2121 | 0.2024 | 0.0098 |
| mixed | 0.5563 | 0.5747 | -0.0185 |
| outlier | 0.2316 | 0.2229 | 0.0087 |

These differences are descriptive and are not transition or predictive evidence.

## Decision

1. Do not run 5,000-permutation confirmation because Phase 3A produced no candidate.
2. Keep AR/ARIMA out of scope.
3. Do not add regime or change-point weighting.
4. Treat pattern-type Markov dependence as unsupported by Phase 3A.
5. Keep v2.6 production scoring frozen.
6. Move only to a cheap Phase-4 direct dynamic-evidence audit of transition and
   momentum, because Phase 1 did not establish either component and Phase 3A found
   no prerequisite time-order structure.
