# v2.7 Phase 2 Calibration Screening Result

Status: screening complete; calibration rejected for production promotion.

## Run identity

- benchmark commit: `ecbbd02235b9ed8f6940ae47aa46e7d4c0e53499`
- runner commit: `966145b966b3c8dacebed90725d23f070ec24e2f`
- data rows: `1235`
- latest reflected draw: `1235`
- data fingerprint prefix: `fef021c1efdc...`
- walk-forward targets: `1135`
- calibration candidates per target: `1000`
- evaluation candidates per target: `500`
- block-bootstrap repetitions: `2000`
- block size: `20`
- minimum same-type calibration coverage: `160`
- regression tests before the run: `52 / 52 OK`

## Results

| Model | Raw overall | Calibrated overall | Delta | Recent 300 delta | Recent 100 delta | 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| `base_only` | 50.2986 | 49.3099 | -0.9887 | -3.0667 | -6.2110 | [-2.8296, 1.0651] |
| `v26_full` | 51.2216 | 51.1914 | -0.0302 | -1.6597 | -2.3670 | [-1.2486, 1.1986] |

Neither model satisfies the predeclared promotion gate. Both overall effects are
non-positive, both recent windows deteriorate, and both confidence intervals
include zero. Therefore the expensive `3000 x 2000` confirmation run is not
justified.

## What calibration did

Calibration successfully removed most of the raw score-scale separation between
pattern types, but that normalization did not improve future winning-combination
ranking.

### `base_only`

- normal: 71.6212 -> 50.8083
- mixed: 28.4553 -> 49.2569
- outlier: 85.1055 -> 48.0217

### `v26_full`

- normal: 71.0909 -> 53.6315
- mixed: 34.8516 -> 50.1952
- outlier: 73.6165 -> 51.3862

This separates two questions that Phase 1 could not answer alone:

1. Raw score scales were indeed not comparable across pattern types.
2. Equalizing those scales is not itself predictive evidence.

The large Mixed-vs-Normal/Outlier raw disparity should therefore not be corrected
in production merely because it looks unbalanced.

## Decision

1. Reject type-wise empirical calibration as a production change.
2. Do not run the `3000 x 2000` calibration confirmation.
3. Keep v2.6 production scoring frozen.
4. Keep AR/ARIMA out of scope until serial dependence is demonstrated.
5. Move to a cheap Phase 3A null/stationarity screen before spending compute on
   dynamic-model confirmation.
