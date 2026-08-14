# v2.7 Phase 4 Dynamic-Evidence Screening Result

Status: screening complete; momentum survives to targeted confirmation.

## Run identity

- runner commit: `5964aec0a3b1f385ef3a20d3134adefa58c535f4`
- data rows: `1235`
- latest reflected draw: `1235`
- data fingerprint prefix: `fef021c1efdc...`
- walk-forward start index: `100`
- targets: `1135`
- fair random combinations per target: `200`
- block-bootstrap repetitions: `2000`
- block size: `20`
- targeted Phase-4 tests before run: `4 / 4 OK`

## Primary results

| Component | Overall percentile | Recent 300 | Recent 100 | 95% CI for percentile-50 | Decision |
|---|---:|---:|---:|---:|---|
| transition_exact | 50.1634 | 51.4100 | 55.0700 | [-1.7100, 1.9610] | fail |
| momentum | 52.2302 | 52.3133 | 50.0675 | [0.7436, 3.7683] | confirmation candidate |

`transition_exact` does not satisfy the predeclared gate because its interval includes
zero. No larger transition run is justified.

`momentum` satisfies all predeclared screening conditions: overall percentile above
50, a bootstrap interval entirely above zero, and non-negative recent-300 and
recent-100 windows.

## Transition hierarchy diagnostics

- coarse minus type: overall `+0.6767`, CI `[-0.9616, 2.3787]`
- exact minus coarse: overall `+0.1264`, CI `[-0.6673, 1.0212]`
- exact minus type: overall `+0.8031`, CI `[-0.7735, 2.3642]`

No hierarchical transition increment has an interval entirely above zero. Phase 3A
also found no pattern-type transition dependence, so transition research is stopped
here rather than rescued by retuning.

## Momentum diagnostics

- overall percentile: `52.2302`
- recent 300: `52.3133`
- recent 100: `50.0675`
- fraction above random median: `0.5436`
- mean actual log-lift: `-0.2285`
- by actual type:
  - normal: `54.2168`
  - mixed: `52.8965`
  - outlier: `48.6663`

The negative mean actual log-lift does not contradict the percentile result: the
screening question is relative ranking against fair candidates under the same
historical state. However, it motivates a stricter confirmation check because exact
families absent from history receive the neutral production fallback lift `1.0`.

## Decision

1. Stop transition confirmation.
2. Do not change production dynamic weights.
3. Confirm momentum only with an independent 1,000-candidate fair baseline.
4. During confirmation, separate the momentum effect from exact-family coverage by
   requiring a positive seen-family conditional result as well as the full fair
   baseline result.
5. If either confirmation gate fails, momentum is not promoted and is not rescued by
   weight or decay retuning.
