# v2.7.1 Combination Ranking Evidence Result

Status: screening complete; no Bayesian posterior track survived.

## Run identity

- runner commit: `903a3e0f056946e1b2fab40d041360e66990e5f8`
- branch: `dev/v2.7-validation-rebuild`
- data rows: `1235`
- latest reflected draw: `1235`
- start index: `100`
- strict walk-forward targets: `1135`
- fair baseline combinations per target: `500`
- circular block-bootstrap reps: `2000`
- block size: `20`
- pattern type: not used
- regression tests before the audit: `82`, all passed

## Number track

- overall mean percentile: `48.6213`
- recent 300: `48.7750`
- recent 100: `48.6380`
- above-random-median ratio: `0.4775`
- percentile-minus-50 95% block-bootstrap CI: `[-2.9364, 0.2837]`
- screening candidate: `false`

Decision: reject. The track is below the fair-combination median in overall and both recent windows.

## Pair track

- overall mean percentile: `49.7819`
- recent 300: `51.3820`
- recent 100: `50.9560`
- above-random-median ratio: `0.5084`
- percentile-minus-50 95% block-bootstrap CI: `[-1.7670, 1.5478]`
- screening candidate: `false`

Decision: reject. The recent windows are mildly positive, but the overall mean is below 50 and the confidence interval crosses zero. Do not promote the recent slice or run confirmation from this screen.

## Equal-family fusion

Frozen screen formula:

```text
0.5 * number_raw_log_lift + 0.5 * pair_raw_log_lift
```

Results:

- overall mean percentile: `49.3711`
- recent 300: `50.6620`
- recent 100: `50.4060`
- above-random-median ratio: `0.4881`
- percentile-minus-50 95% block-bootstrap CI: `[-2.2251, 1.1569]`
- screening candidate: `false`

Decision: reject.

## Final Phase decision

Screening survivors: none.

Therefore:

1. do not run the 2,000-fair-candidate confirmation step;
2. do not replace production reliability with the diagnostic positive pair/fusion rank reliabilities;
3. keep the current full-history Bayesian number/pair posterior family neutral in recommendation ranking;
4. do not retune prior strengths, decay horizons, or number/pair fusion weights from these results;
5. move to a genuinely different model class rather than recombining the rejected posterior tracks.

## Next research class

The next class is the original reverse-learning idea expressed without target leakage:

- draw 101 becomes the first solved historical ranking problem using only draws 1..100 to construct its candidate features;
- after the answer is known, its actual-vs-fair candidate differences become training evidence for later draws;
- draw 102 may learn from draw 101 but may not use the draw-102 answer to fit its own coefficients;
- the process repeats sequentially;
- after a predeclared minimum number of solved historical targets, every later draw is a genuine outer walk-forward test;
- model coefficients are fitted only from previously solved targets and strongly regularized.

This is a nested rolling ranking problem, not a static frequency heuristic and not a retrospective fit to the target being scored.
