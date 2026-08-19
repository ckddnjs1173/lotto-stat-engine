# v3.1 component influence result and CLEAN3 decision

## Dataset / target

The decisive component audit was run locally with draw 1237 included:

```text
latest reflected draw = 1237
target draw = 1238
historical outer tests = 1037
historical baseline candidates / target = 500
bootstrap reps = 2000
latest candidates = all 8,145,060 combinations
latest diagnostic pool = TOP-1000
```

## Full-11 historical result

```text
mean percentile = 48.2984
recent-300 = 49.4920
recent-100 = 50.5360
percentile-50 CI95 = [-3.4923, 0.0945]
```

This does not establish a historical ranking edge over fair alternatives.

## Focused leave-one-out results

`full-minus-without` is positive when retaining the feature helped the full model and negative when removing it was better.

| feature | full-minus-without | recent300 | recent100 | bootstrap CI95 | decision |
|---|---:|---:|---:|---|---|
| previous_draw_overlap | +0.1489 | -0.4020 | -0.1460 | [-0.5188, 0.8932] | remove from current fit |
| number_full_log_lift | +0.7256 | +1.2607 | +1.0820 | [-0.0768, 1.5165] | retain |
| consecutive_pairs | -0.0473 | +0.0800 | +0.4620 | [-0.6156, 0.5393] | remove from current fit |
| number_range | -0.0488 | +0.5180 | +1.5980 | [-0.4413, 0.3535] | remove from current fit |

None of the four intervals individually excludes zero. The decision is therefore not a claim that one coefficient is statistically proven. It combines historical value with the magnitude of the latest ranking distortion.

## Latest full-11 TOP-1000 distortion

```text
23 inclusion rate = 1.000
20 inclusion rate = 0.488
9 inclusion rate = 0.432

mean previous-draw overlap = 2.007
any previous-draw overlap = 1.000
mean consecutive pairs = 0.000
any consecutive pair = 0.000
mean number range = 21.904
```

Fair 6/45 references:

```text
expected previous-draw overlap = 0.800
any previous-draw overlap = 0.599435...
expected consecutive pairs = 0.666667...
expected number range = 32.857143...
```

Key counterfactuals from the exhaustive audit:

- without `previous_draw_overlap`, mean previous overlap fell by about 1.02;
- without `consecutive_pairs`, the TOP-1000 gained about 1.50 consecutive pairs per combination;
- without `number_range`, mean range rose by about 10.8–11.0, almost exactly to the fair reference.

The old full model was therefore using those features to impose strong structural concentration despite no established individual OOS benefit.

## Production decision

The current personal-use model is CLEAN3:

```text
disabled =
  previous_draw_overlap
  consecutive_pairs
  number_range
```

The model is **retrained on the remaining eight feature columns**. This is not a cosmetic post-hoc subtraction from final scores.

`number_full_log_lift` remains active. It can still concentrate the latest TOP pool, but among the four focused terms it showed the most consistent historical positive direction, so concentration alone is not used as a reason to delete it.

## Limits of the decision

- CLEAN3 is not claimed to prove increased jackpot probability.
- The full-11 historical mean remained below the fair median.
- This cleanup removes demonstrated unsupported structural forcing while preserving the personal raw-ranking research path.
- Future completed draws should be appended and tracked without silently retuning the model after every result.
