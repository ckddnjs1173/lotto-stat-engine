# v3.1 component influence result and withdrawn provisional CLEAN3 decision

## Dataset / target

The focused component audit was run locally with draw 1237 included:

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

| feature | full-minus-without | recent300 | recent100 | bootstrap CI95 | current interpretation |
|---|---:|---:|---:|---|---|
| previous_draw_overlap | +0.1489 | -0.4020 | -0.1460 | [-0.5188, 0.8932] | inconclusive |
| number_full_log_lift | +0.7256 | +1.2607 | +1.0820 | [-0.0768, 1.5165] | inconclusive, positive direction |
| consecutive_pairs | -0.0473 | +0.0800 | +0.4620 | [-0.6156, 0.5393] | inconclusive |
| number_range | -0.0488 | +0.5180 | +1.5980 | [-0.4413, 0.3535] | inconclusive |

None of the four bootstrap intervals excludes zero. Therefore none of these four features has a decisive individual historical keep/remove result from this audit.

## Latest full-11 TOP-1000 distribution

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

The counterfactual audit showed that the named features strongly influence these latest TOP-pool statistics. That is evidence of **influence**, not by itself evidence of **error**. A predictive top tail is allowed to differ from the fair population.

## Why the provisional CLEAN3 promotion was withdrawn

An earlier repository state promoted a reduced model that jointly removed:

```text
previous_draw_overlap
consecutive_pairs
number_range
```

That step was too early for two reasons:

1. every focused single-feature result above was statistically inconclusive;
2. the three-feature reduced model had not been jointly evaluated before being labeled the current model.

Therefore CLEAN3 is now only:

```text
experimental_candidate_requires_joint_and_stability_audit
```

The implementation still retrains the reduced ridge system correctly; the withdrawal concerns model-selection evidence, not the subset linear algebra.

## Additional issue found in repository-wide review

The current fair-null transform uses only 1,024 sampled fair combinations per target while final ranking evaluates 8,145,060 combinations. Extreme values can therefore saturate at `z=-1/+1`, creating score plateaus. The previous TOP-1000 distribution was also ordered with a seed-dependent hash tie-break.

Because of this, latest-pool concentration must be reinterpreted only after:

- reference seed/size stability audit;
- `z` saturation diagnostics;
- fixed RNG-free tie handling;
- TOP cutoff tie multiplicity checks.

The shared v3.1 core now uses fixed combination ordering for exact score ties. Existing FULL11/CLEAN3 sample-based fair-null representations remain available as explicit scenarios so old results can still be reproduced conceptually while the representation is audited.

## Next decision rule

Do not choose a feature subset because its TOP pool looks closer to random. Instead:

1. stabilize the fair-null representation numerically;
2. audit all 11 features, not only four;
3. audit correlated groups such as long-run number/pair and recent20/recent100;
4. inspect ridge condition/coefficient stability;
5. only then decide which feature set, if any, is frozen for future independent draws.
