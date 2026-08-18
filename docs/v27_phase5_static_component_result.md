# v2.7 Phase 5A Static Component Evidence Result

## Status

Phase 5A screening completed on local Lotto data through draw 1235.

- runner commit: `d2399347b6ca7a18f9af8eed38029dbe48bce30f`
- rows / latest draw: `1235 / 1235`
- walk-forward start index: `100`
- targets: `1135`
- same-type fair samples per target: `100`
- circular block-bootstrap reps: `2000`
- block size: `20`

The predeclared screening gate required the component to beat the same-pattern-type
fair baseline with overall mean percentile above 50, a 95% bootstrap interval for
`percentile - 50` entirely above zero, non-negative recent-300 and recent-100
stability, and applicable pattern-type support.

## Current branch-base benchmark

| model | overall | recent300 | recent100 | CI95 for percentile-50 | candidate |
|---|---:|---:|---:|---|---|
| current_branch_base | 49.3630 | 49.2067 | 45.7300 | [-2.3953, 1.1491] | no |

By actual pattern type:

| type | overall | recent300 | recent100 | n |
|---|---:|---:|---:|---:|
| normal | 50.8465 | 48.2941 | 40.9524 | 241 |
| mixed | 49.2563 | 51.1125 | 51.2500 | 640 |
| outlier | 48.2244 | 45.8333 | 38.8148 | 254 |

## Normal / outlier components

| component | overall | recent300 | recent100 | CI95 for percentile-50 | candidate |
|---|---:|---:|---:|---|---|
| normal_structure_score | 47.8525 | 45.4357 | 47.5417 | [-4.9053, 0.6005] | no |
| outlier_survival_score | 49.9899 | 48.0714 | 39.5417 | [-2.9297, 3.1601] | no |
| historical_pattern_score | 48.3434 | 48.3286 | 49.8229 | [-3.9600, 0.6313] | no |
| type_balance_score | 47.6919 | 45.6107 | 48.0521 | [-5.0354, 0.2627] | no |
| number_dynamics_score | 49.0424 | 50.0143 | 43.8750 | [-3.4519, 1.6751] | no |

## Mixed components

| component | overall | recent300 | recent100 | CI95 for percentile-50 | candidate |
|---|---:|---:|---:|---|---|
| mixed_lift_score | 50.2008 | 50.7969 | 50.8462 | [-2.0118, 2.4704] | no |
| mixed_interaction_score | 49.8852 | 51.8156 | 56.5385 | [-2.2720, 1.9407] | no |
| normal_backbone_score | 49.4742 | 52.2125 | 46.2885 | [-3.1204, 1.9774] | no |
| controlled_extreme_score | 48.7305 | 46.9719 | 45.7019 | [-3.3906, 0.7120] | no |
| recency_consistency_score | 49.5250 | 50.3281 | 48.2692 | [-2.6324, 1.7458] | no |

## Screening decision

```text
normal/outlier common survivors: []
mixed survivors: []
```

No component advances to the predeclared 500-same-type confirmation stage. The
predeclared rule therefore prohibits rescuing a legacy static component by retuning
its weight after seeing this result.

## Architectural consequence

The result does not mean that every historical calculation must be deleted. It means
that the legacy static structure components are not justified as the production
ranking backbone under this screening protocol.

v2.7.1 therefore changes their role:

- legacy static components remain available for historical audit and diagnostics;
- Normal / Mixed / Outlier remains descriptive metadata;
- the recommendation entry point no longer branches to separate static formulas by
  pattern type;
- number/pair Bayesian evidence is connected to recommendation ranking through a
  separate reliability-shrunk research path;
- all 8,145,060 valid combinations remain eligible.

This Phase 5A result supersedes the earlier statement that Phase 5 could be deferred
without affecting the v2.7 release decision.
