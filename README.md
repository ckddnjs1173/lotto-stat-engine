# Lotto Stat Engine Final - Type-Separated Dynamic Portfolio

This is the production 6/45 engine. The user adds the newest winning draw to
`data/lotto.xlsx`; the engine automatically detects that draw, rebuilds the
historical state through it, exhaustively evaluates all 8,145,060 valid
combinations, and recommends exactly 10 combinations for `latest_draw + 1`.

No draw number is hardcoded. Bonus numbers are not scored. There are no
structural hard filters, popularity-avoidance rules, or aesthetic-number rules.

## Final architecture

The production engine deliberately separates three jobs that were previously
mixed together.

### 1. Next pattern-type budget

The actual chronological normal/mixed/outlier transitions are estimated from
historical `t -> t+1` pairs. The latest draw's pattern type determines the
next-draw type distribution. That distribution is converted by largest remainder
into an exact 10-ticket portfolio budget, for example:

```text
normal 2
mixed 5
outlier 3
```

This transition evidence is not added again to normal/outlier candidate scores.
It is used once, at the type-budget layer.

### 2. Ranking inside each type

Scores from different pattern types are not treated as if they were on one
common probability scale.

Normal and outlier candidates are ranked only against candidates of the same
type using static structural evidence. The final static rank excludes type
rarity, type-transition score, and hot/cold number dynamics so those signals are
not double-counted.

Mixed candidates keep the established empirical mixed model:

```text
mixed_base =
0.35 * mixed_lift_score
+ 0.25 * mixed_interaction_score
+ 0.20 * normal_backbone_score
+ 0.15 * controlled_extreme_score
+ 0.05 * recency_consistency_score
```

For mixed candidates only, family dynamics remain active:

```text
mixed_within_type_score =
0.70 * mixed_base
+ 0.15 * transition_lift_score
+ 0.15 * momentum_lift_score
```

Transition evidence uses real adjacent draws with hierarchical shrinkage.
Momentum uses exponential decay anchored at the global latest draw. Lift 1 is
neutral at score 50.

### 3. Portfolio construction

All 8,145,060 combinations are still evaluated. Independent candidate pools are
retained for normal, mixed, and outlier. After exhaustive scoring, the engine
selects exactly the dynamically calculated number of tickets from each type.

Within each type, bounded diversity penalties reduce redundant number/structure
exposure. Within mixed, the v2.4/v2.5 subtype-family allocation also diversifies
historically supported families.

The type budget is a portfolio policy after exhaustive evaluation, not a hard
candidate filter. Every valid 6/45 combination remains eligible for evaluation.

## Why the architecture is separated

The previous implementation compared normal, mixed, and outlier scores globally
even though mixed used a different structural formula. It also included type
transition inside the base score and again in the dynamic/portfolio layers. This
could make one pattern type dominate the final ten tickets despite a different
dynamic allocation target.

The final architecture removes that double counting:

```text
latest historical data
        -> next pattern-type budget
        -> exhaustive candidate evaluation
        -> within-type ranking
        -> mixed family dynamics/allocation
        -> type-budget portfolio selection
        -> final 10 tickets
```

## Production workflow

1. Add the newest winning draw to `data/lotto.xlsx`.
2. Activate the project virtual environment.
3. Run the tests.
4. Run the recommendation engine.
5. Confirm that `evaluated combination count` is `8145060` and that type
   allocation target equals selected.

```powershell
.\.venv\Scripts\Activate.ps1
python -m unittest discover -s tests -v
python scripts\validate_data.py
python scripts\run_recommend.py
```

A limited smoke run can verify wiring before the exhaustive run:

```powershell
python -c "from lotto_engine.recommender import generate_recommendations, print_recommendations; print_recommendations(generate_recommendations(exhaustive=False, candidate_count=100000))"
```

## Evidence policy

`prediction_score`/`within_type_score` are internal empirical ranking scores, not
actual lottery winning probabilities.

Standalone hot/cold marginal number frequency remains excluded from production
because its strict walk-forward proper-score tests did not improve on the fair
6/45 null. Pair-frequency evidence remains research-only until it demonstrates
out-of-sample value.

Historical v2.2-v2.5 tests and diagnostic scripts remain in the repository as
regression/evidence history; they are not separate production engines.
