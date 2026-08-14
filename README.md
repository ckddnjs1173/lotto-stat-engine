# Lotto Stat Engine Final - Latest-Data Dynamic Structure

This is the production 6/45 engine. `prediction_score` is an internal empirical
ranking score, not an actual winning probability.

## Production workflow

1. Add the newest winning draw to `data/lotto.xlsx`.
2. Run the engine; it automatically detects `latest_draw`.
3. It rebuilds every historical structure, subtype, family, adjacent-transition,
   and decay-momentum statistic through that draw.
4. It exhaustively scores all 8,145,060 valid 6/45 combinations.
5. It recommends exactly 10 combinations for `latest_draw + 1`.

No draw number is hardcoded. The latest real draw is always the prediction
state, and adding a valid row requires no source-code changes.

## Production score

```text
prediction_score =
0.70 * base_score
+ 0.15 * transition_lift_score
+ 0.15 * momentum_lift_score
```

The base score preserves the established structural evidence: empirical and
interaction lift, robust backbone, controlled extremes, and subtype/family
structure.

Transition evidence uses only real chronological `t -> t+1` pairs. Sparse rows
are not trusted directly: target-family probabilities are hierarchically shrunk
from the global family prior through pattern type and coarse family to the exact
latest family. The current transition prior strength is a conservative research
value, not a fitted winning-probability parameter.

Production momentum is anchored at the global latest draw and uses all draw
types. Exponentially decayed recent family mass is shrunk toward the long-run
family distribution before converting recent/long-run lift to the 0-100 score.
A lift of 1 maps to the neutral score 50.

Type and mixed-family allocations are recomputed on every run and applied only
as bounded portfolio preferences. Candidate retention is partitioned by pattern
type so one high-scoring type cannot erase all other types before final portfolio
selection. Every valid combination remains eligible; there are no structural
hard filters, bonus-number scores, or aesthetic number rules.

## Evidence policy

New statistical ideas are researched independently before they are allowed into
the production score. A component is not promoted because it sounds plausible or
because it matches the latest draw. Evaluation uses strict rolling-origin
walk-forward tests: target draw `t` may use only draws `< t`.

### Rejected: standalone number hot/cold frequency

`number_bayes_evidence_v1` tested full-history and exponentially decayed Bayesian
marginal number frequencies against the fair `6/45` null across 1,135 strict
walk-forward targets. All predeclared variants produced negative Brier skill and
worse log loss overall, in the most recent 300 targets, and in the most recent
100 targets. Therefore standalone hot/cold or recent-number frequency is not a
production feature.

### Research: pair frequency / interaction

`pair_bayes_evidence_v1` tests all 990 unordered number pairs against the fair
pair-inclusion probability `1/66`. It is research-only. Pair evidence must improve
out-of-sample proper scores before any pair-derived component can be considered
for production scoring.

## Commands

```powershell
python -m unittest discover -s tests -v
python scripts\validate_data.py
python scripts\analyze_mixed_subtypes.py
python scripts\run_mixed_backtest.py
python scripts\run_number_evidence_backtest.py
python scripts\run_pair_evidence_backtest.py
python scripts\run_recommend.py
```

Use a limited smoke run to verify output wiring without performing the full
exhaustive production run:

```powershell
python -c "from lotto_engine.recommender import generate_recommendations, print_recommendations; print_recommendations(generate_recommendations(exhaustive=False, candidate_count=10000))"
```

Historical diagnostic scripts and the v2.2-v2.5 regression tests remain in the
repository as supporting evidence; they are not separate production engines.
