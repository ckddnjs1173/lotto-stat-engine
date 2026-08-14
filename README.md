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

## Final score

```text
prediction_score =
0.70 * base_score
+ 0.15 * transition_lift_score
+ 0.15 * momentum_lift_score
```

All three components use a comparable 0-100 scale. The base score preserves
the established structural evidence: empirical and interaction lift, robust
backbone, controlled extremes, and subtype/family structure. Transition
evidence uses only real chronological `t -> t+1` pairs with Bayesian smoothing
and hierarchical fallback. Momentum uses the global latest draw as its decay
origin. Both lift components map lift 1 to the neutral score 50 and clip sparse
extremes.

Type and mixed-family allocations are recomputed on every run and applied only
as bounded portfolio preferences. Every valid combination remains eligible;
there are no hard filters, random recommendation layers, bonus-number scores,
or aesthetic number rules.

## Commands

```powershell
python -m unittest discover -s tests -v
python scripts\validate_data.py
python scripts\analyze_mixed_subtypes.py
python scripts\run_mixed_backtest.py
python scripts\run_recommend.py
```

Use a limited smoke run to verify output wiring without performing the full
exhaustive production run:

```powershell
python -c "from lotto_engine.recommender import generate_recommendations, print_recommendations; print_recommendations(generate_recommendations(exhaustive=False, candidate_count=10000))"
```

Historical diagnostic scripts and the v2.2-v2.5 regression tests remain in the
repository as supporting evidence; they are not separate production engines.
