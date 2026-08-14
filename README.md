# Lotto Stat Engine v2.7

Lotto 6/45 statistical ranking engine. `score` / `prediction_score` is an internal
ranking score, not an actual winning probability.

## v2.7 production policy

- Every valid 6/45 combination remains eligible: `C(45, 6) = 8,145,060`.
- There are no hard number, pattern, aesthetic, or portfolio filters.
- Production selection is pure global TOP-10 by the frozen v2.7 static score.
- Transition and momentum are disabled in production after v2.7 validation.
- Type-wise score calibration is disabled because Phase 2 did not improve ranking.
- AR/ARIMA, regime, and change-point logic are not part of v2.7 production.
- Research modules remain in the repository for reproducibility but are not called by
  the v2.7 production entrypoints.

See `docs/v27_release_candidate.md` for the freeze decision and validation routing.

## Production score

### Normal / outlier branch

The established branch components are reused, but the positive legacy
`transition_score` contribution is removed. The remaining positive static weights
are renormalized:

```text
outlier_survival_score  0.45
type_balance_score      0.25
normal_structure_score  0.15
```

`historical_pattern_score` and `number_dynamics_score` remain zero-weight diagnostic
components.

### Mixed branch

The existing Mixed static base is used with dynamic Markov/momentum context removed:

```text
0.35 * mixed_lift_score
+ 0.25 * mixed_interaction_score
+ 0.20 * normal_backbone_score
+ 0.15 * controlled_extreme_score
+ 0.05 * recency_consistency_score
```

No transition or momentum term is added after the Mixed base score.

## Production workflow

1. Update `data/lotto.xlsx` with the newest completed draw.
2. Run the recommendation command.
3. The engine detects the latest draw automatically.
4. It evaluates all 8,145,060 valid combinations by default.
5. It returns exactly TOP-10 for `latest_draw + 1`.

```powershell
python scripts\run_recommend.py
```

To write the site/API JSON contract in the same run:

```powershell
python scripts\run_recommend.py --output-json data\cache\v27_recommendations.json
```

Development smoke only:

```powershell
python scripts\run_recommend.py --sampled --candidate-count 2000 --top-k 10 --output-json data\cache\v27_rc_smoke.json
```

## Site integration

`lotto_engine.v27_release.public_recommendation_payload()` is the stable v2.7 site
contract. The JSON contains:

- `model_version`
- `release_status`
- `generated_at_kst`
- `latest_draw`
- `target_draw`
- `recommendation_mode`
- `evaluated_count`
- `selection_strategy`
- `candidate_policy`
- explicit transition/momentum disabled state
- ranked recommendations with six numbers, score, pattern type, score origin, and
  active component values

`streamlit_app.py` already consumes the same v2.7 production module.

## Validation summary

The v2.7 research program used strict walk-forward evaluation and separate screening
/ confirmation gates. Key release decisions:

- Phase 2: type-wise calibration rejected.
- Phase 3A: no serial-dependence, recent-shift, change-point, or pattern-transition
  confirmation candidate.
- Phase 4: transition rejected; momentum showed a positive long-run displacement but
  failed the predeclared recent-100 confirmation gate.
- Phase 5 static-component research was stopped when v2.7 moved from research to
  release completion.

This means v2.7 should be described as a statistical ranking system, not a proven
increase in the mathematical probability of a lottery draw.

## Release verification

Implementation work uses only targeted tests. At the release boundary run the full
suite once, then one exhaustive end-to-end recommendation run.

Targeted release test:

```powershell
python -m unittest discover -s tests -p "test_v27_release.py" -v
```

Release-boundary regression:

```powershell
python -m unittest discover -s tests -v
```
