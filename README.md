# Lotto Stat Engine Research Branch

Lotto 6/45 statistical ranking research engine. Any displayed `score` or
`prediction_score` is an internal ranking value, not an actual jackpot probability.

## Current research status

The former v2.7 static release path has been superseded by later validation.

Completed findings:

- all `C(45,6) = 8,145,060` valid combinations remain eligible;
- no hard structural/aesthetic filter is justified;
- Phase 3A did not establish serial dependence, regime shift, or first-order pattern transition evidence;
- transition was rejected;
- momentum failed its predeclared recent-100 confirmation gate;
- Phase 5A found no survivor among the legacy static scoring components;
- v2.7.1 number/pair Bayesian posterior evidence was worse than the fair null on Brier reliability;
- the same number/pair posterior family also produced no survivor when tested directly on actual-winning-combination rank versus fair random combinations.

The current production-facing v2.7.1 recommendation path therefore remains neutral when its validated reliability is zero. A 50-point tied recommendation output must not be interpreted as predictive ordering.

## v2.8 reverse-learning research

The next hypothesis implements the original reverse-statistics idea with strict target isolation.

Historical draw 101 is treated as the first solved ranking problem using only draws 1..100 to construct its candidate features. Once draw 101 has already been scored, its known answer can become training evidence for draw 102 and later. The answer of the target currently being scored is never used to fit its own model.

The frozen first model is a small NumPy-only pairwise ridge ranker with ten features:

- full-history number Bayesian log-lift;
- full-history pair Bayesian log-lift;
- recent-20 number excess frequency;
- recent-100 number excess frequency;
- recent-100 pair excess frequency;
- previous-draw overlap;
- absolute sum deviation from 138;
- odd-count imbalance from 3:3;
- number range;
- consecutive-pair count.

Structural features have no hard-coded favorable sign. Because the learner compares historical winners against fair random valid combinations, it must learn any direction from prior solved targets.

Frozen screen settings:

```text
history start index       = 100
minimum solved targets    = 100
training negatives/target = 64
evaluation negatives      = 500
ridge lambda              = 2.0
bootstrap reps            = 2000
bootstrap block size      = 20
```

See `docs/v28_reverse_ranking_spec.md` for the exact equations and leakage rules.

## Current validation commands

After pulling the branch, first run the regression suite:

```powershell
python -m unittest discover -s tests -v
```

Then run the frozen v2.8 screening audit:

```powershell
python scripts\run_v28_reverse_ranking.py --baseline-samples 500 --bootstrap-reps 2000 --progress-every 50 --output-json data\cache\v28_reverse_ranking_screen.json
```

Screening gate:

- overall mean winning-combination percentile > 50;
- block-bootstrap 95% CI for percentile-minus-50 has lower bound > 0;
- recent-300 mean >= 50;
- recent-100 mean >= 50.

If and only if the frozen screen survives, rerun the exact same model with 2,000 fair evaluation combinations per outer target using the confirmation protocol. Do not tune lambda, remove features, change horizons, or select a recent slice from the screening result.

## Historical result documents

- `docs/v27_phase5_static_component_result.md` — legacy static component rejection;
- `docs/v271_brier_reliability_result.md` — number/pair Brier reliability result;
- `docs/v271_ranking_evidence_result.md` — direct combination-ranking rejection of number, pair, and frozen 50:50 fusion;
- `docs/v28_reverse_ranking_spec.md` — current reverse-learning nested ranking specification.

## Recommendation command

`scripts/run_recommend.py` remains available for reproducibility/site-contract work, but while validated evidence reliability is zero its tied 50-point output is neutral and should not be presented as predictive TOP-K.

The research audit and the production-facing recommendation command are intentionally separate until a frozen model passes both screening and independent confirmation.
