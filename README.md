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
- v2.7.1 number/pair Bayesian posterior evidence failed both Brier reliability and direct actual-winning-combination ranking screening;
- v2.8 strict nested reverse-learning with a ten-feature **linear** pairwise ridge ranker also failed its frozen outer screen: mean percentile `48.9486`, recent-300 `48.2933`, recent-100 `47.3860`, CI95 for percentile-minus-50 `[-2.9488, 0.8963]`.

The production-facing recommendation path therefore remains neutral. A tied 50-point output must not be interpreted as predictive ordering.

## v2.9 quadratic reverse-learning research

v2.9 tests one final predeclared model class on the current 1,235-draw dataset.
It keeps every v2.8 data/leakage/sampling decision fixed and changes only the function
class from a linear ten-term score to the complete degree-2 polynomial basis of the
same ten inputs.

Base inputs remain:

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

Quadratic representation:

```text
10 linear terms
+ 10 squared terms
+ 45 pairwise interactions
= 65 total terms
```

Frozen mechanics remain identical to v2.8:

```text
history start index       = 100
minimum solved targets    = 100
training negatives/target = 64
evaluation negatives      = 500
ridge lambda              = 2.0
bootstrap reps            = 2000
bootstrap block size      = 20
pattern type              = not used
fair sample seed paths    = same as v2.8
```

The target answer is revealed only after that target's outer score is recorded.
Appending a future draw must not alter any earlier outer target result.

See:

- `docs/v28_reverse_ranking_result.md`
- `docs/v29_quadratic_reverse_ranking_spec.md`

## Research-budget rule

v2.9 is the last new model class to be screened on the current fixed 1,235-draw data
without new draw evidence. Repeatedly inventing additional classes after seeing each
aggregate historical result would overfit the research process itself.

If v2.9 fails, do not tune lambda, select interactions, change polynomial degree,
change horizons, or create another model class from this same data. Freeze the model
search and require new/future evidence or a separately declared research program.

If v2.9 passes, it earns a frozen 2,000-fair-candidate confirmation only. Because the
class was selected after observing prior family results, historical passage alone is
still not independent production validation.

## Current validation commands

After pulling the branch:

```powershell
python -m unittest discover -s tests -v
```

Then run the frozen v2.9 exploratory screen:

```powershell
python scripts\run_v29_quadratic_reverse_ranking.py --baseline-samples 500 --bootstrap-reps 2000 --progress-every 50 --output-json data\cache\v29_quadratic_reverse_ranking_screen.json
```

Historical gate reported by the runner:

- overall mean winning-combination percentile > 50;
- block-bootstrap 95% CI for percentile-minus-50 has lower bound > 0;
- recent-300 mean >= 50;
- recent-100 mean >= 50.

A gate pass is labeled exploratory, not production-approved.

## Historical result documents

- `docs/v27_phase5_static_component_result.md` — legacy static component rejection;
- `docs/v271_brier_reliability_result.md` — number/pair Brier reliability rejection;
- `docs/v271_ranking_evidence_result.md` — direct ranking rejection of number, pair, and frozen 50:50 fusion;
- `docs/v28_reverse_ranking_spec.md` — linear nested reverse-learning specification;
- `docs/v28_reverse_ranking_result.md` — linear nested reverse-learning rejection;
- `docs/v29_quadratic_reverse_ranking_spec.md` — final fixed-data nonlinear screen specification.

## Recommendation command

`scripts/run_recommend.py` remains available for reproducibility/site-contract work,
but while validated evidence reliability is zero its tied 50-point output is neutral
and should not be presented as predictive TOP-K.

Research audits and the production-facing recommendation command remain intentionally
separate until a frozen model has sufficient evidence for an explicit promotion
decision.
