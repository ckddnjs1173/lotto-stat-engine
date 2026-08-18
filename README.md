# Lotto Stat Engine v2.7.1

Lotto 6/45 statistical ranking research candidate. `score` / `prediction_score` is an internal ranking score, not an actual winning probability.

## Current research status

Phase 5A completed the previously deferred static-component audit. No legacy Normal/Outlier or Mixed static component survived the predeclared strict walk-forward same-type screen. The former v2.7 frozen static release path is therefore superseded.

The v2.7.1 recommendation path now connects the newer Bayesian evidence modules to the actual recommendation entry point:

- all `C(45, 6) = 8,145,060` valid combinations remain eligible;
- no hard number, structure, aesthetic, subtype, or portfolio filters;
- one common ranking equation for Normal/Mixed/Outlier candidates;
- number posterior evidence is measured against the fair `6/45` marginal null;
- pair posterior evidence is measured against the fair `1/66` pair null;
- strict walk-forward validation controls how much each evidence family can affect ranking;
- transition and momentum remain disabled after the v2.7 validation program;
- pattern type is attached only after ranking as descriptive metadata.

See:

- `docs/v27_phase5_static_component_result.md`
- `docs/v271_evidence_integration_spec.md`
- `docs/v271_brier_reliability_result.md`

## Current production-facing research equation

For candidate combination `c`:

```text
number_log_lift(c)
  = mean over the six numbers of log(P(number | history) / (6/45))

pair_log_lift(c)
  = mean over the fifteen unordered pairs of log(P(pair | history) / (1/66))

ranking_evidence(c)
  = number_reliability * number_log_lift(c)
  + pair_reliability   * pair_log_lift(c)
```

The displayed score is a monotone transform only:

```text
prediction_score = 50 + 50 * tanh(ranking_evidence)
```

Ranking uses the unrounded `ranking_evidence` value.

### Brier reliability result on data through draw 1235

The first local v2.7.1 verification produced:

```text
number Brier skill
  overall   -0.0013900204
  recent300 -0.0009973398
  recent100 -0.0011491355
  reliability 0

pair Brier skill
  overall   -0.0006724232
  recent300 -0.0004667738
  recent100 -0.0004752381
  reliability 0
```

Therefore the current Brier-controlled recommendation score is neutral (`50`) for every combination. Any TOP-K shown while both reliabilities are zero is only deterministic tie handling and must not be interpreted as predictive ordering.

This is not treated as the final verdict on ranking usefulness, because Brier score evaluates marginal probability calibration while the application target is combination ranking.

## Combination-ranking audit

The next frozen validation directly tests the application target: whether the actual historical winning six-number combination ranks above fair random valid combinations using the raw Bayesian evidence.

Tracks:

```text
number
pair
equal_family_fusion = 0.5 * number + 0.5 * pair
```

Each historical target uses only earlier draws. Pattern type is not used. Screening uses deterministic unique fair combinations from the full 6/45 universe and circular block-bootstrap uncertainty.

Run:

```powershell
python scripts\run_v271_ranking_evidence_audit.py --baseline-samples 500 --bootstrap-reps 2000 --progress-every 50 --output-json data\cache\v271_ranking_screen.json
```

Interpretation:

- a screening survivor earns a separate 2,000-fair-combination confirmation;
- the screening result does not directly change production weights;
- no survivor means the current full-history number/pair Bayesian posterior family remains neutral in the recommendation path;
- no post-hoc horizon or prior tuning is allowed to rescue a failed screen.

## Recommendation workflow

Update `data/lotto.xlsx` with the newest completed draw, then run a sampled smoke before any exhaustive recommendation:

```powershell
python scripts\run_recommend.py --sampled --candidate-count 2000 --top-k 10 --output-json data\cache\v271_evidence_smoke.json
```

When an evidence configuration has been accepted for exhaustive research ranking:

```powershell
python scripts\run_recommend.py --top-k 10 --output-json data\cache\v271_recommendations.json
```

## Site integration

`lotto_engine.v27_release.public_recommendation_payload()` remains the site-facing contract. It includes:

- model version and research status;
- latest reflected and target draw;
- evaluation mode and candidate count;
- evidence diagnostics including number/pair specifications, Brier skills, and reliabilities;
- ranked numbers, score, raw ranking evidence, pattern metadata, and component breakdown.

`streamlit_app.py` consumes the same recommendation module.

## Validation policy

Strict rolling-origin rules remain mandatory:

```text
target draw 101 -> history 1..100 only
target draw 102 -> history 1..101 only
...
```

Research signals may be computed, but they affect recommendation ranking only through an explicitly documented validation/reliability layer. A failed or neutral validation must never be silently replaced by legacy static weights.

Run the full test suite after pulling changes:

```powershell
python -m unittest discover -s tests -v
```
