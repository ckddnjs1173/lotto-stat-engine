# v2.9 Raw Personal Recommendation Path

Status: personal research path; historical validation is reported separately from next-draw ranking.

## Why this path exists

The frozen v2.9 quadratic reverse-learning model did not pass its historical promotion gate on the current fixed dataset. That result must remain visible as validation evidence, but this project is a private research tool whose primary requirement is to expose the model's actual calculated next-draw values without neutralizing them.

The former v2.7.1 recommendation path multiplies evidence by validation-derived reliability. When both reliability values are zero, all candidates collapse to a tied 50-point display score. That behavior is appropriate for a conservative production claim, but it hides the model output needed for private experimentation.

## Ranking rule

The personal v2.9 path ranks candidates by the frozen quadratic model's raw linear predictor only:

```text
raw_model_score(c) = w^T phi(x(c))
```

where:

```text
x(c) = the same frozen 10 v2.8 candidate inputs
phi(x) = complete degree-2 basis
       = 10 linear + 10 squared + 45 pairwise interaction terms
w      = the frozen ridge(lambda=2.0) weights learned from all completed solved targets
```

No historical validation value is multiplied into this score.

No reliability factor can force the score to 50.

No Normal/Mixed/Outlier quota, pattern-type score, section-distribution preference, odd/even preference, or portfolio-diversity term participates in ranking.

Pattern type and structure statistics are attached only after TOP-K selection for explanation in the personal site.

## Training for the next unknown draw

The next-draw model is rebuilt from the local lotto history using the exact v2.9 training sequence.

For every completed target beginning after the first 100 history rows:

1. Build target features from strictly earlier draws only.
2. Generate the same deterministic 64 fair training negatives used by v2.9.
3. Add actual-minus-negative quadratic differences to the pairwise ridge sufficient statistics.
4. Reveal/append that completed target to history.

After the latest completed draw has been processed, all completed solved targets are valid training evidence for the next unknown draw.

## Validation stays separate

If `data/cache/v29_quadratic_reverse_ranking_screen.json` exists, its `outer_summary` is loaded into output metadata.

It is explicitly marked:

```text
role = diagnostic_only_not_used_in_ranking
ranking_influenced = false
```

This preserves the distinction between:

- what the model calculates now; and
- how well that model ranked historical future draws.

The raw score is not an actual lottery winning probability.

## Commands

First verify the branch and regression suite:

```powershell
git pull --ff-only
python -m unittest discover -s tests -v
```

Then do a sampled wiring check. This is not the final recommendation:

```powershell
python scripts\run_v29_raw_recommend.py --sampled --candidate-count 20000 --top-k 10 --progress-every 5000 --output-json data\cache\v29_raw_personal_sample.json
```

After the sampled path is confirmed, run the exhaustive personal ranking over all 8,145,060 combinations:

```powershell
python scripts\run_v29_raw_recommend.py --top-k 10 --progress-every 1000000 --output-json data\cache\v29_raw_personal.json
```

The exhaustive run may be slow because it evaluates the exact frozen v2.9 feature equation for every valid 6-of-45 combination. Performance optimization must preserve bitwise-equivalent ranking semantics and should be treated separately from model changes.
