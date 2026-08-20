# v2.7 / v2.7.1 Release-Candidate Status

Status: historical release candidate superseded; current branch is research-only.

## Why release promotion is blocked

The original v2.7 static release freeze was invalidated by the completed Phase 5A audit: none of the legacy static Normal/Outlier or Mixed components survived the predeclared strict walk-forward screen.

The subsequent v2.7.1 attempt connected Bayesian number/pair evidence to the recommendation path, but both validation layers were also negative:

- Brier reliability versus the fair marginal null was non-positive in all overall/recent windows for both number and pair models, producing zero production-facing reliability;
- direct actual-winning-combination rank versus fair random valid combinations produced no screening survivor for number, pair, or the frozen 50:50 fusion.

Therefore the current recommendation output remains neutral when reliability is zero. No v2.7 or v2.7.1 score should be promoted as a validated predictive release.

## Preserved decisions

- all `8,145,060` valid combinations remain eligible;
- no hard structural/aesthetic filter;
- no pattern-type ranking branch;
- transition remains disabled;
- momentum remains disabled;
- failed static/Bayesian components are not rescued by post-hoc weight tuning;
- pattern metadata is descriptive only.

## Current research direction: v2.8

The next model class implements nested reverse learning.

For each historical target, candidate features are constructed from older draws only. Once that target has already been scored, the known winning answer may be used as one solved ranking problem for later targets. The current target answer is never allowed to fit its own coefficients.

The frozen v2.8 screen is documented in `docs/v28_reverse_ranking_spec.md` and executed by:

```powershell
python scripts\run_v28_reverse_ranking.py --baseline-samples 500 --bootstrap-reps 2000 --progress-every 50 --output-json data\cache\v28_reverse_ranking_screen.json
```

The production recommendation path is not modified by the v2.8 research module. Promotion requires the predeclared screen and a separate 2,000-fair-candidate confirmation with the exact same features, ridge lambda, horizons, and target-isolation rules.

## Historical evidence records

- `docs/v27_release_validation_result.md` — historical v2.7 static release run, explicitly superseded;
- `docs/v27_phase5_static_component_result.md` — Phase 5A rejection;
- `docs/v271_brier_reliability_result.md` — zero Brier reliability result;
- `docs/v271_ranking_evidence_result.md` — direct combination-ranking rejection;
- `docs/v28_reverse_ranking_spec.md` — current research specification.

PR #1 should remain draft until a genuinely validated ranking model exists.
