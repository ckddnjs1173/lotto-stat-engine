# v2.7 / v2.7.1 Research Candidate Status

Status: draft research candidate; former v2.7 static release freeze superseded.

## Why the former release freeze was withdrawn

Phase 5A completed the previously deferred strict walk-forward static-component audit. No tested legacy Normal/Outlier or Mixed static component survived the predeclared same-pattern-type fair-baseline screen.

The former v2.7 recommendation entry point also did not consume the newer Bayesian number/pair evidence modules. Research calculations and recommendation ranking had become disconnected.

Therefore the old `pure_static_score_top_k` release decision is no longer the active research direction.

## v2.7.1 unified evidence path

The current draft research path:

- keeps all 8,145,060 valid combinations eligible;
- applies no hard structural/aesthetic filters;
- uses one common number/pair Bayesian evidence formula for all candidate types;
- treats Normal/Mixed/Outlier only as descriptive metadata after ranking;
- keeps transition and momentum disabled;
- does not fall back to failed legacy static weights when evidence is neutral.

## Brier reliability result through draw 1235

Local verification at head `6abdd4c` passed 78 tests and produced:

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

The Brier-controlled v2.7.1 recommendation score is therefore currently neutral. A TOP-K emitted while both reliabilities are zero is only deterministic tie handling and is not a predictive recommendation.

## Next validation: direct combination ranking

Brier score is a proper probability-calibration metric, but the application target is ranking the actual six-number winning combination above alternative valid combinations. A target-aligned strict walk-forward ranking audit is therefore required before making a final decision on the fixed number/pair posterior family.

Frozen tracks:

```text
number
pair
equal_family_fusion = 0.5 * number + 0.5 * pair
```

Screening:

```text
start index: 100
fair baseline combinations per target: 500
block bootstrap reps: 2000
block size: 20
```

Gate:

```text
overall mean percentile > 50
95% block-bootstrap CI for percentile - 50 entirely > 0
recent300 >= 50
recent100 >= 50
```

A survivor earns only a separate 2,000-combination-per-target confirmation. No screening result changes production reliability automatically.

## Required local sequence

```powershell
git pull --ff-only
python -m unittest discover -s tests -v
python scripts\run_v271_ranking_evidence_audit.py --baseline-samples 500 --bootstrap-reps 2000 --progress-every 50 --output-json data\cache\v271_ranking_screen.json
```

Do not run the full 8,145,060-combination recommendation while both current production-facing reliabilities are zero.

## Anti-overfitting constraints

- no post-hoc prior-strength retuning to rescue the ranking screen;
- no post-hoc half-life search to rescue the ranking screen;
- no type-specific ranking branch;
- no structural fallback score;
- no transition/momentum reopening;
- no confirmation skip for a screening survivor.

The exact equations are documented in `docs/v271_evidence_integration_spec.md`. The Brier result is recorded in `docs/v271_brier_reliability_result.md`, and Phase 5A is recorded in `docs/v27_phase5_static_component_result.md`.
