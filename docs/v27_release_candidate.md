# v2.7 Static Release Candidate — Superseded

Status: **superseded by Phase 5A and v2.7.1 evidence integration**.

This document originally froze the v2.7 static recommendation path and stated that
Phase 5 static-component research could be deferred without blocking release. That
statement is no longer current.

## Why the freeze was withdrawn

Phase 5A was subsequently executed on data through draw 1235 using the predeclared
same-pattern-type fair baseline. No Normal/Outlier or Mixed static component passed
the screening gate, and the current branch-base benchmark itself did not establish a
same-type ranking advantage.

See:

- `docs/v27_phase5_static_component_spec.md`
- `docs/v27_phase5_static_component_result.md`

The original static release path also left newer number/pair Bayesian evidence modules
outside `scripts/run_recommend.py`, creating a research-to-production disconnect.

## Decisions that remain valid

The following v2.7 conclusions are retained:

- all 8,145,060 valid Lotto 6/45 combinations remain eligible;
- no hard structural filter is applied;
- transition remains disabled after the v2.7 validation gate;
- momentum remains disabled after failing the predeclared recent-100 confirmation
  gate;
- type-wise calibration remains rejected as a predictive improvement;
- AR/ARIMA, regime, and change-point logic remain outside the recommendation path
  because their prerequisites were not established.

## Decisions that are superseded

The following former release decisions must not be treated as current production
policy:

- pure global TOP-K by the frozen legacy static score;
- Normal/Outlier ranking by the renormalized static component blend;
- Mixed ranking by the separate static Mixed base;
- Phase 5 being optional for the release decision;
- describing v2.7 as statistically complete.

## Current path: v2.7.1 research candidate

`lotto_engine.v27_release.generate_release_recommendations()` now routes ranking
through the unified Bayesian evidence layer documented in
`docs/v271_evidence_integration_spec.md`.

Current behavior:

1. load the latest local `data/lotto.xlsx`;
2. strict-walk-forward test the fixed number and pair Bayesian models against their
   fair uniform nulls;
3. convert OOS Brier skill into continuous reliability shrinkage;
4. fit the same fixed posteriors on all completed draws;
5. rank every candidate with one common number + pair evidence equation;
6. attach Normal/Mixed/Outlier and other structure fields only after ranking as
   metadata.

v2.7.1 remains a `research_candidate`. It is not a claim that an individual Lotto
combination has a proven mathematical probability advantage.

## Required next verification

After pulling the v2.7.1 integration, run:

```powershell
git pull --ff-only
python -m unittest discover -s tests -v
python scripts\run_recommend.py --sampled --candidate-count 2000 --top-k 10 --output-json data\cache\v271_evidence_smoke.json
```

The smoke output must be reviewed for the actual number/pair Brier skills and derived
reliabilities from the current local data before an exhaustive run is treated as a
release candidate result.
