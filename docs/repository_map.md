# Repository map

The repository contains several generations of lottery-ranking research. No v3.1 scenario is currently promoted as the final personal model.

## Stable entry points

These commands now require an explicit v3.1 scenario:

```powershell
python main.py --scenario full11
python main.py --scenario clean3

python scripts\run_recommend.py --scenario full11
python scripts\run_recommend.py --scenario clean3
```

`main.py` and `scripts/run_recommend.py` both delegate to `scripts/run_v31_final_recommend.py`. The runner refuses to silently choose a scenario.

## Shared v3.1 calculation core

- `lotto_engine/v31_model_spec.py`
  - frozen FULL11 baseline and CLEAN3 candidate specs
  - priors, windows, ridge lambda, reference count, RNG offsets, tie policy
  - canonical model-spec SHA-256
- `lotto_engine/v31_core.py`
  - history state
  - number/pair evidence context
  - 11 raw features
  - sampled fair-null midrank transform
  - pairwise ridge sufficient statistics and fitting
  - fixed RNG-free combination tie key
  - latest model fitting
  - ridge moment eigenvalue/condition diagnostics
- `lotto_engine/v31_scenario_recommendation.py`
  - explicit FULL11/CLEAN3 scenario ranking
  - full/sampled candidate enumeration
  - tie and distribution diagnostics
  - strict downstream portfolio
- `lotto_engine/strict_portfolio.py`
  - pairwise overlap and per-number exposure constraints
  - no raw score modification and no fallback relaxation
- `lotto_engine/loader.py`
  - strict local history validation
  - integer-only round/number parsing
  - normalized data fingerprint
- `lotto_engine/candidates.py`
  - exhaustive `C(45,6)` iterator and deterministic sampled candidate generation

## v3.1 scenarios

### FULL11 baseline

- `lotto_engine/v31_final_directional_recommendation.py`
- status: `experimental_baseline_not_promoted`

The historical filename is retained because component/joint audits import this module. Its mathematical primitives now route through `v31_core.py`.

### CLEAN3 candidate

- `lotto_engine/v31_clean3_recommendation.py`
- status: `experimental_candidate_requires_joint_and_stability_audit`

CLEAN3 removes `previous_draw_overlap`, `number_range`, and `consecutive_pairs` and solves the reduced ridge system. It is no longer the default/current model.

`lotto_engine/v31_final_portfolio.py` is retained for compatibility with the former CLEAN3-specific execution path. The stable runner no longer uses it as the authoritative current model.

## Current v3.1 diagnostics

### Focused component audit

- `lotto_engine/v31_component_influence_audit.py`
- `scripts/run_v31_component_influence_audit.py`
- `tests/test_v31_component_influence_audit.py`

This reproduces the four-feature focused FULL11 leave-one-out study.

### Joint ablation audit

- `lotto_engine/v31_joint_ablation_audit.py`
- `scripts/run_v31_joint_ablation_audit.py`
- `tests/test_v31_joint_ablation_audit.py`

This compares FULL11 with CLEAN3/CLEAN4 multi-feature removals. It is diagnostic, not a promotion gate by itself.

### Reference stability audit

- `lotto_engine/v31_reference_stability_audit.py`
- `scripts/run_v31_reference_stability_audit.py`
- `tests/test_v31_reference_stability_audit.py`

This holds fitted weights fixed and perturbs/enlarges only the latest fair reference. It measures score correlation, rank correlation, TOP-10/100/1000 Jaccard, exact-score duplication, coordinate movement, and feature saturation. It must not be used to pick whichever reference count happens to score historical winners best.

### Full 11-feature / correlated-group audit

- `lotto_engine/v31_full_feature_audit.py`
- `scripts/run_v31_full_feature_audit.py`
- `tests/test_v31_full_feature_audit.py`

This performs strict walk-forward comparisons of FULL11 against every single-feature removal and the predefined correlated groups:

- long-run number + pair
- all recency
- number marginal family
- pair family
- all structural features

It also records coefficient mean/std, sign fractions, sign flips, recent-100 means, and conditioning diagnostics. It is diagnostic only and does not automatically promote/delete features.

### Exact structural nulls

- `lotto_engine/v31_exact_structural_null.py`
- `tests/test_v31_exact_structural_null.py`

Exact whole-universe fair-null distributions are available for overlap, sum, high-minus-low zones, odd count, range, and consecutive-pair count. These exact coordinates are not yet wired into FULL11/CLEAN3; they are a separately audited representation change.

## Tests and CI

- `.github/workflows/tests.yml` runs `python -m unittest discover -s tests -v` on branch pushes and pull requests.
- Legacy v2.2/v2.4/v2.5 integration classes explicitly skip when local `data/lotto.xlsx` is absent; with the workbook present locally, they still execute normally.
- Latest v3.1 stabilization CI passed after this distinction was added.
- Real-data validation and model audits still require the user's local workbook because it is intentionally not stored in GitHub.

## Historical research retained for reproducibility

The following families remain intentionally versioned:

- legacy structural scoring: `features.py`, `profiles.py`, `scoring.py`, `recommender.py`, `mixed_scoring.py`, `mixed_subtypes.py`, `backtest.py`, `mixed_backtest.py`, `weights.py`
- v2.7 validation/release research: `v27_*`, `v271_*`
- v2.8 linear reverse ranking: `v28_reverse_ranking.py`
- v2.9 quadratic reverse ranking: `v29_*`
- v3.0 null-bounded quadratic research: `v30_*`

The stable v3.1 scenario path no longer depends on private functions from the v27/v28 research modules for its ranking mathematics. Historical audit files can still retain old dependencies when required for reproducibility.

## Current documents

Read these for the current audit state:

1. `README.md`
2. `docs/v31_component_influence_result.md`
3. `docs/v31_final_formula.md` — despite the filename, now records that final status is withdrawn
4. this repository map

Older result documents describe the specific historical model version named in their filename.

## Current workflow

Before any new model promotion:

1. run all unit tests;
2. validate local draw data;
3. run reference stability audit;
4. inspect saturation/tie diagnostics;
5. run full feature/group ablations;
6. inspect ridge coefficient/condition stability;
7. audit number-vs-pair marginal duplication/residualization;
8. only then freeze a scenario for future independent draws.
