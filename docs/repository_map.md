# Repository map

The repository contains several generations of lottery-ranking research. No v3.1 scenario is currently promoted as the final personal model.

## Stable entry points

These commands require an explicit v3.1 scenario:

```powershell
python main.py --scenario full11
python main.py --scenario clean3

python scripts/run_recommend.py --scenario full11
python scripts/run_recommend.py --scenario clean3
```

`main.py` and `scripts/run_recommend.py` delegate to `scripts/run_v31_final_recommend.py`. The runner refuses to silently choose a model and records model-spec SHA-256, data SHA-256, runtime identity, and deterministic tie policy.

## Shared v3.1 calculation core

- `lotto_engine/v31_model_spec.py`
  - frozen FULL11 baseline and CLEAN3 candidate specs
  - priors, windows, ridge lambda, reference/negative counts, RNG offsets, tie policy
  - canonical model-spec SHA-256
- `lotto_engine/v31_core.py`
  - history state
  - number/pair evidence context
  - 11 raw features
  - sampled fair-null midrank transform
  - pairwise ridge sufficient statistics and fitting
  - fixed RNG-free combination tie key
  - latest model fitting
- `lotto_engine/v31_scenario_recommendation.py`
  - explicit FULL11/CLEAN3 scenario ranking
  - full/sampled candidate enumeration
  - tie and ranking-distribution diagnostics
  - strict downstream portfolio
- `lotto_engine/strict_portfolio.py`
  - pairwise overlap and per-number exposure constraints
  - no raw score modification and no fallback relaxation
- `lotto_engine/loader.py`
  - strict local history validation
  - finite integer-only round/number parsing
  - normalized data fingerprint
- `lotto_engine/candidates.py`
  - exhaustive `C(45,6)` iterator and deterministic sampled candidate generation
- `lotto_engine/v31_audit_utils.py`
  - common tie-safe percentile/bootstrap/rank comparison helpers
  - weight/ridge diagnostics
  - runtime identity
  - JSON-safe deterministic audit output

## v3.1 scenarios

### FULL11 baseline

- `lotto_engine/v31_final_directional_recommendation.py`
- status: `experimental_baseline_not_promoted`

The historical filename remains because older component/joint audit files import it. Its ranking mathematics routes through the shared v3.1 core.

### CLEAN3 candidate

- `lotto_engine/v31_clean3_recommendation.py`
- status: `experimental_candidate_requires_joint_and_stability_audit`

CLEAN3 removes `previous_draw_overlap`, `number_range`, and `consecutive_pairs` and solves the reduced ridge system. It is not the default/current model.

`lotto_engine/v31_final_portfolio.py` is retained for compatibility with the former CLEAN3-specific execution path. The stable runner no longer treats it as authoritative.

## Complete v3.1 audit layer

### Fixed-weight reference stability

- `lotto_engine/v31_reference_stability_audit.py`
- `scripts/run_v31_reference_stability_audit.py`
- `tests/test_v31_reference_stability_audit.py`

Holds fitted weights fixed while latest fair-reference seed/count changes. Reports score/rank correlation, TOP-10/100/1000 Jaccard, exact-score duplicates, coordinate movement, and `z=±1` saturation.

### Retrained reference stability

- `lotto_engine/v31_retrained_reference_audit.py`
- `scripts/run_v31_retrained_reference_audit.py`
- `tests/test_v31_retrained_reference_audit.py`

Reruns the complete training history under nested deterministic reference streams/counts and compares final coefficients plus latest ranking against the current reference policy. This is numerical stability analysis, not a search for the reference count with the best winner score.

### Focused component audit

- `lotto_engine/v31_component_influence_audit.py`
- `scripts/run_v31_component_influence_audit.py`
- `tests/test_v31_component_influence_audit.py`

Preserves the original four-feature FULL11 leave-one-out study.

### CLEAN3 / CLEAN4 joint audit

- `lotto_engine/v31_joint_ablation_audit.py`
- `scripts/run_v31_joint_ablation_audit.py`
- `tests/test_v31_joint_ablation_audit.py`

Compares FULL11 with multi-feature removals. It is diagnostic and cannot promote a model by itself.

### Full 11-feature / correlated-group audit

- `lotto_engine/v31_full_feature_audit.py`
- `scripts/run_v31_full_feature_audit.py`
- `tests/test_v31_full_feature_audit.py`

Strict walk-forward FULL11 comparison against every single-feature removal and predefined groups:

- long-run number + pair
- all recency
- number marginal family
- pair family
- all structural features

Also records weight mean/std, sign fractions, sign flips, recent-100 means, and unregularized second-moment conditioning.

### Pair residualization audit

- `lotto_engine/v31_pair_residual_audit.py`
- `scripts/run_v31_pair_residual_audit.py`
- `tests/test_v31_pair_residual_audit.py`

Projects 990 pair-edge values onto the 45-number vertex-main-effect design:

```text
pair_edge(i,j) = a_i + a_j + residual_ij
```

The residual representation removes pair variation explained by individual-number main effects. Original and residualized pair representations use the same target/reference/negative streams and are separately retrained.

### Exact structural null implementation and audit

- `lotto_engine/v31_exact_structural_null.py`
- `tests/test_v31_exact_structural_null.py`
- `lotto_engine/v31_exact_null_audit.py`
- `scripts/run_v31_exact_null_audit.py`
- `tests/test_v31_exact_null_audit.py`

Exact whole-universe null distributions exist for:

- previous-draw overlap
- sum
- high-minus-low zones
- odd count
- number range
- consecutive adjacent-pair count

The exact coordinates are **not** silently wired into FULL11/CLEAN3. The audit separately retrains sampled-structural and exact-structural representations and compares historical/latest ranking behavior.

### Ridge / feature dependency audit

- `lotto_engine/v31_dependency_audit.py`
- `scripts/run_v31_dependency_audit.py`
- `tests/test_v31_dependency_audit.py`

Reports:

- unregularized scaled second-moment condition
- actual `scaled_second + lambda*I` ridge-system condition
- coefficient sign/dispersion stability
- strongest pairwise training-difference second-moment dependencies
- predeclared number/pair, recency, sum/zone, range/consecutive dependencies

The dependency metric is explicitly a second-moment cosine on actual-minus-fair difference rows, not a centered Pearson correlation.

### Complete audit suite

- `scripts/run_v31_audit_suite.py`
- `tests/test_v31_audit_suite.py`

Decision-grade run:

```powershell
python scripts/run_v31_audit_suite.py --mode full --progress-every 50
```

The suite runs, in order:

1. fixed-weight reference stability
2. retrained reference stability
3. focused component audit
4. CLEAN3/CLEAN4 joint audit
5. full 11-feature/group audit
6. pair residualization
7. exact structural-null comparison
8. ridge/feature dependency audit

It writes a checkpoint JSON after every step. If a step fails, the suite records the failed step and traceback and exits nonzero. It never automatically promotes a model.

## Reproducibility / eligibility tests

- `tests/test_v31_reproducibility_and_eligibility.py`
  - unusual but valid combinations remain scorable
  - identical data/spec reproduces identical fitted weights and latest reference
- `tests/test_v31_core.py`
  - model-spec identity
  - explicit scenario choice
  - fixed RNG-free tie behavior
- exact/pair/reference/dependency tests verify their mathematical invariants without requiring the private real-data workbook.

## Tests and CI

- `.github/workflows/tests.yml` runs `python -m unittest discover -s tests -v` on pushes and pull requests.
- v2.2/v2.4/v2.5 legacy integration classes skip only when local `data/lotto.xlsx` is absent; with the workbook present they execute normally.
- real-data validation and heavy audits remain local because the workbook is intentionally not committed.

## Historical research retained for reproducibility

The following families remain intentionally versioned:

- legacy structural scoring: `features.py`, `profiles.py`, `scoring.py`, `recommender.py`, `mixed_scoring.py`, `mixed_subtypes.py`, `backtest.py`, `mixed_backtest.py`, `weights.py`
- v2.7 validation/release research: `v27_*`, `v271_*`
- v2.8 linear reverse ranking: `v28_reverse_ranking.py`
- v2.9 quadratic reverse ranking: `v29_*`
- v3.0 null-bounded quadratic research: `v30_*`

Stable v3.1 scenario ranking no longer depends on private v27/v28 functions for its mathematical primitives. Historical research files may retain old dependencies when required for exact reproduction.

## Current workflow

Before any v3.1 promotion:

1. run all unit tests;
2. validate the local draw workbook;
3. run `run_v31_audit_suite.py --mode full`;
4. inspect reference stability and saturation/ties;
5. inspect all feature/group ablations;
6. inspect pair residualization and exact-null representation results;
7. inspect actual ridge-system and coefficient stability;
8. decide one immutable feature/reference representation;
9. mark only that ModelSpec frozen/promoted;
10. only then run all 8,145,060 combinations for the final RAW ranking;
11. treat future completed draws after the freeze as the new independent validation period.
