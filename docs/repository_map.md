# Repository map

The repository contains several generations of research. This file separates the **current personal calculation path** from historical experiments.

## Current personal-use path

These files define what `python main.py` and `python scripts/run_recommend.py` execute:

- `lotto_engine/loader.py` — strict local draw-history loading, validation, and data fingerprint
- `lotto_engine/candidates.py` — deterministic samples and exhaustive 6/45 iterator
- `lotto_engine/v31_clean3_recommendation.py` — current CLEAN3 fair-null additive reverse-ridge model
- `lotto_engine/v31_final_portfolio.py` — strict downstream ticket diversification
- `scripts/run_v31_final_recommend.py` — canonical runner
- `scripts/run_recommend.py` — stable alias to the canonical runner
- `main.py` — stable root alias to the canonical runner
- `streamlit_app.py` — old web calculation disabled; CLI guidance only

If the stable entry points disagree, that is a repository bug.

## Current diagnostics

- `lotto_engine/v31_component_influence_audit.py`
- `scripts/run_v31_component_influence_audit.py`
- `tests/test_v31_component_influence_audit.py`

These reproduce the former full-11 component influence study that led to CLEAN3.

- `lotto_engine/v31_joint_ablation_audit.py`
- `scripts/run_v31_joint_ablation_audit.py`
- `tests/test_v31_joint_ablation_audit.py`

These are diagnostic comparisons of the former full-11 baseline against multi-feature ablations. They are **not** the default recommendation engine.

`lotto_engine/v31_final_directional_recommendation.py` is retained as the full-11 representation/audit baseline used by those diagnostics; despite its historical filename, it is no longer the stable personal-use entry point.

## Historical research retained for reproducibility

The following families remain intentionally versioned and are not imported by the stable recommendation entry points:

- legacy structural scoring: `scoring.py`, `recommender.py`, `mixed_scoring.py`, `mixed_subtypes.py`, `backtest.py`, `mixed_backtest.py`, `weights.py`
- v2.7 validation/release research: `v27_*`, `v271_*`
- v2.8 linear reverse ranking: `v28_reverse_ranking.py`
- v2.9 quadratic reverse ranking: `v29_*`
- v3.0 null-bounded quadratic research: `v30_*`

Their matching runners and tests are kept so historical findings can be reproduced and old assumptions cannot silently disappear from the record.

## Historical result documents

`docs/` contains specifications/results that motivated each transition. Older documents describe the version named in their filename, not the current model.

For current state, read:

1. `README.md`
2. `docs/v31_component_influence_result.md`
3. `docs/v31_final_formula.md`

## What not to do

Do not use an old versioned runner as the normal next-draw command simply because it is still present. In particular, v2.7/v2.8/v2.9/v3.0 scripts are research artifacts.

Use:

```powershell
python main.py
```

or:

```powershell
python scripts\run_recommend.py
```

for the current calculation.
