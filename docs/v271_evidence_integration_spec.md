# v2.7.1 Unified Evidence Integration Specification

## Objective

Repair the missing research-to-recommendation connection without reintroducing the legacy aesthetic/static structure bias exposed by Phase 5A.

The ranking target is a single comparable evidence value for every valid 6-of-45 combination. Pattern type is not allowed to choose a different scoring equation.

## Frozen principles

1. All `C(45,6) = 8,145,060` combinations remain eligible.
2. No hard structural filters.
3. Target draw `t` may use only draws `< t` during walk-forward testing.
4. Transition and momentum remain disabled after v2.7 validation.
5. Phase 5A legacy static components do not drive the new ranking.
6. Reliability is continuous shrinkage, not post-hoc weight tuning.
7. A model worse than the fair null is not automatically reversed into a predictor.
8. Normal/Mixed/Outlier is metadata only and never selects a different ranking formula.

## Fixed number model

The fair marginal inclusion probability is:

```text
p0 = 6 / 45
```

The fixed next-draw research model is `full_prior_120`:

```text
p_i = (hits_i + 120 * p0) / (N + 120)
L_i = log(p_i / p0)
E_number(c) = mean(L_i for i in c)
```

No decay horizon is selected after seeing the target. The full-history specification is consistent with Phase 3A, which did not establish a recent regime/shift.

## Fixed pair model

A six-number draw contains 15 unordered pairs out of 990 possible pairs, so:

```text
q0 = 15 / 990 = 1 / 66
```

The fixed pair model is `full_prior_330`:

```text
q_ij = (pair_hits_ij + 330 * q0) / (N + 330)
L_ij = log(q_ij / q0)
E_pair(c) = mean(L_ij for the 15 unordered pairs in c)
```

## Brier reliability layer

The fixed number and pair models are first evaluated against their fair marginal nulls using strict walk-forward Brier skill for overall, recent-300, and recent-100 windows.

For skills `s_all`, `s_300`, and `s_100`:

```text
positive_mean
  = (max(s_all,0) + max(s_300,0) + max(s_100,0)) / 3

positive_fraction
  = count(skill > 0 among all/300/100) / 3

reliability
  = positive_mean * positive_fraction
```

The production-facing research ranking key is:

```text
E(c) = r_number * E_number(c) + r_pair * E_pair(c)
```

Display only:

```text
score(c) = 50 + 50 * tanh(E(c))
```

Ranking uses the unrounded `E(c)`.

## Observed Brier result through draw 1235

The first local verification after integration produced:

```text
number
  overall   -0.0013900204
  recent300 -0.0009973398
  recent100 -0.0011491355
  reliability 0

pair
  overall   -0.0006724232
  recent300 -0.0004667738
  recent100 -0.0004752381
  reliability 0
```

Therefore the Brier-controlled score is currently neutral. This confirms that the evidence wiring is active and that marginal calibration does not support positive influence for these fixed full-history models.

This is not yet the final answer to the application question, because Brier evaluates marginal probability calibration while the application target is six-number combination ranking.

## Combination-ranking audit

A second, target-aligned audit is frozen before any attempt to replace Brier reliability.

For historical target `t`:

1. use only draws `< t`;
2. construct the fixed number and pair posteriors above;
3. score the actual winning combination;
4. sample deterministic unique fair valid 6/45 combinations from the full candidate universe;
5. compute the tie-safe actual-vs-fair percentile.

Predeclared tracks:

```text
number = E_number(c)
pair = E_pair(c)
equal_family_fusion = 0.5 * E_number(c) + 0.5 * E_pair(c)
```

The fusion is fixed before seeing the audit result. Pattern type is not conditioned on.

Screening budget:

```text
start index: 100
fair combinations per target: 500
circular block-bootstrap reps: 2000
block size: 20
```

A screening candidate must satisfy all of:

```text
overall mean percentile > 50
95% block-bootstrap CI for percentile - 50 entirely > 0
recent300 mean percentile >= 50
recent100 mean percentile >= 50
```

A survivor earns only a separate confirmation with `2,000` fair combinations per target. The screening result does not directly change production reliability.

## Continuous ranking reliability diagnostic

The audit also reports a diagnostic continuous rank reliability from overall/recent300/recent100 percentile excess:

```text
e_i = (percentile_i - 50) / 50
positive_i = max(e_i, 0)
positive_fraction = count(e_i > 0) / 3
r_rank = mean(positive_i) * positive_fraction
```

This remains diagnostic until the ranking audit and any required confirmation are accepted.

## Exact ties

If two production candidates have exactly equal `E(c)`, selection uses a deterministic BLAKE2b hash of `(seed, sorted numbers)` as a structure-neutral tie-break. This exists only to avoid accidental lexicographic preference when evidence is neutral or tied.

## Pattern metadata

Only after TOP-K evidence ranking is determined does the engine attach:

- Normal / Mixed / Outlier;
- sum;
- odd count;
- range;
- min/max gap;
- section distribution.

These fields explain a selected ticket; they do not alter its score.

## Anti-overfitting rules

- no target draw may influence its own posterior;
- no post-hoc prior-strength search to rescue a failed screen;
- no post-hoc half-life search to rescue a failed screen;
- no structural fallback score when evidence reliability is neutral;
- no Normal/Mixed/Outlier-specific ranking branch;
- no dynamic transition or momentum reopening inside this stage;
- no screening survivor may enter production without its declared confirmation.

## Commands

Brier-controlled recommendation smoke:

```powershell
python scripts\run_recommend.py --sampled --candidate-count 2000 --top-k 10 --output-json data\cache\v271_evidence_smoke.json
```

Direct combination-ranking screen:

```powershell
python scripts\run_v271_ranking_evidence_audit.py --baseline-samples 500 --bootstrap-reps 2000 --progress-every 50 --output-json data\cache\v271_ranking_screen.json
```

Only if a track survives the screen should it be rerun with the declared 2,000-combination confirmation baseline.
