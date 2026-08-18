# Lotto Stat Engine v2.7.1

Lotto 6/45 statistical ranking research engine. `score` / `prediction_score` is an
internal ranking score, **not** an actual winning probability.

## Current research-candidate policy

- Every valid 6/45 combination remains eligible: `C(45, 6) = 8,145,060`.
- There are no hard number, pattern, aesthetic, or portfolio filters.
- Normal / Mixed / Outlier is metadata only and does not select a scoring branch.
- All candidates are ranked by one common number + pair Bayesian evidence equation.
- Number and pair influence is recomputed from strict walk-forward Brier skill on the
  same local `data/lotto.xlsx` used for the recommendation.
- Transition and momentum remain disabled after the v2.7 validation gates.
- The former v2.7 static structure components remain in the repository for audit and
  reproducibility, but Phase 5A found no screening survivor and they no longer drive
  the recommendation path.

## Why v2.7.1 exists

The v2.7 research code and the v2.7 recommendation entry point had become separated.
`number_evidence.py`, `pair_evidence.py`, and the Phase 5 validation path could produce
OOS evidence, but `scripts/run_recommend.py` still ranked combinations with the frozen
legacy static scorer. v2.7.1 adds the missing evidence -> recommendation bridge.

This does **not** claim that a predictive lottery edge has been proven. It ensures that
whatever number/pair evidence the engine computes is reflected in ranking only in
proportion to its own OOS reliability.

## Unified evidence equation

### Number posterior

For number `i`, the production research spec uses the full completed history with a
fair 6/45 prior and prior strength 120:

```text
p_i = (hits_i + 120 * (6/45)) / (draws + 120)
number_log_lift_i = log(p_i / (6/45))
```

Candidate number evidence is the mean log-lift of its six numbers.

### Pair posterior

For unordered pair `(i,j)`, the fair-null inclusion probability is `1/66`. The fixed
production research spec uses prior strength 330:

```text
q_ij = (pair_hits_ij + 330 * (1/66)) / (draws + 330)
pair_log_lift_ij = log(q_ij / (1/66))
```

Candidate pair evidence is the mean log-lift of its 15 unordered pairs.

### Reliability

Before scoring the next draw, the same fixed number and pair specs are tested by
strict rolling-origin / walk-forward validation against the fair uniform null.
For each evidence class the engine reads Brier skill for:

- all walk-forward targets;
- recent 300 targets;
- recent 100 targets.

The continuous reliability is:

```text
positive_mean = mean(max(skill_window, 0))
positive_fraction = fraction of {overall, recent300, recent100} with skill > 0
reliability = positive_mean * positive_fraction
```

Negative skill is never inverted into a predictive signal. A weak or unstable model
therefore contributes little or nothing rather than being treated as a full-strength
predictor.

### Candidate ranking

```text
E(c)
  = number_reliability * mean_number_log_lift(c)
  + pair_reliability   * mean_pair_log_lift(c)
```

`E(c)` is the exact ranking key. The displayed score is only a monotone bounded view:

```text
prediction_score = 50 + 50 * tanh(E(c))
```

If evidence is weak, scores remain close to 50. Pattern type, odd/even balance, sum,
sections, and other aesthetic structure do not modify this ranking.

Exact evidence ties use a deterministic seed-based structure-neutral hash tie-break;
this prevents lexicographic number order from becoming an accidental preference.

## Workflow

1. Update `data/lotto.xlsx` with the newest completed draw.
2. Run the recommendation command.
3. The engine detects the latest draw automatically.
4. It reruns number/pair walk-forward reliability on the current data.
5. It fits the next-draw Bayesian posteriors using all completed draws.
6. It evaluates all 8,145,060 valid combinations by default with one common equation.
7. It returns TOP-10 for `latest_draw + 1`.

```powershell
python scripts\run_recommend.py
```

Site/API JSON:

```powershell
python scripts\run_recommend.py --output-json data\cache\v271_recommendations.json
```

Development smoke:

```powershell
python scripts\run_recommend.py --sampled --candidate-count 2000 --top-k 10 --output-json data\cache\v271_evidence_smoke.json
```

## Output diagnostics

The public payload includes:

- model/research status and reflected/target draws;
- number and pair model specifications;
- number and pair Brier skill for overall/recent300/recent100;
- derived number and pair reliability;
- each candidate's raw number/pair log-lift and weighted contributions;
- pattern type as metadata only.

## Validation history

- Phase 2: type-wise calibration rejected; equalizing score scales did not improve OOS
  ranking.
- Phase 3A: no confirmed serial dependence, recent shift, change point, or first-order
  pattern transition.
- Phase 4: transition rejected; momentum showed weak long-run displacement but failed
  the predeclared recent-100 confirmation gate.
- Phase 5A: no legacy static component passed the same-type walk-forward screening
  gate. See `docs/v27_phase5_static_component_result.md`.

The correct interpretation remains conservative: this engine can rank statistical
evidence, but current validation does not establish an increase in the mathematical
jackpot probability of an individual Lotto 6/45 combination.

## Verification after pulling changes

```powershell
git pull --ff-only
python -m unittest discover -s tests -v
python scripts\run_recommend.py --sampled --candidate-count 2000 --top-k 10 --output-json data\cache\v271_evidence_smoke.json
```

Inspect the printed number/pair Brier skills and reliabilities before running the full
8,145,060-combination evaluation.
