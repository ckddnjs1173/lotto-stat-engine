# v2.7.1 Unified Evidence Integration Specification

Status: implemented and tested; evidence family rejected for predictive promotion.

## Purpose

This document records the v2.7.1 experiment that connected Bayesian number/pair evidence to the actual recommendation path after the legacy v2.7 static components failed Phase 5A.

The implementation remains in the repository for reproducibility, but the evidence family is now closed because it failed both probability-calibration and direct combination-ranking validation.

## Number model

Fair marginal inclusion probability:

```text
p0 = 6 / 45
```

Frozen model:

```text
full_prior_120
p_i = (hits_i + 120 * p0) / (N + 120)
L_i = log(p_i / p0)
E_number(c) = mean(L_i for i in candidate c)
```

## Pair model

Fair pair inclusion probability:

```text
q0 = 15 / 990 = 1 / 66
```

Frozen model:

```text
full_prior_330
q_ij = (pair_hits_ij + 330 * q0) / (N + 330)
L_ij = log(q_ij / q0)
E_pair(c) = mean(L_ij for the 15 pairs in c)
```

## Original production-facing reliability

The first integration used strict walk-forward Brier skill over overall/recent-300/recent-100 windows. Only positive stable skill could contribute; all-negative skill produced zero reliability instead of reversing the model.

On data through draw 1235, both number and pair skills were negative in all three windows, so both reliabilities became zero. See `docs/v271_brier_reliability_result.md`.

## Direct ranking audit

Because Brier calibration is not identical to candidate ranking, a second frozen audit tested the raw evidence directly against fair random valid 6/45 combinations.

Tracks:

```text
number
pair
equal_family_fusion = 0.5 * number + 0.5 * pair
```

Results on 1,135 strict walk-forward targets with 500 fair alternatives per target:

```text
number
  overall   48.6213
  recent300 48.7750
  recent100 48.6380
  CI95      [-2.9364, 0.2837]

pair
  overall   49.7819
  recent300 51.3820
  recent100 50.9560
  CI95      [-1.7670, 1.5478]

50:50 fusion
  overall   49.3711
  recent300 50.6620
  recent100 50.4060
  CI95      [-2.2251, 1.1569]
```

No track passed the predeclared gate. See `docs/v271_ranking_evidence_result.md`.

## Final decision

The v2.7.1 full-history Bayesian posterior family is neutralized for production and closed as the active research hypothesis.

Do not:

- run 2,000-sample confirmation for a failed screen;
- promote the mildly positive recent pair slice;
- tune the number/pair prior strengths from these outcomes;
- choose decay horizons post hoc;
- retune fusion weights after seeing the result;
- reinterpret negative evidence by simply flipping its sign.

The next research class is materially different: nested reverse-learning ranking, documented in `docs/v28_reverse_ranking_spec.md`.
