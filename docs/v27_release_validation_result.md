# v2.7 Release Validation Result

> Historical record only. This validation was completed before Phase 5A. The former
> statement that the static production candidate had passed its final gate is
> superseded by the later Phase 5A result and v2.7.1 evidence integration work.
> Do not use the TOP-10 below as the current recommendation model.

## Historical validation identity

- validated code commit: `d2399347b6ca7a18f9af8eed38029dbe48bce30f`
- branch: `dev/v2.7-validation-rebuild`
- data rows: `1235`
- latest reflected draw: `1235`
- target draw: `1236`
- historical production selection: `pure_static_score_top_k`
- dynamic transition: disabled
- dynamic momentum: disabled
- hard filters: none

## Historical full regression

Command:

```powershell
python -m unittest discover -s tests -v
```

Result:

- tests: `72`
- failures: `0`
- errors: `0`
- runtime: `2.965s`
- status: `OK`

## Historical exhaustive end-to-end

Command:

```powershell
python scripts\run_recommend.py --top-k 10 --output-json data\cache\v27_release_1236.json
```

Result:

- evaluation mode: `exhaustive_all_8,145,060`
- evaluated combinations: `8,145,060`
- JSON output created successfully
- target draw: `1236`

### Historical TOP 10

| Rank | Numbers | Score | Pattern type |
|---:|---|---:|---|
| 1 | 5 7 12 14 27 38 | 83.8372 | normal |
| 2 | 4 6 14 21 23 37 | 83.8372 | normal |
| 3 | 3 10 13 15 28 36 | 83.8325 | normal |
| 4 | 5 7 14 16 25 38 | 83.8240 | normal |
| 5 | 2 4 11 25 28 35 | 83.8145 | normal |
| 6 | 2 4 15 17 30 35 | 83.7999 | normal |
| 7 | 4 10 13 15 29 34 | 83.7973 | normal |
| 8 | 3 8 13 15 30 36 | 83.7836 | normal |
| 9 | 5 7 12 14 25 38 | 83.7781 | normal |
| 10 | 5 7 12 14 27 40 | 83.7762 | normal |

At the time, all final TOP-10 combinations being `normal` was an observed result of
the frozen cross-type raw score, not a quota or hard filter.

## Why this result is superseded

Phase 5A later performed the strict same-pattern-type fair-baseline audit that had
previously been deferred. It found no surviving static component and also showed the
current branch-base benchmark itself had no validated same-type ranking advantage.
See `docs/v27_phase5_static_component_result.md`.

The former production path also did not consume the Bayesian number/pair evidence
modules. v2.7.1 therefore replaced the active research direction with a unified
number/pair evidence path and explicit reliability control.

The first v2.7.1 Brier reliability run through draw 1235 produced zero reliability
for both fixed number and pair models. See `docs/v271_brier_reliability_result.md`.

Because Brier probability calibration and six-number combination ranking are not the
same target, the next frozen audit directly measures actual-winning-combination rank
versus fair alternative combinations:

```powershell
python scripts\run_v271_ranking_evidence_audit.py --baseline-samples 500 --bootstrap-reps 2000 --progress-every 50 --output-json data\cache\v271_ranking_screen.json
```

Until that ranking audit and any required confirmation are reviewed, there is no
accepted final production predictor in this branch.
