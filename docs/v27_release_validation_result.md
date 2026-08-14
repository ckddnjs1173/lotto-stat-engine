# v2.7 Release Validation Result

Status: release validation complete; production candidate passed the final gate.

## Validation identity

- validated code commit: `d2399347b6ca7a18f9af8eed38029dbe48bce30f`
- branch: `dev/v2.7-validation-rebuild`
- data rows: `1235`
- latest reflected draw: `1235`
- target draw: `1236`
- production selection: `pure_static_score_top_k`
- dynamic transition: disabled
- dynamic momentum: disabled
- hard filters: none

## Full regression

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

## Exhaustive production end-to-end

Command:

```powershell
python scripts\run_recommend.py --top-k 10 --output-json data\cache\v27_release_1236.json
```

Result:

- evaluation mode: `exhaustive_all_8,145,060`
- evaluated combinations: `8,145,060`
- JSON output created successfully
- target draw: `1236`

### TOP 10

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

The fact that all final TOP-10 combinations are `normal` is not a quota or filter outcome. v2.7 deliberately uses global score order with no pattern-type allocation, so this is the observed result of the frozen score on the full candidate universe.

## Release decision

The v2.7 validation/rebuild program is complete for this release scope.

- Do not run additional Phase-5 research as a release blocker.
- Do not retune transition, momentum, calibration, or static component weights for this release.
- Keep research modules and historical diagnostics for reproducibility, but production entry points use the v2.7 static release path.
- Further statistical research belongs to a later research version and must not delay this release.
- Next work is deployment/site integration and performance/operational improvements, not another prediction-model validation cycle.
