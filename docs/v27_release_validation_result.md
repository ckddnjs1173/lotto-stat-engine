# v2.7 Release Validation Result — Historical / Superseded

> **Current status:** this file preserves the successful execution record of the old
> v2.7 static path at commit `d2399347...`. Its former release conclusion was
> superseded after Phase 5A was actually executed and produced no static-component
> screening survivor. The current recommendation research path is v2.7.1; see
> `docs/v27_phase5_static_component_result.md` and
> `docs/v271_evidence_integration_spec.md`.

At the time of this run, the v2.7 static implementation completed its then-declared
release validation successfully. The execution facts below remain reproducible
historical facts; the statement that the static model passed the final statistical
release gate is no longer current.

## Validation identity

- validated code commit: `d2399347b6ca7a18f9af8eed38029dbe48bce30f`
- branch: `dev/v2.7-validation-rebuild`
- data rows: `1235`
- latest reflected draw: `1235`
- target draw: `1236`
- production selection at that commit: `pure_static_score_top_k`
- dynamic transition: disabled
- dynamic momentum: disabled
- hard filters: none

## Full regression at the historical validation commit

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

## Historical exhaustive end-to-end run

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

All ten being `normal` was not caused by a quota or hard filter. It was the result of
the then-frozen global static score. Phase 5A later showed that the relevant legacy
static components did not pass the predeclared same-type predictive-evidence gate, so
this TOP-10 must not be interpreted as evidence that Normal combinations are more
predictive.

## Superseding decision

The old release conclusion is withdrawn for statistical-model purposes:

- Phase 5A is no longer deferred; it was completed and recorded.
- No legacy static component advances to confirmation.
- The frozen v2.7 static score no longer drives `scripts/run_recommend.py`.
- Transition and momentum remain disabled; those earlier decisions are unchanged.
- v2.7.1 connects number/pair Bayesian evidence to the recommendation path with
  continuous OOS reliability shrinkage and one common scoring equation.
- v2.7.1 remains a research candidate pending local-data smoke verification.
