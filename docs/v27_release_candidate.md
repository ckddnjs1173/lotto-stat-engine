# v2.7 Release Candidate Production Freeze

Status: research closed for v2.7; production completion in progress.

## Frozen production decisions

The v2.7 validation program is closed for this release candidate. New statistical
components are deferred to a later research version.

Production policy:

- all 8,145,060 valid Lotto 6/45 combinations remain eligible;
- no hard structural filter is applied;
- final selection is pure global TOP-K by the release static score;
- portfolio quotas, subtype allocation, and diversity penalties are not used;
- transition is disabled after Phase 3/4 failed to establish production evidence;
- momentum is disabled because confirmation failed the predeclared recent-100 gate;
- type-wise calibration is disabled because Phase 2 reduced predictive ranking;
- AR/ARIMA, regime, and change-point logic remain out of scope after Phase 3A;
- Phase 5 static-component research is deferred and does not block this release.

## Static score policy

### Normal / outlier

The existing branch score is reused, but the positive legacy `transition_score`
contribution is removed. The remaining positive static weights are renormalized:

- outlier survival: 0.45
- type balance: 0.25
- normal structure: 0.15

Historical-pattern and number-dynamics components retain zero production weight.
No new weight optimization is performed.

### Mixed

The existing Mixed static base is reused with `dynamic_markov_decay` removed from
the scoring profile. Therefore `mixed_slot_score` equals the static Mixed base and
contains no final transition/momentum composition.

## Selection

The production iterator streams either all valid combinations or, for development
smoke only, a deterministic sampled candidate set. A heap retains only the globally
highest `TOP_K_RECOMMENDATIONS` static scores.

No pattern-type allocation or portfolio adjustment changes the final order.

## Public JSON contract

`scripts/run_recommend.py --output-json <path>` writes a compact site-facing payload
containing:

- model version and release status;
- generated time in Asia/Seoul;
- latest reflected and target draw;
- evaluation mode and evaluated combination count;
- selection/candidate policy;
- explicit dynamic-component disabled status;
- rank, six numbers, score, pattern type, score origin, and active components.

The public payload does not label the score as a probability.

## Release verification policy

During implementation, run only the targeted `test_v27_release.py` module and a
small sampled smoke. The full test suite and one exhaustive end-to-end recommendation
run are reserved for the final release boundary.

Recommended pre-release sequence:

```powershell
git pull --ff-only
python -m unittest discover -s tests -p "test_v27_release.py" -v
python scripts\run_recommend.py --sampled --candidate-count 2000 --top-k 10 --output-json data\cache\v27_rc_smoke.json
```

If those pass, perform the single release-boundary full regression and exhaustive
end-to-end run. No additional research backtest is required for v2.7.
