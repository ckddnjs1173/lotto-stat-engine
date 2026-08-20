# v3.0 Final Personal Recommendation Usage

This repository is being used as a private calculation tool. There is no frontend or deployment requirement for this path.

## Final calculation path

The personal next-draw calculation uses the v3.0 fair-null bounded reverse-ranking model:

1. read every completed draw from the local lotto data file;
2. reproduce the strict historical reverse-learning sequence from draw 101 onward;
3. fit the final 65-term bounded quadratic ridge model using all completed solved targets;
4. build the fair-null reference for the next unknown draw;
5. score either a sample or all 8,145,060 valid 6-of-45 combinations;
6. rank strictly by the model's calculated score;
7. save TOP-K and score explanations to JSON.

Historical validation is metadata only. It is never multiplied into the next-draw score and never collapses rankings to a neutral 50.

Pattern type, odd/even balance, section distribution, and other structural labels are output metadata only. They do not filter or rerank candidates.

## One-time verification after pulling code

```powershell
git pull --ff-only
python -m unittest discover -s tests -v
```

## Quick wiring check

```powershell
python scripts\run_v30_personal_recommend.py --sampled --candidate-count 20000 --top-k 10 --progress-every 5000 --output-json data\cache\v30_personal_sample.json
```

## Final exhaustive next-draw calculation

```powershell
python scripts\run_v30_personal_recommend.py --top-k 10 --progress-every 1000000 --output-json data\cache\v30_personal.json
```

Expected exhaustive count:

```text
8,145,060
```

After the local Excel/history data is updated with a newly completed draw, rerun the same final exhaustive command. The engine detects the latest completed draw and targets the next draw automatically.

## What is intentionally not part of the final workflow

- no frontend;
- no web deployment;
- no production-release gate;
- no repeated 1,035-target historical backtest on every recommendation run;
- no Normal/Mixed/Outlier quota;
- no odd/even or section balancing rule;
- no score neutralization from failed validation.

The stored v3.0 historical validation remains useful for interpreting the model, but the personal recommendation command exists to expose the model calculation itself.
