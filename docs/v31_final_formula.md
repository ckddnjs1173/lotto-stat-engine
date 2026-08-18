# v3.1 final personal prediction formula

This is the terminal personal-use calculation path. It does not use Normal/Mixed/Outlier allocation, structural quotas, validation multipliers, or quadratic interactions.

## Training sequence

For each solved historical target after the first 100 draws:

1. build the context using only earlier draws;
2. calculate 11 candidate features for the actual draw;
3. create 1,024 deterministic fair 6/45 reference combinations from that past-only context;
4. map every feature to a tie-safe fair-null coordinate in `[-1,1]`;
5. compare the actual draw with 64 fair negative combinations;
6. add the pairwise difference to ridge training only after the target-time features are fixed;
7. advance history with the solved target.

After the latest completed draw is included, the same process fits the next-draw model and builds the next-draw fair-null reference.

## 11 features

Evidence features:

- full-history number log lift
- full-history pair log lift
- recent-20 number excess
- recent-100 number excess
- recent-100 pair excess
- previous-draw overlap

Directional/structural features:

- signed sum center: `sum(numbers) - 138`
- high-minus-low zone count: `count(31..45) - count(1..15)`
- signed odd count: `odd_count - 3`
- number range
- consecutive-pair count

The signed features intentionally preserve low/high direction. No absolute sum deviation is used.

## Fair-null coordinate

For raw feature `x_j` and its pre-target fair reference distribution:

```text
z_j = 2 * F_mid,j(x_j) - 1
```

Every `z_j` is bounded to `[-1,1]`.

## Final score

The final model is additive:

```text
score(c) = w^T z(c)
```

`w` is learned by pairwise ridge from solved historical targets versus fair negative combinations. There are no squared terms and no feature-feature interaction terms.

All 8,145,060 valid 6/45 combinations remain eligible. Ranking is strictly by `score(c)`.

## Bias audit

During exhaustive ranking the engine retains the top 1,000 candidates for diagnostics and reports:

- inclusion rate for every number 1..45;
- one-digit-number presence rate;
- mean/min/max combination sum;
- slot shares for 1..15, 16..30, 31..45.

The audit never changes ranking. It exists only to expose concentration such as one number becoming nearly mandatory.

## Final command

```powershell
python scripts\run_v31_final_recommend.py --top-k 10 --progress-every 1000000 --output-json data\cache\v31_final_personal.json
```

The runner automatically detects the latest completed draw in the local dataset and predicts the next draw.
