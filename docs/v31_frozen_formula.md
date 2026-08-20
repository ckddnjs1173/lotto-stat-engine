# v3.1 Frozen Personal Model

## Status

The frozen personal-use model is `v31_frozen7_exact4096_neg256_v1`.

It was selected after the completed v3.1 diagnostic suite and the final pre-freeze nested candidate audit on draws 1..1237. Historical validation is strict target-isolated walk-forward, but feature/model decisions were made after reviewing the same historical record. Therefore this is a **post-selection exploratory personal model**, not evidence of a proven lottery edge.

## Candidate universe

Every valid Lotto 6/45 combination is eligible:

```text
C(45,6) = 8,145,060
```

There are no hard filters for odd/even balance, sum, range, consecutive numbers, previous-draw overlap, section balance, or visual normality.

## Frozen active features

Exactly seven coordinates are active:

1. `number_full_log_lift`
2. `number_recent20_excess`
3. `number_recent100_excess`
4. `previous_draw_overlap`
5. `high_minus_low_zone_count`
6. `number_range`
7. `consecutive_pairs`

The following four coordinates remain defined for historical compatibility but have zero fitted weight in the frozen model:

- `pair_full_log_lift`
- `pair_recent100_excess`
- `sum_signed_center_138`
- `odd_count_signed_center_3`

## Raw features

For candidate `c={n1,...,n6}` and history available strictly before the target draw:

```text
number_full_log_lift
  = mean_i log(posterior_number(ni) / (6/45))

number_recent20_excess
  = mean_i recent20_rate(ni) - 6/45

number_recent100_excess
  = mean_i recent100_rate(ni) - 6/45

previous_draw_overlap
  = |c intersect previous_draw|

high_minus_low_zone_count
  = count(31..45) - count(1..15)

number_range
  = max(c) - min(c)

consecutive_pairs
  = count of adjacent sorted gaps equal to 1
```

The long-run number posterior uses prior strength 120. The 20-draw and 100-draw windows are frozen.

## Fair-null coordinates

### Evidence coordinates

Non-structural evidence coordinates use a deterministic nested fair-reference stream with exactly 4096 unique 6/45 candidates per historical target.

For raw feature value `x_j` and the corresponding sorted reference column:

```text
z_j = 2 * F_mid,j(x_j) - 1
```

where `F_mid` is the empirical midrank CDF. The stream is count-independent and frozen at stream delta 0.

### Structural coordinates

The structural coordinates do **not** use the sampled reference. They use exact whole-universe 6/45 distributions over all 8,145,060 combinations.

For a discrete structural value `x` with exact count table `N(v)`:

```text
F_mid(x)
  = (sum_{v<x} N(v) + 0.5*N(x)) / 8,145,060

z = 2*F_mid(x) - 1
```

Exact distributions are used for:

- previous-draw overlap
- sum
- high-minus-low zone count
- odd count
- range
- consecutive pairs

Only the active structural coordinates contribute to the frozen score.

## Training objective

Historical training begins after the first 100 completed draws.

For each solved historical target `t`:

1. build context using only draws `< t`;
2. build the target's 4096 nested evidence reference;
3. transform the actual target combination to `z(actual_t)`;
4. draw 256 unique fair negatives from the frozen count-independent nested training stream;
5. transform each negative to `z(negative)`;
6. accumulate pairwise differences

```text
d = z(actual_t) - z(negative)
```

The negative stream is frozen at stream delta 0. Larger tested counts preserved smaller-count prefixes; 256 was selected as the numerical-stability checkpoint, not by maximizing historical winner performance.

## Ridge fit

For active coordinates `A`:

```text
S = E[d_A d_A^T]
m = E[d_A]
rms_i = sqrt(S_ii + 1e-12)

S_scaled = S / (rms rms^T)
m_scaled = m / rms

w_scaled = (S_scaled + 2 I)^(-1) m_scaled
w = w_scaled / rms
```

Frozen ridge lambda:

```text
lambda = 2.0
```

There is no intercept and no interaction term.

## Final score

For candidate `c`:

```text
score(c) = sum_{j in A} w_j * z_j(c)
```

The exhaustive raw ranking sorts all 8,145,060 combinations by:

```text
1. model_score descending
2. fixed lexicographic combination order for exact score ties
```

No RNG is used for exhaustive tie ordering.

## Frozen identity

The immutable specification fixes:

- seven active features
- exact structural null policy
- 4096 nested evidence-reference samples
- reference stream delta 0
- 256 nested training negatives per historical target
- negative stream delta 0
- number prior strength 120
- pair prior strength 330 (historical context compatibility; pair features inactive)
- recent windows 20 / 100
- ridge lambda 2.0
- history start index 100
- fixed RNG offsets
- fixed lexicographic tie policy

Each recommendation artifact records the model-spec SHA-256 and normalized data SHA-256.

## Audit decision basis

Final pre-freeze historical scenario means were:

```text
exact_full11                48.3853
exact_no_pair9              48.8176
exact_no_pair_sum8          49.0060
exact_no_pair_sum_odd7      49.4031
```

Incremental removal of odd count after pair and sum removal produced:

```text
mean delta      +0.3971
recent300       +0.1600
recent100       +0.0440
bootstrap 95%   [0.1603, 0.6504]
```

The 7-feature candidate was therefore selected.

Training-negative convergence for the 7-feature finalist improved from 64->128 to 128->256. For 128->256:

```text
weight correlation      0.998645
relative weight delta   0.049765
sign disagreements      0
rank Spearman           0.998192
TOP10 Jaccard           0.818182
TOP100 Jaccard          0.801802
```

256 is frozen as the highest predeclared tested numerical-stability checkpoint.

## Stable commands

Exhaustive raw ranking:

```powershell
python main.py `
  --top-k 10 `
  --progress-every 1000000 `
  --output-json data\cache\v31_frozen_1238.json
```

Equivalent stable alias:

```powershell
python scripts\run_recommend.py `
  --top-k 10 `
  --progress-every 1000000 `
  --output-json data\cache\v31_frozen_1238.json
```

A sampled smoke test is available with `--sampled --candidate-count 20000`, but the final raw prediction uses exhaustive evaluation.

## Portfolio

The strict portfolio is downstream only. It never changes raw scores or the raw ranking.

```text
pairwise shared numbers <= 2
individual number exposure <= 40%
no fallback relaxation
```
