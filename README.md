# Lotto Stat Engine

개인용 Lotto 6/45 통계 랭킹 연구 엔진입니다.

`model_score`는 실제 당첨확률이 아니라 과거 데이터에서 학습한 계산식이 후보 조합을 정렬하기 위한 내부 점수입니다. 과거 검증은 strict target-isolated walk-forward지만, feature/model 선택 자체는 같은 역사 데이터를 반복 검토한 뒤 이뤄졌기 때문에 **post-selection exploratory validation**으로 해석합니다. 검증된 당첨 우위를 주장하지 않습니다.

## 현재 frozen 모델

현재 stable personal model:

```text
v31_frozen7_exact4096_neg256_v1
```

최종 pre-freeze audit에서 `exact_no_pair_sum_odd7`이 nested 후보 중 가장 일관되게 우세했고, training-negative 수는 winner performance가 아니라 numerical convergence 기준으로 256을 선택했습니다.

Frozen identity:

```text
active features = 7
structural null = exact whole-universe midrank
non-structural evidence reference = deterministic nested 4,096
training negatives per historical target = deterministic nested 256
ridge lambda = 2.0
history start index = 100
reference stream delta = 0
negative stream delta = 0
exhaustive tie policy = fixed lexicographic, RNG-free
```

정식 계산식은 `docs/v31_frozen_formula.md`를 기준으로 합니다.

## Frozen 7 features

사용:

- `number_full_log_lift`
- `number_recent20_excess`
- `number_recent100_excess`
- `previous_draw_overlap`
- `high_minus_low_zone_count`
- `number_range`
- `consecutive_pairs`

비활성(역사 호환용 feature 정의는 유지, fitted weight=0):

- `pair_full_log_lift`
- `pair_recent100_excess`
- `sum_signed_center_138`
- `odd_count_signed_center_3`

## Stable 실행

### 최종 전수 RAW ranking

`--sampled`를 붙이지 않으면 모든 조합을 평가합니다.

```powershell
python main.py `
  --top-k 10 `
  --progress-every 1000000 `
  --output-json data\cache\v31_frozen_1238.json
```

동일한 alias:

```powershell
python scripts\run_recommend.py `
  --top-k 10 `
  --progress-every 1000000 `
  --output-json data\cache\v31_frozen_1238.json
```

직접 versioned runner를 실행해도 동일합니다.

```powershell
python scripts\run_v31_frozen_recommend.py `
  --top-k 10 `
  --progress-every 1000000 `
  --output-json data\cache\v31_frozen_1238.json
```

### 빠른 smoke test

```powershell
python main.py --sampled --candidate-count 20000 --top-k 10
```

샘플 실행은 wiring 확인용입니다. 최종 RAW prediction은 exhaustive `C(45,6)=8,145,060` ranking을 사용합니다.

## 핵심 계산 흐름

```text
past-only history state
→ raw candidate features
→ evidence coordinates: 4,096 nested empirical midrank
→ structural coordinates: exact 8,145,060-universe midrank
→ pairwise actual-minus-fair ridge training
→ frozen 7-feature score(c)=w^T z(c)
→ all 8,145,060 candidates RAW ranking
→ optional strict portfolio downstream
```

## 후보 정책

모든 유효한 6/45 조합이 RAW ranking에 참가합니다.

- hard filter 없음
- all-odd/all-even 허용
- 극단 합계 허용
- 좁거나 넓은 range 허용
- 연속수 많은 조합 허용
- 이전 회차 번호가 많이 겹치는 조합도 허용
- 구조가 보기 좋거나 이상하다는 이유로 보정하지 않음

구조 특성은 오직 frozen 계산식의 learned score를 통해서만 영향을 줍니다.

## Fair-null transform

Non-structural evidence coordinates:

```text
z_j = 2 * F_mid,j(x_j) - 1
```

`F_mid`는 target별 deterministic nested fair-reference 4,096개의 empirical midrank입니다.

Structural coordinates는 Monte Carlo reference를 쓰지 않고 전체 8,145,060 조합의 exact discrete distribution을 사용합니다.

```text
F_mid(x)
  = (count(v < x) + 0.5 * count(v = x)) / 8,145,060

z = 2*F_mid(x) - 1
```

Exact structural distributions:

- previous-draw overlap
- sum
- high-minus-low zone count
- odd count
- number range
- consecutive pairs

Frozen7에서는 이 중 `previous_draw_overlap`, `high_minus_low_zone_count`, `number_range`, `consecutive_pairs`만 active입니다.

## Training

각 역사 target은 반드시 target 이전 history만 사용합니다.

```text
actual vector z+
256 nested fair negative vectors z-
d = z+ - z-
```

누적 통계:

```text
S = E[d d^T]
m = E[d]
```

RMS scaling 후:

```text
w_scaled = (S_scaled + 2I)^(-1) m_scaled
w = w_scaled / rms
```

최종 후보 score:

```text
score(c) = sum_{j in active7} w_j z_j(c)
```

intercept와 interaction term은 없습니다.

## Final pre-freeze audit 결과

1..1237회를 사용한 최종 nested comparison:

```text
exact_full11                mean 48.3853  recent300 49.5247  recent100 50.4260
exact_no_pair9              mean 48.8176  recent300 49.9900  recent100 50.6080
exact_no_pair_sum8          mean 49.0060  recent300 50.1160  recent100 50.7700
exact_no_pair_sum_odd7      mean 49.4031  recent300 50.2760  recent100 50.8140
```

`exact_no_pair_sum_odd7 - exact_no_pair_sum8`:

```text
mean delta     +0.3971
recent300      +0.1600
recent100      +0.0440
bootstrap 95%  [0.1603, 0.6504]
```

Training-negative convergence for frozen7, `128 -> 256`:

```text
weight correlation      0.998645
relative weight delta   0.049765
sign disagreements      0
rank Spearman           0.998192
TOP10 Jaccard           0.818182
TOP100 Jaccard          0.801802
```

256은 predeclared audit에서 테스트한 가장 높은 numerical-stability checkpoint로 고정합니다.

## 데이터 무결성

```powershell
python scripts\validate_data.py
```

Loader는 다음을 hard validation합니다.

- 1회부터 최신까지 연속 회차
- 중복 회차 없음
- 유한한 정수 회차/번호
- fractional value truncation 금지
- 회차별 6개 번호 중복 없음
- 번호 범위 1..45

모든 recommendation artifact는 normalized data SHA-256을 기록합니다.

## 재현성

Frozen model spec은 다음을 SHA-256 identity에 포함합니다.

- active feature set
- prior/window constants
- reference count/policy/stream
- training negative count/policy/stream
- exact structural-null policy
- ridge lambda
- RNG offsets
- tie policy

Stable production stream이 pre-freeze audit에서 사용한 nested stream과 동일한지는 unit test로 고정합니다.

```powershell
python -m unittest discover -s tests -v
```

## Strict portfolio

Portfolio는 RAW ranking 이후 별도 단계입니다.

```text
pairwise shared numbers <= 2
individual number exposure <= 40%
no fallback relaxation
```

- RAW score 수정 없음
- RAW ranking 수정 없음
- source pool은 상위 50,000개 retained entries
- 조건을 만족하는 K장을 못 채우면 `complete=false`

## Historical experimental runners

FULL11/CLEAN3와 기존 v2.x/v3.0 runner는 재현성을 위해 남겨둡니다. Stable entrypoint는 더 이상 이 실험 시나리오를 자동 선택하지 않습니다.

필요하면 직접 역사 runner를 호출할 수 있습니다.

```powershell
python scripts\run_v31_final_recommend.py --scenario full11 --sampled --candidate-count 20000
python scripts\run_v31_final_recommend.py --scenario clean3 --sampled --candidate-count 20000
```

이 경로는 historical/experimental comparison용이며 현재 frozen personal model이 아닙니다.

## Audit artifacts

주요 v3.1 진단:

- `v31_reference_stability_audit.py`
- `v31_retrained_reference_audit.py`
- `v31_component_influence_audit.py`
- `v31_joint_ablation_audit.py`
- `v31_full_feature_audit.py`
- `v31_pair_residual_audit.py`
- `v31_exact_null_audit.py`
- `v31_dependency_audit.py`
- `v31_freeze_candidate_audit.py`

Complete audit suite와 final pre-freeze audit은 자동 승격을 하지 않도록 설계되어 있으며, frozen7 결정은 결과 검토 후 명시적으로 코드에 고정했습니다.
