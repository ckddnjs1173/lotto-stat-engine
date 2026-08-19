# Lotto Stat Engine

개인용 Lotto 6/45 통계 랭킹 연구 엔진입니다.

`model_score`는 실제 당첨확률이 아니라 과거 데이터에서 학습한 계산식이 후보 조합을 정렬하기 위한 내부 점수입니다. 과거 walk-forward 결과는 재현 가능한 우위를 입증하지 못했으며, 현재 v3.1의 어떤 feature subset도 최종 모델로 승격되어 있지 않습니다.

## 현재 상태

v3.1은 **모델 가족을 감사 중인 단계**입니다.

공통 구조:

```text
raw feature
→ fair-null coordinate z_j
→ pairwise actual-vs-fair ridge
→ score(c)=w^T z(c)
→ 모든 C(45,6)=8,145,060 조합 ranking
```

현재 비교 시나리오:

- `full11`: 기존 11-feature v3.1 baseline. 상태는 `experimental_baseline_not_promoted`.
- `clean3`: `previous_draw_overlap`, `number_range`, `consecutive_pairs`를 동시에 제거하고 재학습한 후보. 상태는 `experimental_candidate_requires_joint_and_stability_audit`.

**CLEAN3는 더 이상 현재/production 모델로 취급하지 않습니다.** 개별 leave-one-out 결과가 모두 불확정인 상태에서 세 feature를 묶어 먼저 승격한 것은 절차상 너무 빨랐기 때문입니다.

## 실행

안전장치로 시나리오를 반드시 명시해야 합니다.

```powershell
python main.py --scenario full11
python main.py --scenario clean3
```

동일한 stable alias:

```powershell
python scripts\run_recommend.py --scenario full11
python scripts\run_recommend.py --scenario clean3
```

빠른 wiring 확인:

```powershell
python scripts\run_recommend.py --scenario full11 --sampled --candidate-count 20000
```

최종 전수평가가 필요한 경우에만 `--sampled`를 제거합니다.

## 현재 계산식

기존 v3.1 baseline은 매 역사 target에서 1,024개의 deterministic fair 6/45 reference를 만들고 feature별 empirical midrank를 사용합니다.

```text
z_j = 2 * F_mid,j(x_j) - 1
score(c) = w_A^T z_A(c)
```

`A`는 선택한 시나리오의 active feature 집합입니다.

역사 학습에서 target 정답은 target의 context/reference/outer score가 만들어진 뒤에만 training state에 추가됩니다. 따라서 회차 내부 target leakage는 막혀 있습니다. 다만 모델 클래스와 feature 선택이 같은 역사 데이터를 반복해서 본 뒤 결정되어 왔으므로 이 검증은 독립적인 최종 OOS 증명이 아니라 post-selection exploratory walk-forward로 해석해야 합니다.

## 중요한 재검토 항목

### 1. 1,024 reference 해상도

814만 조합을 1,024개 empirical CDF로 변환하면 극단 꼬리에서 `z=-1/+1` 포화와 score plateau가 생길 수 있습니다. TOP-10/TOP-1000 세부 순위가 reference sample 선택에 안정적인지 별도 감사가 필요합니다.

### 2. tie policy

과거에는 동점 조합을 seed 기반 BLAKE2 hash로 순서화했습니다. 현재 공통 core는 RNG를 사용하지 않고 고정 조합 순서로 동점을 처리합니다.

```text
primary   = model_score
secondary = fixed combination order
```

`--seed-offset`은 sampled candidate 생성에만 영향을 주며 exhaustive ranking 동점 순서에는 영향을 주지 않습니다.

### 3. exact structural fair-null

다음 여섯 구조 feature는 1,024개 Monte Carlo reference 없이 전체 8,145,060 조합의 정확한 분포를 계산할 수 있습니다.

- `previous_draw_overlap`
- `sum_signed_center_138`
- `high_minus_low_zone_count`
- `odd_count_signed_center_3`
- `number_range`
- `consecutive_pairs`

`lotto_engine/v31_exact_structural_null.py`에 정확한 조합수/CDF 구현을 추가했습니다. **아직 FULL11/CLEAN3 baseline 계산에는 연결하지 않았습니다.** 기존 결과를 보존한 상태에서 별도 stability audit 후 representation 변경 여부를 결정합니다.

### 4. feature correlation

`number_full`/`pair_full`, `recent20`/`recent100`, number/pair recency는 서로 독립 축이 아닙니다. 특히 pair hit에는 개별 번호 hit의 marginal 효과가 구조적으로 포함됩니다. 전체 11-feature LOO와 group ablation, coefficient sign stability를 별도 감사합니다.

## ModelSpec / 재현성

v3.1 계산 상수는 `lotto_engine/v31_model_spec.py`의 frozen dataclass로 모았습니다.

포함 항목:

- feature 목록과 active subset
- history start index
- number/pair Bayesian prior
- recent window
- training negative 수
- fair reference 수
- ridge lambda
- RNG seed offsets
- tie policy

각 실행은 canonical model spec의 SHA-256을 출력합니다. 데이터도 normalized draw history SHA-256을 기록합니다.

## 후보 정책

- 1~45 중 서로 다른 6개인 모든 조합이 유효합니다.
- hard filter가 없습니다.
- all-odd/all-even, 극단 합계, 좁은 range, 연속수 많은 조합도 제거하지 않습니다.
- Normal/Mixed/Outlier 같은 구조 분류는 ranking 규칙이 아닙니다.
- ranking distribution이 fair 평균과 멀다는 사실만으로 feature를 삭제하지 않습니다.
- validation 결과를 raw score에 곱해 중립화하지 않습니다.

## 포트폴리오

구매용 분산은 모델 score 이후 단계입니다.

현재 strict selector는 raw-score 상위 pool을 순서대로 보면서:

```text
티켓 간 공통번호 <= 2
AND
한 번호의 노출 <= 전체 티켓의 40%
```

를 만족하는 티켓만 선택합니다.

- raw score 수정 없음
- fallback relaxation 없음
- pool에서 K장을 못 채우면 적은 수를 반환하고 `complete=false`

공용 구현은 `lotto_engine/strict_portfolio.py`입니다.

## 데이터 무결성

로컬 데이터:

```text
data/lotto.xlsx
```

필수 컬럼:

```text
회차, 번호1, 번호2, 번호3, 번호4, 번호5, 번호6
```

loader는 다음을 hard validation합니다.

- 1회부터 최신까지 누락 없는 연속 회차
- 중복 회차 없음
- 회차/번호가 유한한 정수인지 확인
- `12.5 -> 12` 같은 암묵적 truncation 금지
- 각 회차 6개 번호 중복 없음
- 번호 범위 1~45

검증:

```powershell
python scripts\validate_data.py
```

## 감사 runner

### Reference stability

fitted weight를 고정하고 latest fair-reference만 교란/확장합니다. 성능 튜닝용이 아니라 수치 수렴성 진단용입니다.

```powershell
python scripts\run_v31_reference_stability_audit.py \
  --scenario full11 \
  --candidate-count 20000
```

주요 출력:

- Pearson score correlation
- Spearman rank correlation
- TOP-10/100/1000 Jaccard
- exact-score duplicate count
- feature별 `z=±1` saturation rate
- reference count 증가에 따른 nested convergence

### Full 11-feature / group audit

모든 단일 feature와 상관 group을 동일한 strict walk-forward 조건에서 leave-out합니다.

```powershell
python scripts\run_v31_full_feature_audit.py \
  --baseline-samples 500 \
  --bootstrap-reps 2000
```

포함 group:

- long-run number + pair
- all recency
- number marginal family
- pair family
- all structural features

또한 FULL11 coefficient mean/std, 부호 비율, sign-flip count, 최근100 평균을 기록합니다.

## 테스트 / CI

로컬 전체 회귀 테스트:

```powershell
python -m unittest discover -s tests -v
```

PR/push에서는 `.github/workflows/tests.yml`이 동일한 unittest suite를 GitHub Actions에서 실행하도록 추가했습니다. 다만 실제 `data/lotto.xlsx`는 저장소에 포함하지 않으므로 최신 로컬 데이터 검증과 실데이터 smoke/audit는 로컬에서 별도로 실행해야 합니다.

## 다음 감사 순서

1. reference seed/size stability
2. z saturation 및 score tie/cutoff multiplicity
3. exact structural null representation 비교
4. 11개 전체 leave-one-out
5. long-run / recency / structure group ablation
6. number-vs-pair 중복 및 pair residualization
7. ridge coefficient/condition stability
8. 그 결과 이후에만 feature subset 확정
9. 모델 동결 후 새로운 미래 회차를 독립 검증 구간으로 누적
