# Lotto Stat Engine

개인용 Lotto 6/45 통계 랭킹 연구 엔진입니다.

`model_score`는 실제 당첨확률이 아니라 과거 데이터에서 학습한 계산식이 후보 조합을 정렬하기 위한 내부 점수입니다. 과거 walk-forward 결과는 재현 가능한 우위를 입증하지 못했으며, 현재 v3.1의 어떤 시나리오도 최종 모델로 승격되어 있지 않습니다.

## 현재 상태

v3.1은 **모델 가족을 감사하고 수치 안정성을 확정하는 단계**입니다.

공통 구조:

```text
raw feature
→ fair-null coordinate z_j
→ pairwise actual-vs-fair ridge
→ score(c)=w^T z(c)
→ 모든 C(45,6)=8,145,060 조합 ranking
```

현재 비교 시나리오:

- `full11`: 기존 11-feature v3.1 baseline. `experimental_baseline_not_promoted`.
- `clean3`: `previous_draw_overlap`, `number_range`, `consecutive_pairs`를 동시에 제거하고 reduced ridge를 다시 적합한 후보. `experimental_candidate_requires_joint_and_stability_audit`.

**CLEAN3는 현재/production 모델이 아닙니다.** focused leave-one-out의 bootstrap interval이 모두 0을 포함했고, joint CLEAN3를 실행 검토하기 전에 먼저 승격한 것은 절차상 너무 빨랐기 때문에 승격을 철회했습니다.

## stable 실행 경로

시나리오를 반드시 명시해야 합니다.

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

최종 전수평가가 필요한 경우에만 `--sampled`를 제거합니다. 모델이 동결되기 전에는 8,145,060 전수 결과를 최종 예측값으로 취급하지 않습니다.

## 현재 baseline 계산식

기존 v3.1 baseline은 매 역사 target에서 deterministic fair 6/45 reference 1,024개를 만들고 feature별 empirical midrank를 사용합니다.

```text
z_j = 2 * F_mid,j(x_j) - 1
score(c) = w_A^T z_A(c)
```

`A`는 선택한 시나리오의 active feature 집합입니다.

역사 target의 정답은 target 시점의 context/reference/score를 계산한 뒤에만 state에 추가됩니다. 따라서 회차 내부 target leakage는 막혀 있습니다. 다만 모델 클래스와 feature 선택은 동일한 역사 구간을 반복해서 본 뒤 결정되어 왔으므로 historical 결과는 독립적인 최종 OOS 증명이 아니라 **post-selection exploratory walk-forward validation**으로 해석합니다.

## 재검토에서 수정된 핵심

### 데이터 무결성

loader는 다음을 hard validation합니다.

- 1회부터 최신까지 누락 없는 연속 회차
- 중복 회차 없음
- 회차/번호가 유한한 정수인지 확인
- `12.5 -> 12` 같은 암묵적 truncation 금지
- 각 회차 6개 번호 중복 없음
- 번호 범위 1~45

```powershell
python scripts\validate_data.py
```

각 실행은 normalized history SHA-256을 기록합니다.

### ModelSpec / 재현성

`lotto_engine/v31_model_spec.py`의 frozen dataclass가 다음을 고정합니다.

- feature 목록과 active subset
- history start index / minimum meta targets
- number/pair prior
- recent windows
- training negatives
- fair reference count
- ridge lambda
- RNG offsets
- tie policy

각 scenario 실행은 model spec SHA-256, data SHA-256, Python/NumPy runtime identity를 JSON에 남깁니다.

### deterministic tie policy

seed 기반 BLAKE2 tie-break는 제거했습니다.

```text
primary   = model_score
secondary = fixed combination order
```

`--seed-offset`은 sampled candidate 생성에만 영향을 주며 exhaustive tie order에는 영향을 주지 않습니다.

### 후보 정책

- 1~45 중 서로 다른 6개인 모든 조합이 유효합니다.
- hard filter가 없습니다.
- all-odd/all-even, 극단 합계, 좁은 range, 연속수 많은 조합도 제거하지 않습니다.
- 구조가 fair 평균과 멀다는 이유만으로 score를 깎지 않습니다.
- validation 결과를 raw score에 곱해 중립화하지 않습니다.

### strict portfolio

구매용 분산은 raw ranking 이후 단계입니다.

```text
티켓 간 공통번호 <= 2
AND
한 번호 노출 <= 전체 티켓의 40%
```

- raw score 수정 없음
- fallback relaxation 없음
- K장을 못 채우면 `complete=false`

공용 구현은 `lotto_engine/strict_portfolio.py`입니다.

## 완성된 v3.1 감사 계층

### 1. Fixed-weight reference stability

최종 fitted weight는 고정하고 latest fair-reference만 seed/count별로 바꿉니다.

```powershell
python scripts\run_v31_reference_stability_audit.py --scenario full11 --candidate-count 20000
```

측정:

- Pearson score correlation
- Spearman rank correlation
- TOP-10/100/1000 Jaccard
- exact score duplication
- feature별 `z=±1` saturation
- 1024→4096→16384 nested convergence

### 2. Retrained reference stability

reference 정책을 역사 전체에서 바꾸고 **처음부터 다시 학습**한 뒤 weight/ranking 안정성을 비교합니다.

```powershell
python scripts\run_v31_retrained_reference_audit.py `
  --reference-counts 1024 4096 `
  --stream-deltas 0 1 `
  --latest-candidate-count 20000
```

측정:

- fitted weight correlation
- absolute / relative L2 weight delta
- coefficient sign disagreement
- 최신 score/rank correlation
- TOP-10/100/1000 Jaccard
- 실제 ridge system condition number

### 3. Full 11-feature / correlated-group audit

모든 단일 feature와 사전 정의한 상관 group을 strict walk-forward로 leave-out합니다.

```powershell
python scripts\run_v31_full_feature_audit.py `
  --baseline-samples 500 `
  --bootstrap-reps 2000
```

포함 group:

- long-run number + pair
- all recency
- number marginal family
- pair family
- all structural features

weight mean/std, 부호 비율, sign flip, recent100도 기록합니다.

### 4. Focused + joint ablation

기존 4-feature focused audit와 CLEAN3/CLEAN4 joint audit를 보존합니다.

```powershell
python scripts\run_v31_component_influence_audit.py
python scripts\run_v31_joint_ablation_audit.py --sampled-latest --latest-candidate-count 200000
```

이 결과는 단독 promotion gate가 아닙니다.

### 5. Pair residualization audit

pair edge의 개별번호 주효과를 least-squares vertex projection으로 제거합니다.

```text
pair_edge(i,j) = a_i + a_j + residual_ij
```

원본 pair feature와 residual pair feature를 같은 target/reference/negative candidate stream으로 각각 재학습해 비교합니다.

```powershell
python scripts\run_v31_pair_residual_audit.py `
  --baseline-samples 500 `
  --bootstrap-reps 2000 `
  --latest-candidate-count 20000
```

이는 `number_full`과 `pair_full`, number-recency와 pair-recency가 같은 개별번호 hot/cold 효과를 중복 반영하는 정도를 진단합니다.

### 6. Exact structural null audit

다음 여섯 feature는 전체 8,145,060 조합의 정확한 null distribution을 계산합니다.

- `previous_draw_overlap`
- `sum_signed_center_138`
- `high_minus_low_zone_count`
- `odd_count_signed_center_3`
- `number_range`
- `consecutive_pairs`

정확한 조합수 구현은 `lotto_engine/v31_exact_structural_null.py`에 있습니다. baseline 계산에는 아직 자동 적용하지 않고, sampled structural CDF와 exact structural CDF를 각각 재학습하여 비교합니다.

```powershell
python scripts\run_v31_exact_null_audit.py `
  --baseline-samples 500 `
  --bootstrap-reps 2000 `
  --latest-candidate-count 20000
```

### 7. Ridge / feature dependency audit

학습 pairwise difference 공간에서 다음 값을 분리합니다.

- unregularized scaled second-moment condition
- 실제 `scaled_second + lambda*I` ridge system condition
- weight sign/dispersion stability
- feature pair second-moment cosine

```powershell
python scripts\run_v31_dependency_audit.py
```

사전 추적 pair:

- `number_full_log_lift ↔ pair_full_log_lift`
- `number_recent20_excess ↔ number_recent100_excess`
- `number_recent100_excess ↔ pair_recent100_excess`
- `sum ↔ high-minus-low`
- `range ↔ consecutive`

## 통합 감사 runner

위 감사를 한 manifest 아래에서 순서대로 실행합니다. 중간 단계가 실패하면 partial JSON에 실패 step/traceback을 기록하고 즉시 중단합니다. **실패를 무시하고 다음 단계로 진행하지 않습니다.**

결정용 전체 감사:

```powershell
python scripts\run_v31_audit_suite.py `
  --mode full `
  --progress-every 50 `
  --output-json data\cache\v31_complete_audit_suite.json
```

빠른 wiring 감사:

```powershell
python scripts\run_v31_audit_suite.py --mode quick
```

`full` suite 순서:

1. fixed-weight reference stability
2. retrained reference stability
3. focused component audit
4. CLEAN3/CLEAN4 joint audit
5. full 11-feature/group audit
6. pair residualization
7. exact structural-null comparison
8. ridge/feature dependency audit

**suite는 어떤 scenario도 자동 승격하지 않습니다.**

## 테스트 / CI

로컬 전체 회귀 테스트:

```powershell
python -m unittest discover -s tests -v
```

`.github/workflows/tests.yml`도 같은 suite를 GitHub Actions에서 실행합니다. 실데이터가 필요한 v2.2/v2.4/v2.5 legacy 테스트는 `data/lotto.xlsx`가 없는 CI에서만 명시적으로 skip되며, 로컬 파일이 있으면 정상 실행됩니다.

새 v3.1 테스트는 다음도 고정합니다.

- unusual valid combinations eligibility
- 같은 data/spec의 deterministic fitted weights/reference
- fixed RNG-free tie order
- exact structural combinatorics
- pair residual의 vertex-design 직교성
- reference nested stream prefix
- full feature/group scenario coverage
- dependency pair ordering
- audit suite completeness

## 모델 승격 규칙

코드가 특정 결과를 보고 자동으로 model을 선택하지 않습니다.

다음 조건을 **전부 검토한 뒤** 사람이 feature/reference representation을 동결합니다.

1. reference seed/size에 TOP ranking이 과도하게 흔들리지 않는가
2. reference를 바꿔 재학습해도 coefficient 방향이 안정적인가
3. `z=±1` saturation/score tie가 TOP tail을 지배하지 않는가
4. single-feature/group ablation의 historical 방향이 무엇인가
5. pair residualization이 중복 marginal 신호를 얼마나 제거하는가
6. exact structural null이 sampled approximation을 실질적으로 바꾸는가
7. ridge system과 coefficient sign이 안정적인가
8. CLEAN3/CLEAN4 같은 subset이 joint 조건에서도 정당화되는가

그 이후에만 하나의 model spec을 `promoted/frozen` 상태로 만들고, 그때 처음 8,145,060개 전체를 최종 RAW ranking으로 계산합니다. 모델 동결 후 새로 나오는 미래 회차는 별도의 독립 검증 구간으로 누적합니다.
