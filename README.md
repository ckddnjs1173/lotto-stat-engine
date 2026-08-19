# Lotto Stat Engine

개인용 Lotto 6/45 통계 랭킹 연구 엔진입니다.

`model_score`는 실제 당첨확률이 아니라 **현재 계산식이 후보 조합을 정렬하기 위한 내부 점수**입니다. 과거 walk-forward 검증은 재현 가능한 우위를 입증하지 못했으며, 이 저장소는 그 사실과 별개로 계산식이 만든 raw ranking을 그대로 보존합니다.

## 현재 실행 경로

현재 기본 실행은 하나로 통일되어 있습니다.

```powershell
python main.py
```

또는:

```powershell
python scripts\run_recommend.py
```

둘 다 `scripts/run_v31_final_recommend.py`의 현재 개인용 모델을 실행합니다.

기본 모드는 `C(45,6) = 8,145,060`개 전체 조합 전수평가입니다. 빠른 확인용으로만:

```powershell
python scripts\run_recommend.py --sampled --candidate-count 20000
```

를 사용합니다.

## 현재 모델: v3.1 CLEAN3

현재 점수식은 과거 정답을 target 시점 이후에만 학습에 추가하는 역산(pairwise ridge) 구조입니다.

각 후보의 raw feature는 해당 시점의 공정 6/45 reference 조합 1,024개와 비교해:

```text
z_j = 2 * F_mid,j(x_j) - 1
```

로 `[-1, 1]`의 fair-null 좌표로 변환됩니다.

최종 점수는 활성 feature만 다시 적합한 additive ridge입니다.

```text
score(c) = w_A^T z_A(c)
```

### 활성 feature 8개

- `number_full_log_lift`
- `pair_full_log_lift`
- `number_recent20_excess`
- `number_recent100_excess`
- `pair_recent100_excess`
- `sum_signed_center_138`
- `high_minus_low_zone_count`
- `odd_count_signed_center_3`

### 제외 feature 3개

1237회까지 반영한 1238회 전수 TOP-1000 component audit에서 다음 세 항목은 최신 ranking을 강하게 왜곡했지만 독립 historical OOS 기여가 확인되지 않았습니다.

- `previous_draw_overlap`
- `number_range`
- `consecutive_pairs`

따라서 단순히 결과에서 contribution을 0으로 숨기는 것이 아니라 **세 feature를 제외한 8개 feature로 ridge를 다시 적합**합니다.

`number_full_log_lift`는 강한 concentration을 만들 수 있지만, focused leave-one-out audit에서 네 항목 중 전체/최근300/최근100이 가장 일관되게 양의 OOS 방향이어서 유지합니다.

자세한 숫자는 `docs/v31_component_influence_result.md`를 참고하세요.

## 후보 정책

- 1~45 중 서로 다른 6개인 모든 조합이 유효합니다.
- hard filter가 없습니다.
- 합계, 홀짝, 구간, 연속수 등을 “예쁘게” 맞추는 quota가 없습니다.
- Normal/Mixed/Outlier는 출력 설명용 metadata일 뿐 ranking에 사용하지 않습니다.
- 검증 실패를 이유로 raw score에 confidence multiplier를 곱하거나 50점으로 중립화하지 않습니다.

## 포트폴리오

RAW TOP-K와 구매용 분산 포트폴리오는 분리되어 있습니다.

현재 portfolio는 상위 raw-score 후보 50,000개를 순서대로 보면서:

```text
티켓 간 공통번호 <= 2
AND
한 번호의 노출 <= 전체 티켓의 40%
```

를 만족하는 티켓만 고릅니다.

- raw model score는 수정하지 않습니다.
- 제약을 만족하지 못한다고 규칙을 완화하지 않습니다.
- 후보 pool에서 K장을 채울 수 없으면 `complete=false`와 함께 적은 수를 반환합니다.

## 데이터

로컬 데이터 파일:

```text
data/lotto.xlsx
```

필수 컬럼:

```text
회차, 번호1, 번호2, 번호3, 번호4, 번호5, 번호6
```

loader는 다음을 hard validation합니다.

- 1회부터 최신 회차까지 누락 없는 연속 회차
- 중복 회차 없음
- 각 회차 6개 번호의 중복 없음
- 번호 범위 1~45

각 실행 결과에는 normalize된 draw history의 SHA-256 fingerprint가 포함됩니다.

```powershell
python scripts\validate_data.py
```

## 검증

전체 회귀 테스트:

```powershell
python -m unittest discover -s tests -v
```

빠른 계산 smoke:

```powershell
python scripts\run_recommend.py --sampled --candidate-count 20000 --output-json data\cache\v31_final_personal.json
```

최종 전수 계산:

```powershell
python scripts\run_recommend.py --top-k 10 --progress-every 1000000 --output-json data\cache\v31_final_personal.json
```

## 저장소 구조

현재 사용 파일과 과거 연구 재현용 파일을 구분해 두었습니다.

- 현재 모델/실행: `lotto_engine/v31_clean3_recommendation.py`, `lotto_engine/v31_final_portfolio.py`, `scripts/run_v31_final_recommend.py`
- component 감사: `lotto_engine/v31_component_influence_audit.py`, `lotto_engine/v31_joint_ablation_audit.py`
- v2.7~v3.0 파일: 과거 실험 재현과 테스트를 위해 보존
- 상세 지도: `docs/repository_map.md`

과거 버전 runner는 연구 재현용입니다. **일상 실행은 `main.py` 또는 `scripts/run_recommend.py`만 사용하세요.**

## 해석 원칙

로또 6/45의 특정 조합은 공정 추첨에서 다른 특정 조합과 같은 확률을 가집니다. 이 엔진은 그 사실을 바꾸지 않습니다. 통계 모델은 과거 데이터에서 학습된 ranking 신호를 계산할 뿐이며, historical validation 결과는 별도로 기록합니다.
