# Lotto Stat Engine v2.5

6/45의 모든 `8,145,060`개 조합을 제거 없이 평가하는 mixed-structure prediction engine입니다.
보너스 번호, 공동당첨 회피, 인기번호 회피는 평가에 사용하지 않습니다.

> `prediction_score`는 실제 당첨확률이 아니라 내부 예측확률점수입니다.

## v2.2 score

```text
prediction_score =
outlier_survival_score * 0.45
+ type_balance_score * 0.25
+ transition_score * 0.15
+ normal_structure_score * 0.15
```

`historical_pattern_score` and `number_dynamics_score` remain diagnostic
components, but their final weight is zero because their actual-vs-random
backtest percentiles did not beat the random baseline.

- `normal_structure_score`: 기존 weighted structure score
- `outlier_survival_score`: 역사적으로 생존한 특이 패턴의 지지도
- `historical_pattern_score`: 패턴 flag 및 count의 역대 분포 적합도
- `type_balance_score`: normal/mixed/outlier 한 유형의 점수 독점 완화
- `transition_score`: 직전 회차 유형에서 다음 유형으로의 역사적 전이
- `number_dynamics_score`: 번호별 장기·최근 출현 및 미출현 기간 동역학

어떤 component도 조합을 제외하는 hard filter로 사용하지 않습니다.

## v2.3 mixed-slot score

Mixed portfolio slots are ranked against the exact distribution of all
`8,145,060` valid combinations:

```text
mixed_slot_score =
0.35 * mixed_lift_score
+ 0.25 * mixed_interaction_score
+ 0.20 * normal_backbone_score
+ 0.15 * controlled_extreme_score
+ 0.05 * recency_consistency_score
```

Single-feature and supported interaction lifts use Bayesian smoothing.
The backbone uses historical mixed-draw medians and MADs. Transition scores do
not rank candidates inside mixed slots; they are used for portfolio allocation.
For a latest outlier draw, the allocation is normal 2 / mixed 6 / outlier 2.

## v2.4 mixed subtype analysis

The v2.4 analysis layer tags every historical draw and candidate with structural
mixed subtypes. It derives the latest and target draw from `data/lotto.xlsx`,
and keeps the v2.3.1 recommendation score unchanged. The exact all-combination
mixed-subtype baseline is versioned and cached under `data/cache`.

v2.4.1 keeps those definitions and the v2.3.1 ranking score unchanged, while
making subtype allocation lift-, information-, recency-, and signature-aware.

v2.5 integrates the dynamically generated signature-family allocation into the
final mixed-six portfolio as soft fit bonuses and duplicate/overfill penalties.
No valid combination is filtered, and the v2.3.1 empirical score remains the
base ranking component.

## 실행

```powershell
python scripts\validate_data.py
python scripts\run_recommend.py
python scripts\run_backtest.py
python scripts\run_mixed_backtest.py
python scripts\analyze_mixed_subtypes.py
```

분석 도구:

```powershell
python scripts\analyze_pattern_types.py
python scripts\analyze_draw_transitions.py
python scripts\analyze_number_dynamics.py
python scripts\diagnose_prediction_failures.py --rounds 100 --baseline-samples 500
```

테스트:

```powershell
python -m unittest discover -s tests -v
```
