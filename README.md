# Lotto Stat Engine

한국 로또 6/45 과거 당첨번호 데이터를 기반으로 10게임 추천번호를 생성하는 통계 엔진입니다.

## 중요한 안내

로또 6/45의 모든 6개 조합은 공정한 독립 시행에서 1등 확률이 동일합니다. 이 프로그램은 당첨을 보장하지 않으며, 당첨 확률 상승을 보장하지 않습니다. 목적은 과거 당첨번호의 구조적 특성을 참고해 10게임 번호 포트폴리오를 생성하는 것입니다.

## 계산 원칙

- 번호 적중 개수 예측이 아니라 구조 feature 설명력을 평가합니다.
- 1~100회 데이터로 101회 구조를 평가하고, 1~101회 데이터로 102회 구조를 평가하는 워크포워드 방식을 사용합니다.
- base weight 70% + walk-forward learned weight 30%를 혼합합니다.
- 보너스 번호, 동반 출현 관계성, 공동 당첨 위험 회피, 생일 번호 회피, 인기 번호 회피는 점수에 넣지 않습니다.
- Weighted Luck에는 점수 하한선을 두지 않습니다. 단, hard filter는 통과해야 합니다.

## 데이터 파일

아래 경로에 엑셀 파일을 넣어야 합니다.

```text
data/lotto.xlsx
```

필수 컬럼:

```text
회차, 번호1, 번호2, 번호3, 번호4, 번호5, 번호6
```

선택 컬럼:

```text
보너스, 추첨일
```

## 설치

Windows PowerShell 기준입니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

실행 정책 에러가 나오면 한 번만 실행합니다.

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

## 실행

데이터 검증:

```powershell
python scripts\validate_data.py
```

워크포워드 백테스트:

```powershell
python scripts\run_backtest.py
```

추천번호 생성:

```powershell
python scripts\run_recommend.py
```

또는:

```powershell
python main.py
```

Streamlit 앱:

```powershell
streamlit run streamlit_app.py
```

## 추천 전략

1. Core Structure
2. Core Structure
3. Balanced Structure
4. High Entropy
5. Wide Gap
6. Recent Soft Match
7. Cluster Diversity
8. Weighted Luck
9. Weighted Luck
10. Coverage Optimizer
