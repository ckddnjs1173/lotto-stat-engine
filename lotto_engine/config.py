from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
LOTTO_XLSX_PATH = DATA_DIR / "lotto.xlsx"
WEIGHTS_PATH = CACHE_DIR / "feature_weights.json"

NUMBER_COLUMNS = ["번호1", "번호2", "번호3", "번호4", "번호5", "번호6"]
ROUND_COLUMN = "회차"

# 샘플링 추천을 쓸 때의 기본 후보 수입니다. 최종 추천은 exhaustive=True일 때 전체 조합을 전수 평가합니다.
DEFAULT_CANDIDATE_COUNT = 100_000
BACKTEST_START_INDEX = 100
RECENT_WINDOW = 20

# 워크포워드 actual-vs-random 백테스트에서 각 회차마다 비교할 랜덤 기준선 후보 수입니다.
# 개인 PC에서 먼저 돌릴 수 있도록 과하게 크게 잡지 않습니다.
BASELINE_SAMPLE_COUNT = 2_000

# 최종 추천에서 전체 8,145,060개 조합을 전수 평가할지 여부입니다.
DEFAULT_EXHAUSTIVE_RECOMMENDATION = True
TOP_K_RECOMMENDATIONS = 10

BASE_WEIGHTS = {
    "sum": 0.15,
    "odd_even": 0.15,
    "section": 0.15,
    "gap": 0.12,
    "entropy": 0.13,
    "consecutive": 0.08,
    "ending": 0.07,
    "recent": 0.05,
    "cluster": 0.10,
}
