from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
LOTTO_XLSX_PATH = DATA_DIR / "lotto.xlsx"
WEIGHTS_PATH = CACHE_DIR / "feature_weights.json"

NUMBER_COLUMNS = ["번호1", "번호2", "번호3", "번호4", "번호5", "번호6"]
ROUND_COLUMN = "회차"

DEFAULT_CANDIDATE_COUNT = 100_000
BACKTEST_START_INDEX = 100
RECENT_WINDOW = 20

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
