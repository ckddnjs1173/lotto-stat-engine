from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lotto_engine.loader import LottoDataError, dataset_fingerprint, load_lotto_data


def main() -> None:
    print("LOTTO DATA VALIDATION")
    try:
        df = load_lotto_data()
    except LottoDataError as exc:
        print(f"검증 실패: {exc}")
        raise SystemExit(1) from exc

    rounds = df["회차"].astype(int)
    print(f"총 회차 수: {len(df)}")
    print(f"첫 회차: {int(rounds.min())}")
    print(f"최신 회차: {int(rounds.max())}")
    print("누락 회차: 없음")
    print("번호 범위 오류: 없음")
    print("중복 번호 오류: 없음")
    print(f"data sha256: {dataset_fingerprint(df)}")
    print("검증 완료")


if __name__ == "__main__":
    main()
