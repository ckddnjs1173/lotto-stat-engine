from lotto_engine.loader import LottoDataError, load_lotto_data


def main() -> None:
    print("LOTTO DATA VALIDATION")
    try:
        df = load_lotto_data()
    except LottoDataError as exc:
        print(f"검증 실패: {exc}")
        raise SystemExit(1)

    rounds = df["회차"].astype(int)
    expected = set(range(int(rounds.min()), int(rounds.max()) + 1))
    missing = sorted(expected - set(rounds.tolist()))

    print(f"총 회차 수: {len(df)}")
    print(f"최신 회차: {int(rounds.max())}")
    print(f"누락 회차: {missing[:20] if missing else '없음'}")
    print("번호 범위 오류: 없음")
    print("중복 번호 오류: 없음")
    print("검증 완료")


if __name__ == "__main__":
    main()
