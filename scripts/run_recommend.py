from lotto_engine.config import DEFAULT_CANDIDATE_COUNT
from lotto_engine.loader import LottoDataError
from lotto_engine.recommender import generate_recommendations, print_recommendations


def main() -> None:
    try:
        payload = generate_recommendations(seed_offset=0, candidate_count=DEFAULT_CANDIDATE_COUNT)
    except (LottoDataError, ValueError) as exc:
        print(f"추천 실패: {exc}")
        raise SystemExit(1)
    print_recommendations(payload)


if __name__ == "__main__":
    main()
