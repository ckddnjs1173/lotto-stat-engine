from lotto_engine.config import DEFAULT_CANDIDATE_COUNT
from lotto_engine.recommender import generate_recommendations, print_recommendations


def main() -> None:
    payload = generate_recommendations(seed_offset=0, candidate_count=DEFAULT_CANDIDATE_COUNT)
    print_recommendations(payload)


if __name__ == "__main__":
    main()
