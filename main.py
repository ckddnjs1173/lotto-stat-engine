from lotto_engine.config import DEFAULT_CANDIDATE_COUNT
from lotto_engine.v27_release import (
    generate_release_recommendations,
    print_release_recommendations,
)


def main() -> None:
    payload = generate_release_recommendations(
        seed_offset=0,
        candidate_count=DEFAULT_CANDIDATE_COUNT,
        exhaustive=True,
    )
    print_release_recommendations(payload)


if __name__ == "__main__":
    main()
