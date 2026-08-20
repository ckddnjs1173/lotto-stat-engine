from __future__ import annotations

import itertools
import math
import random
from collections.abc import Iterator


TOTAL_COMBINATION_COUNT = math.comb(45, 6)


def make_seed(latest_round: int, seed_offset: int = 0) -> int:
    return int(latest_round) + int(seed_offset) * 100_000


def random_combination(rng: random.Random) -> list[int]:
    return sorted(rng.sample(range(1, 46), 6))


def generate_candidates(count: int, seed: int) -> list[list[int]]:
    count = int(count)
    if count <= 0:
        raise ValueError("candidate count must be positive")
    if count > TOTAL_COMBINATION_COUNT:
        raise ValueError(f"candidate count cannot exceed {TOTAL_COMBINATION_COUNT:,}")

    rng = random.Random(int(seed))
    seen: set[tuple[int, ...]] = set()
    candidates: list[list[int]] = []
    max_attempts = max(10_000, count * 20)
    attempts = 0
    while len(candidates) < count and attempts < max_attempts:
        attempts += 1
        nums = tuple(random_combination(rng))
        if nums in seen:
            continue
        seen.add(nums)
        candidates.append(list(nums))

    if len(candidates) != count:
        raise RuntimeError(
            f"could not generate the requested {count:,} unique candidates "
            f"(generated {len(candidates):,})"
        )
    return candidates


def iter_all_combinations() -> Iterator[list[int]]:
    for combo in itertools.combinations(range(1, 46), 6):
        yield list(combo)
