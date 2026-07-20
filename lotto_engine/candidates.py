from __future__ import annotations

import random


def make_seed(latest_round: int, seed_offset: int = 0) -> int:
    return int(latest_round) + int(seed_offset) * 100_000


def generate_candidates(count: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    seen: set[tuple[int, ...]] = set()
    candidates: list[list[int]] = []

    attempts = 0
    max_attempts = count * 5
    while len(candidates) < count and attempts < max_attempts:
        attempts += 1
        nums = tuple(sorted(rng.sample(range(1, 46), 6)))
        if nums in seen:
            continue
        seen.add(nums)
        candidates.append(list(nums))
    return candidates
