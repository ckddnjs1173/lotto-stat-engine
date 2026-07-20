from __future__ import annotations

from collections import Counter
from itertools import combinations


def sum_band(total: int) -> str:
    if total <= 125:
        return "low"
    if total <= 155:
        return "mid"
    return "high"


def portfolio_report(recommendations: list[dict]) -> dict:
    all_numbers = [n for item in recommendations for n in item["numbers"]]
    counts = Counter(all_numbers)
    overlaps = []
    for a, b in combinations(recommendations, 2):
        overlaps.append(len(set(a["numbers"]) & set(b["numbers"])))

    return {
        "unique_number_count": len(set(all_numbers)),
        "average_overlap": round(sum(overlaps) / len(overlaps), 4) if overlaps else 0.0,
        "max_overlap": max(overlaps) if overlaps else 0,
        "most_repeated_numbers": dict(counts.most_common(10)),
        "sum_band_distribution": dict(Counter(sum_band(int(item["features"]["sum"])) for item in recommendations)),
        "odd_even_distribution": dict(Counter(f"{item['features']['odd_count']}:{item['features']['even_count']}" for item in recommendations)),
        "strategy_distribution": dict(Counter(item["strategy"] for item in recommendations)),
    }
