from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lotto_engine.loader import LottoDataError
from lotto_engine.v31_final_portfolio import generate_recommendations, write_json


def _print_items(title: str, items: list[dict], portfolio: bool = False) -> None:
    print(title)
    print("-" * 100)
    for item in items:
        rank = item.get("portfolio_rank") if portfolio else item.get("rank")
        nums = " ".join(str(n) for n in item["numbers"])
        suffix = ""
        if portfolio:
            suffix = f" raw_pool_rank={item['raw_rank_within_retained_pool']}"
        print(f"#{rank}  {nums}{suffix}")
        print(f"model_score={item['model_score']:.12f} type={item['pattern_type']}")
        active_terms = [
            term for term in item["feature_contributions"] if term.get("active", True)
        ]
        for term in active_terms[:5]:
            print(
                f"  {term['feature']}: contribution={term['contribution']:.12f} "
                f"weight={term['weight']:.12f} bounded={term['bounded_value']:.12f}"
            )
        print()


def _print_audit(label: str, audit: dict) -> None:
    if not audit:
        return
    rates = audit["number_inclusion_rates"]
    most = sorted(rates.items(), key=lambda kv: kv[1], reverse=True)[:8]
    fair = audit.get("fair_references", {})
    print(f"{label}: {audit['pool_size']}")
    print(
        f"mean sum={audit['mean_sum']:.3f} "
        f"(fair={fair.get('expected_sum', float('nan')):.3f}) "
        f"range=[{audit['min_sum']}, {audit['max_sum']}]"
    )
    print(
        f"mean odd count={audit['mean_odd_count']:.3f} "
        f"(fair={fair.get('expected_odd_count', float('nan')):.3f})"
    )
    print(
        f"mean number range={audit['mean_number_range']:.3f} "
        f"(fair={fair.get('expected_number_range', float('nan')):.3f})"
    )
    print(
        f"mean consecutive pairs={audit['mean_consecutive_pairs']:.3f} "
        f"(fair={fair.get('expected_consecutive_pairs', float('nan')):.3f})"
    )
    if "mean_previous_draw_overlap" in audit:
        print(
            f"mean previous-draw overlap={audit['mean_previous_draw_overlap']:.3f} "
            f"(fair={fair.get('expected_previous_draw_overlap', float('nan')):.3f}); "
            f"any={audit['any_previous_draw_overlap_rate']:.3f} "
            f"(fair={fair.get('any_previous_draw_overlap_rate', float('nan')):.3f})"
        )
    print(
        "zone slot rates: "
        + ", ".join(f"{key}={value:.4f}" for key, value in audit["zone_slot_rates"].items())
    )
    print("most frequent numbers: " + ", ".join(f"{n}:{rate:.3f}" for n, rate in most))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Current personal CLEAN3 fair-null reverse-ranking recommendation"
    )
    parser.add_argument("--sampled", action="store_true")
    parser.add_argument("--candidate-count", type=int, default=20_000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=1_000_000)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/cache/v31_final_personal.json"),
    )
    args = parser.parse_args()

    try:
        payload = generate_recommendations(
            exhaustive=not args.sampled,
            candidate_count=args.candidate_count,
            top_k=args.top_k,
            seed_offset=args.seed_offset,
            progress_every=args.progress_every,
        )
    except (LottoDataError, ValueError, RuntimeError) as exc:
        print(f"추천 실패: {exc}")
        raise SystemExit(1) from exc

    meta = payload["meta"]
    print("=" * 100)
    print("LOTTO STAT ENGINE v3.1 CLEAN3 - PERSONAL FAIR-NULL REVERSE RANKING")
    print("=" * 100)
    print(f"model: {meta['model_version']}")
    print(f"latest reflected draw: {meta['latest_draw']}")
    print(f"target draw: {meta['target_draw']}")
    print(f"data: rows={meta['data']['rows']} sha256={meta['data']['sha256'][:16]}...")
    print(f"evaluation mode: {meta['evaluation_mode']}")
    print(f"evaluated combinations: {meta['evaluated_count']:,}")
    print(f"formula: {meta['formula']}")
    print("active features: " + ", ".join(meta["active_feature_names"]))
    print("disabled after component audit: " + ", ".join(meta["disabled_feature_names"]))
    print("pattern type is metadata only; no pattern quota; all valid 6/45 combinations remain eligible")
    print()

    _print_items("RAW MODEL TOP-K (CLEAN3 score order unchanged)", payload["recommendations"])

    portfolio_meta = meta["portfolio"]
    print(
        "PORTFOLIO RULE: raw-score order, pairwise shared numbers <= "
        f"{portfolio_meta['max_shared_numbers']}, each number <= "
        f"{portfolio_meta['max_number_ticket_count']}/{portfolio_meta['requested_count']} tickets; "
        "model score is never modified"
    )
    print(
        "fair references: P(two 6/45 tickets share >=3 numbers)="
        f"{portfolio_meta['fair_random_pair_overlap_ge_3_rate']:.4%}; "
        "P(a fixed number exceeds exposure cap in independent fair tickets)="
        f"{portfolio_meta['fair_random_fixed_number_exceeds_exposure_cap_rate']:.4%}"
    )
    print(
        f"portfolio source pool={portfolio_meta['source_pool_size']:,}; "
        f"complete={portfolio_meta['complete']} "
        f"selected={portfolio_meta['selected_count']}/{portfolio_meta['requested_count']} "
        f"fallback_relaxed={portfolio_meta['fallback_relaxed']}"
    )
    print()
    _print_items(
        "STRICT DIVERSIFIED PORTFOLIO TOP-K",
        payload["portfolio_recommendations"],
        portfolio=True,
    )

    _print_audit("bias audit raw TOP pool", payload.get("bias_audit_top_pool", {}))
    _print_audit("portfolio audit", payload.get("portfolio_bias_audit", {}))

    saved = write_json(payload, args.output_json)
    print(f"personal JSON saved: {saved}")


if __name__ == "__main__":
    main()
