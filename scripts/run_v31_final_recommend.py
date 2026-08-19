from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lotto_engine.loader import LottoDataError
from lotto_engine.v31_audit_utils import runtime_identity, write_json
from lotto_engine.v31_scenario_recommendation import (
    SCENARIO_SPECS,
    generate_scenario_recommendations,
)


def _print_items(title: str, items: list[dict], portfolio: bool = False) -> None:
    print(title)
    print("-" * 100)
    for item in items:
        rank = item.get("portfolio_rank") if portfolio else item.get("rank")
        nums = " ".join(str(number) for number in item["numbers"])
        suffix = ""
        if portfolio:
            suffix = f" raw_pool_rank={item['raw_rank_within_retained_pool']}"
        print(f"#{rank}  {nums}{suffix}")
        print(f"model_score={item['model_score']:.12f}")
        active_terms = [
            term for term in item["feature_contributions"] if term.get("active", True)
        ]
        for term in active_terms[:5]:
            print(
                f"  {term['feature']}: contribution={term['contribution']:.12f} "
                f"weight={term['weight']:.12f} bounded={term['bounded_value']:.12f}"
            )
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "v3.1 experimental recommendation scenarios. No scenario is currently promoted; "
            "choose the calculation explicitly."
        )
    )
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIO_SPECS),
        required=True,
        help="full11=historical baseline, clean3=unpromoted joint-ablation candidate",
    )
    parser.add_argument("--sampled", action="store_true")
    parser.add_argument("--candidate-count", type=int, default=20_000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="affects sampled candidate generation only; exhaustive tie ordering is RNG-free",
    )
    parser.add_argument("--progress-every", type=int, default=1_000_000)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/cache/v31_experimental_scenario.json"),
    )
    args = parser.parse_args()

    try:
        payload = generate_scenario_recommendations(
            scenario=args.scenario,
            exhaustive=not args.sampled,
            candidate_count=args.candidate_count,
            top_k=args.top_k,
            seed_offset=args.seed_offset,
            progress_every=args.progress_every,
        )
    except (LottoDataError, ValueError, RuntimeError) as exc:
        print(f"계산 실패: {exc}")
        raise SystemExit(1) from exc

    payload["meta"]["runtime"] = runtime_identity()
    meta = payload["meta"]
    print("=" * 100)
    print("LOTTO STAT ENGINE v3.1 - EXPERIMENTAL SCENARIO CALCULATION")
    print("NO v3.1 SCENARIO IS CURRENTLY PROMOTED")
    print("=" * 100)
    print(f"scenario: {meta['scenario']}")
    print(f"model: {meta['model_version']}")
    print(f"status: {meta['model_status']}")
    print(f"model spec sha256: {meta['model_spec']['sha256']}")
    print(f"latest reflected draw: {meta['latest_draw']}")
    print(f"target draw: {meta['target_draw']}")
    print(f"data: rows={meta['data']['rows']} sha256={meta['data']['sha256'][:16]}...")
    print(
        f"runtime: python={meta['runtime']['python']} numpy={meta['runtime']['numpy']} "
        f"platform={meta['runtime']['platform']}"
    )
    print(f"evaluation mode: {meta['evaluation_mode']}")
    print(f"evaluated combinations: {meta['evaluated_count']:,}")
    print(f"tie policy: {meta['tie_policy']}")
    print("active features: " + ", ".join(meta["active_feature_names"]))
    print(
        "distribution diagnostics are descriptive only; distance from fair does not delete or penalize a feature"
    )
    print()

    _print_items("RAW MODEL TOP-K", payload["recommendations"])

    portfolio_meta = meta["portfolio"]
    print(
        "STRICT PORTFOLIO: pairwise shared numbers <= "
        f"{portfolio_meta['max_shared_numbers']}, each number <= "
        f"{portfolio_meta['max_number_ticket_count']}/{portfolio_meta['requested_count']} tickets; "
        "no fallback relaxation"
    )
    print(
        f"complete={portfolio_meta['complete']} "
        f"selected={portfolio_meta['selected_count']}/{portfolio_meta['requested_count']}"
    )
    print()
    _print_items("STRICT DIVERSIFIED PORTFOLIO", payload["portfolio_recommendations"], portfolio=True)

    tie = payload.get("tie_diagnostics", {})
    if tie:
        print("TIE DIAGNOSTICS")
        print("-" * 100)
        print(
            f"raw top-k cutoff score={tie['raw_top_k_cutoff_score']:.12f}; "
            f"equal cutoff within retained pool={tie['equal_cutoff_score_count_within_retained_pool']}; "
            f"top-k duplicate scores={tie['exact_score_duplicate_count_inside_raw_top_k']}"
        )
        print()

    saved = write_json(payload, args.output_json)
    print(f"scenario JSON saved: {saved}")


if __name__ == "__main__":
    main()
