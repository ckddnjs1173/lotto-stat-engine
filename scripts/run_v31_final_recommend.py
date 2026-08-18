from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lotto_engine.v31_final_directional_recommendation import generate_recommendations, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Final personal directional fair-null lotto recommendation")
    parser.add_argument("--sampled", action="store_true")
    parser.add_argument("--candidate-count", type=int, default=20_000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=1_000_000)
    parser.add_argument("--output-json", type=Path, default=Path("data/cache/v31_final_personal.json"))
    args = parser.parse_args()

    payload = generate_recommendations(
        exhaustive=not args.sampled,
        candidate_count=args.candidate_count,
        top_k=args.top_k,
        seed_offset=args.seed_offset,
        progress_every=args.progress_every,
    )
    meta = payload["meta"]
    print("=" * 100)
    print("LOTTO STAT ENGINE v3.1 - FINAL DIRECTIONAL FAIR-NULL REVERSE RANKING")
    print("=" * 100)
    print(f"latest reflected draw: {meta['latest_draw']}")
    print(f"target draw: {meta['target_draw']}")
    print(f"evaluation mode: {meta['evaluation_mode']}")
    print(f"evaluated combinations: {meta['evaluated_count']:,}")
    print(f"formula: {meta['formula']}")
    print("direction preserved; bounded fair-null coordinates; additive model; no pattern quota")
    print()

    for item in payload["recommendations"]:
        nums = " ".join(str(n) for n in item["numbers"])
        print(f"#{item['rank']}  {nums}")
        print(f"model_score={item['model_score']:.12f} type={item['pattern_type']}")
        for term in item["feature_contributions"][:5]:
            print(
                f"  {term['feature']}: contribution={term['contribution']:.12f} "
                f"weight={term['weight']:.12f} bounded={term['bounded_value']:.12f}"
            )
        print()

    audit = payload.get("bias_audit_top_pool", {})
    if audit:
        rates = audit["number_inclusion_rates"]
        most = sorted(rates.items(), key=lambda kv: kv[1], reverse=True)[:8]
        print(f"bias audit pool: TOP {audit['pool_size']}")
        print(f"mean sum={audit['mean_sum']:.3f} range=[{audit['min_sum']}, {audit['max_sum']}]")
        print(f"one-digit present rate={audit['one_digit_present_rate']:.4f}")
        print("zone slot rates: " + ", ".join(f"{k}={v:.4f}" for k, v in audit["zone_slot_rates"].items()))
        print("most frequent numbers in audit pool: " + ", ".join(f"{n}:{rate:.3f}" for n, rate in most))

    saved = write_json(payload, args.output_json)
    print(f"personal JSON saved: {saved}")


if __name__ == "__main__":
    main()
