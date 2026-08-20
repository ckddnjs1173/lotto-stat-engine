import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lotto_engine.loader import load_lotto_data
from lotto_engine.mixed_subtypes import (
    SUBTYPE_ORDER,
    build_exact_mixed_subtype_baseline,
    build_historical_subtype_records,
    latest_target_draw,
    suggest_mixed_subtype_allocation,
    suggest_mixed_signature_allocation,
    subtype_information_diagnostics,
)


def _tag_counts(records):
    return Counter(tag for record in records if record["pattern_type"] == "mixed" for tag in record["subtype_tags"])


def main():
    df = load_lotto_data()
    latest, target = latest_target_draw(df)
    records = build_historical_subtype_records(df)
    mixed = [record for record in records if record["pattern_type"] == "mixed"]
    baseline = build_exact_mixed_subtype_baseline()
    full_counts = _tag_counts(records)
    recent_300 = _tag_counts(records[-300:])
    recent_100 = _tag_counts(records[-100:])
    recent_300_total = sum(r["pattern_type"] == "mixed" for r in records[-300:])
    recent_100_total = sum(r["pattern_type"] == "mixed" for r in records[-100:])
    type_counts = Counter(record["pattern_type"] for record in records)

    information = subtype_information_diagnostics(records, baseline)
    print("LOTTO STAT ENGINE v2.4.1 - LIFT-AWARE MIXED SUBTYPE ANALYSIS")
    print(f"latest reflected draw: {latest}")
    print(f"target draw: {target}")
    print(f"total draw count: {len(records)}")
    for kind in ("normal", "mixed", "outlier"):
        print(f"{kind}: {type_counts[kind]} ({100 * type_counts[kind] / len(records):.4f}%)")
    print(f"total mixed draw count: {len(mixed)}")
    print("\nSUBTYPE STATISTICS")
    for tag in SUBTYPE_ORDER:
        historical = full_counts[tag] / max(1, len(mixed))
        baseline_ratio = baseline["tag_counts"].get(tag, 0) / max(1, baseline["mixed_total"])
        lift = historical / baseline_ratio if baseline_ratio else 0.0
        r300 = recent_300[tag] / max(1, recent_300_total)
        r100 = recent_100[tag] / max(1, recent_100_total)
        info = information[tag]
        print(f"{tag}: count={full_counts[tag]}, historical={historical:.4%}, baseline={baseline_ratio:.4%}, lift={lift:.4f}, information_score={info['information_score']:.6f}, support_count={info['support_count']}, recent300={r300:.4%}, recent100={r100:.4%}, recent_trend={info['recent_trend']:.4f}")

    print("\nSUBTYPE CO-OCCURRENCE MATRIX")
    print("subtype," + ",".join(SUBTYPE_ORDER))
    for left in SUBTYPE_ORDER:
        values = []
        for right in SUBTYPE_ORDER:
            if left == right:
                value = full_counts[left]
            else:
                value = sum(left in r["subtype_tags"] and right in r["subtype_tags"] for r in mixed)
            values.append(str(value))
        print(left + "," + ",".join(values))

    signatures = Counter(record["subtype_signature"] for record in mixed)
    print("\nTOP 30 MIXED SUBTYPE SIGNATURES")
    for signature, count in signatures.most_common(30):
        print(f"{signature}: {count} ({count / len(mixed):.4%})")
    latest_record = records[-1]
    print("\nLATEST DRAW SUBTYPE")
    print(f"draw: {latest_record['draw_no']}")
    print(f"tags: {', '.join(latest_record['subtype_tags']) or 'none'}")
    print(f"signature: {latest_record['subtype_signature']}")
    print(f"\nA. RAW LIFT-AWARE SUBTYPE ALLOCATION FOR DRAW {target}")
    for subtype, count in suggest_mixed_subtype_allocation(records, baseline=baseline).items():
        print(f"{subtype}: {count}")
    print(f"\nB. RECOMMENDED SIGNATURE/SUBTYPE-FAMILY ALLOCATION FOR DRAW {target}")
    for signature, count in suggest_mixed_signature_allocation(records, baseline=baseline).items():
        print(f"{signature}: {count}")


if __name__ == "__main__":
    main()
