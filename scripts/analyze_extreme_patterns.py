import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from lotto_engine.features import extract_features
from lotto_engine.loader import LottoDataError, load_lotto_data, row_numbers


def pct(count: int, total: int) -> float:
    return count / total * 100 if total else 0.0


def label_extreme_reason(features: dict) -> list[str]:
    reasons: list[str] = []
    odd = int(features["odd_count"])
    total = int(features["sum"])
    section_counts = [int(features[f"section_{i}"]) for i in range(1, 6)]
    lmh = [int(features["low_count"]), int(features["mid_count"]), int(features["high_count"])]
    max_run = int(features["max_consecutive_run"])
    consecutive_pairs = int(features["consecutive_pairs"])
    dup_end = int(features["duplicate_endings"])
    max_gap = int(features["max_gap"])
    min_gap = int(features["min_gap"])
    num_range = int(features["range"])

    if odd in {0, 6}:
        reasons.append("홀짝 6:0/0:6")
    elif odd in {1, 5}:
        reasons.append("홀짝 5:1/1:5")

    if total <= 90:
        reasons.append("합계 매우 낮음(<=90)")
    elif total <= 105:
        reasons.append("합계 낮음(<=105)")
    elif total >= 185:
        reasons.append("합계 매우 높음(>=185)")
    elif total >= 170:
        reasons.append("합계 높음(>=170)")

    if max(section_counts) >= 4:
        reasons.append("10번대 구간 4개 이상 몰림")
    if 0 in section_counts and section_counts.count(0) >= 2:
        reasons.append("빈 10번대 구간 2개 이상")
    if max(lmh) >= 5:
        reasons.append("저/중/고 15구간 5개 이상 몰림")

    if max_run >= 4:
        reasons.append("연속 run 4개 이상")
    elif consecutive_pairs >= 2:
        reasons.append("연속쌍 2개 이상")

    if dup_end >= 3:
        reasons.append("끝수 중복 3개 이상")
    elif dup_end >= 2:
        reasons.append("끝수 중복 2개")

    if max_gap >= 20:
        reasons.append("최대 간격 20 이상")
    if min_gap == 1:
        reasons.append("최소 간격 1")
    if num_range <= 20:
        reasons.append("범위 좁음(<=20)")
    elif num_range >= 42:
        reasons.append("범위 넓음(>=42)")

    return reasons


def print_distribution(title: str, counter: Counter, total: int, sort_key=None) -> None:
    print()
    print(title)
    print("-" * len(title))
    items = counter.items()
    if sort_key:
        items = sorted(items, key=sort_key)
    else:
        items = sorted(items)
    for key, count in items:
        print(f"{str(key):>12}: {count:4d}회  {pct(count, total):6.2f}%")


def theoretical_odd_distribution(total_draws: int) -> dict[int, tuple[int, float]]:
    denom = len(list(combinations(range(1, 46), 6)))
    out = {}
    for odd in range(7):
        even = 6 - odd
        cases = len(list(combinations(range(1, 46, 2), odd))) * len(list(combinations(range(2, 46, 2), even)))
        out[odd] = (cases, cases / denom * 100)
    return out


def main() -> None:
    try:
        df = load_lotto_data()
    except LottoDataError as exc:
        print(f"분석 실패: {exc}")
        raise SystemExit(1)

    total = len(df)
    feature_rows = []
    reason_counter: Counter[str] = Counter()
    extreme_rows: list[tuple[int, list[int], list[str], dict]] = []

    for _, row in df.iterrows():
        numbers = row_numbers(row)
        features = extract_features(numbers)
        features["round"] = int(row["회차"])
        features["numbers"] = numbers
        feature_rows.append(features)
        reasons = label_extreme_reason(features)
        for reason in reasons:
            reason_counter[reason] += 1
        if reasons:
            extreme_rows.append((int(row["회차"]), numbers, reasons, features))

    print("LOTTO EXTREME PATTERN ANALYSIS")
    print("=" * 72)
    print(f"총 회차 수: {total}")
    print(f"최신 회차: {int(df['회차'].max())}")

    odd_counter = Counter(int(f["odd_count"]) for f in feature_rows)
    print()
    print("홀짝 분포: 실제 vs 이론")
    print("-" * 72)
    theo = theoretical_odd_distribution(total)
    for odd in range(7):
        actual_count = odd_counter.get(odd, 0)
        _, theo_pct = theo[odd]
        print(
            f"홀수 {odd} / 짝수 {6-odd}: "
            f"실제 {actual_count:4d}회 {pct(actual_count, total):6.2f}% | "
            f"이론 {theo_pct:6.2f}% | 차이 {pct(actual_count, total) - theo_pct:+6.2f}%p"
        )

    print_distribution("합계 구간 분포", Counter(
        "<=90" if int(f["sum"]) <= 90 else
        "91~105" if int(f["sum"]) <= 105 else
        "106~125" if int(f["sum"]) <= 125 else
        "126~155" if int(f["sum"]) <= 155 else
        "156~169" if int(f["sum"]) <= 169 else
        "170~184" if int(f["sum"]) <= 184 else
        ">=185"
        for f in feature_rows
    ), total)

    print_distribution("최대 연속 run 분포", Counter(int(f["max_consecutive_run"]) for f in feature_rows), total)
    print_distribution("연속쌍 개수 분포", Counter(int(f["consecutive_pairs"]) for f in feature_rows), total)
    print_distribution("끝수 중복 개수 분포", Counter(int(f["duplicate_endings"]) for f in feature_rows), total)

    print_distribution("번호 범위(range) 구간 분포", Counter(
        "<=20" if int(f["range"]) <= 20 else
        "21~30" if int(f["range"]) <= 30 else
        "31~40" if int(f["range"]) <= 40 else
        ">=41"
        for f in feature_rows
    ), total)

    print_distribution("최대 gap 구간 분포", Counter(
        "<=10" if int(f["max_gap"]) <= 10 else
        "11~15" if int(f["max_gap"]) <= 15 else
        "16~19" if int(f["max_gap"]) <= 19 else
        ">=20"
        for f in feature_rows
    ), total)

    print()
    print("극단/특이 구조 발생 빈도")
    print("-" * 72)
    for reason, count in reason_counter.most_common():
        print(f"{reason:<28}: {count:4d}회  {pct(count, total):6.2f}%")

    print()
    print("극단 구조 포함 회차 수")
    print("-" * 72)
    print(f"극단/특이 조건 하나 이상 포함: {len(extreme_rows)}회  {pct(len(extreme_rows), total):.2f}%")

    print()
    print("최근 30회 중 극단/특이 구조")
    print("-" * 72)
    for round_no, numbers, reasons, features in extreme_rows[-30:]:
        nums = " ".join(str(n) for n in numbers)
        print(f"{round_no:4d}회 | {nums:<18} | {', '.join(reasons)}")

    print()
    print("핵심 결론")
    print("-" * 72)
    print("1. 극단 구조는 드물지만 역대 당첨번호 안에 반복적으로 존재합니다.")
    print("2. 따라서 hard filter로 제거하면 실제 당첨 가능한 구조를 스스로 버리게 됩니다.")
    print("3. 평균 유사도만 강하게 쓰면 한두 feature가 극단적인 당첨번호가 하위로 밀릴 수 있습니다.")
    print("4. 다음 예측식은 평균형 점수와 별도로 outlier survival 점수를 검토하는 것이 합리적입니다.")


if __name__ == "__main__":
    main()
