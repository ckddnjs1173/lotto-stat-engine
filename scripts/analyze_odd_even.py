import sys
from collections import Counter
from math import comb
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lotto_engine.loader import LottoDataError, load_lotto_data, row_numbers


def theoretical_distribution() -> dict[int, float]:
    total = comb(45, 6)
    # 1~45에는 홀수 23개, 짝수 22개가 있습니다.
    return {
        odd: comb(23, odd) * comb(22, 6 - odd) / total
        for odd in range(7)
        if odd <= 23 and 6 - odd <= 22
    }


def main() -> None:
    try:
        df = load_lotto_data()
    except LottoDataError as exc:
        print(f"분석 실패: {exc}")
        raise SystemExit(1)

    counts: Counter[int] = Counter()
    examples: dict[int, list[int]] = {}

    for _, row in df.iterrows():
        nums = row_numbers(row)
        odd_count = sum(1 for n in nums if n % 2 == 1)
        counts[odd_count] += 1
        examples.setdefault(odd_count, []).append(int(row["회차"]))

    total_draws = len(df)
    theory = theoretical_distribution()

    print("ODD/EVEN HISTORICAL ANALYSIS")
    print(f"총 회차 수: {total_draws}")
    print(f"최신 회차: {int(df['회차'].max())}")
    print()
    print("홀수 개수별 분포")
    print("홀수 | 짝수 | 실제횟수 | 실제비율 | 이론비율 | 실제-이론")
    for odd in range(7):
        even = 6 - odd
        actual_count = counts.get(odd, 0)
        actual_rate = actual_count / total_draws * 100 if total_draws else 0.0
        theory_rate = theory.get(odd, 0.0) * 100
        diff = actual_rate - theory_rate
        print(f"{odd:>2}   | {even:>2}   | {actual_count:>6}   | {actual_rate:>7.3f}% | {theory_rate:>7.3f}% | {diff:>+8.3f}%p")

    extreme_count = counts.get(0, 0) + counts.get(6, 0)
    extreme_rate = extreme_count / total_draws * 100 if total_draws else 0.0
    extreme_theory = (theory.get(0, 0.0) + theory.get(6, 0.0)) * 100

    print()
    print("극단 홀짝 구조")
    print(f"0:6 또는 6:0 실제 횟수: {extreme_count}회")
    print(f"0:6 또는 6:0 실제 비율: {extreme_rate:.3f}%")
    print(f"0:6 또는 6:0 이론 비율: {extreme_theory:.3f}%")

    print()
    print("회차 예시")
    for odd in [0, 1, 5, 6]:
        rounds = examples.get(odd, [])
        preview = ", ".join(str(r) for r in rounds[:20])
        suffix = " ..." if len(rounds) > 20 else ""
        print(f"홀수 {odd}개: {len(rounds)}회" + (f" / {preview}{suffix}" if rounds else ""))


if __name__ == "__main__":
    main()
