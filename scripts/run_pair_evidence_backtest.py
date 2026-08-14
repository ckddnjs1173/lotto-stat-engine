from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lotto_engine.loader import load_lotto_data
from lotto_engine.pair_evidence import run_pair_evidence_backtest


def main() -> None:
    df = load_lotto_data()
    payload = run_pair_evidence_backtest(df)
    baseline = payload["baseline"]
    print("=" * 80)
    print("PAIR BAYES EVIDENCE BACKTEST - STRICT WALK FORWARD")
    print("=" * 80)
    print(f"tests: {payload['total_tests']}")
    print(f"pair count: {payload['pair_count']}")
    print(f"pairs per draw: {payload['pairs_per_draw']}")
    print(f"uniform pair probability: {payload['uniform_probability']:.8f}")
    print(f"random TOP15 expected matches: {payload['random_top15_expected_matches']:.6f}")
    print(f"uniform mean Brier: {baseline['mean_brier']:.8f}")
    print(f"uniform mean log loss: {baseline['mean_log_loss']:.8f}")
    print()
    for name, result in payload["models"].items():
        print(name)
        print(f"  prior_strength: {result['prior_strength']}")
        print(f"  half_life: {result['half_life']}")
        print(f"  mean Brier: {result['mean_brier']:.8f}")
        print(f"  Brier skill vs uniform: {result['brier_skill_vs_uniform'] * 100:+.4f}%")
        print(f"  mean log loss: {result['mean_log_loss']:.8f}")
        print(f"  log-loss improvement vs uniform: {result['log_loss_improvement_vs_uniform']:+.8f}")
        print(f"  mean winner pair probability: {result['mean_winner_pair_probability']:.8f}")
        print(f"  mean TOP15 pair matches: {result['mean_top15_pair_matches']:.4f}")
        for window in (300, 100):
            values = result["windows"][str(window)]
            print(
                f"  recent {window}: Brier skill={values['brier_skill_vs_uniform'] * 100:+.4f}% "
                f"logloss_delta={values['log_loss_improvement_vs_uniform']:+.8f} "
                f"winner_pair_p={values['mean_winner_pair_probability']:.8f} "
                f"TOP15={values['mean_top15_pair_matches']:.4f}"
            )
        print()


if __name__ == "__main__":
    main()
