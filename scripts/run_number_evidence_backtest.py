from __future__ import annotations

from lotto_engine.loader import load_lotto_data
from lotto_engine.number_evidence import run_number_evidence_backtest


def main() -> None:
    df = load_lotto_data()
    payload = run_number_evidence_backtest(df)
    baseline = payload["baseline"]
    print("=" * 80)
    print("NUMBER BAYES EVIDENCE BACKTEST - STRICT WALK FORWARD")
    print("=" * 80)
    print(f"tests: {payload['total_tests']}")
    print(f"uniform marginal probability: {payload['uniform_probability']:.8f}")
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
        print(f"  mean winner probability: {result['mean_winner_probability']:.8f}")
        print(f"  mean TOP6 matches: {result['mean_top6_matches']:.4f}")
        for window in (300, 100):
            values = result["windows"][str(window)]
            print(
                f"  recent {window}: Brier skill={values['brier_skill_vs_uniform'] * 100:+.4f}% "
                f"logloss_delta={values['log_loss_improvement_vs_uniform']:+.8f} "
                f"winner_p={values['mean_winner_probability']:.8f} "
                f"TOP6={values['mean_top6_matches']:.4f}"
            )
        print()


if __name__ == "__main__":
    main()
