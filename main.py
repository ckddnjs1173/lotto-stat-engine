"""Stable CLI entry point for explicit v3.1 experimental scenarios.

No v3.1 scenario is currently promoted. The delegated runner requires
``--scenario full11`` or ``--scenario clean3`` explicitly.
"""

from scripts.run_v31_final_recommend import main


if __name__ == "__main__":
    main()
