"""Stable CLI entry point for the frozen v3.1 personal model.

Historical experimental FULL11/CLEAN3 runners remain available for reproducibility,
but the unversioned stable entry point now delegates to the immutable frozen model.
"""

from scripts.run_v31_frozen_recommend import main


if __name__ == "__main__":
    main()
