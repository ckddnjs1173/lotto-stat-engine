"""Stable recommendation command.

Historical versioned runners remain in ``scripts/`` for reproducibility.
This unversioned command always delegates to the current personal model.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_v31_final_recommend import main


if __name__ == "__main__":
    main()
