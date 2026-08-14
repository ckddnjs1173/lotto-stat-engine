from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from .config import NUMBER_COLUMNS, ROUND_COLUMN


def dataframe_fingerprint(df) -> dict[str, Any]:
    """Return a stable SHA-256 fingerprint for the draw rows used by an audit."""
    columns = [ROUND_COLUMN, *NUMBER_COLUMNS]
    canonical = df[columns].copy().sort_values(ROUND_COLUMN).reset_index(drop=True)
    lines = [
        ",".join(str(int(row[column])) for column in columns)
        for _, row in canonical.iterrows()
    ]
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    first_draw = int(canonical.iloc[0][ROUND_COLUMN]) if len(canonical) else None
    latest_draw = int(canonical.iloc[-1][ROUND_COLUMN]) if len(canonical) else None
    return {
        "row_count": int(len(canonical)),
        "first_draw": first_draw,
        "latest_draw": latest_draw,
        "columns": columns,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def current_git_commit(project_root: Path) -> str | None:
    """Best-effort runner commit capture; returns None outside a git checkout."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return value or None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json_artifact(payload: dict, output_path: Path) -> Path:
    """Write strict JSON (no NaN/Infinity) and return the resolved output path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            _json_safe(payload),
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
    return path.resolve()
