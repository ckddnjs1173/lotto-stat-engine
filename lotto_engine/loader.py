from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .config import LOTTO_XLSX_PATH, NUMBER_COLUMNS, ROUND_COLUMN


class LottoDataError(Exception):
    pass


def _strict_integer_series(series: pd.Series, label: str) -> pd.Series:
    """Parse a spreadsheet column without silently truncating fractional values."""
    try:
        numeric = pd.to_numeric(series, errors="raise")
    except (TypeError, ValueError) as exc:
        raise LottoDataError(f"{label} 컬럼에 숫자가 아닌 값이 있습니다: {exc}") from exc

    values = numeric.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise LottoDataError(f"{label} 컬럼에 유한한 숫자가 아닌 값이 있습니다.")

    fractional_mask = values != np.floor(values)
    if np.any(fractional_mask):
        bad = numeric.iloc[np.flatnonzero(fractional_mask)[:10]].tolist()
        raise LottoDataError(f"{label} 컬럼에는 정수만 허용됩니다: {bad}")

    return numeric.astype(int)


def load_lotto_data(path: Path = LOTTO_XLSX_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise LottoDataError(
            f"로또 데이터 파일을 찾을 수 없습니다: {path}\n"
            "data/lotto.xlsx 파일을 프로젝트의 data 폴더에 넣어주세요."
        )

    try:
        df = pd.read_excel(path)
    except Exception as exc:
        raise LottoDataError(f"로또 데이터 파일을 읽을 수 없습니다: {path}: {exc}") from exc

    missing = [col for col in [ROUND_COLUMN, *NUMBER_COLUMNS] if col not in df.columns]
    if missing:
        raise LottoDataError(f"필수 컬럼이 없습니다: {', '.join(missing)}")

    df = df.copy()
    df[ROUND_COLUMN] = _strict_integer_series(df[ROUND_COLUMN], ROUND_COLUMN)
    for col in NUMBER_COLUMNS:
        df[col] = _strict_integer_series(df[col], col)

    df = df.sort_values(ROUND_COLUMN).reset_index(drop=True)
    validate_lotto_data(df)
    return df


def row_numbers(row) -> list[int]:
    return sorted(int(row[col]) for col in NUMBER_COLUMNS)


def validate_lotto_data(df: pd.DataFrame) -> None:
    if df.empty:
        raise LottoDataError("로또 데이터가 비어 있습니다.")

    duplicate_rounds = df[df[ROUND_COLUMN].duplicated()][ROUND_COLUMN].tolist()
    if duplicate_rounds:
        raise LottoDataError(f"중복 회차가 있습니다: {duplicate_rounds[:10]}")

    rounds = [int(value) for value in df[ROUND_COLUMN].tolist()]
    if rounds[0] != 1:
        raise LottoDataError(f"데이터는 1회부터 연속되어야 합니다. 첫 회차: {rounds[0]}")
    expected = list(range(1, rounds[-1] + 1))
    if rounds != expected:
        present = set(rounds)
        missing = [round_no for round_no in expected if round_no not in present]
        raise LottoDataError(
            "회차가 연속적이지 않습니다. "
            f"누락 회차: {missing[:20]}{' ...' if len(missing) > 20 else ''}"
        )

    for _, row in df.iterrows():
        nums = row_numbers(row)
        round_no = int(row[ROUND_COLUMN])
        if len(set(nums)) != 6:
            raise LottoDataError(f"{round_no}회 번호에 중복이 있습니다: {nums}")
        if min(nums) < 1 or max(nums) > 45:
            raise LottoDataError(f"{round_no}회 번호가 1~45 범위를 벗어났습니다: {nums}")


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """Stable SHA-256 for the normalized draw history used by a run."""
    validate_lotto_data(df)
    digest = hashlib.sha256()
    for _, row in df.iterrows():
        round_no = int(row[ROUND_COLUMN])
        nums = row_numbers(row)
        digest.update(f"{round_no}:{','.join(map(str, nums))}\n".encode("ascii"))
    return digest.hexdigest()
