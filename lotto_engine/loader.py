from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import LOTTO_XLSX_PATH, NUMBER_COLUMNS, ROUND_COLUMN


class LottoDataError(Exception):
    pass


def load_lotto_data(path: Path = LOTTO_XLSX_PATH) -> pd.DataFrame:
    if not path.exists():
        raise LottoDataError(
            f"로또 데이터 파일을 찾을 수 없습니다: {path}\n"
            "data/lotto.xlsx 파일을 프로젝트의 data 폴더에 넣어주세요."
        )

    df = pd.read_excel(path)
    missing = [col for col in [ROUND_COLUMN, *NUMBER_COLUMNS] if col not in df.columns]
    if missing:
        raise LottoDataError(f"필수 컬럼이 없습니다: {', '.join(missing)}")

    df = df.copy()
    df[ROUND_COLUMN] = pd.to_numeric(df[ROUND_COLUMN], errors="raise").astype(int)
    for col in NUMBER_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)

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

    for _, row in df.iterrows():
        nums = row_numbers(row)
        round_no = int(row[ROUND_COLUMN])
        if len(set(nums)) != 6:
            raise LottoDataError(f"{round_no}회 번호에 중복이 있습니다: {nums}")
        if min(nums) < 1 or max(nums) > 45:
            raise LottoDataError(f"{round_no}회 번호가 1~45 범위를 벗어났습니다: {nums}")
