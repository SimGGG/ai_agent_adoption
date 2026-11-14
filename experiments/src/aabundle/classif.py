"""산업(한국표준산업분류 10차)과 직종(한국고용직업분류 2025) 코드를 정리하는 함수.

분류표 파일은 엑셀이며, 모두 문자열로 읽는다(앞자리 0이 사라지지 않게 하기 위해서다).
"""
from pathlib import Path

import pandas as pd

KSIC_NOT_IN_TABLE = "분류표에 없음"

_HEADER_ROWS = 3


def _read_sheet(path: Path) -> pd.DataFrame:
    """첫 시트를 머리 3줄을 건너뛰고 문자열로 읽는다. 파일이 없으면 파일 이름을 적은 예외를 낸다."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"분류표 파일이 없습니다: {path.name} (찾은 위치: {path.parent})")
    raw = pd.read_excel(path, sheet_name=0, header=None, dtype=str)
    return raw.iloc[_HEADER_ROWS:].reset_index(drop=True)


def load_ksic_sub(path: Path) -> set[str]:
    """산업분류표에서 소분류 3자리 코드의 집합을 돌려준다(열 4 중 정확히 숫자 3자리인 값)."""
    table = _read_sheet(path)
    if table.shape[1] <= 4:
        raise KeyError(f"{Path(path).name}에 소분류 코드 열(열 4)이 없습니다. 열 수: {table.shape[1]}")
    codes = table[4].dropna().astype(str).str.strip()
    return set(codes[codes.str.fullmatch(r"\d{3}")])


def load_keco(path: Path) -> pd.DataFrame:
    """직종분류표에서 세분류 4자리 코드와 중분류 코드·이름을 돌려준다.

    열은 keco4, keco4_name, keco2, keco2_name이다. 중분류 칸은 묶음의 첫 줄에만
    값이 있으므로 아래로 채우고, 세분류 코드가 정확히 숫자 4자리인 줄만 남겨
    맨 아래 합계 줄을 뺀다.
    """
    table = _read_sheet(path)
    if table.shape[1] <= 7:
        raise KeyError(f"{Path(path).name}에 필요한 열(열 2, 3, 6, 7)이 없습니다. 열 수: {table.shape[1]}")
    out = pd.DataFrame({
        "keco4": table[6],
        "keco4_name": table[7],
        "keco2": table[2].ffill(),
        "keco2_name": table[3].ffill(),
    })
    out["keco4"] = out["keco4"].astype("string").str.strip()
    out = out[out["keco4"].str.fullmatch(r"\d{4}").fillna(False).astype(bool)]
    return out.astype(object).reset_index(drop=True)


def normalize_ksic(s: pd.Series) -> pd.Series:
    """산업 코드를 숫자 5자리 문자열로 맞춘다. 변환할 수 없으면 결측으로 둔다.

    대문자 한 글자 + 숫자 5자리는 글자를 떼고, 숫자 4자리는 앞에 0을 붙이고,
    숫자 5자리는 그대로 둔다. 소문자 글자는 받아들이지 않는다.
    """
    t = s.astype("string")
    letter5 = t.str.extract(r"^[A-Z](\d{5})$")[0]
    digits4 = t.str.extract(r"^(\d{4})$")[0].radd("0")
    digits5 = t.str.extract(r"^(\d{5})$")[0]
    return digits5.fillna(letter5).fillna(digits4)


def ksic_sub(s: pd.Series, valid: set[str], not_in_table: str = "category") -> pd.Series:
    """산업 코드의 소분류 3자리를 돌려준다.

    분류표(valid)에 없는 소분류는 `not_in_table`이 `"category"`이면 KSIC_NOT_IN_TABLE로,
    `"missing"`이면 결측으로 둔다(설계 문서 결정 28). 원래 결측이거나 변환할 수 없는 코드는 결측으로 둔다.
    """
    if not_in_table not in ("category", "missing"):
        raise ValueError(f"not_in_table은 'category'나 'missing'이어야 합니다: {not_in_table!r}")
    sub = normalize_ksic(s).str[:3]
    in_table = sub.isna() | sub.isin(valid)
    return sub.where(in_table, KSIC_NOT_IN_TABLE if not_in_table == "category" else pd.NA)


def keco_mid(s: pd.Series) -> pd.Series:
    """직종 세분류 코드의 앞 2자리(중분류)를 돌려준다. 결측은 결측으로 둔다."""
    return s.astype("string").str[:2]
