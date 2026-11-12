"""KED 기업 자료에서 사업자번호별 네 가지 속성을 뽑는다.

원자료는 `|`로 구분하고 CP949로 저장된 머리 줄 없는 파일이며, 한 줄에 48개 항목(구분자 47개)이 있다.
18번 항목(0부터 센 번호)은 개인사업자의 주민등록번호이므로 읽지 않는다. 이를 지키기 위해
  1. 읽을 항목 번호에 18번을 넣지 않고(모듈을 불러올 때 확인한다),
  2. 구분자 수가 47개가 아닌 줄은 항목이 밀려 18번 값이 다른 항목 자리로 올 수 있으므로,
     글자로 바꾸기 전에 바이트 상태에서 줄 전체를 버리고,
  3. 줄 내용을 예외 메시지, 로그, 화면 출력에 쓰지 않는다.
"""
import csv
import io
from pathlib import Path

import pandas as pd

N_FIELDS = 48
FORBIDDEN_INDEX = 18
FIELDS = {"ked_size": 5, "ked_venture": 12, "estab_date": 19, "ked_ipo": 20, "info_date": 31, "bzno": 32}

assert FORBIDDEN_INDEX not in FIELDS.values()

_DELIMITER = b"|"
_ESTAB_YEAR_RANGE = (1800, 2026)
_CODE_COLUMNS = ["ked_size", "ked_ipo", "ked_venture"]
_OUT_COLUMNS = ["bzno", "ked_estab_year", *_CODE_COLUMNS]


def _kept_lines(path: Path):
    """구분자가 정확히 47개인 줄만 줄 끝 문자를 뗀 바이트로 내보낸다. 나머지 줄은 내용을 보지 않고 버린다."""
    with open(path, "rb") as f:
        for raw in f:
            line = raw.rstrip(b"\r\n")
            if line.count(_DELIMITER) == N_FIELDS - 1:
                yield line


def _chunks(path: Path, chunksize: int):
    buf = []
    for line in _kept_lines(path):
        buf.append(line)
        if len(buf) >= chunksize:
            yield buf
            buf = []
    if buf:
        yield buf


def _parse(lines: list) -> pd.DataFrame:
    df = pd.read_csv(
        io.BytesIO(b"\n".join(lines) + b"\n"),
        sep="|",
        header=None,
        names=range(N_FIELDS),
        usecols=sorted(FIELDS.values()),
        dtype=str,
        encoding="cp949",
        encoding_errors="replace",
        quoting=csv.QUOTE_NONE,
        keep_default_na=False,
        na_values=[""],
    )
    df = df.rename(columns={idx: name for name, idx in FIELDS.items()})
    return df[df["bzno"].str.fullmatch(r"\d{10}", na=False)]


def _latest_per_firm(df: pd.DataFrame) -> pd.DataFrame:
    """같은 사업자번호가 여러 행이면 정보일자가 가장 큰 행만 남긴다."""
    df = df.sort_values("info_date", kind="stable", na_position="first")
    return df.drop_duplicates("bzno", keep="last")


def extract_ked_firm(path: Path, chunksize: int = 500_000) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"KED 기업 자료 파일이 없습니다: {path}")

    parts = [_latest_per_firm(_parse(lines)) for lines in _chunks(path, chunksize)]
    if parts:
        df = _latest_per_firm(pd.concat(parts, ignore_index=True))
    else:
        df = pd.DataFrame({c: pd.Series(dtype=str) for c in FIELDS})

    is_date = df["estab_date"].str.fullmatch(r"\d{8}", na=False)
    year = pd.to_numeric(df["estab_date"].where(is_date).str[:4], errors="coerce")
    lo, hi = _ESTAB_YEAR_RANGE
    year = year.where((year >= lo) & (year <= hi))
    df = df.assign(ked_estab_year=year.astype("Int16"))

    return df.sort_values("bzno").reset_index(drop=True)[_OUT_COLUMNS]
