"""KED 기업 자료에서 사업자번호별 네 가지 속성을 뽑는다.

원자료는 `|`로 구분하고 CP949로 저장된 머리 줄 없는 파일이며, 한 줄에 48개 항목(구분자 47개)이 있다.
18번 항목(0부터 센 번호)은 개인사업자의 주민등록번호이므로 읽지 않는다. 이를 지키기 위해
  1. 읽을 항목 번호에 18번을 넣지 않고(모듈을 불러올 때 확인한다),
  2. 줄 끝 문자를 뗀 뒤에도 캐리지 리턴 문자(CR)가 남아 있거나 구분자 수가 47개가 아닌 줄은 항목이 밀리거나 둘로 갈라져
     18번 값이 다른 항목 자리로 올 수 있으므로, 글자로 바꾸기 전에 바이트 상태에서 줄 전체를 버리고,
  3. 줄을 나누는 일과 항목을 나누는 일을 이 모듈이 한 번씩만 하며(표 읽기 도구에 맡기지 않는다),
     항목 수를 확인한 바로 그 나눔에서 필요한 여섯 항목만 글자로 바꾸고,
  4. 줄 내용을 예외 메시지, 로그, 화면 출력에 쓰지 않는다. 줄 수만 `attrs`에 남긴다.
"""
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


def _split_line(raw: bytes):
    """한 줄(바이트)을 항목 48개로 나눈 목록을 돌려준다. 버려야 하는 줄이면 None을 돌려준다.

    버리는 경우: 줄 끝 문자를 뗀 뒤에도 `\r`가 있는 줄, 구분자가 47개가 아닌 줄.
    """
    if raw.endswith(b"\r\n"):
        line = raw[:-2]
    elif raw.endswith(b"\n"):
        line = raw[:-1]
    else:
        line = raw
    if b"\r" in line or line.count(_DELIMITER) != N_FIELDS - 1:
        return None
    return line.split(_DELIMITER)


def _decode(part: bytes):
    text = part.decode("cp949", errors="replace").strip()
    return text or None


def _chunks(path: Path, chunksize: int, counts: dict):
    """읽을 여섯 항목만 글자로 바꾼 행을 `chunksize`줄씩 모아 내보낸다."""
    idx = list(FIELDS.values())
    buf = []
    with open(path, "rb") as f:
        for raw in f:
            counts["n_lines"] += 1
            parts = _split_line(raw)
            if parts is None:
                counts["n_dropped_lines"] += 1
                continue
            buf.append([_decode(parts[i]) for i in idx])
            if len(buf) >= chunksize:
                yield buf
                buf = []
    if buf:
        yield buf


def _to_frame(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=list(FIELDS), dtype="string")
    return df[df["bzno"].str.fullmatch(r"\d{10}", na=False)]


def _latest_per_firm(df: pd.DataFrame) -> pd.DataFrame:
    """같은 사업자번호가 여러 행이면 정보일자가 가장 큰 행만 남긴다."""
    df = df.sort_values("info_date", kind="stable", na_position="first")
    return df.drop_duplicates("bzno", keep="last")


def extract_ked_firm(path: Path, chunksize: int = 500_000) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"KED 기업 자료 파일이 없습니다: {path}")

    counts = {"n_lines": 0, "n_dropped_lines": 0}
    parts = [_latest_per_firm(_to_frame(rows)) for rows in _chunks(path, chunksize, counts)]
    if parts:
        df = _latest_per_firm(pd.concat(parts, ignore_index=True))
    else:
        df = pd.DataFrame({c: pd.Series(dtype="string") for c in FIELDS})

    is_date = df["estab_date"].str.fullmatch(r"\d{8}", na=False)
    year = pd.to_numeric(df["estab_date"].where(is_date).str[:4], errors="coerce")
    lo, hi = _ESTAB_YEAR_RANGE
    year = year.where((year >= lo) & (year <= hi))
    df = df.assign(ked_estab_year=year.astype("Int16"))

    out = df.sort_values("bzno").reset_index(drop=True)[_OUT_COLUMNS]
    out.attrs.update(counts)
    return out
