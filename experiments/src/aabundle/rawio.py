"""공고 파일(parquet)을 행 묶음 단위로 읽고, 중간 산출물을 쓰는 함수.

원자료가 있는 `data` 폴더에는 아무것도 쓰지 않는다.
"""
from pathlib import Path
from typing import Iterator

import pandas as pd
import pyarrow.parquet as pq


def posting_path(postings_dir: Path, year: int) -> Path:
    """연도별 공고 파일 경로 `pfile_{year}.parquet`를 돌려준다."""
    return Path(postings_dir) / f"pfile_{year}.parquet"


def iter_postings(postings_dir: Path, year: int, columns: list[str]) -> Iterator[pd.DataFrame]:
    """공고 파일에서 지정한 열만, 행 묶음(row group) 하나씩 읽어 돌려준다.

    파일이 없으면 파일 이름을 적은 FileNotFoundError를, 파일에 없는 열이 있으면
    그 열 이름을 적은 KeyError를 낸다.
    """
    path = posting_path(postings_dir, year)
    if not path.exists():
        raise FileNotFoundError(f"공고 파일이 없습니다: {path.name} (찾은 위치: {path.parent})")
    pf = pq.ParquetFile(path)
    missing = [c for c in columns if c not in pf.schema_arrow.names]
    if missing:
        raise KeyError(f"{path.name}에 없는 열: {missing}")
    for i in range(pf.num_row_groups):
        yield pf.read_row_group(i, columns=columns).to_pandas()


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    """상위 폴더를 만들고 parquet로 쓴다. 경로에 `data` 폴더가 있으면 ValueError를 낸다."""
    path = Path(path)
    if "data" in path.parts:
        raise ValueError(f"원자료 폴더(data) 아래에는 쓸 수 없습니다: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """parquet 파일을 읽는다. 파일이 없으면 파일 이름을 적은 FileNotFoundError를 낸다."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"파일이 없습니다: {path}")
    return pd.read_parquet(path, columns=columns)
