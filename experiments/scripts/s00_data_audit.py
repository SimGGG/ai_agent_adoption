"""자료 점검(S0) 스크립트.

1단계 `scan`: 연도별 공고 파일을 행 묶음(row group) 단위로 한 번 훑어, 공고 단위 중간 자료와
작은 집계 자료, 1단계 기록(`scan_meta.json`: 실행 시각, 코드 시점, 1단계의 결과를 바꾸는 설정 값)을
`out_dir`에 쓴다. 공고 원문 열은 처치 판별, 키워드 표현별 공고 수, 본문 글자 수를 세는 데에만 메모리에서 쓰고,
어떤 파일과 화면 출력에도 남기지 않는다(글자 수만 `base`에 쓴다).

2단계 `report`: 1단계 기록의 설정 값이 지금 설정과 같은지 확인한 뒤, 1단계의 중간 자료로 집계표
(`build_tables`), 결과모형 한 번의 학습 시간(`time_outcome_model`), 자료 점검 보고서(`write_report`)를
만든다. 집계표에는 공고 번호, 사업자번호, 공고 원문을 넣지 않는다.

실행: 저장소 최상위 폴더에서 `python experiments/scripts/s00_data_audit.py scan|report|all`
"""
import argparse
import datetime as dt
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/ 폴더의 paths를 불러오기 위해서다

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

import paths
from aabundle import classif, confounders, ked, rawio, sample, skills, treatment, validate

OUTPUT_NAMES = ("occ_year_all", "base", "treat", "skills_long", "scan_stats", "kw_pattern_hits",
                "post_start_missing", "sido_other_words", "post_end_gap")
# 날짜를 채운 공고(설계 문서 결정 38)를 앞뒤 날짜의 간격 구간별로 센 중간 자료. 설정 `period.fill.enabled`가 참일
# 때만 쓰고, 거짓이면 쓰지 않는다(이전 실행의 파일은 지운다).
FILL_OUTPUT_NAME = "date_fill"
# 1단계를 실행한 때, 코드 시점, 1단계의 결과를 바꾸는 설정 항목을 적은 기록(JSON). 위 파일들과 함께 쓰고 지운다.
META_NAME = "scan_meta"
# 1단계 기록에 남기는 설정 항목. `scan`이 읽어 결과가 달라지는 항목만이다. 나머지 항목(`data.keco`, `period.unit`,
# `period.fill.max_gap_days`, `occupations.min_use_postings`, `occupations.exclude_keco2`,
# `occupations.group_min_treated`, `sample`, `treatment.definition`, `treatment.keyword_dictionary`, `skills`,
# `estimation` 등)은 2단계(`report`)에서만 쓰므로, 바꾼 뒤 `report`만 다시 실행하면 된다.
SCAN_CONFIG_KEYS = ("data.years", "data.codebook", "data.ksic", "data.ked_firm", "period.start", "period.end",
                    "period.date_columns", "period.groups", "period.fill.enabled", "occupations.keco4",
                    "industry.not_in_table", "region.merge_sido", "treatment.text_columns", "treatment.keywords",
                    "treatment.agent_skills", "salary_band_edges")
# 마감일에서 게시일을 뺀 일수의 구간(`post_end_gap`): (구간 이름, 아래 끝, 위 끝). 두 끝을 포함하고, None은 끝이 없다는 뜻이다.
POST_END_GAP_BINS = (("음수", None, -1), ("0~7일", 0, 7), ("8~30일", 8, 30), ("31~60일", 31, 60),
                     ("61~90일", 61, 90), ("91~180일", 91, 180), ("181일 이상", 181, None))
# 날짜를 채운 공고의 앞뒤 날짜 간격(`fill_gap_days`)의 구간(`date_fill`, `t01d`, `t02e`): (구간 이름, 아래 끝, 위 끝).
# 두 끝을 포함하고, None은 끝이 없다는 뜻이다. 뒤에 날짜가 있는 공고가 없어 간격이 결측이면 `FILL_GAP_NONE`이다.
FILL_GAP_BINS = (("7일 이내", 0, 7), ("8~30일", 8, 30), ("31~90일", 31, 90), ("91~365일", 91, 365),
                 ("365일 초과", 366, None))
FILL_GAP_NONE = "뒤 날짜 없음"
FILL_GAP_LABELS = tuple(label for label, _, _ in FILL_GAP_BINS) + (FILL_GAP_NONE,)
# 결과 파일의 이름을 바꿀 때 PermissionError가 나면 기다렸다가 다시 해 보는 횟수(첫 시도 포함)와 기다리는 시간(초).
# 구글 드라이브 동기화 프로그램이 파일을 잠깐 잡고 있으면 이름 바꾸기가 실패하기 때문이다.
RENAME_ATTEMPTS = 5
RENAME_WAIT_SECONDS = 2.0
# 시도가 `기타`인 공고의 근무지 첫 낱말 표에 남기는 낱말 수와 낱말의 최대 글자 수
SIDO_OTHER_WORDS_TOP = 200
SIDO_OTHER_WORD_MAX_CHARS = 20

_STR = pa.string()
_INT = pa.int64()
_BOOL = pa.bool_()

# 공고 기본 자료에서 새로 만든 열의 형식. 나머지 열은 원자료 열을 그대로 옮긴 것이라 원자료 파일의 형식을 따른다.
_BASE_DERIVED_TYPES = {
    "year": _INT, "month": _INT, "quarter": _STR, "period": _STR, "date_source": _STR, "fill_gap_days": _INT, "bzno": _STR,
    "ksic5": _STR, "ksic_sub": _STR, "firm_emp_log": pa.float64(), "keco4": _STR, "keco2": _STR, "salary_band": _STR, "sido": _STR,
    "ked_matched": _BOOL, "firm_age": _INT, "ked_size": _STR, "ked_ipo": _STR, "ked_venture": _STR,
    "in_period": _BOOL, "in_occ": _BOOL, "has_bzno": _BOOL, "n_skills": _INT, "text_len": _INT,
}
# `sample.BASE_COLUMNS` 뒤에 붙이는 열: 숙련 수, 본문(`treatment.text_columns`의 열을 이은 문자열)의 글자 수
_BASE_EXTRA_COLUMNS = ["n_skills", "text_len"]
_TREAT_FLAGS = ["kw_strict", "kw_broad", "kw_use", "has_dev", "has_use", "treat_code", "treat_all"]
# 키워드 사전마다, 표현별 공고 수를 셀 공고를 정하는 처치 표시 열. 표현별 공고 수는 그 표시가 참인 공고 안에서 센다.
_KW_COUNT_FLAG = {"strict": "kw_broad", "broad_extra": "kw_broad", "use": "kw_use"}
# 처치 표시 열마다, 표현별 공고 수를 셀 표현 목록(`treatment.keyword_lists`의 이름)
_KW_FLAG_PATTERNS = {"kw_broad": "broad", "kw_use": "use"}

OCC_SCHEMA = pa.schema([("year", _INT), ("keco4", _STR), ("n", _INT), ("n_skill", _INT),
                        ("n_use", _INT), ("n_dev", _INT)])
STATS_SCHEMA = pa.schema([("year", _INT), ("n_raw", _INT), ("n_in_period", _INT), ("n_in_occ", _INT),
                          ("n_tokens", _INT), ("n_unmatched", _INT), ("n_length_mismatch_rows", _INT)])
KW_SCHEMA = pa.schema([("year", _INT), ("dictionary", _STR), ("pattern", _STR), ("n", _INT)])
_PSM_COUNTS = ["n_raw", "n_no_post_start", "n_no_date", "n_end_unusable",
               "n_in_occ", "n_in_occ_no_post_start", "n_in_occ_no_date", "n_in_occ_end_unusable",
               "n_in_occ_from_end_in_period",
               "n_in_occ_use", "n_in_occ_use_no_post_start", "n_in_occ_use_no_date", "n_in_occ_use_from_end_in_period",
               "n_filled", "n_in_occ_filled", "n_in_occ_filled_in_period", "n_in_occ_fill_dates",
               "n_in_occ_use_filled_in_period"]
PSM_SCHEMA = pa.schema([("year", _INT), ("source", _STR)] + [(c, _INT) for c in _PSM_COUNTS])
WORDS_SCHEMA = pa.schema([("word", _STR), ("n", _INT)])
GAP_SCHEMA = pa.schema([("year", _INT), ("bin", _STR), ("n", _INT)])
_FILL_COUNTS = ["n", "n_in_occ", "n_in_occ_in_period"]
FILL_SCHEMA = pa.schema([("year", _INT), ("source", _STR), ("gap_bin", _STR)] + [(c, _INT) for c in _FILL_COUNTS])

_STAT_KEYS = ["n_tokens", "n_unmatched", "n_length_mismatch_rows"]


def _plain(t: pa.DataType) -> pa.DataType:
    """문자열 계열 형식은 모두 `string`으로 맞춘다."""
    if pa.types.is_string(t) or pa.types.is_large_string(t) or pa.types.is_string_view(t):
        return _STR
    return t


def _promote(types: list[pa.DataType]) -> pa.DataType | None:
    """여러 연도의 자료형을 하나로 맞춘다. 맞출 수 없으면 None을 돌려준다.

    null 형식(값이 하나도 없는 열)은 다른 연도의 형식을 따른다. 문자열 계열은 `string`으로,
    정수와 실수가 섞이면 `float64`로, 너비가 다른 정수만 있으면 `int64`로 맞춘다.
    """
    ts = [_plain(t) for t in types if not pa.types.is_null(t)]
    if all(t == ts[0] for t in ts):
        return ts[0]
    if all(pa.types.is_integer(t) or pa.types.is_floating(t) for t in ts):
        return pa.float64() if any(pa.types.is_floating(t) for t in ts) else _INT
    return None


def _passthrough_types(files: dict) -> dict[str, pa.DataType]:
    """원자료에서 그대로 옮기는 열의 자료형을 모든 연도 파일의 열 구성으로 정한다(훑기 전에 한 번).

    연도 간에 하나로 맞출 수 없는 열이나 모든 연도에서 null 형식인 열이 있으면, 열 이름과
    연도별 자료형을 적은 ValueError를 낸다.
    """
    out = {}
    for c in [c for c in sample.BASE_COLUMNS if c not in _BASE_DERIVED_TYPES]:
        by_year = {y: pf.schema_arrow.field(c).type for y, pf in files.items()}
        desc = ", ".join(f"{y}년 {t}" for y, t in by_year.items())
        if all(pa.types.is_null(t) for t in by_year.values()):
            raise ValueError(f"{c} 열은 모든 연도에서 값이 하나도 없는 형식(null)입니다: {desc}")
        t = _promote(list(by_year.values()))
        if t is None:
            raise ValueError(f"{c} 열의 자료형을 연도 간에 하나로 맞출 수 없습니다: {desc}")
        out[c] = t
    return out


def _schemas(passthrough: dict[str, pa.DataType]) -> dict[str, pa.Schema]:
    """공고 단위 세 파일의 열 구성을 정한다. 연도나 행 묶음에 따라 형식이 바뀌지 않게 하기 위해서다."""
    id_type = passthrough["posting_id"]
    base = [(c, _BASE_DERIVED_TYPES[c] if c in _BASE_DERIVED_TYPES else passthrough[c])
            for c in [*sample.BASE_COLUMNS, *_BASE_EXTRA_COLUMNS]]
    return {
        "base": pa.schema(base),
        "treat": pa.schema([("posting_id", id_type)] + [(c, _BOOL) for c in _TREAT_FLAGS]),
        "skills_long": pa.schema([("posting_id", id_type), ("skill_code", _STR), ("src", _STR)]),
    }


def _to_table(df: pd.DataFrame, schema: pa.Schema) -> pa.Table:
    if list(df.columns) != schema.names:
        raise RuntimeError(f"열 구성이 정해진 것과 다릅니다: {list(df.columns)} != {schema.names}")
    return pa.Table.from_pandas(df, schema=schema, preserve_index=False).replace_schema_metadata(None)


def _joined_text(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """원문 열들을 공백 하나로 잇는다. 결측인 열은 건너뛰고, 모두 결측이면 결측으로 둔다."""
    out = df[columns[0]].astype("string")
    for c in columns[1:]:
        part = df[c].astype("string")
        out = (out + " " + part).fillna(out).fillna(part)
    return out.reset_index(drop=True)


def _part_path(path: Path) -> Path:
    return path.with_name(path.name + ".part")


def _replace_with_retry(src: Path, dst: Path) -> None:
    """`src`의 이름을 `dst`로 바꾼다. PermissionError가 나면 `RENAME_WAIT_SECONDS`초 기다렸다가 다시 하고,
    `RENAME_ATTEMPTS`번 모두 실패하면 마지막 예외를 그대로 낸다."""
    for attempt in range(1, RENAME_ATTEMPTS + 1):
        try:
            src.replace(dst)
            return
        except PermissionError:
            if attempt == RENAME_ATTEMPTS:
                raise
            time.sleep(RENAME_WAIT_SECONDS)


def _now_iso() -> str:
    """지금 시각(이 컴퓨터의 시간대, 초 단위)을 ISO 형식으로 돌려준다."""
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _json_value(value):
    """설정 값을 JSON으로 썼다가 읽은 값으로 바꾼다. 기록한 값과 지금 값을 같은 형태로 비교하기 위해서다."""
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _config_at(cfg: dict, key: str):
    d = cfg
    for part in key.split("."):
        d = d[part]
    return d


def _scan_config(cfg: dict) -> dict:
    """설정에서 1단계의 결과를 바꾸는 항목(`SCAN_CONFIG_KEYS`)만 {항목 이름: 값}으로 뽑는다."""
    return {key: _json_value(_config_at(cfg, key)) for key in SCAN_CONFIG_KEYS}


def _differences(recorded, current, key: str) -> list[str]:
    """두 값이 다른 곳의 항목 이름을 돌려준다. 둘 다 사전이면 안쪽 항목까지 내려가 이름을 `.`으로 잇는다."""
    if isinstance(recorded, dict) and isinstance(current, dict):
        out = []
        for k in list(recorded) + [k for k in current if k not in recorded]:
            out += _differences(recorded.get(k), current.get(k), f"{key}.{k}")
        return out
    return [] if recorded == current else [key]


def _empty_frame(schema: pa.Schema) -> pd.DataFrame:
    return pd.DataFrame({f.name: pd.Series([], dtype=f.type.to_pandas_dtype()) for f in schema})


def _post_start_counts(raw: pd.DataFrame, b: pd.DataFrame, has_use: np.ndarray,
                       date_columns: list[str]) -> pd.DataFrame:
    """행 묶음 하나의 모든 공고를 수집 사이트별로 세어, 게시일과 관련된 공고 수를 함께 돌려준다.

    `b`는 `sample.build_posting_base`가 `raw`로 만든 기본 자료(같은 순서)이고, 그 `date_source`, `in_period`,
    `in_occ`를 쓴다. `has_use`는 생성형 AI 활용 숙련 보유 여부이며 `raw`와 위치로 맞춘다. `date_columns`는
    설정 `period.date_columns`이다. 세는 공고는 다음과 같다(`n_in_occ_*`는 그 가운데 분석 대상 세분류의 공고,
    `n_in_occ_use_*`는 다시 그 가운데 생성형 AI 활용 숙련이 있는 공고).
    - `no_post_start`: 첫 날짜 열에 게시일로 쓸 날짜가 없는 공고(비었거나 읽을 수 없는 값)
    - `no_date`: `date_columns`의 어느 열에도 게시일로 쓸 날짜가 없는 공고(날짜를 채우지 않는 설정에서는 분석 기간
      단계에서 빠진다. 채우는 설정에서는 그 가운데 파일에서 앞에 날짜 있는 공고가 없어 채울 날짜가 없는 공고만 빠진다)
    - `end_unusable`: 첫 날짜 열에 날짜가 없고, 둘째 날짜 열에 값은 있지만 날짜로 쓸 수 없는 공고
      (`sample.unusable_dates`. 날짜 열이 하나뿐이면 0)
    - `from_end_in_period`: 첫 날짜 열에 날짜가 없어 뒤 열의 날짜를 게시일로 썼고, 그 날짜가 분석 기간 안인 공고
    - `filled`: `no_date` 가운데 같은 연도 파일에서 행 순서로 가장 가까운 앞 공고의 날짜를 게시일로 받은 공고
      (`date_source` = `fill`, 설계 문서 결정 38). `filled_in_period`는 그 날짜가 분석 기간 안인 공고이다.

    `no_date`는 날짜를 채우기 전, 공고 자신의 날짜 열로 게시일을 정할 수 없는 공고이다. 그래서 `n_filled <=
    n_no_date`이고, 차이는 파일의 첫 날짜 있는 공고보다 앞에 있어 받을 날짜가 없는 공고이다.
    `n_in_occ_fill_dates`(날짜를 채운 분석 대상 세분류 공고가 받은 서로 다른 날짜의 수)는 행 묶음 사이에 더할 수
    없으므로 여기서는 0으로 두고, `scan`이 연도 파일 하나를 다 읽은 뒤 수집 사이트별로 채운다.

    `end_unusable`은 둘째 날짜 열만 본다. 그래서 `n_end_unusable <= n_no_date`(둘째 열의 값을 날짜로 쓸 수 없으면
    게시일을 정할 수 없다)는 날짜 열이 둘 이하일 때만 성립한다. 셋째 열이 있으면 그 열의 날짜를 게시일로 쓴 공고도
    `end_unusable`에 들 수 있다.
    """
    date_source = b["date_source"]
    filled = (date_source == sample.FILL_SOURCE).fillna(False).to_numpy(dtype=bool)
    no_date = date_source.isna().to_numpy(dtype=bool) | filled
    no_first = (date_source != date_columns[0]).fillna(True).to_numpy(dtype=bool)
    if len(date_columns) > 1:
        end_unusable = no_first & sample.unusable_dates(raw, date_columns[1:2])[date_columns[1]].to_numpy(dtype=bool)
    else:
        end_unusable = np.zeros(len(raw), dtype=bool)
    in_period = b["in_period"].to_numpy(dtype=bool)
    from_end_in_period = no_first & ~no_date & in_period
    in_occ = b["in_occ"].to_numpy(dtype=bool)
    occ_use = in_occ & has_use
    df = pd.DataFrame({
        "source": _source_labels(raw),
        "n_raw": np.ones(len(raw), dtype="int64"),
        "n_no_post_start": no_first.astype("int64"),
        "n_no_date": no_date.astype("int64"),
        "n_end_unusable": end_unusable.astype("int64"),
        "n_in_occ": in_occ.astype("int64"),
        "n_in_occ_no_post_start": (in_occ & no_first).astype("int64"),
        "n_in_occ_no_date": (in_occ & no_date).astype("int64"),
        "n_in_occ_end_unusable": (in_occ & end_unusable).astype("int64"),
        "n_in_occ_from_end_in_period": (in_occ & from_end_in_period).astype("int64"),
        "n_in_occ_use": occ_use.astype("int64"),
        "n_in_occ_use_no_post_start": (occ_use & no_first).astype("int64"),
        "n_in_occ_use_no_date": (occ_use & no_date).astype("int64"),
        "n_in_occ_use_from_end_in_period": (occ_use & from_end_in_period).astype("int64"),
        "n_filled": filled.astype("int64"),
        "n_in_occ_filled": (in_occ & filled).astype("int64"),
        "n_in_occ_filled_in_period": (in_occ & filled & in_period).astype("int64"),
        "n_in_occ_fill_dates": np.zeros(len(raw), dtype="int64"),
        "n_in_occ_use_filled_in_period": (occ_use & filled & in_period).astype("int64"),
    })
    return df.groupby("source", as_index=False).sum()


def _source_labels(raw: pd.DataFrame) -> np.ndarray:
    return raw["source"].astype("string").fillna(MISSING_LABEL).to_numpy()


def _fill_gap_bin(gap: pd.Series) -> np.ndarray:
    """앞뒤 날짜의 간격(일)을 구간 이름(`FILL_GAP_BINS`, 결측은 `FILL_GAP_NONE`)으로 바꾼다."""
    g = pd.to_numeric(gap).astype("Float64")
    out = np.full(len(g), FILL_GAP_NONE, dtype=object)
    for label, low, high in FILL_GAP_BINS:
        hit = g.notna()
        if low is not None:
            hit &= g >= low
        if high is not None:
            hit &= g <= high
        out[hit.fillna(False).to_numpy(dtype=bool)] = label
    return out


def _date_fill_counts(raw: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """행 묶음 하나에서 날짜를 채운 공고(`date_source` = `fill`)를 수집 사이트 × 앞뒤 날짜 간격의 구간별로 센다.
    `b`는 `raw`로 만든 기본 자료(같은 순서)이다. 열: `source`, `gap_bin`, `n`, `n_in_occ`, `n_in_occ_in_period`."""
    filled = (b["date_source"] == sample.FILL_SOURCE).fillna(False).to_numpy(dtype=bool)
    in_occ = b["in_occ"].to_numpy(dtype=bool)[filled]
    df = pd.DataFrame({"source": _source_labels(raw)[filled], "gap_bin": _fill_gap_bin(b["fill_gap_days"][filled]),
                       "n": 1, "n_in_occ": in_occ.astype("int64"),
                       "n_in_occ_in_period": (in_occ & b["in_period"].to_numpy(dtype=bool)[filled]).astype("int64")})
    return df.groupby(["source", "gap_bin"], as_index=False)[_FILL_COUNTS].sum()


def _date_fill_table(pieces: list[pd.DataFrame]) -> pd.DataFrame:
    """연도별 `_date_fill_counts`의 합을 `date_fill`의 열 구성으로 맞추고, 연도, 수집 사이트, 간격 구간
    (`FILL_GAP_LABELS`의 순서)으로 정렬한다. 조각이 없으면 열만 있는 빈 표이다."""
    pieces = [p for p in pieces if not p.empty]
    if not pieces:
        return _empty_frame(FILL_SCHEMA)
    out = pd.concat(pieces, ignore_index=True)[FILL_SCHEMA.names]
    order = out["gap_bin"].map({b: i for i, b in enumerate(FILL_GAP_LABELS)})
    return (out.assign(_o=order).sort_values(["year", "source", "_o"], kind="stable")
            .drop(columns="_o").reset_index(drop=True))


def _file_fill(pf: pq.ParquetFile, date_columns: list[str]) -> pd.DataFrame:
    """연도 파일 하나의 날짜 열만 모든 행에서 읽어, 행마다 채울 날짜와 앞뒤 날짜의 간격(`sample.fill_posting_dates`)을
    파일의 행 순서대로 돌려준다(열 `fill_date`, `fill_gap_days`, 새 RangeIndex)."""
    dates = pf.read(columns=date_columns).to_pandas()
    own, _ = sample.posting_date(dates, date_columns)
    fill_date, gap = sample.fill_posting_dates(own.reset_index(drop=True))
    return pd.DataFrame({"fill_date": fill_date, "fill_gap_days": gap})


def _post_end_gap_counts(raw: pd.DataFrame, in_occ: np.ndarray, date_columns: list[str]) -> dict[str, int]:
    """분석 대상 세분류의 공고(`in_occ`, `raw`와 위치로 맞춘다) 가운데 `date_columns`의 첫 열과 둘째 열이 모두
    날짜로 읽히는 공고를, 둘째 날짜에서 첫 날짜를 뺀 일수의 구간(`POST_END_GAP_BINS`)별로 센다.

    {구간 이름: 공고 수}를 돌려주며, 공고가 없는 구간도 0으로 넣는다. `date_columns`에 열이 둘 이상 있어야 한다.
    """
    first, _ = sample.posting_date(raw, date_columns[:1])
    second, _ = sample.posting_date(raw, date_columns[1:2])
    both = in_occ & first.notna().to_numpy(dtype=bool) & second.notna().to_numpy(dtype=bool)
    days = (second - first).dt.days.to_numpy()[both]
    out = {}
    for label, low, high in POST_END_GAP_BINS:
        hit = np.ones(len(days), dtype=bool)
        if low is not None:
            hit &= days >= low
        if high is not None:
            hit &= days <= high
        out[label] = int(hit.sum())
    return out


def scan(cfg: dict, root: Path, out_dir: Path) -> dict:
    """원자료를 한 번 훑어 중간 자료 아홉 파일(`OUTPUT_NAMES`)과 1단계 기록(`scan_meta.json`)을 `out_dir`에 쓰고
    {이름: 경로}를 돌려준다.

    `base`의 `text_len`은 본문(`treatment.text_columns`의 열을 공백 하나로 이은 문자열)의 글자 수이고, 본문이
    없으면 0이다. `occ_year_all`의 `year`는 게시일(`sample.posting_date`)의 연도이며, 게시일을 정할 수 없으면 파일
    연도이다. `kw_pattern_hits`는 키워드 사전(`strict`, `broad_extra`, `use`)의 표현마다 걸린 공고 수를 게시 연도별로
    센 표이다. `strict`와 `broad_extra`는 `kw_broad`가 참인 공고 안에서, `use`는 `kw_use`가 참인 공고 안에서
    제외 표현을 지운 뒤 센다. `post_start_missing`은 공고 파일의 모든 공고를 파일 연도 × 수집 사이트별로 세어
    게시일과 관련된 공고 수(`_post_start_counts`)를 적은 표이다. `post_end_gap`은 분석 대상 세분류의 공고
    가운데 `period.date_columns`의 첫 열과 둘째 열이 모두 날짜로 읽히는 공고를 파일 연도 × 두 날짜의 간격
    구간(`POST_END_GAP_BINS`, 공고가 없는 구간도 0)별로 센 표이고, 날짜 열이 하나뿐이면 빈 표이다.

    설정 `period.fill.enabled`가 참이면 연도 파일마다 행 묶음을 읽기 전에 날짜 열만 파일 전체에서 읽어, 두 날짜가 모두
    없는 공고에 같은 파일에서 행 순서로 가장 가까운 앞 공고의 날짜를 채운다(설계 문서 결정 38, `sample.fill_posting_dates`).
    행 묶음마다 파일 안의 행 위치로 자른 부분을 `sample.build_posting_base`에 넘기므로, 앞 공고가 앞 행 묶음에 있어도
    날짜를 받는다. 날짜 열로 읽은 행 수와 행 묶음으로 읽은 행 수가 다르면 RuntimeError를 낸다. 이때 `date_fill`
    (`FILL_OUTPUT_NAME`)도 쓴다. 날짜를 채운 공고를 파일 연도 × 수집 사이트 × 앞뒤 날짜 간격의 구간(`FILL_GAP_BINS`,
    `FILL_GAP_NONE`)별로 센 표이며, 공고가 있는 조합만 행이 있다(채운 공고가 없으면 빈 표). 거짓이면 날짜 열을 따로
    읽지 않고 `date_fill`도 쓰지 않는다.
    `sido_other_words`는 `base`에 들어간 공고 가운데 시도가 `기타`인 공고의 근무지 첫 낱말
    (`sample.region_first_word`, 20자까지)별 공고 수를 많은 순으로 200개까지 적은 표이다. 근무지 열 자체는
    `base`에 쓰지 않는다. `scan_meta.json`에는 실행을 시작하고 끝낸 시각, 실행을 시작할 때의 git 커밋과 git에 기록된
    파일에 커밋하지 않은 변경이 있었는지, 설정 가운데 `SCAN_CONFIG_KEYS` 항목의 값만 적는다(공고 단위 값은 적지 않는다).

    설정을 확인하고(`validate.validate_config`) 입력을 확인한 뒤 훑기 전에, `out_dir`에 있는 이 함수의 결과 파일
    (`date_fill` 포함 열한 개)과 그 `.part` 파일을 지운다(폴더의 다른 파일은 건드리지 않는다). 각 파일은 먼저 `.part`가 붙은
    이름으로 쓰고, 모든 연도를 끝까지 처리한 뒤에 정해진 이름으로 바꾼다(PermissionError가 나면 몇 번 다시
    해 본다). 이름 바꾸기까지 어느 단계에서든 멈추면 이번 실행의 결과 파일과 `.part` 파일을 모두 지우고
    예외를 다시 낸다. 그래서 `out_dir`에 남은 결과 파일은 끝까지 처리된 한 번의 실행에서 나온 것뿐이다.
    """
    validate.validate_config(cfg)
    # 결과를 만드는 코드는 실행을 시작할 때 읽혔으므로, 코드 시점도 시작할 때 한 번 적는다(훑는 도중에 커밋이 생길 수 있다)
    started = _now_iso()
    git_commit, git_changed = _git_commit(), _git_uncommitted_changes()
    root, out_dir = Path(root), Path(out_dir)
    years = [int(y) for y in cfg["data"]["years"]]
    postings_dir = root / "data" / "postings"
    text_cols = list(cfg["treatment"]["text_columns"])
    date_cols = list(cfg["period"]["date_columns"])
    # 설정의 날짜 열도 읽는다. `sample.RAW_COLUMNS`에 이미 있는 열이면 열 목록과 순서는 바뀌지 않는다.
    read_cols = list(dict.fromkeys(sample.RAW_COLUMNS + date_cols + skills.raw_skill_columns()))
    if not text_cols:
        raise ValueError("cfg['treatment']['text_columns']가 비어 있습니다")

    # 오래 훑기 전에 모든 연도 파일과 필요한 열이 있는지 확인한다.
    files = {y: rawio.open_postings(postings_dir, y, read_cols + text_cols) for y in years}
    schemas = _schemas(_passthrough_types(files))

    ked_firm = ked.extract_ked_firm(root / cfg["data"]["ked_firm"])
    ksic_valid = classif.load_ksic_sub(root / cfg["data"]["ksic"])
    codebook = skills.load_codebook(root / cfg["data"]["codebook"])
    codes = treatment.agent_skill_codes(codebook, cfg)
    kws = treatment.keyword_lists(cfg)
    kw_cfg = cfg["treatment"]["keywords"]
    labelled = [(d, p) for d in _KW_COUNT_FLAG for p in kw_cfg[d]]

    fill_on = bool(cfg["period"]["fill"]["enabled"])
    out = {name: out_dir / f"{name}.parquet" for name in OUTPUT_NAMES}
    if fill_on:
        out[FILL_OUTPUT_NAME] = out_dir / f"{FILL_OUTPUT_NAME}.parquet"
    out[META_NAME] = out_dir / f"{META_NAME}.json"
    parts = {name: _part_path(p) for name, p in out.items()}
    # 날짜를 채우지 않는 실행에서도 이전 실행의 `date_fill`을 지운다(이번 실행의 결과로 보이지 않게)
    fill_path = out_dir / f"{FILL_OUTPUT_NAME}.parquet"

    def remove_outputs():
        for p in [*out.values(), *parts.values(), fill_path, _part_path(fill_path)]:
            p.unlink(missing_ok=True)

    remove_outputs()  # 이전 실행의 결과 파일이 이번 실행의 결과로 보이지 않게 한다
    writers = {}
    try:
        for name in ("base", "treat", "skills_long"):
            writers[name] = rawio.parquet_writer(parts[name], schemas[name])

        occ_pieces, stats_rows, base_ids, psm_pieces, fill_pieces = [], [], [], [], []
        hits = defaultdict(int)   # {(게시 연도, 처치 표시 열, 표현): 공고 수}
        other_words = defaultdict(int)
        gap_counts = {}           # {파일 연도: {간격 구간: 공고 수}}. 날짜 열이 하나뿐이면 비어 있다
        for year in years:
            t0 = time.perf_counter()
            pf = files[year]
            acc = {"base": [], "treat": [], "skills_long": []}
            occ_year, psm_year, fill_year = [], [], []
            fill_dates = defaultdict(set)   # {수집 사이트: 날짜를 채운 분석 대상 세분류 공고가 받은 날짜의 집합}
            if len(date_cols) > 1:
                gap_counts[year] = dict.fromkeys([label for label, _, _ in POST_END_GAP_BINS], 0)
            st = {"year": year, "n_raw": 0, "n_in_period": 0, "n_in_occ": 0, **{k: 0 for k in _STAT_KEYS}}
            # 날짜를 채울 때: 파일 전체의 날짜 열로 행마다 채울 날짜를 먼저 정한다(행 묶음 경계를 넘어 채우기 위해서다)
            file_fill = _file_fill(pf, date_cols) if fill_on else None
            offset = 0

            for i in range(pf.num_row_groups):
                raw = pf.read_row_group(i, columns=read_cols).to_pandas()
                st["n_raw"] += len(raw)
                fill = None
                if file_fill is not None:
                    if offset + len(raw) > len(file_fill):
                        raise RuntimeError(f"{year}년 파일: 날짜 열로 읽은 행 수({len(file_fill)})가 행 묶음으로 읽은 "
                                           f"행 수보다 적습니다(행 묶음 {i}까지 {offset + len(raw)}행)")
                    fill = file_fill.iloc[offset:offset + len(raw)].reset_index(drop=True)
                offset += len(raw)

                # 모든 공고: 숙련 긴 표와 연도 × 세분류 집계
                long, s = skills.explode_skills(raw, codebook)
                for k in _STAT_KEYS:
                    st[k] += s[k]
                flags = treatment.skill_flags(long, codes, raw["posting_id"])
                b = sample.build_posting_base(raw, ked_firm, ksic_valid, cfg, fill=fill)
                cal_year = b["year"].fillna(year).astype("int64")   # 게시일의 연도. 게시일을 정할 수 없으면 파일 연도
                occ_year.append(pd.DataFrame({
                    "year": cal_year.to_numpy(),
                    "keco4": raw["keco_2025"].astype("string").fillna("NA").to_numpy(),
                    "n": np.ones(len(raw), dtype="int64"),
                    "n_skill": raw["posting_id"].isin(long["posting_id"]).to_numpy(dtype="int64"),
                    "n_use": flags["has_use"].to_numpy(dtype="int64"),
                    "n_dev": flags["has_dev"].to_numpy(dtype="int64"),
                }).groupby(["year", "keco4"], as_index=False).sum())

                in_occ = b["in_occ"].to_numpy(dtype=bool)
                # 모든 공고: 게시일과 관련된 공고 수(게시일을 정할 수 없는 공고는 분석 기간 단계에서 빠진다. 날짜를
                # 채우는 설정에서는 파일에서 앞에 날짜 있는 공고가 없어 채울 날짜가 없는 공고만 빠진다)
                psm_year.append(_post_start_counts(raw, b, flags["has_use"].to_numpy(dtype=bool), date_cols))
                # 날짜를 채운 공고: 간격 구간별 공고 수와, 분석 대상 세분류에서 수집 사이트별로 받은 날짜
                if fill_on:
                    fill_year.append(_date_fill_counts(raw, b))
                    occ_filled = in_occ & (b["date_source"] == sample.FILL_SOURCE).fillna(False).to_numpy(dtype=bool)
                    got = pd.DataFrame({"source": _source_labels(raw)[occ_filled],
                                        "day": fill["fill_date"].to_numpy()[occ_filled]})
                    for src, days in got.groupby("source")["day"]:
                        fill_dates[src].update(days.tolist())
                # 분석 대상 세분류의 공고: 게시일과 마감일의 간격
                if len(date_cols) > 1:
                    for label, n in _post_end_gap_counts(raw, in_occ, date_cols).items():
                        gap_counts[year][label] += n

                # 분석 기간과 분석 대상 직종에 드는 공고만 남긴다
                keep = b["in_period"].to_numpy(dtype=bool) & in_occ
                st["n_in_period"] += int(b["in_period"].sum())
                st["n_in_occ"] += int(keep.sum())
                if not keep.any():
                    continue
                b = b[keep].reset_index(drop=True)
                # base에 들어가는 공고 가운데 시도가 `기타`인 공고의 근무지 첫 낱말을 센다. 근무지 열은 쓰지 않는다.
                other = (b["sido"] == sample.SIDO_OTHER).fillna(False).to_numpy(dtype=bool)
                if other.any():
                    region = raw["region"][keep].reset_index(drop=True)[other]
                    words = sample.region_first_word(region).str.slice(0, SIDO_OTHER_WORD_MAX_CHARS)
                    for w, n in words.value_counts().items():
                        other_words[str(w)] += int(n)
                n_skills = long["posting_id"].value_counts()
                b["n_skills"] = n_skills.reindex(b["posting_id"], fill_value=0).to_numpy(dtype="int64")
                long = long[long["posting_id"].isin(b["posting_id"])].reset_index(drop=True)

                # 남은 공고가 있는 행 묶음에서만 원문 열을 다시 읽는다. 원문은 여기서만 쓰고 글자 수만 저장한다.
                text = _joined_text(pf.read_row_group(i, columns=text_cols).to_pandas(), text_cols)[keep]
                text = text.reset_index(drop=True)
                b["text_len"] = text.str.len().fillna(0).astype("int64").to_numpy()
                tr = treatment.build_treat(b["posting_id"], text, long, codebook, cfg)
                post_year = b["year"].astype("int64").to_numpy()
                for flag, patterns in _KW_FLAG_PATTERNS.items():
                    sel = tr[flag].to_numpy(dtype=bool)
                    if not sel.any():
                        continue
                    t_sel = text[sel].reset_index(drop=True)
                    y_sel = post_year[sel]
                    for y in np.unique(y_sel):
                        counts = treatment.pattern_counts(t_sel[y_sel == y], kws[patterns], kws["exclude"])
                        for p, n in counts.items():
                            hits[(int(y), flag, p)] += n
                del text

                acc["base"].append(_to_table(b, schemas["base"]))
                acc["treat"].append(_to_table(tr, schemas["treat"]))
                acc["skills_long"].append(_to_table(long, schemas["skills_long"]))

            if file_fill is not None and offset != len(file_fill):
                raise RuntimeError(f"{year}년 파일: 날짜 열로 읽은 행 수({len(file_fill)})와 행 묶음으로 읽은 행 수"
                                   f"({offset})가 다릅니다")
            # 연도가 끝나면 그 연도의 행을 각 파일에 이어 쓴다
            for name, tables in acc.items():
                if tables:
                    writers[name].write_table(pa.concat_tables(tables))
            base_ids += [t.column("posting_id") for t in acc["base"]]
            ids = pa.chunked_array([c for col in base_ids for c in col.chunks], type=schemas["base"].field("posting_id").type)
            n_dup = len(ids) - pc.count_distinct(ids, mode="all").as_py()
            if n_dup:
                raise RuntimeError(f"posting_id가 겹치는 공고가 있습니다: 겹치는 행 {n_dup}개 ({year}년 파일까지)")
            if occ_year:
                occ_pieces.append(pd.concat(occ_year, ignore_index=True)
                                  .groupby(["year", "keco4"], as_index=False).sum())
            if psm_year:
                g = pd.concat(psm_year, ignore_index=True).groupby("source", as_index=False).sum()
                g["n_in_occ_fill_dates"] = g["source"].map(lambda s: len(fill_dates.get(s, ()))).astype("int64")
                g.insert(0, "year", year)
                psm_pieces.append(g)
            if fill_year:
                g = pd.concat(fill_year, ignore_index=True).groupby(["source", "gap_bin"], as_index=False).sum()
                g.insert(0, "year", year)
                fill_pieces.append(g)
            stats_rows.append(st)
            print(f"{year}년 파일: {st['n_raw']:,}행 읽음, {time.perf_counter() - t0:.1f}초", flush=True)

        for w in writers.values():
            w.close()
        writers = {}

        if occ_pieces:
            occ = (pd.concat(occ_pieces, ignore_index=True)
                   .groupby(["year", "keco4"], as_index=False).sum()
                   .sort_values(["year", "keco4"], ignore_index=True))
        else:
            occ = _empty_frame(OCC_SCHEMA)
        stats = pd.DataFrame(stats_rows, columns=STATS_SCHEMA.names)
        kw_years = sorted(set(years) | {y for y, _, _ in hits})
        kw_hits = pd.DataFrame(
            [(y, d, p, hits.get((y, _KW_COUNT_FLAG[d], p), 0)) for y in kw_years for d, p in labelled],
            columns=KW_SCHEMA.names,
        )
        gap = pd.DataFrame([(y, label, n) for y, by_bin in gap_counts.items() for label, n in by_bin.items()],
                           columns=GAP_SCHEMA.names)
        if gap.empty:
            gap = _empty_frame(GAP_SCHEMA)
        if psm_pieces:
            psm = (pd.concat(psm_pieces, ignore_index=True)[PSM_SCHEMA.names]
                   .sort_values(["year", "source"], ignore_index=True))
        else:
            psm = _empty_frame(PSM_SCHEMA)
        if other_words:
            words = (pd.DataFrame(list(other_words.items()), columns=WORDS_SCHEMA.names)
                     .sort_values(["n", "word"], ascending=[False, True], ignore_index=True)
                     .head(SIDO_OTHER_WORDS_TOP))
        else:
            words = _empty_frame(WORDS_SCHEMA)
        small = [("occ_year_all", occ, OCC_SCHEMA), ("scan_stats", stats, STATS_SCHEMA),
                 ("kw_pattern_hits", kw_hits, KW_SCHEMA), ("post_start_missing", psm, PSM_SCHEMA),
                 ("sido_other_words", words, WORDS_SCHEMA), ("post_end_gap", gap, GAP_SCHEMA)]
        if fill_on:
            small.append((FILL_OUTPUT_NAME, _date_fill_table(fill_pieces), FILL_SCHEMA))
        for name, df, schema in small:
            with rawio.parquet_writer(parts[name], schema) as w:
                w.write_table(_to_table(df, schema))

        meta = {"scan_started": started, "scan_finished": _now_iso(), "git_commit": git_commit,
                "git_uncommitted_changes": git_changed, "config": _scan_config(cfg)}
        # 같은 폴더의 parquet 파일을 쓸 때 `data` 폴더 아래가 아닌지 이미 확인했다
        parts[META_NAME].write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        for name in out:
            _replace_with_retry(parts[name], out[name])
    except BaseException:
        for w in writers.values():
            w.close()
        remove_outputs()
        raise
    return out


# ---------------------------------------------------------------------------
# 2단계: 집계표, 학습 시간 측정, 보고서
# ---------------------------------------------------------------------------

# 희소 필터 기준별로 남는 숙련 수를 보기 위한 격자(출현율 기준 세 값 × 처치 공고 수 기준 세 값의 모든 조합). 분석에 쓰는 기준은
# config.yaml의 skills.sparse_filter에 있고, 그 값도 격자에 더한다.
SPARSE_GRID_MIN_RATE = (0.0005, 0.001, 0.002)
SPARSE_GRID_MIN_TREATED = (10, 30, 50)
SKILL_COUNT_QUANTILES = {"min": 0.0, "p25": 0.25, "p50": 0.5, "p75": 0.75, "p90": 0.9, "p99": 0.99, "max": 1.0}
SALARY_QUANTILES = {"p10": 0.1, "p25": 0.25, "p50": 0.5, "p75": 0.75, "p90": 0.9}
KSIC_NOT_IN_TABLE_TOP = 30
REPORT_MAX_ROWS = 30
# `t17_firm_treated`에서 처치 공고가 많은 기업 몇 개의 비중을 볼지
FIRM_TOP = 10
# `t16_detail_by_arm`에 싣는 숙련 수의 분위수
ARM_SKILL_QUANTILES = {"p50": 0.5, "p75": 0.75, "p90": 0.9}
# `t15_aiie_coverage`에 싣는 AIIE 값의 분위수
AIIE_QUANTILES = {"p10": 0.1, "p25": 0.25, "p50": 0.5, "p75": 0.75, "p90": 0.9}
# `t15_aiie_coverage`를 만드는 외부 자료의 설정 항목(`data.` 아래)
AIIE_DATA_KEYS = ("aiie", "naics_isic", "ksic_isic")
# 외부 자료가 없어 `t15_aiie_coverage`를 만들지 않았을 때, 빈 표의 attrs에 없는 파일의 목록을 두는 이름
MISSING_EXTERNAL_ATTR = "missing_external"

KIND_LABELS = {"dev": "개발", "use": "생성형 AI 활용"}
# `t10b_occ_list_check`의 상태. 마지막 값은 자료 쪽 목록에서 `occupations.exclude_keco2`로 뺀 세분류이다.
OCC_STATUS = ("설정에만 있음", "자료에만 있음", "양쪽에 있음", "설정에서 뺀 중분류")
# `t16_detail_by_arm`의 행 이름
ARM_LABELS = {1: "처치 공고", 0: "비교 공고"}
# `t03e`, `t04`, `t15`에서 처치 공고와 비교 공고의 행 앞에 두는, 대상 공고 모두의 행 이름
ALL_LABEL = "전체"
ARM_WITHIN_LABEL = "같은 세분류·연도 안의 차이"
MISSING_LABEL = "결측"
GIT_UNKNOWN = "알 수 없음(git으로 커밋을 확인하지 못함)"

# 학습 시간 측정의 설명 변수. `text_len_log`는 base.parquet의 `text_len`으로 만든다(log1p).
TIMING_FEATURES = ["firm_emp_log", "ked_size", "ked_ipo", "ked_venture", "firm_age", "ksic_sub", "keco4",
                   "career_type", "career_years_min", "edu_level", "salary_band", "emp_type", "sido", "quarter",
                   "source", "text_len_log"]
# base.parquet에 없고 다른 열로 만드는 설명 변수: {설명 변수: 만드는 데 쓰는 base.parquet의 열}
_TIMING_DERIVED = {"text_len_log": "text_len"}
# 제외 규칙에 쓰는 base.parquet의 열. 뒤의 두 열은 `period.fill.max_gap_days`를 적용할 때 쓴다(`sample.apply_exclusions`)
_EXCLUSION_COLUMNS = ["year", "in_period", "in_occ", "has_bzno", "date_source", "fill_gap_days"]
# 집계표에 쓰는 base.parquet의 열(`bzno`는 기업별 집계에만 쓰고 표에는 넣지 않는다)
_TABLE_BASE_COLUMNS = ["posting_id", "bzno", "year", "month", "quarter", "period", "date_source", "fill_gap_days",
                       "source", "keco4", "keco2", "ksic5", "ksic_sub", "salary_annual_min", "sido", "ai_peripheral",
                       "ked_matched", "in_period", "in_occ", "has_bzno", "n_skills", "text_len"]
# 날짜를 채우지 않는 설정(`period.fill.enabled` = false)에서는 만들지 않는 표
FILL_TABLES = ("t01d_date_fill", "t02e_fill_gap_by_period")
# 학습 시간 측정에 쓰는 base.parquet의 열
_TIMING_BASE_COLUMNS = list(dict.fromkeys(["posting_id", "bzno", "n_skills", *_EXCLUSION_COLUMNS,
                                           *[_TIMING_DERIVED.get(c, c) for c in TIMING_FEATURES]]))


def _require_interim(s0_dir: Path, cfg: dict) -> None:
    """1단계의 중간 자료 아홉 파일(`OUTPUT_NAMES`), 날짜를 채우는 설정이면 `date_fill.parquet`, 1단계 기록이 모두 있는지
    확인한다. 없으면 없는 파일 이름을 적은 예외를 낸다."""
    fill = [FILL_OUTPUT_NAME] if cfg["period"]["fill"]["enabled"] else []
    names = [f"{n}.parquet" for n in [*OUTPUT_NAMES, *fill]] + [f"{META_NAME}.json"]
    missing = [n for n in names if not (s0_dir / n).is_file()]
    if missing:
        raise FileNotFoundError(
            f"자료 점검 중간 자료가 없습니다: {', '.join(missing)} (찾은 위치: {s0_dir}). 먼저 scan 명령을 실행해야 합니다.")


def _read_scan_meta(s0_dir: Path) -> dict:
    """1단계 기록(`scan_meta.json`)을 읽는다. 읽을 수 없거나 필요한 내용이 없으면 scan을 다시 하라는 예외를 낸다."""
    path = s0_dir / f"{META_NAME}.json"
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise RuntimeError(f"{path.name}을 읽을 수 없습니다({type(e).__name__}). scan 명령을 다시 실행해야 합니다.") from None
    if not isinstance(meta, dict) or not isinstance(meta.get("config"), dict):
        raise RuntimeError(f"{path.name}에 설정 기록(config)이 없습니다. scan 명령을 다시 실행해야 합니다.")
    return meta


def _check_scan_config(cfg: dict, meta: dict) -> None:
    """1단계 기록의 설정 값과 지금 설정을 비교한다. 다르면 다른 항목을 모두 적고 scan을 다시 하라는 예외를 낸다."""
    recorded = meta["config"]
    current = _scan_config(cfg)
    diff = []
    for key in list(recorded) + [k for k in current if k not in recorded]:
        diff += _differences(recorded.get(key), current.get(key), key)
    if diff:
        raise RuntimeError(
            "중간 자료를 만들 때(scan)의 설정과 지금 설정이 다른 항목이 있습니다: " + ", ".join(diff)
            + ". 이 항목은 중간 자료의 내용을 바꾸므로 scan 명령을 다시 실행해야 합니다.")


def _keyword_col(cfg: dict) -> str:
    return f"kw_{cfg['treatment']['keyword_dictionary']}"


def _as_arrow(values) -> pa.Array:
    if isinstance(values, pa.ChunkedArray):
        return values.combine_chunks()
    if isinstance(values, pa.Array):
        return values
    return pa.array(values)


def _positions(ids, base_ids) -> np.ndarray:
    """`ids` 각각이 `base_ids`의 몇 번째 값인지(0부터) 돌려준다. `base_ids`에 없는 값이 있으면 멈춘다."""
    values, value_set = _as_arrow(ids), _as_arrow(base_ids)
    if values.type != value_set.type:
        values = values.cast(value_set.type)
    pos = pc.index_in(values, value_set=value_set)
    if pos.null_count:
        raise RuntimeError(f"숙련 긴 표에 공고 기본 자료에 없는 공고가 있습니다: {pos.null_count}행")
    return pos.to_numpy(zero_copy_only=False)


def _check_same_postings(treat: pd.DataFrame, base: pd.DataFrame) -> None:
    """처치 표시 자료와 공고 기본 자료의 공고가 같은 순서로 같은지 확인한다. 집계는 위치로 맞추기 때문이다."""
    a = treat["posting_id"].to_numpy(dtype=object)
    b = base["posting_id"].to_numpy(dtype=object)
    if len(a) != len(b) or not np.array_equal(a, b):
        raise RuntimeError(f"treat와 base의 공고가 같은 순서로 같지 않습니다: treat {len(a)}행, base {len(b)}행")


def _exclusions(base: pd.DataFrame, cfg: dict) -> tuple[np.ndarray, pd.DataFrame]:
    """제외 규칙 네 단계를 적용해, 남는 공고의 표시(base와 같은 길이)와 연도·단계별 건수 표를 돌려준다."""
    has_skill = pd.Series(base["n_skills"].to_numpy() > 0)
    # 날짜를 채우는 규칙의 두 열이 없는 표(시험에서 손으로 만든 표 등)도 받는다. 그 열이 필요한 설정이면
    # `sample.apply_exclusions`가 없는 열 이름을 적은 예외를 낸다
    slim = base[[c for c in _EXCLUSION_COLUMNS if c in base.columns]].reset_index(drop=True)
    kept, counts = sample.apply_exclusions(slim, cfg, has_skill)
    alive = np.zeros(len(base), dtype=bool)
    alive[kept.index.to_numpy()] = True
    return alive, counts


def _read_long_positions(s0_dir: Path, base_ids: pa.ChunkedArray) -> pd.DataFrame:
    """숙련 긴 표를 읽어 열 `pos`(base에서의 위치), `skill_code`, `src`(둘 다 category)로 돌려준다.

    공고 번호는 pandas로 바꾸지 않고 위치로만 바꾼다. 행이 수천만 개여도 메모리를 적게 쓰기 위해서다.
    """
    t = pq.read_table(s0_dir / "skills_long.parquet", columns=["posting_id", "skill_code", "src"],
                      read_dictionary=["skill_code", "src"])
    pos = _positions(t.column("posting_id"), base_ids)
    return pd.DataFrame({"pos": pos, "skill_code": t.column("skill_code").to_pandas(),
                         "src": t.column("src").to_pandas()})


def _rate(num, den):
    return num / den.where(den > 0)


def _flag_label(s: pd.Series) -> pd.Series:
    """참·거짓 열을 `1`, `0`, `결측` 문자열로 바꾼다. 참·거짓으로 바꿀 수 없는 열은 값을 그대로 문자열로 쓴다."""
    try:
        b = s.astype("boolean")
    except (TypeError, ValueError):
        return s.astype("string").fillna(MISSING_LABEL)
    lab = np.where(b.fillna(False).to_numpy(dtype=bool), "1", "0")
    return pd.Series(np.where(b.isna().to_numpy(), MISSING_LABEL, lab), index=s.index)


def _crosstab(labels: pd.DataFrame, title: str) -> pd.DataFrame:
    """여러 표시 열의 모든 값 조합별 공고 수. 공고가 없는 조합도 0으로 넣는다."""
    cols = list(labels.columns)
    g = labels.groupby(cols).size()
    levels = [sorted({"0", "1"} | set(labels[c].unique())) for c in cols]
    full = pd.MultiIndex.from_product(levels, names=cols)
    out = g.reindex(full, fill_value=0).rename("n").reset_index()
    out.insert(0, "cross", title)
    return out


# --- 표 하나씩 -------------------------------------------------------------

# `t01_exclusions`의 단계 순서. `fill_gap`은 `period.fill.max_gap_days`에 값이 있을 때만 있는 행으로, base.parquet의
# 공고 가운데 날짜를 채웠지만 앞뒤 날짜의 간격이 그 값보다 크거나 결측이어서 분석 기간 단계에서 빠진 공고를 센다.
T01_STEPS = ("in_period", "in_occ", "fill_gap", "has_bzno", "has_skill")


def _t01_exclusions(stats: pd.DataFrame, counts: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    cols = ["year", "step", "n_before", "n_dropped", "n_after"]
    first = pd.concat([
        pd.DataFrame({"year": stats["year"], "step": "in_period",
                      "n_before": stats["n_raw"], "n_after": stats["n_in_period"]}),
        pd.DataFrame({"year": stats["year"], "step": "in_occ",
                      "n_before": stats["n_in_period"], "n_after": stats["n_in_occ"]}),
    ], ignore_index=True)
    first["n_dropped"] = first["n_before"] - first["n_after"]
    later = counts[counts["step"].isin(["has_bzno", "has_skill"])]
    pieces = [first[cols], later[cols]]
    if cfg["period"]["fill"]["max_gap_days"] is not None:
        # base.parquet의 공고는 모두 분석 기간 안이므로, base에서 센 분석 기간 단계의 빠진 공고는 이 규칙으로 빠진 공고이다
        pieces.append(counts[counts["step"] == "in_period"].assign(step="fill_gap")[cols])
    out = pd.concat(pieces, ignore_index=True)
    out = out.astype({c: "int64" for c in cols if c != "step"})
    order = out["step"].map({s: i for i, s in enumerate(T01_STEPS)})
    return out.assign(_o=order).sort_values(["year", "_o"], kind="stable").drop(columns="_o").reset_index(drop=True)


_T01B_COLUMNS = ["level", "year", "source", "n_raw", "n_no_post_start", "n_no_date", "no_date_rate", "n_end_unusable",
                 "n_filled", "n_in_occ", "n_in_occ_no_post_start", "n_in_occ_no_date", "n_in_occ_end_unusable",
                 "n_in_occ_from_end_in_period", "n_in_occ_filled", "n_in_occ_filled_in_period", "n_in_occ_fill_dates",
                 "n_in_occ_use", "n_in_occ_use_no_post_start", "n_in_occ_use_no_date",
                 "n_in_occ_use_from_end_in_period", "n_in_occ_use_filled_in_period", "in_occ_use_no_date_rate"]


def _t01b_post_start_missing(psm: pd.DataFrame) -> pd.DataFrame:
    """공고 파일의 모든 공고 가운데 게시일과 관련된 공고 수(`post_start_missing.parquet`의 열).

    연도별 행(`level` = 연도)과, 첫 날짜 열에 날짜가 없는 공고가 있는 연도 × 수집 사이트별 행. 연도 행의
    `n_in_occ_fill_dates`는 수집 사이트별 값의 합이다(사이트마다 받은 날짜를 따로 센다).
    """
    by_year = psm.groupby("year", as_index=False)[_PSM_COUNTS].sum()
    by_year.insert(0, "level", "연도")
    by_year.insert(2, "source", pd.Series(pd.NA, index=by_year.index, dtype="string"))
    site = psm[psm["n_no_post_start"] > 0].sort_values(["year", "n_no_post_start", "source"],
                                                       ascending=[True, False, True])
    site = site.assign(source=site["source"].astype("string"))
    site.insert(0, "level", "연도·수집 사이트")
    pieces = [d for d in (by_year, site) if not d.empty]
    out = pd.concat(pieces, ignore_index=True) if pieces else by_year
    out = out.astype({"year": "int64", **{c: "int64" for c in _PSM_COUNTS}})
    out["no_date_rate"] = _rate(out["n_no_date"], out["n_raw"])
    out["in_occ_use_no_date_rate"] = _rate(out["n_in_occ_use_no_date"], out["n_in_occ_use"])
    return out[_T01B_COLUMNS]


def _t01c_post_end_gap(gap: pd.DataFrame) -> pd.DataFrame:
    """분석 대상 세분류의 공고 가운데 게시일과 마감일이 모두 날짜로 읽히는 공고를 파일 연도 × 두 날짜의 간격 구간별로
    센 표(`post_end_gap.parquet`)에, 그 연도 공고 가운데 그 구간 공고의 비율(`share_in_year`)을 더한다."""
    out = gap[["year", "bin", "n"]].astype({"year": "int64", "n": "int64"}).reset_index(drop=True)
    out["share_in_year"] = _rate(out["n"], out.groupby("year")["n"].transform("sum"))
    return out


def _t01d_date_fill(fill: pd.DataFrame) -> pd.DataFrame:
    """날짜를 채운 공고를 파일 연도 × 수집 사이트 × 앞뒤 날짜 간격의 구간별로 센 표(`date_fill.parquet`). 연도 순,
    연도 안에서는 날짜를 채운 공고가 많은 사이트부터(같으면 이름 순), 사이트 안에서는 `FILL_GAP_LABELS`의 순서이다."""
    cols = ["year", "source", "gap_bin", *_FILL_COUNTS]
    out = fill[cols].astype({"year": "int64", **{c: "int64" for c in _FILL_COUNTS}}).reset_index(drop=True)
    out["source"] = out["source"].astype("string")
    total = out.groupby(["year", "source"])["n"].transform("sum")
    order = out["gap_bin"].map({b: i for i, b in enumerate(FILL_GAP_LABELS)})
    out = out.assign(_t=-total, _o=order).sort_values(["year", "_t", "source", "_o"], kind="stable")
    return out.drop(columns=["_t", "_o"]).reset_index(drop=True)


def _t02a_treated_by_period(base: pd.DataFrame, flags: pd.DataFrame, arm: np.ndarray, alive: np.ndarray,
                            cfg: dict) -> pd.DataFrame:
    """제외 규칙 네 단계를 적용한 공고의 기간별 공고 수, 설정의 처치 정의로 나눈 처치 공고 수(`n_treated`)와 비교 공고
    수(`n_control`, `_arm`), 처치 표시별 공고 수. 기간은 연, 반기, 분기와 설정의 기간(`period.groups`, `unit` = period,
    설정의 순서)이다."""
    b = base.loc[alive, ["year", "month", "quarter", "period"]]
    f = flags.loc[alive].astype("int64")
    f.insert(0, "n_treated", (arm[alive] == 1).astype("int64"))
    f.insert(1, "n_control", (arm[alive] == 0).astype("int64"))
    year = b["year"].astype("int64").astype("string")
    half = pd.Series(np.where(b["month"].to_numpy() <= 6, "H1", "H2"), index=b.index, dtype="string")
    names = [str(g["name"]) for g in cfg["period"]["groups"]]
    pieces = []
    for unit, period in (("year", year), ("half", year + half), ("quarter", b["quarter"].astype("string")),
                         ("period", b["period"].astype("string"))):
        g = f.groupby(period).sum()
        g.insert(0, "n", f.groupby(period).size())
        if unit == "period":   # 설정의 순서. 설정에 없는 이름은 뒤에 붙인다
            g = g.reindex([n for n in names if n in g.index] + [n for n in g.index if n not in names])
        pieces.append(g.rename_axis("period").reset_index().assign(unit=unit))
    out = pd.concat(pieces, ignore_index=True)
    counts = ["n", "n_treated", "n_control", *_TREAT_FLAGS]
    return out[["unit", "period", *counts]].astype({c: "int64" for c in counts})


def _t02b_treated_by_occ(base: pd.DataFrame, flags: pd.DataFrame, keco: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    names4 = dict(zip(keco["keco4"], keco["keco4_name"]))
    names2 = dict(zip(keco["keco2"], keco["keco2_name"]))
    occs4 = sorted(set(cfg["occupations"]["keco4"]) | set(base["keco4"].dropna()))
    occs2 = sorted({c[:2] for c in occs4})
    f = flags.astype("int64")
    pieces = []
    for level, col, codes, names in (("중분류", "keco2", occs2, names2), ("세분류", "keco4", occs4, names4)):
        key = base[col].astype("string").fillna(MISSING_LABEL)
        g = f.groupby(key).sum()
        g.insert(0, "n", f.groupby(key).size())
        g = g.reindex(sorted(set(codes) | set(g.index)), fill_value=0).rename_axis("code").reset_index()
        g.insert(0, "level", level)
        g.insert(2, "name", g["code"].map(names))
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True)


def _arm(treat: pd.DataFrame, alive: np.ndarray, cfg: dict) -> np.ndarray:
    """base와 같은 길이의 배열: 제외 규칙 네 단계를 통과한 공고 가운데 설정의 처치 정의(`treatment.definition`)로
    처치 공고는 1, 비교 공고는 0, 그 밖의 공고(어느 쪽도 아니거나 네 단계에서 빠진 공고)는 -1."""
    out = np.full(len(alive), -1, dtype="int64")
    idx = np.flatnonzero(alive)
    arm = treatment.select_arms(treat.iloc[idx], cfg["treatment"]["definition"], _keyword_col(cfg))
    out[idx] = arm.astype("Int64").fillna(-1).to_numpy(dtype="int64")
    return out


def _t02c_occ_group_treated(base: pd.DataFrame, arm: np.ndarray, alive: np.ndarray, keco: pd.DataFrame,
                            cfg: dict) -> pd.DataFrame:
    """제외 규칙 네 단계를 적용한 공고를 직종군(중분류 `keco2`)별로 센다. 처치 공고와 비교 공고는 설정의 처치
    정의로 나눈 것이다(`_arm`). 분석 대상 세분류(`occupations.keco4`)가 속한 직종군은 공고가 없어도 싣는다."""
    key = base.loc[alive, "keco2"].astype("string").fillna(MISSING_LABEL).to_numpy()
    a = arm[alive]
    df = pd.DataFrame({"keco2": key, "n": 1, "n_treated": (a == 1).astype("int64"), "n_control": (a == 0).astype("int64")})
    g = df.groupby("keco2")[["n", "n_treated", "n_control"]].sum()
    groups = sorted({str(c)[:2] for c in cfg["occupations"]["keco4"]} | set(g.index))
    g = g.reindex(groups, fill_value=0).astype("int64").rename_axis("keco2").reset_index()
    g.insert(1, "name", g["keco2"].map(dict(zip(keco["keco2"], keco["keco2_name"]))))
    g["meets_min"] = g["n_treated"] >= int(cfg["occupations"]["group_min_treated"])
    return g


def _period_names(cfg: dict) -> list[str]:
    return [str(p["name"]) for p in cfg["period"]["groups"]]


def _t02d_date_source_by_period(base: pd.DataFrame, arm: np.ndarray, alive: np.ndarray, cfg: dict) -> pd.DataFrame:
    """제외 규칙 네 단계를 적용한 공고를 설정의 기간(`period.groups`, 설정의 순서) × 게시일을 가져온 곳(`date_source`:
    `period.date_columns`의 순서, 날짜를 채우는 설정이면 그 뒤에 `fill`)별로 센다. 공고가 없는 조합도 0으로 싣고,
    설정에 없는 값은 뒤에 붙인다.

    `n_treated`는 설정의 처치 정의로 나눈 처치 공고 수(`_arm`)이다. `share_in_period`는 그 기간 공고 가운데 그 행의
    공고 비율, `treated_share_in_period`는 그 기간 처치 공고 가운데 그 행의 처치 공고 비율이다.
    """
    df = pd.DataFrame({
        "period": base.loc[alive, "period"].astype("string").fillna(MISSING_LABEL).to_numpy(),
        "date_source": base.loc[alive, "date_source"].astype("string").fillna(MISSING_LABEL).to_numpy(),
        "n": 1, "n_treated": (arm[alive] == 1).astype("int64")})
    g = df.groupby(["period", "date_source"])[["n", "n_treated"]].sum()
    sources = [str(c) for c in cfg["period"]["date_columns"]]
    if cfg["period"]["fill"]["enabled"]:
        sources.append(sample.FILL_SOURCE)
    levels = []
    for col, configured in (("period", _period_names(cfg)), ("date_source", sources)):
        levels.append(configured + sorted(set(df[col]) - set(configured)))
    full = pd.MultiIndex.from_product(levels, names=["period", "date_source"])
    out = g.reindex(full, fill_value=0).astype("int64").reset_index()
    by_period = out.groupby("period", sort=False)
    out["share_in_period"] = _rate(out["n"], by_period["n"].transform("sum"))
    out["treated_share_in_period"] = _rate(out["n_treated"], by_period["n_treated"].transform("sum"))
    return out


def _t02e_fill_gap_by_period(base: pd.DataFrame, arm: np.ndarray, alive: np.ndarray, cfg: dict) -> pd.DataFrame:
    """제외 규칙 네 단계를 적용한 공고 가운데 날짜를 채운 공고(`date_source` = `fill`)를 설정의 기간(설정의 순서) ×
    앞뒤 날짜 간격의 구간(`FILL_GAP_LABELS`의 순서)별로 센다. 공고가 없는 조합도 0으로 싣는다.

    `n_treated`는 그 가운데 설정의 처치 정의로 나눈 처치 공고 수(`_arm`)이고, `share_in_period`는 `n`을 그 기간의
    모든 공고(제외 규칙 네 단계를 적용한 공고) 수로 나눈 값이다.
    """
    period = base.loc[alive, "period"].astype("string").fillna(MISSING_LABEL).to_numpy()
    filled = (base.loc[alive, "date_source"].astype("string") == sample.FILL_SOURCE).fillna(False).to_numpy(dtype=bool)
    df = pd.DataFrame({"period": period[filled], "gap_bin": _fill_gap_bin(base.loc[alive, "fill_gap_days"][filled]),
                       "n": 1, "n_treated": (arm[alive][filled] == 1).astype("int64")})
    g = df.groupby(["period", "gap_bin"])[["n", "n_treated"]].sum()
    names = _period_names(cfg)
    periods = names + sorted(set(period) - set(names))
    full = pd.MultiIndex.from_product([periods, list(FILL_GAP_LABELS)], names=["period", "gap_bin"])
    out = g.reindex(full, fill_value=0).astype("int64").reset_index()
    n_period = pd.Series(period).value_counts()
    out["share_in_period"] = _rate(out["n"], out["period"].map(n_period).fillna(0))
    return out


def _t03a_skills_per_posting(base: pd.DataFrame) -> pd.DataFrame:
    s = base.loc[base["n_skills"] > 0, "n_skills"].astype("float64")
    row = {"n_postings": len(s)}
    row.update({k: (s.quantile(q) if len(s) else np.nan) for k, q in SKILL_COUNT_QUANTILES.items()})
    row["mean"] = s.mean() if len(s) else np.nan
    return pd.DataFrame([row])


def _sparse_filter_sample(alive: np.ndarray, treat: pd.DataFrame, pos: np.ndarray, skill_code: pd.Series,
                          cfg: dict) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """희소 필터를 적용할 공고와 숙련을 정한다. `t03b_sparse_filter_grid`, `t13`, `t14`, `time_outcome_model`이 함께 쓴다.

    대상 공고는 제외 규칙 네 단계를 통과한 공고(`alive`, base와 같은 길이) 가운데 설정의 처치 정의(`_arm`)로 처치
    공고이거나 비교 공고인 공고이다(추정에 들어가는 공고. 어느 쪽도 아닌 공고는 뺀다). `pos`와 `skill_code`는 숙련 긴
    표의 행별 base 위치와 숙련 코드이다. 돌려주는 값은 대상 공고의 행만 남긴 숙련 표(열 `posting_id`는 base에서의
    위치, `skill_code`), 대상 공고의 base 위치, 대상 공고의 처치 여부(1 처치 공고, 0 비교 공고, 대상 공고와 같은 순서)이다.
    """
    arm_all = _arm(treat, alive, cfg)
    member = arm_all >= 0
    idx = np.flatnonzero(member)
    keep = member[pos]
    lk = pd.DataFrame({"posting_id": pos[keep], "skill_code": skill_code[keep].reset_index(drop=True)})
    return lk, idx, arm_all[idx]


def _t03b_sparse_filter_grid(long: pd.DataFrame, treat: pd.DataFrame, alive: np.ndarray,
                             agent_codes: set[str], cfg: dict) -> pd.DataFrame:
    outcome = skills.drop_codes(long[["pos", "skill_code"]], agent_codes)
    lk, sample_idx, arm = _sparse_filter_sample(alive, treat, outcome["pos"].to_numpy(), outcome["skill_code"], cfg)
    treated = set(sample_idx[arm == 1].tolist())
    n_postings = len(sample_idx)
    sf = cfg["skills"]["sparse_filter"]
    rates = sorted(set(SPARSE_GRID_MIN_RATE) | {float(sf["min_rate"])})
    mins = sorted(set(SPARSE_GRID_MIN_TREATED) | {int(sf["min_treated"])})
    rows = []
    for r in rates:
        for m in mins:
            kept = skills.sparse_filter(lk, treated, n_postings, r, m) if n_postings > 0 else []
            rows.append({"min_rate": r, "min_treated": m, "n_skills_kept": len(kept),
                         "n_postings": n_postings, "n_treated": len(treated)})
    return pd.DataFrame(rows)


def _t03c_skills_by_source_column(long: pd.DataFrame, codebook: pd.DataFrame) -> pd.DataFrame:
    g = long.groupby(["skill_code", "src"], observed=True).size().rename("n").reset_index()
    g["skill_code"] = g["skill_code"].astype("string")
    g["src"] = g["src"].astype("string")
    g = skills.attach_hierarchy(g, codebook)
    g["domain"] = g["domain"].astype("string").fillna(MISSING_LABEL)
    out = pd.DataFrame(index=pd.Index(sorted(g["domain"].unique()), name="domain"))
    for kind in skills.SKILL_COLUMNS:
        hit = g["src"].str.contains(rf"(?:^|\+){kind}(?:\+|$)", regex=True).fillna(False).to_numpy(dtype=bool)
        out[kind] = g[hit].groupby("domain")["n"].sum().reindex(out.index, fill_value=0)
    out["total"] = g.groupby("domain")["n"].sum().reindex(out.index, fill_value=0)
    out = out.astype("int64").reset_index()
    total = out.drop(columns="domain").sum().to_frame().T.assign(domain="전체")
    return pd.concat([out, total[out.columns]], ignore_index=True)


def _arm_groups(scope: np.ndarray, arm: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """(행 이름, 공고 표시) 세 쌍: `전체`(`scope`의 모든 공고), `처치 공고`, `비교 공고`(`arm`, `_arm`의 값)."""
    return [(ALL_LABEL, scope), (ARM_LABELS[1], scope & (arm == 1)), (ARM_LABELS[0], scope & (arm == 0))]


def _t04_bzno_ked_by_treatment(base: pd.DataFrame, arm_base: np.ndarray) -> pd.DataFrame:
    """base.parquet의 모든 공고를 `전체`, 설정의 처치 정의로 나눈 처치 공고, 비교 공고(`arm_base`)별로 보고, 사업자번호
    보유 비율과 KED 결합 비율을 센다."""
    has_bzno = base["has_bzno"].to_numpy(dtype=bool)
    ked = base["ked_matched"].to_numpy(dtype=bool) & has_bzno
    rows = []
    for label, member in _arm_groups(np.ones(len(base), dtype=bool), arm_base):
        rows.append({"arm": label, "n": int(member.sum()), "n_has_bzno": int((member & has_bzno).sum()),
                     "n_ked_matched": int((member & ked).sum())})
    g = pd.DataFrame(rows)
    g["has_bzno_rate"] = _rate(g["n_has_bzno"], g["n"])
    g["ked_matched_rate"] = _rate(g["n_ked_matched"], g["n_has_bzno"])
    return g[["arm", "n", "n_has_bzno", "has_bzno_rate", "n_ked_matched", "ked_matched_rate"]]


def _t03e_no_outcome_skill(long: pd.DataFrame, arm: np.ndarray, alive: np.ndarray,
                           agent_codes: set[str]) -> pd.DataFrame:
    """제외 규칙 네 단계를 통과한 공고 가운데, AI 에이전트 숙련코드를 빼면 숙련이 하나도 남지 않는 공고 수.
    행은 `전체`와 설정의 처치 정의로 나눈 처치 공고, 비교 공고(`arm`, `_arm`)이다."""
    outcome = skills.drop_codes(long[["pos", "skill_code"]], agent_codes)
    has_outcome = np.zeros(len(alive), dtype=bool)
    has_outcome[outcome["pos"].to_numpy()] = True
    none_left = alive & ~has_outcome
    rows = []
    for group, member in _arm_groups(alive, arm):
        rows.append({"group": group, "n": int(member.sum()), "n_no_outcome_skill": int((member & none_left).sum())})
    out = pd.DataFrame(rows)
    out["no_outcome_skill_rate"] = _rate(out["n_no_outcome_skill"], out["n"])
    return out


def _t04b_ked_lines(ked_firm: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"n_lines": [int(ked_firm.attrs["n_lines"])],
                         "n_dropped_lines": [int(ked_firm.attrs["n_dropped_lines"])],
                         "n_masked_values": [int(ked_firm.attrs["n_masked_values"])],
                         "n_firms": [len(ked_firm)]})


def _t05_source(base: pd.DataFrame, flags: pd.DataFrame, arm_base: np.ndarray) -> pd.DataFrame:
    """연도 × 수집 사이트별 공고 수와 비율. `treated_rate`는 설정의 처치 정의로 나눈 처치 공고(`arm_base`)의 비율이다."""
    df = pd.DataFrame({"year": base["year"].to_numpy(), "source": base["source"].astype("string").fillna(MISSING_LABEL),
                       "use": flags["has_use"].to_numpy(dtype=bool), "treated": arm_base == 1,
                       "skill": base["n_skills"].to_numpy() > 0})
    g = df.groupby(["year", "source"]).agg(n=("use", "size"), n_use=("use", "sum"), n_treated=("treated", "sum"),
                                           n_skill=("skill", "sum"))
    g = g.reset_index()
    g["share_in_year"] = _rate(g["n"], g.groupby("year")["n"].transform("sum"))
    g["treated_rate"] = _rate(g["n_treated"], g["n"])
    g["has_use_rate"] = _rate(g["n_use"], g["n"])
    g["has_skill_rate"] = _rate(g["n_skill"], g["n"])
    # 연도 안에서는 공고가 많은 사이트부터 싣는다(보고서에서 행이 잘려도 큰 사이트가 남게 하기 위해서다)
    g = g.sort_values(["year", "n", "source"], ascending=[True, False, True], ignore_index=True)
    return g[["year", "source", "n", "share_in_year", "treated_rate", "has_use_rate", "has_skill_rate"]]


def _t06a_overlap(base: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    lab = {c: _flag_label(flags[c]) for c in ("kw_strict", "kw_use", "treat_code", "has_use", "has_dev")}
    lab["ai_peripheral"] = _flag_label(base["ai_peripheral"])
    sets = (["kw_strict", "treat_code", "ai_peripheral"], ["kw_strict", "has_use", "has_dev"],
            ["kw_use", "has_use", "has_dev"])
    out = pd.concat([_crosstab(pd.DataFrame({c: lab[c] for c in s}), " × ".join(s)) for s in sets],
                    ignore_index=True)
    return out[["cross", "kw_strict", "kw_use", "treat_code", "ai_peripheral", "has_use", "has_dev", "n"]]


def _t06b_agent_skill_by_year(long: pd.DataFrame, base: pd.DataFrame, codebook: pd.DataFrame,
                              codes: dict[str, set[str]], years: list[int]) -> pd.DataFrame:
    kind = {c: KIND_LABELS["dev"] for c in codes["dev"]} | {c: KIND_LABELS["use"] for c in codes["use"]}
    agent = sorted(kind)
    sel = long["skill_code"].isin(agent).to_numpy(dtype=bool)
    hits = pd.DataFrame({"skill_code": long.loc[sel, "skill_code"].astype("string").to_numpy(),
                         "year": base["year"].to_numpy()[long.loc[sel, "pos"].to_numpy()]})
    wide = hits.groupby(["skill_code", "year"]).size().unstack(fill_value=0)
    wide = wide.reindex(index=agent, columns=years, fill_value=0).astype("int64")
    wide.columns = [str(y) for y in years]
    names = dict(zip(codebook["skill_code"], codebook["skill_name"]))
    out = wide.rename_axis("skill_code").reset_index()
    out.insert(1, "skill_name", out["skill_code"].map(names))
    out.insert(2, "kind", out["skill_code"].map(kind))
    order = out["kind"].map({KIND_LABELS["use"]: 0, KIND_LABELS["dev"]: 1})
    return out.assign(_o=order).sort_values(["_o", "skill_code"]).drop(columns="_o").reset_index(drop=True)


def _t06c_keyword_pattern_hits(kw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """키워드 사전(`strict`, `broad_extra`, `use`)의 표현마다 걸린 공고 수(`kw_pattern_hits.parquet`)를 게시 연도별로
    펼친 표. 표현은 사전마다 설정의 순서로 싣는다."""
    k = cfg["treatment"]["keywords"]
    labelled = [(d, p) for d in ("strict", "broad_extra", "use") for p in k[d]]
    years = sorted(int(y) for y in kw["year"].unique())
    wide = kw.groupby(["dictionary", "pattern", "year"])["n"].sum().unstack(fill_value=0)
    wide = wide.reindex(index=pd.MultiIndex.from_tuples(labelled, names=["dictionary", "pattern"]),
                        columns=years, fill_value=0).astype("int64")
    wide.columns = [str(y) for y in years]
    wide["total"] = wide.sum(axis=1)
    return wide.reset_index()


def _t08_last_year_monthly(base: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    last = max(int(y) for y in cfg["data"]["years"])
    m = base["year"].to_numpy() == last
    df = pd.DataFrame({"month": base.loc[m, "month"].to_numpy(), "skill": base.loc[m, "n_skills"].to_numpy() > 0})
    g = df.groupby("month").agg(n=("skill", "size"), n_has_skill=("skill", "sum")).astype("int64").reset_index()
    g.insert(0, "year", last)
    g["has_skill_rate"] = _rate(g["n_has_skill"], g["n"])
    return g[["year", "month", "n", "n_has_skill", "has_skill_rate"]]


def _t09a_by_ksic_sub(base: pd.DataFrame, flags: pd.DataFrame, arm_base: np.ndarray) -> pd.DataFrame:
    """산업 소분류별 공고 수, 생성형 AI 활용 숙련 공고 수, 설정의 처치 정의로 나눈 처치 공고 수(`arm_base`)."""
    df = pd.DataFrame({"ksic_sub": base["ksic_sub"].astype("string").fillna(MISSING_LABEL),
                       "use": flags["has_use"].to_numpy(dtype=bool), "treated": arm_base == 1})
    g = (df.groupby("ksic_sub").agg(n=("use", "size"), n_has_use=("use", "sum"), n_treated=("treated", "sum"))
         .astype("int64").reset_index())
    return g.sort_values(["n", "ksic_sub"], ascending=[False, True], ignore_index=True)


def _t09b_ksic_not_in_table(base: pd.DataFrame, ksic_valid: set[str]) -> pd.DataFrame:
    """base.parquet의 공고 가운데 5자리로 맞춘 산업코드(`ksic5`)가 있고 그 앞 3자리가 소분류표(`ksic_valid`)에 없는
    공고를 `ksic5`별로 센다. `ksic_sub`로 고르지 않는 것은 설정에 따라 이 공고들의 `ksic_sub`가 결측이기 때문이다."""
    ksic5 = base["ksic5"].astype("string")
    m = (ksic5.notna() & ~ksic5.str[:3].isin(ksic_valid)).fillna(False).to_numpy(dtype=bool)
    vc = ksic5[m].value_counts()
    out = vc.rename("n").rename_axis("ksic5").reset_index().astype({"n": "int64"})
    out = out.sort_values(["n", "ksic5"], ascending=[False, True], ignore_index=True)
    return out.head(KSIC_NOT_IN_TABLE_TOP)


def _t10a_use_by_occ_year(base: pd.DataFrame, flags: pd.DataFrame, alive: np.ndarray, keco: pd.DataFrame,
                          cfg: dict, years: list[int]) -> pd.DataFrame:
    occs = list(cfg["occupations"]["keco4"])
    df = pd.DataFrame({"keco4": base.loc[alive, "keco4"].astype("string").to_numpy(),
                       "year": base.loc[alive, "year"].to_numpy(),
                       "use": flags.loc[alive, "has_use"].to_numpy(dtype=bool)})
    g = df.groupby(["keco4", "year"]).agg(n=("use", "size"), n_use=("use", "sum"))
    g = g.reindex(pd.MultiIndex.from_product([occs, years], names=["keco4", "year"]), fill_value=0)
    out = pd.DataFrame({"keco4": occs})
    out["keco4_name"] = out["keco4"].map(dict(zip(keco["keco4"], keco["keco4_name"])))
    for y in years:
        part = g.xs(y, level="year").reindex(occs)
        out[f"{y}_n_use"] = part["n_use"].astype("int64").to_numpy()
        out[f"{y}_use_rate"] = _rate(part["n_use"], part["n"]).to_numpy()
    return out


def _t10b_occ_list_check(occ: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """`occ_year_all.parquet`(공고 파일의 모든 공고)에서 센 자료 쪽 세분류 목록과 설정의 목록(`occupations.keco4`)을
    비교한다. 자료 쪽 목록에서 `occupations.exclude_keco2`의 중분류에 속한 세분류는 빼서 따로 싣는다."""
    o = occ[occ["keco4"] != "NA"].groupby("keco4")["n_use"].sum()
    in_data = set(o.index[o >= cfg["occupations"]["min_use_postings"]])
    in_cfg = set(cfg["occupations"]["keco4"])
    excluded_groups = {str(g) for g in cfg["occupations"]["exclude_keco2"]}
    dropped = {c for c in in_data - in_cfg if str(c)[:2] in excluded_groups}
    in_data -= dropped
    rows = sorted(in_data | in_cfg | dropped)
    status = [OCC_STATUS[3] if c in dropped else OCC_STATUS[2] if c in in_data and c in in_cfg
              else OCC_STATUS[0] if c in in_cfg else OCC_STATUS[1] for c in rows]
    out = pd.DataFrame({"keco4": rows, "n_use": o.reindex(rows, fill_value=0).astype("int64").to_numpy(),
                        "status": status})
    order = out["status"].map({s: i for i, s in enumerate(OCC_STATUS)})
    return out.assign(_o=order).sort_values(["_o", "keco4"]).drop(columns="_o").reset_index(drop=True)


def _t11_salary(base: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame({"year": base["year"].to_numpy(),
                       "salary": pd.to_numeric(base["salary_annual_min"]).astype("float64").to_numpy()})
    g = df.groupby("year")["salary"]
    out = pd.DataFrame({"n": g.size(), "n_stated": g.count()}).astype("int64")
    out["stated_rate"] = _rate(out["n_stated"], out["n"])
    q = g.quantile(list(SALARY_QUANTILES.values())).unstack()
    for name, p in SALARY_QUANTILES.items():
        out[name] = q[p] if len(q) else np.nan
    return out.rename_axis("year").reset_index()


def _t12_sido_other(base: pd.DataFrame) -> pd.DataFrame:
    vc = base["sido"].astype("string").fillna(MISSING_LABEL).value_counts()
    out = vc.rename("n").rename_axis("sido").reset_index().astype({"n": "int64"})
    return out.sort_values(["n", "sido"], ascending=[False, True], ignore_index=True)


def _t12b_sido_other_words(words: pd.DataFrame) -> pd.DataFrame:
    out = words[["word", "n"]].astype({"n": "int64"})
    return out.sort_values(["n", "word"], ascending=[False, True], ignore_index=True)


# 후보 숙련 표의 설정 항목과, 그 항목의 이름을 찾는 숙련 설명 파일의 열
_CANDIDATE_SETTINGS = (("skills.adjacent_candidate_details", "detail"), ("skills.placebo_candidate_domains", "domain"))


def _absent_names(codebook: pd.DataFrame, column: str, names: list[str]) -> list[str]:
    """`names` 가운데 숙련 설명 파일의 `column`에 한 행도 없는 이름."""
    return [n for n in names if not (codebook[column] == n).any()]


def _candidate_rows(codebook: pd.DataFrame, column: str, names: list[str], key: str) -> pd.DataFrame:
    """숙련 설명 파일에서 `column`의 값이 `names`에 든 행을 돌려준다. 설명 파일에 한 행도 없는 이름이 있으면
    그 이름을 모두 적은 ValueError를 낸다(`key`는 오류 문장에 쓰는 설정 항목 이름이다)."""
    absent = _absent_names(codebook, column, names)
    if absent:
        raise ValueError(f"`{key}`의 이름 가운데 숙련 설명 파일에 없는 것이 있습니다: {absent}")
    return codebook[codebook[column].isin(names)]


def _check_report_inputs(cfg: dict, root: Path, codebook: pd.DataFrame | None = None) -> None:
    """2단계(`report`)의 입력 가운데 중간 자료 없이 확인할 수 있는 것을 확인한다. `build_tables`가 처음에 부르고,
    `all` 명령은 오래 걸리는 1단계(`scan`)를 시작하기 전에 부른다.

    - 후보 숙련의 이름(`skills.adjacent_candidate_details`, `skills.placebo_candidate_domains`)이 숙련 설명 파일에
      있는지. 없는 이름이 있으면 두 항목의 없는 이름을 모두 적은 ValueError를 낸다.
    - 직종 분류표(`data.keco`) 파일이 있는지. 없으면 FileNotFoundError를 낸다.

    산업별 AI 노출 지수의 외부 자료가 없는 것은 오류가 아니다(`t15_aiie_coverage`만 만들지 않는다).
    `codebook`을 주지 않으면 설정의 숙련 설명 파일을 읽는다.
    """
    validate.validate_config(cfg)
    if codebook is None:
        codebook = skills.load_codebook(root / cfg["data"]["codebook"])
    problems = []
    for key, column in _CANDIDATE_SETTINGS:
        absent = _absent_names(codebook, column, list(_config_at(cfg, key)))
        if absent:
            problems.append(f"`{key}` {absent}")
    if problems:
        raise ValueError("후보 숙련의 이름 가운데 숙련 설명 파일에 없는 것이 있습니다: " + ", ".join(problems))
    keco = root / cfg["data"]["keco"]
    if not keco.exists():
        raise FileNotFoundError(f"직종 분류표 파일이 없습니다: {keco.name} (찾은 위치: {keco.parent}, 설정 `data.keco`)")


def _skill_counts(outcome: pd.DataFrame, treat: pd.DataFrame, alive: np.ndarray,
                  cfg: dict) -> tuple[pd.DataFrame, int, int, set[str]]:
    """제외 규칙 네 단계를 적용한 공고에서 숙련마다 공고 수를 센다. `outcome`은 AI 에이전트 숙련코드를 뺀 숙련 긴 표
    (열 `pos`, `skill_code`)이다.

    대상 공고(설정의 처치 정의로 처치 공고이거나 비교 공고인 공고)와 처치 공고는 `t03b_sparse_filter_grid`와 같다
    (`_sparse_filter_sample`). 돌려주는 값: 숙련 코드를 번호로 한 표(열 `n`, `n_treated`), 대상 공고 수, 처치 공고 수,
    설정의 희소 필터(`skills.sparse_filter`)를 통과한 숙련 코드의 집합.
    """
    lk, sample_idx, arm = _sparse_filter_sample(alive, treat, outcome["pos"].to_numpy(), outcome["skill_code"], cfg)
    treated = set(sample_idx[arm == 1].tolist())
    n_postings = len(sample_idx)
    count = lambda d: d.groupby("skill_code", observed=True)["posting_id"].nunique()
    counts = pd.DataFrame({"n": count(lk), "n_treated": count(lk[lk["posting_id"].isin(treated)])})
    counts = counts.fillna(0).astype("int64")
    counts.index = counts.index.astype(str)
    sf = cfg["skills"]["sparse_filter"]
    kept = set(skills.sparse_filter(lk, treated, n_postings, sf["min_rate"], sf["min_treated"])) if n_postings else set()
    return counts, n_postings, len(treated), kept


def _t13_adjacent_candidates(codebook: pd.DataFrame, counts: pd.DataFrame, n_postings: int, n_treated: int,
                             kept: set[str], agent_codes: set[str], cfg: dict) -> pd.DataFrame:
    """인접 숙련 후보: 숙련 설명 파일에서 세부 분류가 `skills.adjacent_candidate_details`에 든 숙련 가운데 AI 에이전트
    숙련코드가 아닌 것마다 한 행. 공고 수는 제외 규칙 네 단계를 적용한 공고 가운데 설정의 처치 정의로 처치 공고이거나
    비교 공고인 공고에서 센 것(`_skill_counts`)이고, 공고가 없는 후보도 0으로 싣는다. 공고가 많은 순."""
    cand = _candidate_rows(codebook, "detail", list(cfg["skills"]["adjacent_candidate_details"]),
                           "skills.adjacent_candidate_details")
    cand = cand[~cand["skill_code"].isin(agent_codes)]
    out = cand[["skill_code", "skill_name", "detail"]].reset_index(drop=True)
    c = counts.reindex(out["skill_code"], fill_value=0)
    out["n"] = c["n"].to_numpy(dtype="int64")
    out["rate"] = out["n"] / n_postings if n_postings else np.nan
    out["n_treated"] = c["n_treated"].to_numpy(dtype="int64")
    out["treated_rate"] = out["n_treated"] / n_treated if n_treated else np.nan
    out["passes_sparse_filter"] = out["skill_code"].isin(kept)
    return out.sort_values(["n", "skill_code"], ascending=[False, True], ignore_index=True)


def _t14_placebo_candidates(codebook: pd.DataFrame, counts: pd.DataFrame, n_postings: int, kept: set[str],
                            cfg: dict) -> pd.DataFrame:
    """placebo 숙련 후보: 제외 규칙 네 단계를 적용한 공고 가운데 처치 공고와 비교 공고에서 설정의 희소 필터를 통과한 숙련(`_skill_counts`,
    `t03b_sparse_filter_grid`와 같은 공고와 같은 처치 공고) 가운데 대분류가 `skills.placebo_candidate_domains`에
    든 것마다 한 행. 처치 공고와 비교 공고의 출현율은 싣지 않는다. 공고가 많은 순."""
    cand = _candidate_rows(codebook, "domain", list(cfg["skills"]["placebo_candidate_domains"]),
                           "skills.placebo_candidate_domains")
    cand = cand[cand["skill_code"].isin(kept)]
    out = cand[["skill_code", "skill_name", "domain", "sub"]].reset_index(drop=True)
    out["n"] = counts.reindex(out["skill_code"], fill_value=0)["n"].to_numpy(dtype="int64")
    out["rate"] = out["n"] / n_postings if n_postings else np.nan
    return out.sort_values(["n", "skill_code"], ascending=[False, True], ignore_index=True)


def _missing_external(cfg: dict, root: Path) -> list[str]:
    """`t15_aiie_coverage`에 쓰는 외부 자료 가운데 설정에 항목이 없거나 파일이 없는 것을 보고서에 쓸 글로 돌려준다."""
    out = []
    for key in AIIE_DATA_KEYS:
        rel = cfg["data"].get(key)
        if rel is None:
            out.append(f"`data.{key}`(설정에 항목이 없음)")
        elif not (root / rel).exists():
            out.append(f"`{rel}`(`data.{key}`)")
    return out


def _t15_aiie_coverage(base: pd.DataFrame, arm: np.ndarray, alive: np.ndarray, cfg: dict, root: Path,
                       ksic_valid: set[str]) -> pd.DataFrame:
    """제외 규칙 네 단계를 적용한 공고에 산업 소분류별 AIIE가 붙는 비율과 값의 분위수(`전체`와, 설정의 처치 정의로
    나눈 처치 공고, 비교 공고. `arm`은 `_arm`의 값).

    외부 자료가 없으면 빈 표를 돌려주고 attrs[MISSING_EXTERNAL_ATTR]에 없는 파일의 목록을 둔다. 값이 붙은 소분류 수와
    산업분류표의 소분류 수는 attrs의 `n_sub_with_aiie`, `n_sub_total`에 둔다(보고서의 설명에 쓴다).
    판단으로 만든 KSIC-ISIC 표(`data.ksic_isic`, `isic`이 빈 줄 포함)의 소분류와 산업분류표(`ksic_valid`)의 소분류가
    다르면, 한쪽에만 있는 소분류를 적은 ValueError를 낸다.
    """
    missing = _missing_external(cfg, root)
    if missing:
        out = pd.DataFrame()
        out.attrs[MISSING_EXTERNAL_ATTR] = missing
        return out
    d = cfg["data"]
    crosswalk_subs = confounders.ksic_isic_subs(root / d["ksic_isic"])
    only_crosswalk, only_table = sorted(crosswalk_subs - ksic_valid), sorted(ksic_valid - crosswalk_subs)
    if only_crosswalk or only_table:
        raise ValueError(f"KSIC-ISIC 표(`data.ksic_isic`)와 산업분류표(`data.ksic`)의 소분류가 다릅니다. "
                         f"KSIC-ISIC 표에만 있는 소분류: {only_crosswalk}, 산업분류표에만 있는 소분류: {only_table}")
    naics_isic = confounders.load_naics_isic(root / d["naics_isic"])
    aiie = confounders.load_aiie(root / d["aiie"], set(naics_isic["naics6"].str[:4]))
    table = confounders.aiie_by_ksic_sub(aiie, naics_isic, confounders.load_ksic_isic(root / d["ksic_isic"]))
    value = confounders.attach_aiie(base["ksic_sub"], table).to_numpy()
    rows = []
    for label, mask in _arm_groups(alive, arm):
        v = pd.Series(value[mask]).dropna()
        n = int(mask.sum())
        row = {"arm": label, "n": n, "n_with_aiie": len(v), "with_aiie_rate": len(v) / n if n else np.nan}
        row.update({k: float(v.quantile(q)) if len(v) else np.nan for k, q in AIIE_QUANTILES.items()})
        rows.append(row)
    out = pd.DataFrame(rows).astype({"n": "int64", "n_with_aiie": "int64"})
    out.attrs["n_sub_total"] = len(ksic_valid)
    out.attrs["n_sub_with_aiie"] = len(ksic_valid & set(table["ksic_sub"]))
    return out


def _t16_detail_by_arm(base: pd.DataFrame, outcome: pd.DataFrame, arm: np.ndarray) -> pd.DataFrame:
    """제외 규칙 네 단계를 적용한 공고를 설정의 처치 정의로 나눈 처치 공고와 비교 공고(`_arm`)의 공고 수, AI 에이전트
    숙련코드를 뺀 숙련 수(`outcome`은 AI 에이전트 숙련코드를 뺀 숙련 긴 표)의 평균과 분위수, 본문 글자 수
    (`text_len`)의 평균과 중앙값.

    마지막 행은 세분류(`keco4`) × 연도마다 처치 공고 평균에서 비교 공고 평균을 뺀 값을 처치 공고 수로 가중 평균한
    것이다. 두 집단이 모두 있는 칸만 쓰고, 이 행의 `n`은 그 칸들의 처치 공고 수이다.
    """
    n_skills = np.bincount(outcome["pos"].to_numpy(), minlength=len(base)).astype("float64")
    text_len = pd.to_numeric(base["text_len"]).astype("float64").to_numpy()
    rows = []
    for a, label in ARM_LABELS.items():
        sel = arm == a
        s, t = pd.Series(n_skills[sel]), pd.Series(text_len[sel])
        row = {"arm": label, "n": int(sel.sum()), "n_skills_mean": s.mean()}
        row.update({f"n_skills_{k}": s.quantile(q) for k, q in ARM_SKILL_QUANTILES.items()})
        row.update({"text_len_mean": t.mean(), "text_len_p50": t.median()})
        rows.append(row)

    sel = arm >= 0
    df = pd.DataFrame({"keco4": base["keco4"].astype("string").to_numpy()[sel],
                       "year": base["year"].to_numpy()[sel], "arm": arm[sel],
                       "n_skills": n_skills[sel], "text_len": text_len[sel]})
    within = {"arm": ARM_WITHIN_LABEL, "n": 0, "n_skills_mean": np.nan, "text_len_mean": np.nan}
    if set(df["arm"]) == {0, 1}:
        cell = df.groupby(["keco4", "year", "arm"])[["n_skills", "text_len"]].mean().unstack("arm")
        weight = df[df["arm"] == 1].groupby(["keco4", "year"]).size().reindex(cell.index, fill_value=0)
        both = cell[("n_skills", 1)].notna() & cell[("n_skills", 0)].notna() & (weight > 0)
        w = weight[both].astype("float64")
        within["n"] = int(w.sum())
        if w.sum() > 0:
            for col in ("n_skills", "text_len"):
                diff = (cell.loc[both, (col, 1)] - cell.loc[both, (col, 0)]).to_numpy()
                within[f"{col}_mean"] = float(np.average(diff, weights=w.to_numpy()))
    rows.append(within)
    cols = ["arm", "n", "n_skills_mean", *[f"n_skills_{k}" for k in ARM_SKILL_QUANTILES], "text_len_mean",
            "text_len_p50"]
    return pd.DataFrame(rows).reindex(columns=cols).astype({"n": "int64"})


def _t17_firm_treated(base: pd.DataFrame, arm: np.ndarray) -> pd.DataFrame:
    """제외 규칙 네 단계를 적용한 공고 가운데 사업자번호가 있는 공고에서, 설정의 처치 정의로 나눈 처치 공고와 비교
    공고(`_arm`)를 기업(`bzno`)별로 본 값. `key`, `설명`, `값` 세 열의 긴 표이며 기업 식별자는 싣지 않는다."""
    has_firm = base["bzno"].notna().to_numpy(dtype=bool)
    df = pd.DataFrame({"bzno": base["bzno"].astype("string").to_numpy(), "year": base["year"].to_numpy(),
                       "keco4": base["keco4"].astype("string").fillna(MISSING_LABEL).to_numpy(), "arm": arm})
    tr = df[(arm == 1) & has_firm]
    comp = df[(arm == 0) & has_firm]
    per_firm = tr.groupby("bzno").size().sort_values(ascending=False)
    n_treated = len(tr)
    fy = pd.MultiIndex.from_frame(tr[["bzno", "year"]])
    same_fy = fy.isin(pd.MultiIndex.from_frame(comp[["bzno", "year"]]))
    same_fyo = pd.MultiIndex.from_frame(tr[["bzno", "year", "keco4"]]).isin(
        pd.MultiIndex.from_frame(comp[["bzno", "year", "keco4"]]))
    rate = lambda k: k / n_treated if n_treated else np.nan
    rows = [
        ("n_treated", "처치 공고 수", n_treated),
        ("n_firms_treated", "처치 공고가 있는 기업 수", len(per_firm)),
        ("treated_per_firm_p50", "기업당 처치 공고 수의 중앙값", float(per_firm.median()) if len(per_firm) else np.nan),
        ("treated_per_firm_max", "기업당 처치 공고 수의 최댓값", int(per_firm.max()) if len(per_firm) else 0),
        ("top_firms_share", f"처치 공고가 많은 상위 {FIRM_TOP}개 기업의 처치 공고 비중",
         rate(int(per_firm.head(FIRM_TOP).sum()))),
        ("n_treated_same_firm_year", "같은 기업·같은 연도에 비교 공고도 있는 처치 공고 수", int(same_fy.sum())),
        ("same_firm_year_rate", "위 처치 공고의 비율(처치 공고 수 대비)", rate(int(same_fy.sum()))),
        ("n_firms_same_firm_year", "같은 연도에 처치 공고와 비교 공고를 함께 낸 기업 수",
         int(tr.loc[same_fy, "bzno"].nunique())),
        ("n_treated_same_firm_year_occ", "같은 기업·같은 연도·같은 세분류에 비교 공고도 있는 처치 공고 수",
         int(same_fyo.sum())),
        ("same_firm_year_occ_rate", "위 처치 공고의 비율(처치 공고 수 대비)", rate(int(same_fyo.sum()))),
    ]
    return pd.DataFrame(rows, columns=["key", "설명", "값"]).astype({"값": object})


def _t18_first_treated_firms(base: pd.DataFrame, arm: np.ndarray) -> pd.DataFrame:
    """제외 규칙 네 단계를 적용한 공고 가운데 사업자번호가 있는 공고에서, 설정의 처치 정의로 처치 공고를 처음 낸
    연도(`first_year`)별 기업 수와, 그 가운데 그 전 연도에 비교 공고(`_arm`)를 낸 기업 수와 그 비교 공고 수."""
    has_firm = base["bzno"].notna().to_numpy(dtype=bool)
    df = pd.DataFrame({"bzno": base["bzno"].astype("string").to_numpy(), "year": base["year"].to_numpy(), "arm": arm})
    first = df[(arm == 1) & has_firm].groupby("bzno")["year"].min()
    comp = df[(arm == 0) & has_firm]
    earlier = comp[comp["year"].to_numpy() < comp["bzno"].map(first).fillna(-1).to_numpy()]
    n_earlier = earlier.groupby("bzno").size().reindex(first.index, fill_value=0)
    firms = pd.DataFrame({"first_year": first.astype("int64"), "n_earlier_postings": n_earlier.astype("int64")})
    firms["with_earlier"] = (firms["n_earlier_postings"] > 0).astype("int64")
    out = firms.groupby("first_year").agg(n_firms=("with_earlier", "size"), n_firms_with_earlier=("with_earlier", "sum"),
                                          n_earlier_postings=("n_earlier_postings", "sum"))
    return out.astype("int64").reset_index()


def build_tables(cfg: dict, root: Path, s0_dir: Path) -> dict[str, pd.DataFrame]:
    """1단계의 중간 자료로 자료 점검 집계표를 만들어 {표 이름: 표}로 돌려준다.

    각 표는 `s0_dir/tables/{표 이름}.csv`(utf-8-sig)로도 쓰고, 그 경로를 표의 `attrs["csv_path"]`에 둔다.
    표에는 집계값만 넣는다(공고 번호, 사업자번호, 공고 원문은 넣지 않는다).
    `t15_aiie_coverage`에 쓰는 외부 자료(`data.aiie`, `data.naics_isic`, `data.ksic_isic`)의 설정 항목이나 파일이
    없으면 그 표만 만들지 않는다. 그때 그 표 자리에는 attrs[MISSING_EXTERNAL_ATTR]에 없는 파일을 적은 빈 표를 두고
    CSV는 쓰지 않는다. 설정 `period.fill.enabled`가 거짓이면 날짜를 채운 공고의 표(`FILL_TABLES`)는 돌려주는 사전에
    넣지 않고 CSV도 쓰지 않는다.
    """
    validate.validate_config(cfg)
    root, s0_dir = Path(root), Path(s0_dir)
    _require_interim(s0_dir, cfg)
    _check_scan_config(cfg, _read_scan_meta(s0_dir))
    codebook = skills.load_codebook(root / cfg["data"]["codebook"])
    codes = treatment.agent_skill_codes(codebook, cfg)
    agent_codes = codes["dev"] | codes["use"]
    # 후보 이름이 숙련 설명 파일에 없거나 직종 분류표가 없으면 오래 걸리는 집계 전에 멈춘다
    _check_report_inputs(cfg, root, codebook)
    keco = classif.load_keco(root / cfg["data"]["keco"])
    ksic_valid = classif.load_ksic_sub(root / cfg["data"]["ksic"])
    ked_lines = _t04b_ked_lines(ked.extract_ked_firm(root / cfg["data"]["ked_firm"]))  # 줄 수와 기업 수만 쓴다

    base_ids = pq.read_table(s0_dir / "base.parquet", columns=["posting_id"]).column("posting_id")
    base = pd.read_parquet(s0_dir / "base.parquet", columns=_TABLE_BASE_COLUMNS)
    treat = pd.read_parquet(s0_dir / "treat.parquet")
    _check_same_postings(treat, base)
    flags = treat[_TREAT_FLAGS].astype(bool)
    long = _read_long_positions(s0_dir, base_ids)
    del base_ids
    stats = pd.read_parquet(s0_dir / "scan_stats.parquet")
    occ = pd.read_parquet(s0_dir / "occ_year_all.parquet")
    kw = pd.read_parquet(s0_dir / "kw_pattern_hits.parquet")
    psm = pd.read_parquet(s0_dir / "post_start_missing.parquet")
    gap = pd.read_parquet(s0_dir / "post_end_gap.parquet")
    other_words = pd.read_parquet(s0_dir / "sido_other_words.parquet")
    fill_on = bool(cfg["period"]["fill"]["enabled"])

    alive, counts = _exclusions(base, cfg)
    arm = _arm(treat, alive, cfg)
    # 사업자번호와 숙련 제외 전의 base.parquet 모든 공고를 설정의 처치 정의로 나눈 값(t04, t05, t09a)
    arm_base = _arm(treat, np.ones(len(base), dtype=bool), cfg)
    years = sorted({int(y) for y in cfg["data"]["years"]} | {int(y) for y in base["year"].unique()})
    outcome = skills.drop_codes(long[["pos", "skill_code"]], agent_codes)   # 결과변수로 쓰는 숙련만 남긴 긴 표
    skill_counts, n_sample, n_sample_treated, kept = _skill_counts(outcome, treat, alive, cfg)

    tables = {
        "t01_exclusions": _t01_exclusions(stats, counts, cfg),
        "t01b_post_start_missing": _t01b_post_start_missing(psm),
        "t01c_post_end_gap": _t01c_post_end_gap(gap),
    }
    if fill_on:
        tables["t01d_date_fill"] = _t01d_date_fill(pd.read_parquet(s0_dir / f"{FILL_OUTPUT_NAME}.parquet"))
    tables |= {
        "t02a_treated_by_period": _t02a_treated_by_period(base, flags, arm, alive, cfg),
        "t02b_treated_by_occ": _t02b_treated_by_occ(base, flags, keco, cfg),
        "t02c_occ_group_treated": _t02c_occ_group_treated(base, arm, alive, keco, cfg),
        "t02d_date_source_by_period": _t02d_date_source_by_period(base, arm, alive, cfg),
    }
    if fill_on:
        tables["t02e_fill_gap_by_period"] = _t02e_fill_gap_by_period(base, arm, alive, cfg)
    tables |= {
        "t03a_skills_per_posting": _t03a_skills_per_posting(base),
        "t03b_sparse_filter_grid": _t03b_sparse_filter_grid(long, treat, alive, agent_codes, cfg),
        "t03c_skills_by_source_column": _t03c_skills_by_source_column(long, codebook),
        "t03d_match_stats": stats[["year", "n_tokens", "n_unmatched", "n_length_mismatch_rows"]].copy(),
        "t03e_no_outcome_skill": _t03e_no_outcome_skill(long, arm, alive, agent_codes),
        "t04_bzno_ked_by_treatment": _t04_bzno_ked_by_treatment(base, arm_base),
        "t04b_ked_lines": ked_lines,
        "t05_source": _t05_source(base, flags, arm_base),
        "t06a_overlap": _t06a_overlap(base, flags),
        "t06b_agent_skill_by_year": _t06b_agent_skill_by_year(long, base, codebook, codes, years),
        "t06c_keyword_pattern_hits": _t06c_keyword_pattern_hits(kw, cfg),
        "t08_2026_monthly": _t08_last_year_monthly(base, cfg),
        "t09a_by_ksic_sub": _t09a_by_ksic_sub(base, flags, arm_base),
        "t09b_ksic_not_in_table": _t09b_ksic_not_in_table(base, ksic_valid),
        "t10a_use_by_occ_year": _t10a_use_by_occ_year(base, flags, alive, keco, cfg, years),
        "t10b_occ_list_check": _t10b_occ_list_check(occ, cfg),
        "t11_salary": _t11_salary(base),
        "t12_sido_other": _t12_sido_other(base),
        "t12b_sido_other_words": _t12b_sido_other_words(other_words),
        "t13_adjacent_candidates": _t13_adjacent_candidates(codebook, skill_counts, n_sample, n_sample_treated, kept,
                                                            agent_codes, cfg),
        "t14_placebo_candidates": _t14_placebo_candidates(codebook, skill_counts, n_sample, kept, cfg),
        "t15_aiie_coverage": _t15_aiie_coverage(base, arm, alive, cfg, root, ksic_valid),
        "t16_detail_by_arm": _t16_detail_by_arm(base, outcome, arm),
        "t17_firm_treated": _t17_firm_treated(base, arm),
        "t18_first_treated_firms": _t18_first_treated_firms(base, arm),
    }
    table_dir = rawio._check_out_path(s0_dir / "tables")
    table_dir.mkdir(parents=True, exist_ok=True)
    for name in FILL_TABLES:   # 날짜를 채우지 않는 설정에서 만들지 않은 표. 앞선 실행의 CSV가 남아 있으면 지운다
        if name not in tables:
            (table_dir / f"{name}.csv").unlink(missing_ok=True)
    for name, df in tables.items():
        path = table_dir / f"{name}.csv"
        if MISSING_EXTERNAL_ATTR in df.attrs:   # 만들지 않은 표. 앞선 실행의 CSV가 남아 있으면 지운다
            path.unlink(missing_ok=True)
            continue
        df.to_csv(path, index=False, encoding="utf-8-sig")
        df.attrs["csv_path"] = str(path)
    return tables


def _timing_features(base: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:
    """학습 시간 측정용 설명 변수 표. 숫자 열은 float64로, 나머지 열은 category로 바꾼다.

    `text_len_log`는 본문 글자 수(`text_len`)에 1을 더한 값의 자연로그이다.
    """
    cols = [_TIMING_DERIVED.get(c, c) for c in TIMING_FEATURES]
    X = base[cols].iloc[rows].reset_index(drop=True)
    X["text_len_log"] = np.log1p(pd.to_numeric(X.pop("text_len")).astype("float64"))
    X = X[TIMING_FEATURES]
    for c in TIMING_FEATURES:
        if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c]):
            X[c] = X[c].astype("float64")
        else:
            X[c] = X[c].astype("string").astype("category")
    return X


def time_outcome_model(base: pd.DataFrame, treat: pd.DataFrame, long: pd.DataFrame, cfg: dict) -> dict:
    """숙련 하나에 대한 결과모형(LightGBM 분류 모형)을 한 번 학습하는 데 걸리는 시간을 잰다.

    대상은 제외 규칙을 모두 적용한 공고 가운데 설정의 처치 정의로 처치 공고(1)나 비교 공고(0)인 공고이다.
    결과변수는 희소 필터(설정의 기준)를 통과한 숙련 가운데 가장 많은 공고에 나온 숙련의 보유 여부이고,
    통과한 숙련이 없으면 전체에서 가장 많이 나온 숙련을 쓴다. 희소 필터의 대상 공고와 처치 공고는
    `t03b_sparse_filter_grid`와 같은 함수(`_sparse_filter_sample`)로 정한다. 교차적합(자료를 여러 묶음으로 나누어,
    한 묶음의 예측값을 나머지 묶음으로 학습한 모형으로 구하는 방법)에서 나눈 묶음 하나를 폴드라고 한다. 기업
    (`bzno`)으로 나눈 폴드 가운데 첫 번째 폴드를 뺀 나머지(학습 쪽)에서 비교 공고만으로 한 번 학습한다.
    `long`에서 AI 에이전트 숙련코드는 부르는 쪽이 미리 뺀다.

    처치 공고가 없거나, 비교 공고의 기업 수가 폴드 수보다 적거나, 학습할 공고가 없거나 결과변수가 한 값뿐이면
    측정하지 않고 `seconds`와 `est_total_hours`를 None으로 둔다.
    """
    est = cfg["estimation"]
    n_folds = int(est["n_folds"])
    sf = cfg["skills"]["sparse_filter"]
    out = {"skill_code": None, "n_train": 0, "n_features": len(TIMING_FEATURES), "seconds": None,
           "n_skills_kept": 0, "est_total_hours": None}

    base = base.reset_index(drop=True)
    treat = treat.reset_index(drop=True)
    _check_same_postings(treat, base)
    alive, _ = _exclusions(base, cfg)
    if not alive.any() or long.empty:
        return out

    # 숙련 긴 표는 공고 번호 대신 base에서의 위치로 다루고, 제외 규칙을 통과한 공고의 행만 남긴다.
    # 희소 필터의 대상은 t03b_sparse_filter_grid와 같은 함수로 정한다.
    pos = _positions(long["posting_id"], base["posting_id"])
    lk, sample_idx, arm = _sparse_filter_sample(alive, treat, pos, long["skill_code"], cfg)
    is_treated = arm == 1
    is_comparison = arm == 0
    if lk.empty:
        return out
    kept_codes = skills.sparse_filter(lk, set(sample_idx[is_treated].tolist()), len(sample_idx),
                                      sf["min_rate"], sf["min_treated"])
    out["n_skills_kept"] = len(kept_codes)
    n_by_code = lk.groupby("skill_code", observed=True)["posting_id"].nunique()
    if kept_codes:
        n_by_code = n_by_code[n_by_code.index.isin(kept_codes)]
    n_by_code = n_by_code.sort_index().sort_values(ascending=False, kind="stable")
    top = str(n_by_code.index[0])
    out["skill_code"] = top

    model_rows = sample_idx[is_treated | is_comparison]
    model_is_comparison = is_comparison[is_treated | is_comparison]
    bzno = base["bzno"].astype("string").iloc[model_rows]
    groups = bzno.fillna(base["posting_id"].astype("string").iloc[model_rows]).to_numpy()
    if not is_treated.any() or pd.unique(groups[model_is_comparison]).size < n_folds:
        return out

    train, _ = next(GroupKFold(n_splits=n_folds).split(np.zeros(len(model_rows)), groups=groups))
    train = train[model_is_comparison[train]]
    out["n_train"] = int(len(train))
    has_top = np.zeros(len(base), dtype=bool)
    has_top[lk.loc[(lk["skill_code"] == top).to_numpy(dtype=bool), "posting_id"].to_numpy()] = True
    y = has_top[model_rows[train]].astype("int64")
    if len(train) == 0 or y.min() == y.max():
        return out

    X = _timing_features(base, model_rows[train])
    model = LGBMClassifier(**est["lgbm"], random_state=est["seed"], verbose=-1)
    t0 = time.perf_counter()
    model.fit(X, y)
    seconds = time.perf_counter() - t0
    out["seconds"] = float(seconds)
    out["est_total_hours"] = float(seconds * n_folds * len(kept_codes) / 3600)
    return out


# --- 보고서 ----------------------------------------------------------------

# (절 번호, 절 제목, 표 이름). 제목은 설계 문서 6.6절의 항목 내용 그대로이다. 다만 1절 제목에는 표에 있는 분석
# 기간 단계를 더했다.
REPORT_SECTIONS = [
    (1, "제외 규칙 단계별 공고 수 (연도별): 분석 기간, 분석 대상 직종 한정, 사업자번호 없음, 숙련 없음. 게시일이 없는 "
        "공고 가운데 마감일로 게시일을 대신한 공고 수와 둘 다 없는 공고 수, 게시일과 마감일이 모두 있는 공고의 두 날짜 "
        "간격 분포, 두 날짜가 모두 없어 날짜를 채운 공고(같은 연도 파일의 앞 공고 날짜)의 앞뒤 날짜 간격 분포",
     ["t01_exclusions", "t01b_post_start_missing", "t01c_post_end_gap", "t01d_date_fill"]),
    (2, "기간별 처치 공고 수 (분기·반기·연 단위): 설정의 처치 정의로 나눈 처치 공고와 비교 공고, 키워드(엄격한 사전, "
        "넓은 사전), 개발 숙련, 생성형 AI 활용 숙련, 숙련코드 전체, 키워드와 숙련코드를 합친 것 각각과, 생성형 AI 활용 "
        "키워드 사전에 걸린 공고(`kw_use`). 분석 대상 세분류별·중분류별 처치 공고 수(처치 정의별). 기간별로 게시일을 "
        "가져온 곳별 공고 수와 날짜를 채운 공고의 앞뒤 날짜 간격별 공고 수",
     ["t02a_treated_by_period", "t02b_treated_by_occ", "t02c_occ_group_treated", "t02d_date_source_by_period",
      "t02e_fill_gap_by_period"]),
    (3, "공고당 숙련 수의 분포와 숙련별 출현율 (희소 필터 기준별 남는 숙련 수), 숙련 열별 숙련 수, "
        "숙련명으로 최종판 코드를 찾지 못한 숙련 수",
     ["t03a_skills_per_posting", "t03b_sparse_filter_grid", "t03c_skills_by_source_column", "t03d_match_stats",
      "t03e_no_outcome_skill"]),
    (4, "처치 여부별 사업자번호 보유 비율과 KED 결합 비율", ["t04_bzno_ked_by_treatment", "t04b_ked_lines"]),
    (5, "연도별 수집 사이트 구성, 사이트별 처치 공고 비율과 숙련 보유 비율", ["t05_source"]),
    (6, "키워드 처치와 숙련코드 처치의 겹침, 두 처치와 `ai_peripheral`의 겹침, 생성형 AI 활용 키워드 사전에 걸린 "
        "공고(`kw_use`)와 두 종류의 숙련코드 처치의 겹침, AI 에이전트 숙련코드 54개 각각의 기간별 공고 수, 키워드 "
        "표현별 공고 수",
     ["t06a_overlap", "t06b_agent_skill_by_year", "t06c_keyword_pattern_hits"]),
    (7, "숙련 1개에 대한 결과모형의 학습 시간 (미도입 공고를 전부 쓴 표본 기준)과 이로부터 계산한 전체 추정 시간", []),
    (8, "2026년의 월별 공고 수와 숙련 보유 비율", ["t08_2026_monthly"]),
    (9, "산업 소분류별·직종 세분류별 공고 수와 처치 공고 수, 소분류표에 없는 산업코드의 건수와 예",
     ["t09a_by_ksic_sub", "t09b_ksic_not_in_table"]),
    (10, "분석 대상 세분류 각각의 생성형 AI 활용 숙련 공고 수와 비율(연도별, 제외 규칙 적용 뒤). "
         "전체 공고에서 세분류별 생성형 AI 활용 숙련 공고 수를 다시 세어 3.4절의 목록과 같은지 확인",
     ["t10a_use_by_occ_year", "t10b_occ_list_check"]),
    (11, "연봉과 근무지", ["t11_salary", "t12_sido_other", "t12b_sido_other_words"]),
    (12, "인접 숙련 후보(6.7절 3번)와 placebo 숙련 후보의 숙련별 공고 수와 출현율",
     ["t13_adjacent_candidates", "t14_placebo_candidates"]),
    (13, "산업별 AI 노출 지수가 붙은 공고의 비율과 값의 분포", ["t15_aiie_coverage"]),
    (14, "처치 여부별 공고당 숙련 수와 본문 글자 수, 처치 공고가 있는 기업 수와 기업별 집중도, 같은 기업·같은 연도에 "
         "비교 공고도 있는 처치 공고의 수, 처치 공고를 처음 낸 연도별 기업 수와 그 전에 낸 비교 공고 수",
     ["t16_detail_by_arm", "t17_firm_treated", "t18_first_treated_firms"]),
]

# 절 제목 바로 아래에 싣는 문장. 그 절에서 처음 나오는 용어의 정의이다.
REPORT_SECTION_INTROS = {
    3: "희소 필터는 출현율이 기준보다 낮거나 처치 공고에서 나온 횟수가 기준보다 적은 숙련을 결과변수에서 빼는 규칙이다.",
    7: ("결과모형은 공고의 특성으로 그 숙련이 공고에 들어 있을 확률을 예측하는 모형이다. 제목의 `미도입 공고`(설계 문서 "
        "6.6절의 표현)는 이 보고서의 비교 공고와 같은 공고이다."),
    12: ("인접 숙련은 AI 에이전트 숙련코드와 내용이 가까워 처치 공고에 함께 나오기 쉬운 숙련이다. 결과변수에 그대로 "
         "두되 결과표에 따로 표시한다(`proposal.md` 3.2절). placebo 숙련은 생성형 AI 활용과 관계가 없어야 하는 숙련이며, "
         "같은 방법으로 추정했을 때 이 숙련에서 효과가 0으로 나오는지를 보아 분석 방법에 문제가 없는지 점검하는 데 쓴다"
         "(`proposal.md` 7장 2번). 두 목록은 이 절의 후보 표를 본 뒤 정한다(`docs/decisions.md` 결정 33)."),
    13: ("교란변수는 처치 여부와 결과 양쪽에 영향을 주어, 통제하지 않으면 처치의 효과로 잘못 읽히는 변수이다. 산업별 AI "
         "노출 정도를 교란변수로 쓰려고 Felten, Raj, Seamans의 산업별 AI 노출 지수(AIIE) 가운데 언어 모형 판을 쓴다"
         "(`docs/decisions.md` 결정 30). AIIE는 직업에 필요한 능력이 AI 응용 분야(이 판에서는 언어 모형)와 얼마나 "
         "관련되는지를 나타내는 직업별 지수를 산업의 직업 구성으로 가중 평균한 값이며, 미국 산업분류(NAICS) 4자리 산업마다 "
         "있다. 이 값을 한국표준산업분류(KSIC) 소분류로 옮기려고 세 표를 잇는다. KSIC 소분류를 판단으로 만든 표"
         "(`experiments/crosswalks/ksic10_sub_to_isic4.csv`)로 국제표준산업분류(ISIC Rev.4) 코드의 앞자리에 잇고, "
         "미국 통계국의 NAICS 2017 → ISIC Rev.4 연계표로 그 ISIC 코드에 이어진 NAICS 6자리 산업을 찾은 뒤, 그 앞 "
         "4자리 산업 가운데 AIIE 값이 있는 서로 다른 산업의 값을 단순 평균한다(`aabundle/confounders.py`)."),
    14: ("이 절의 처치 공고와 비교 공고는 제외 규칙 네 단계를 모두 적용한 공고를 설정의 처치 정의"
         "(`treatment.definition`)로 나눈 것이다(`treatment.select_arms`)."),
}
# 보고서 머리와 `t02a`의 소제목에 쓰는 기간 단위(`period.unit`)의 한국어 이름
PERIOD_UNIT_LABELS = {"quarter": "분기", "half": "반기", "year": "연"}
assert set(PERIOD_UNIT_LABELS) == set(validate.PERIOD_UNITS)

# 보고서에서 이 열의 값마다 따로 나눠 싣는 표. 행이 30개를 넘어 뒤쪽 행이 잘리지 않게 하기 위해서다.
REPORT_SPLIT = {"t01b_post_start_missing": "level", "t02a_treated_by_period": "unit",
                "t02b_treated_by_occ": "level", "t05_source": "year", "t06a_overlap": "cross",
                "t06b_agent_skill_by_year": "kind", "t10b_occ_list_check": "status"}
# 나눠 실을 때 소제목에 값 대신 쓰는 한국어 이름(값은 뒤에 코드 표시로 붙인다)
REPORT_SPLIT_LABELS = {"t02a_treated_by_period": {**PERIOD_UNIT_LABELS, "period": "설정의 기간"}}
# 나눠 실을 때 그 부분에서 값이 하나도 없으면 빼는 열. None이면 모든 열이 대상이다.
# t06a는 두 교차표의 열이 서로 달라서, t01b는 연도 행에 수집 사이트가 없어서다.
REPORT_DROP_EMPTY_COLUMNS = {"t06a_overlap": None, "t01b_post_start_missing": ["source"]}
# 보고서에 싣는 행 수를 표마다 따로 정한 것. None이면 모든 행을 싣는다. 여기 없는 표는 REPORT_MAX_ROWS이다.
# t06b는 설계 문서 6.6절 6번이 AI 에이전트 숙련코드 54개 각각을 보이라고 해서다. t06c는 키워드 사전의 표현 모두를,
# t13과 t14는 후보 숙련 모두를 보고 목록을 정해야 해서다. t01b와 t01d는 연도·수집 사이트별 행이 연도 순이어서, 행 수를
# 줄이면 마지막 연도들의 행이 먼저 빠지기 때문이다.
REPORT_MAX_ROWS_BY_TABLE = {"t01b_post_start_missing": None, "t01d_date_fill": None, "t06b_agent_skill_by_year": None,
                            "t06c_keyword_pattern_hits": None, "t13_adjacent_candidates": None,
                            "t14_placebo_candidates": None}
# 보고서에서 정규표현식 패턴이 든 열. 값을 코드 표시(백틱)로 감싼다.
REPORT_CODE_COLUMNS = {"pattern"}
_PLAIN_INT_COLUMNS = {"year", "month", "min_treated", "first_year"}

_BASE_NOTE = "`base.parquet`의 공고(분석 기간 안, 분석 대상 세분류. 사업자번호와 숙련 제외 전)"
_ALIVE_NOTE = "제외 규칙 네 단계(분석 기간, 분석 대상 세분류, 사업자번호, 숙련)를 모두 적용한 공고"
_POST_YEAR = "연도와 기간은 게시일(`post_start`, 없으면 `post_end`) 기준이다."
# 날짜를 채우는 설정(`period.fill.enabled`)일 때 `_POST_YEAR` 대신 쓰는 문장
_POST_YEAR_FILL = ("연도와 기간은 게시일(`post_start`, 없으면 `post_end`, 둘 다 없으면 같은 연도 파일에서 행 순서로 가장 "
                   "가까운 앞 공고의 날짜) 기준이다.")
# 처치 정의(`treatment.definition`)마다 처치 공고를 한 문장으로 설명한 것. 보고서 머리에 쓴다(`treatment.select_arms`).
DEFINITION_TEXT = {
    "all": "AI 에이전트 숙련코드(개발 숙련이나 생성형 AI 활용 숙련)가 있거나 본문이 AI 에이전트 키워드 사전"
           "(`treatment.keyword_dictionary`로 고른 사전)에 걸린 공고를 처치 공고로 본다.",
    "dev": "개발 숙련(숙련코드)이 있는 공고를 처치 공고로 본다.",
    "use": "생성형 AI 활용 숙련(숙련코드)이 있는 공고를 처치 공고로 본다. 본문이 생성형 AI 활용 키워드 사전에만 걸린 "
           "공고는 처치 공고가 아니다.",
    "keyword": "본문이 AI 에이전트 키워드 사전(`treatment.keyword_dictionary`로 고른 사전)에 걸린 공고를 처치 공고로 본다.",
    "code": "개발 숙련이나 생성형 AI 활용 숙련(숙련코드)이 있는 공고를 처치 공고로 본다.",
    "use_keyword": "본문이 생성형 AI 활용 키워드 사전(`treatment.keywords.use`)에 걸린 공고를 처치 공고로 본다.",
    "use_any": "생성형 AI 활용 숙련(숙련코드)이 있거나 본문이 생성형 AI 활용 키워드 사전(`treatment.keywords.use`)에 "
               "걸린 공고를 처치 공고로 본다.",
}
assert set(DEFINITION_TEXT) == set(treatment.DEFINITIONS)


def _arm_note(cfg: dict) -> str:
    """처치 공고와 비교 공고를 나누는 표의 설명에 넣는 한 문장. 설정의 처치 정의 값을 적는다."""
    return f"처치 공고와 비교 공고는 설정의 처치 정의(`treatment.definition` = `{cfg['treatment']['definition']}`)를 따른다."


_COMPARISON_TEXT = ("비교 공고는 어떤 처치 정의에서도 같다: 본문이 AI 에이전트 키워드 사전(`treatment.keyword_dictionary`로 "
                    "고른 사전)과 생성형 AI 활용 키워드 사전에 모두 걸리지 않고 AI 에이전트 숙련코드도 없는 공고이다.")
_TREAT_ALL_NOTE = ("`treat_all`은 숙련코드 처치(`treat_code`)와 엄격한 사전의 키워드 처치(`kw_strict`)를 합친 것이며, "
                   "`treatment.keyword_dictionary`를 무엇으로 정해도 엄격한 사전을 쓴다.")
_KW_USE_NOTE = ("`kw_use`는 생성형 AI 활용 키워드 사전(`treatment.keywords.use`)에 걸린 공고를 표시한다. 이 표시만 "
                "있는 공고는 처치 정의가 `use_keyword`나 `use_any`일 때 처치 공고이고, 나머지 정의에서는 처치 공고에도 "
                "비교 공고에도 들지 않는다. 어떤 정의에서도 비교 공고는 아니다.")


def _table_notes(cfg: dict) -> dict[str, str]:
    """표마다 대상 공고와 연도 기준을 적은 설명."""
    last = max(int(y) for y in cfg["data"]["years"])
    definition = f"`{cfg['treatment']['definition']}`"
    sf = cfg["skills"]["sparse_filter"]
    sf_text = f"`min_rate` {sf['min_rate']}, `min_treated` {sf['min_treated']}"
    k = cfg["treatment"]["keywords"]
    in_both = [p for p in k["use"] if p in set(k["strict"]) | set(k["broad_extra"])]
    both_note = (" " + ", ".join(f"`{p}`" for p in in_both) + "는 넓은 사전과 생성형 AI 활용 사전에 모두 있어, 두 사전의 "
                 "행에 한 번씩 센다.") if in_both else ""
    not_in_table = cfg["industry"]["not_in_table"]
    ksic_missing = not_in_table == "missing"
    fill_on = bool(cfg["period"]["fill"]["enabled"])
    max_gap = cfg["period"]["fill"]["max_gap_days"]
    post_year = _POST_YEAR_FILL if fill_on else _POST_YEAR
    fill_rule = ("두 날짜가 모두 없는 공고는 같은 연도 파일에서 행 순서로 가장 가까운 앞 공고의 날짜를 받는다"
                 "(`period.fill.enabled`, 설계 문서 결정 38). ")
    if fill_on:
        no_date_text = ("`in_period` 단계에서는 게시일이 분석 기간 밖인 공고와, 두 날짜가 모두 없고 파일에서 그 앞에 날짜 "
                        "있는 공고가 없어 날짜를 채우지 못한 공고가 함께 빠진다. ")
        base_missing = ("두 날짜가 모두 없는 공고는 같은 연도 파일에서 행 순서로 가장 가까운 앞 공고의 날짜를 받는다. 받을 "
                        "날짜가 없는 공고와 게시일(마감일, 채운 날짜 포함)이 분석 기간 밖인 공고는 `in_period` 단계에서 "
                        "빠지므로 `base.parquet`에 없다. ")
    else:
        no_date_text = ("`in_period` 단계에서는 게시일이 분석 기간 밖인 공고와, `post_start`와 `post_end`가 모두 날짜로 "
                        "쓸 수 없어 게시일을 정할 수 없는 공고가 함께 빠진다. ")
        base_missing = ("게시일도 마감일도 쓸 수 없는 공고와 마감일로 정한 게시일이 분석 기간 밖인 공고는 `in_period` "
                        "단계에서 빠지므로 `base.parquet`에 없다. ")
    gap_rule = (f"설정 `period.fill.max_gap_days` = {max_gap}이어서, `step` = `fill_gap` 행은 " + _BASE_NOTE + " 가운데 "
                "날짜를 채웠지만 앞뒤 날짜의 간격이 이 일수보다 크거나 뒤에 날짜 있는 공고가 없어 분석 기간 단계에서 빠진 "
                "공고를 센다(`year`는 채운 날짜의 연도). " if max_gap is not None else "")
    arm_note = _arm_note(cfg)
    # 사업자번호와 숙련 제외 전의 공고(base.parquet)를 처치 정의로 나눈 표(t04, t05, t09a)의 설명
    base_arm_note = (arm_note + " 이 표는 사업자번호와 숙련 제외 전의 공고에 처치 정의를 적용했으므로, 사업자번호가 "
                     "없는 처치 공고도 센다.")
    return {
        "t01_exclusions": (
            "`in_period`, `in_occ` 두 단계는 `scan_stats.parquet`(공고 파일의 모든 공고)에서 왔고 `year`는 공고 파일의 "
            "연도이다. `has_bzno`, `has_skill` 두 단계는 " + _BASE_NOTE + "에 `sample.apply_exclusions`를 적용한 것이고 "
            "`year`는 게시일(`post_start`, 없으면 `post_end`" + (", 둘 다 없으면 채운 날짜" if fill_on else "")
            + ")의 연도이다. 두 연도 기준이 다르므로, 같은 `year` 행이라도 "
            "`in_occ`의 `n_after`와 `has_bzno`의 `n_before`는 같지 않을 수 있다. 게시일은 `post_start`이고, "
            "게시일(`post_start`)을 날짜로 쓸 수 없으면 마감일(`post_end`)을 게시일로 쓴다. "
            + (fill_rule if fill_on else "") + no_date_text + gap_rule
            + ("마감일로 게시일을 정한 공고의 수와 날짜를 채운 공고의 수(`n_filled`)는 `t01b_post_start_missing`에 "
               "있다. 그 표의 `n_no_date`는 날짜를 채우기 전의 수이므로, 게시일을 정할 수 없어 `in_period` 단계에서 "
               "실제로 빠진 공고의 수는 `n_no_date − n_filled`이다." if fill_on else
               "게시일을 정할 수 없는 공고의 수와 마감일로 게시일을 정한 공고의 수는 `t01b_post_start_missing`에 있다.")),
        "t01b_post_start_missing": (
            "대상: 공고 파일의 모든 공고. `year`는 공고 파일의 연도이다. 게시일이 없으면 마감일을 게시일로 쓴다. "
            + base_missing +
            "`n_raw`는 공고 수이다. `n_no_post_start`는 첫 날짜 열(`post_start`)에 날짜로 쓸 수 있는 값이 없는 공고 "
            "수로, `post_start`가 비었거나 값이 날짜로 읽히지 않거나 날짜로 쓸 수 있는 범위 밖인 공고이다(`post_start`에 "
            "그런 값이 없는 파일에서는 `post_start`가 빈 공고 수와 같다). `n_no_date`는 그 가운데 마감일(`post_end`)도 "
            "날짜로 쓸 수 없어 공고 자신의 날짜 열로는 게시일을 정할 수 없는 공고 수이고, `n_end_unusable`은 다시 그 "
            "가운데 마감일 칸에 값은 있으나 날짜로 쓸 수 없는 공고 수(예: `9999-12-31`)이다. `n_filled`는 `n_no_date` "
            "가운데 같은 연도 파일에서 행 순서로 가장 가까운 앞 공고의 날짜로 날짜를 채운 공고 수이다(`n_filled <= n_no_date`. "
            "차이는 파일의 첫 날짜 있는 공고보다 앞에 있어 받을 날짜가 없는 공고이다. 날짜를 채우지 않는 설정에서는 0). "
            "`n_in_occ`는 게시일과 관계없이 분석 대상 세분류"
            "(`occupations.keco4`)에 드는 공고 수이고, `n_in_occ_no_post_start`, `n_in_occ_no_date`, "
            "`n_in_occ_end_unusable`은 그 가운데 앞의 세 경우에 해당하는 공고 수이다. `n_in_occ_from_end_in_period`는 "
            "분석 대상 세분류의 공고 가운데 게시일이 없어 마감일을 게시일로 썼고 그 날짜가 분석 기간 안이어서 남는 공고 "
            "수이다. 게시일을 마감일로 대신했지만 그 날짜가 분석 기간 밖이어서 빠진 공고 수는 "
            "`n_in_occ_no_post_start − n_in_occ_no_date − n_in_occ_from_end_in_period`이다. `n_in_occ_filled`는 "
            "분석 대상 세분류의 공고 가운데 날짜를 채운 공고 수, `n_in_occ_filled_in_period`는 그 가운데 채운 "
            "날짜가 분석 기간 안인 공고 수이다. `n_in_occ_fill_dates`는 날짜를 채운 분석 대상 세분류 공고에 채운 서로 "
            "다른 날짜의 수이다. 이 값이 `n_in_occ_filled`보다 훨씬 작으면, 그 수집 사이트의 공고가 날짜 없이 길게 "
            "이어져 많은 공고에 같은 몇 개의 날짜를 채웠다는 뜻이다(연도 행의 값은 사이트별 값의 합이다). "
            "`n_in_occ_use`는 분석 대상 세분류에 들고 생성형 AI 활용 숙련이 있는 공고 수이고, "
            "`n_in_occ_use_no_post_start`, `n_in_occ_use_no_date`, `n_in_occ_use_from_end_in_period`, "
            "`n_in_occ_use_filled_in_period`는 그 가운데 같은 경우의 공고 수이다. `no_date_rate`는 `n_no_date / n_raw`, "
            "`in_occ_use_no_date_rate`는 `n_in_occ_use_no_date / n_in_occ_use`이다. `level` = 연도는 연도별 합계이고, "
            "`level` = 연도·수집 사이트는 `n_no_post_start`가 0보다 큰 사이트만 연도 안에서 그 수가 많은 순으로 싣는다."),
        "t01c_post_end_gap": (
            "대상: 공고 파일의 공고 가운데 분석 대상 세분류에 들고 게시일(`post_start`)과 마감일(`post_end`)이 모두 날짜로 "
            "쓸 수 있는 공고(분석 기간과 관계없이). `year`는 공고 파일의 연도이다. `bin`은 마감일에서 게시일을 뺀 일수의 "
            "구간이고, `share_in_year`는 그 연도 공고 가운데 그 구간 공고의 비율이다. 게시일이 없는 공고는 마감일을 "
            "게시일로 쓰므로, 그런 공고의 날짜는 실제 게시일보다 이 간격만큼 늦을 수 있다. 마감일 칸에 값은 있으나 날짜로 "
            "쓸 수 없는 공고(예: `9999-12-31`)는 이 표에 없다. 그 가운데 게시일도 없는 분석 대상 세분류의 공고 수는 "
            "`t01b_post_start_missing`의 `n_in_occ_end_unusable`에 있다."),
        "t01d_date_fill": (
            "대상: 공고 파일의 공고 가운데 날짜를 채운 공고(분석 기간과 관계없이). 두 날짜가 모두 없는 공고는 같은 연도 "
            "파일에서 행 순서로 가장 가까운 앞 공고의 날짜를 받는다. `year`는 공고 파일의 연도이다. `gap_bin`은 앞뒤 "
            "날짜의 간격의 구간이다. 앞뒤 날짜의 간격은 날짜를 채운 공고의 바로 앞과 바로 뒤에 있는 날짜 있는 공고 사이의 "
            "일수이다. 간격이 짧으면 채운 날짜가 실제 게시일과 가까울 가능성이 높고, 간격이 길거나 뒤에 날짜가 없으면 한 "
            "수집 사이트의 공고가 날짜 없이 길게 이어진 구간이어서 채운 날짜가 실제 게시일과 다를 수 있다. `n`은 공고 수, "
            "`n_in_occ`는 그 가운데 분석 대상 세분류의 공고 수, `n_in_occ_in_period`는 다시 그 가운데 채운 날짜가 분석 "
            "기간 안인 공고 수이다. 연도 안에서는 날짜를 채운 공고가 많은 수집 사이트부터 싣는다."),
        "t02a_treated_by_period": (
            "대상: " + _ALIVE_NOTE + ". " + post_year + " `unit` = period는 설정의 기간(`period.groups`)이며 설정의 "
            "순서로 싣는다. `n_treated`와 `n_control`은 설정의 처치 정의(" + definition + ")로 나눈 처치 공고와 비교 "
            "공고의 수이다. 나머지 처치 정의 열은 그 기간에 해당 표시가 참인 공고 수이다. " + _TREAT_ALL_NOTE + " "
            + _KW_USE_NOTE + " "
            "자료가 마지막 기간의 도중에 끝나면 마지막 기간의 행에는 그 기간의 일부 달만 들어 있다. 이때 마지막 반기의 "
            "행이 마지막 분기의 행과 같을 수 있다. 마지막 연도의 월별 공고 수는 8절의 `t08_2026_monthly`에 있다."),
        "t02b_treated_by_occ": ("대상: " + _BASE_NOTE + ". `n`은 전체 공고 수, 처치 정의 열은 해당 표시가 참인 공고 수이다. "
                                + _TREAT_ALL_NOTE + " " + _KW_USE_NOTE),
        "t02c_occ_group_treated": (
            "대상: " + _ALIVE_NOTE + ". 직종군은 세분류 코드의 앞 2자리(중분류 `keco2`)이고, `name`은 직종분류표의 "
            "중분류 이름이다. `n_treated`와 `n_control`은 설정의 처치 정의(" + definition + ")로 나눈 처치 공고와 비교 "
            "공고의 수이며, `n`에서 둘을 뺀 나머지는 어느 쪽도 아닌 공고이다. `meets_min`은 `n_treated`가 직종군별 "
            "추정의 처치 공고 수 기준(`occupations.group_min_treated` = "
            f"{cfg['occupations']['group_min_treated']}) 이상인지이다."),
        "t02d_date_source_by_period": (
            "대상: " + _ALIVE_NOTE + ". 설정의 기간(`period.groups`, 설정의 순서)과 게시일을 가져온 열(`date_source`)별 "
            "공고 수이다. `date_source` = `post_start`는 게시일이 있는 공고, `post_end`는 게시일이 없어 마감일을 "
            "게시일로 쓴 공고이다. "
            + ("`fill`은 두 날짜가 모두 없어 같은 연도 파일에서 행 순서로 가장 가까운 앞 공고의 날짜를 게시일로 받은 "
               "공고이며, 그 날짜가 얼마나 믿을 만한지는 `t02e_fill_gap_by_period`의 앞뒤 날짜 간격으로 본다. "
               if fill_on else "")
            + "`n_treated`는 그 가운데 설정의 처치 정의(" + definition + ")로 나눈 처치 공고 수이다. "
            "`share_in_period`는 그 기간 공고 가운데 그 행 공고의 비율이고, `treated_share_in_period`는 그 기간 처치 공고 "
            "가운데 그 행 처치 공고의 비율이다. 이 표는 공고의 게시일을 마감일로 대신한 공고가 분석 표본과 처치 공고에서 "
            "차지하는 몫을 기간별로 보여 준다. 마감일은 실제 게시일보다 늦으므로 이 몫이 큰 기간은 기간 구분이 덜 정확하다."),
        "t02e_fill_gap_by_period": (
            "대상: " + _ALIVE_NOTE + " 가운데 날짜를 채운 공고(`date_source` = `fill`). 설정의 기간(`period.groups`, "
            "설정의 순서)과 앞뒤 날짜 간격의 구간(`gap_bin`)별 공고 수이며, 공고가 없는 조합도 0으로 싣는다. `n_treated`는 "
            "그 가운데 설정의 처치 정의(" + definition + ")로 나눈 처치 공고 수이고, `share_in_period`는 `n`을 그 기간에 "
            "네 단계를 모두 통과한 공고의 수로 나눈 값이다. 이 표는 날짜를 채운 공고가 분석 표본에 얼마나 들어오고, "
            "그 날짜의 앞뒤 간격이 얼마나 긴지를 기간별로 보여 준다. 설정 `period.fill.max_gap_days`에 일수를 주면, "
            "날짜를 채운 공고 가운데 간격이 그 일수보다 크거나 뒤에 날짜가 없는 공고(이 표에서 그 일수보다 위의 구간과 "
            f"`{FILL_GAP_NONE}`)가 분석 기간 단계에서 빠진다(지금 설정: "
            + ("값 없음, 빼지 않음" if max_gap is None else f"{max_gap}일") + ")."),
        "t03a_skills_per_posting": (
            "대상: " + _BASE_NOTE + " 가운데 숙련이 하나 이상 있는 공고. 공고당 숙련 수는 `skills_long.parquet`의 "
            "공고별 행 수이다."),
        "t03b_sparse_filter_grid": (
            "대상: " + _ALIVE_NOTE + " 가운데 처치 공고이거나 비교 공고인 공고(추정에 들어가는 공고이며, 7절의 학습 "
            "시간 측정에서 희소 필터를 적용하는 공고와 같다). " + arm_note + " 어느 쪽도 아닌 공고는 뺐다. "
            "AI 에이전트 숙련코드는 세기 전에 뺐다. `n_postings`는 출현율의 분모(처치 공고와 비교 공고의 수), "
            "`n_treated`는 처치 공고 수이다. "
            f"설정의 기준은 `min_rate` {cfg['skills']['sparse_filter']['min_rate']}, "
            f"`min_treated` {cfg['skills']['sparse_filter']['min_treated']}이다."),
        "t03c_skills_by_source_column": (
            "대상: `skills_long.parquet`의 공고 × 숙련 행(" + _BASE_NOTE + "). 숙련 열별 값은 그 열에 나온 행 수이며, "
            "같은 숙련이 한 공고의 여러 열에 나오면 열마다 한 번씩 센다. `total`은 행 수이다. `domain`은 숙련 설명 "
            "파일의 대분류이다."),
        "t03d_match_stats": (
            "대상: 공고 파일의 모든 공고. `year`는 공고 파일의 연도이다. `n_tokens`는 숙련 코드 열의 값을 `|`로 나눠 "
            "얻은 코드 하나하나(빈 값 제외)의 수, `n_unmatched`는 그 가운데 설명 파일의 코드로도 숙련명으로도 최종판 "
            "코드를 찾지 못한 코드 수, `n_length_mismatch_rows`는 한 숙련 열에서 코드 칸과 숙련명 칸을 `|`로 나눈 "
            "조각 수(빈 조각 포함)가 다른 공고 × 열의 수이다."),
        "t03e_no_outcome_skill": (
            "대상: " + _ALIVE_NOTE + ". `전체`는 이 공고 모두이고, `처치 공고`와 `비교 공고`는 그 가운데 처치 정의로 "
            "나눈 공고이다. " + arm_note + " 어느 쪽도 아닌 공고는 `전체`에만 든다. `n_no_outcome_skill`은 숙련이 있어 숙련 없음 단계를 통과했지만, 결과변수에서 빼는 "
            "AI 에이전트 숙련코드 54개를 빼면 숙련이 하나도 남지 않는 공고 수이고, `no_outcome_skill_rate`는 "
            "`n_no_outcome_skill / n`이다. 이 공고들은 지금 설정에서는 분석에 남아, 모든 숙련의 결과변수가 0인 공고가 "
            "된다. 이 공고를 분석에 남길지는 저자가 정한다(`docs/decisions.md` 6.7-17)."),
        "t04_bzno_ked_by_treatment": (
            "대상: " + _BASE_NOTE + ". `전체`는 이 공고 모두이고, `처치 공고`와 `비교 공고`는 그 가운데 처치 정의로 "
            "나눈 공고이다(어느 쪽도 아닌 공고는 `전체`에만 든다). " + base_arm_note + " `ked_matched_rate`의 분모는 "
            "사업자번호가 있는 공고 수(`n_has_bzno`)이다."),
        "t04b_ked_lines": (
            "KED 기업 자료 파일의 전체 줄 수, 구분자 수가 맞지 않거나 줄 안에 캐리지 리턴 문자가 있어 버린 줄 수, "
            "남은 사업자번호 수이다. `n_masked_values`는 버리지 않은 줄에서 읽는 여섯 항목(사업자번호, 설립일, "
            "정보일자, 기업규모 구분, 기업공개 구분, 벤처 여부) 가운데 주민등록번호와 형태가 같은 값(숫자 13개, 여섯째 숫자 "
            "뒤에 `-`가 있을 수 있음)이 들어 있어 값을 읽지 않고 결측으로 둔 항목 수이다. 0이 아니면 항목이 밀린 "
            "줄이 있다는 뜻이다."),
        "t05_source": (
            "대상: " + _BASE_NOTE + ". " + post_year + " `share_in_year`는 그 연도 공고 가운데 그 사이트 공고의 "
            "비율이다. `treated_rate`는 그 사이트 공고 가운데 처치 공고의 비율이고, `has_use_rate`는 그 사이트 공고 "
            "가운데 생성형 AI 활용 숙련(숙련코드)이 있는 공고의 비율이다. " + base_arm_note),
        "t06a_overlap": ("대상: " + _BASE_NOTE + ". 값 `1`은 참, `0`은 거짓, `결측`은 값이 없는 공고이다. 세 번째 "
                         "교차표는 생성형 AI 활용 키워드 사전에 걸린 공고(`kw_use`)와 두 종류의 AI 에이전트 숙련코드 "
                         "처치(`has_use`, `has_dev`)의 교차표이다."),
        "t06b_agent_skill_by_year": "대상: " + _BASE_NOTE + ". " + post_year,
        "t06c_keyword_pattern_hits": (
            "대상: " + _BASE_NOTE + " 가운데, `dictionary`가 `strict`나 `broad_extra`인 행은 넓은 사전에 걸린 공고"
            "(`kw_broad`), `use`인 행은 생성형 AI 활용 키워드 사전에 걸린 공고(`kw_use`). " + post_year
            + " 제외 패턴을 지운 뒤 센다. 한 공고가 여러 패턴에 걸리면 패턴마다 센다." + both_note),
        "t08_2026_monthly": (
            "대상: " + _BASE_NOTE + " 가운데 `config.yaml`의 `data.years`의 마지막 연도(" + str(last) + ")에 게시된 "
            "공고. 월은 공고 게시일 기준이다."),
        "t09a_by_ksic_sub": (
            "대상: " + _BASE_NOTE + ". `결측`은 산업 코드가 없거나 형식을 맞출 수 없는 공고이다. "
            + ("지금 설정(`industry.not_in_table` = `missing`)에서는 소분류표에 없는 코드의 공고도 `결측`에 든다"
               "(그 건수는 `t09b_ksic_not_in_table`에 있다). " if ksic_missing else
               f"`{classif.KSIC_NOT_IN_TABLE}`은 소분류표에 없는 코드의 공고이다. ")
            + "`n_has_use`는 생성형 AI 활용 숙련(숙련코드)이 있는 공고 수이고, `n_treated`는 처치 공고 수이다. "
            + base_arm_note + " 직종 세분류별 건수는 2절의 `t02b_treated_by_occ`에 있다."),
        "t09b_ksic_not_in_table": (
            "대상: " + _BASE_NOTE + " 가운데 5자리로 맞춘 산업 코드(`ksic5`)가 있고 그 앞 3자리가 소분류표에 없는 공고. "
            f"`ksic5`별 공고 수, 많은 순으로 최대 {KSIC_NOT_IN_TABLE_TOP}개. 지금 설정(`industry.not_in_table` = "
            f"`{not_in_table}`)에서 이 공고들의 `ksic_sub`는 "
            + ("결측이다(공고는 빼지 않는다)." if ksic_missing else f"`{classif.KSIC_NOT_IN_TABLE}`이다.")),
        "t10a_use_by_occ_year": (
            "대상: " + _ALIVE_NOTE + ". " + post_year + " `{연도}_n_use`는 생성형 AI 활용 숙련(숙련코드)이 있는 공고 "
            "수, `{연도}_use_rate`는 그 연도 그 세분류 공고 가운데 그 비율이다. 설정의 처치 정의(" + definition + ")로 "
            "센 처치 공고 수가 아니다."),
        "t10b_occ_list_check": (
            "대상: `occ_year_all.parquet`(공고 파일의 모든 공고, 분석 기간과 제외 규칙 적용 전). `n_use`는 모든 연도의 "
            "생성형 AI 활용 숙련 공고 수의 합이다. 자료 쪽 목록은 `n_use`가 `occupations.min_use_postings`("
            f"{cfg['occupations']['min_use_postings']}) 이상인 세분류이고, 설정 쪽 목록은 `occupations.keco4`이다. "
            "자료 쪽 목록에서 `occupations.exclude_keco2`("
            + ", ".join(f"`{g}`" for g in cfg["occupations"]["exclude_keco2"]) + ")의 중분류에 속한 세분류는 빼고, "
            f"`status` = `{OCC_STATUS[3]}`로 따로 싣는다. 건수는 `data.years`("
            + ", ".join(str(y) for y in cfg["data"]["years"]) + ")의 공고 파일만 센 것이다(다른 연도의 파일은 읽지 않는다)."),
        "t11_salary": (
            "대상: " + _BASE_NOTE + ". " + post_year + " `n_stated`는 연봉(`salary_annual_min`)이 기재된 공고 수이고, "
            "분위수는 기재된 공고의 `salary_annual_min`(원)이다."),
        "t12_sido_other": (
            "대상: " + _BASE_NOTE + ". `기타`는 근무지에서 시도를 알 수 없는 공고, `결측`은 근무지가 없는 공고이다. "
            "`전남광주`는 근무지 첫 낱말이 `전남광주`나 `전남광주통합특별시`인 공고이다. 설정 `region.merge_sido`에 "
            "따라 다음 시도를 한 범주로 합쳤다: "
            + ", ".join(f"`{name}`(" + ", ".join(f"`{m}`" for m in members) + ")"
                        for name, members in cfg["region"]["merge_sido"].items()) + "."),
        "t12b_sido_other_words": (
            "대상: " + _BASE_NOTE + " 가운데 시도가 `기타`인 공고. 근무지(`region`)의 첫 낱말(첫 쉼표 앞에서 처음 나오는 "
            "공백, `>`, `&` 앞까지, 끝의 `전지역`과 `전체`를 뗀 것, 20자까지)별 공고 수를 많은 순으로 싣는다. "
            f"CSV에는 많은 순으로 최대 {SIDO_OTHER_WORDS_TOP}개가 있다."),
        "t13_adjacent_candidates": (
            "대상: " + _ALIVE_NOTE + " 가운데 처치 공고이거나 비교 공고인 공고(`t03b_sparse_filter_grid`와 같다. 어느 "
            "쪽도 아닌 공고는 뺐다). " + arm_note + " 숙련 설명 파일에서 세부 분류(`detail`)가 "
            "`skills.adjacent_candidate_details`에 든 숙련 가운데 AI 에이전트 숙련코드가 아닌 것마다 한 행이며, 공고가 "
            "없는 후보도 싣는다. `n`은 그 숙련이 있는 공고 수, `rate`는 `n`을 대상 공고 수로 나눈 값이다. `n_treated`는 "
            "처치 공고 가운데 그 숙련이 있는 공고 수, `treated_rate`는 `n_treated`를 처치 공고 수로 나눈 값이다. `passes_sparse_filter`는 설정의 희소 필터(" + sf_text + ")를 통과하는지이다. 공고가 많은 "
            "순으로 모든 후보를 싣는다."),
        "t14_placebo_candidates": (
            "대상: " + _ALIVE_NOTE + " 가운데 처치 공고이거나 비교 공고인 공고(3절 `t03b_sparse_filter_grid`와 같다. "
            "어느 쪽도 아닌 공고는 뺐다). " + arm_note + " 설정의 희소 필터(" + sf_text + ")를 통과한 숙련(AI 에이전트 "
            "숙련코드는 뺐다) 가운데 대분류(`domain`)가 "
            "`skills.placebo_candidate_domains`에 든 숙련마다 한 행이다. `sub`는 중분류, `n`은 그 숙련이 있는 공고 수, "
            "`rate`는 `n`을 대상 공고 수로 나눈 값이며, 공고가 많은 순으로 모두 싣는다. placebo 숙련은 생성형 AI 활용과 "
            "관계가 없어야 하는 숙련이며, 이 숙련에서 효과가 0으로 나오는지를 보아 분석 방법에 문제가 없는지 점검하는 데 "
            "쓴다(`proposal.md` 7장 2번). 이 표에 처치 공고와 비교 공고의 출현율을 싣지 않는 것은, placebo 숙련은 추정 "
            "결과를 보기 전에 골라야 하는데 두 집단의 출현율 차이는 추정 결과와 같은 방향의 정보이기 때문이다."),
        "t15_aiie_coverage": (
            "대상: " + _ALIVE_NOTE + ". `arm` = `전체`는 이 공고 모두이고, `처치 공고`와 `비교 공고`는 그 가운데 처치 "
            "정의로 나눈 공고이다(어느 쪽도 아닌 공고는 `전체`에만 든다). " + arm_note + " `n_with_aiie`는 산업 소분류(`ksic_sub`)에 AIIE 값이 붙은 공고 수, "
            "`with_aiie_rate`는 `n_with_aiie / n`이다. 산업코드가 없거나 소분류표에 없는 공고(`ksic_sub` 결측)와, 소분류에 "
            "이어진 산업에 AIIE 값이 없는 공고는 값이 붙지 않은 쪽에 센다. `p10`, `p25`, `p50`, `p75`, `p90`은 값이 "
            "붙은 공고의 AIIE 값의 10, 25, 50, 75, 90% 분위수이다(공고마다 한 번씩 센다)."),
        "t16_detail_by_arm": (
            "대상: " + _ALIVE_NOTE + "를 설정의 처치 정의(" + definition + ")로 나눈 처치 공고와 비교 공고(어느 쪽도 "
            "아닌 공고는 뺐다). " + post_year + " 숙련 수는 공고의 숙련 가운데 AI 에이전트 숙련코드를 뺀 숙련의 수이고, "
            "`n_skills_p50`, `n_skills_p75`, `n_skills_p90`은 50%, 75%, 90% 분위수이다. `text_len_mean`과 "
            "`text_len_p50`은 본문(`treatment.text_columns`의 열을 이은 문자열) 글자 수의 평균과 중앙값이다. 마지막 행"
            f"(`{ARM_WITHIN_LABEL}`)은 세분류 × 연도마다 처치 공고의 평균에서 비교 공고의 평균을 뺀 값을 그 칸의 처치 "
            "공고 수로 가중 평균한 것이다(숙련 수와 글자 수 각각). 처치 공고와 비교 공고가 모두 있는 칸만 쓰며, 이 행의 "
            "`n`은 그 칸들의 처치 공고 수이다."),
        "t17_firm_treated": (
            "대상: " + _ALIVE_NOTE + " 가운데 사업자번호가 있는 공고를 설정의 처치 정의(" + definition + ")로 나눈 처치 "
            "공고와 비교 공고. 기업은 사업자번호로 구분하며, 기업 식별자는 싣지 않는다. " + post_year + " 비율은 처치 "
            "공고 수에 대한 비율이다."),
        "t18_first_treated_firms": (
            "대상: " + _ALIVE_NOTE + " 가운데 사업자번호가 있는 공고를 설정의 처치 정의(" + definition + ")로 나눈 처치 "
            "공고와 비교 공고. " + post_year + " `first_year`는 기업이 처치 공고를 처음 낸 연도, `n_firms`는 그 연도에 "
            "처치 공고를 처음 낸 기업 수, `n_firms_with_earlier`는 그 가운데 `first_year` 전의 연도에 비교 공고를 낸 기업 "
            "수, `n_earlier_postings`는 그 기업들이 `first_year` 전의 연도에 낸 비교 공고 수이다. 분석 기간의 첫 연도에 "
            "처치 공고를 처음 낸 기업은 그 전 연도의 공고가 자료에 없으므로 늘 0이다."),
    }


def _fmt_small(v: float) -> str:
    """0이 아니고 절댓값이 100보다 작은 실수를 지수 표기 없이 쓴다.

    소수 넷째 자리까지 쓰되, 절댓값이 작아 유효숫자가 세 자리보다 적게 남으면 세 자리가 남도록 자리를 늘린다
    (예: 0.00004, 0.00012, 0.00049). 그래서 0이 아닌 값이 `0`으로 보이지 않는다. 끝의 0은 지운다.
    """
    decimals = max(4, 2 - math.floor(math.log10(abs(v))))
    return f"{v:.{decimals}f}".rstrip("0").rstrip(".")


def _fmt_cell(v, column: str) -> str:
    if v is None or (not isinstance(v, (str, list, tuple)) and pd.isna(v)):
        s = ""
    elif isinstance(v, (bool, np.bool_)):
        s = str(bool(v))
    elif isinstance(v, (int, np.integer)):
        s = str(int(v)) if column in _PLAIN_INT_COLUMNS else f"{int(v):,}"
    elif isinstance(v, (float, np.floating)):
        v = float(v)
        if v.is_integer() and abs(v) >= 1000:
            s = f"{v:,.0f}"
        elif abs(v) >= 100:
            s = f"{v:,.1f}"
        elif v == 0:
            s = "0"
        else:
            s = _fmt_small(v)
    else:
        s = str(v)
    if column in REPORT_CODE_COLUMNS and s:
        s = f"`` {s} ``" if "`" in s else f"`{s}`"
    return s.replace("\r", " ").replace("\n", " ").replace("|", "\\|")


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(행 없음)"
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(c.replace("|", "\\|") for c in cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_fmt_cell(v, c) for v, c in zip(row, cols)) + " |")
    return "\n".join(lines)


def _display_path(path: str | None, name: str) -> str:
    if not path:
        return f"experiments/outputs/s0/tables/{name}.csv"
    p = Path(path)
    try:
        return p.resolve().relative_to(paths.ROOT.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def _table_block(name: str, df: pd.DataFrame, note: str) -> list[str]:
    limit = REPORT_MAX_ROWS_BY_TABLE.get(name, REPORT_MAX_ROWS)

    def part(d: pd.DataFrame) -> list[str]:
        if limit is None:
            return [_md_table(d)]
        out = [_md_table(d.head(limit))]
        if len(d) > limit:
            out.append(f"전체 {len(d):,}행 가운데 위 {limit}행만 싣는다. 전체 표: "
                       f"`{_display_path(df.attrs.get('csv_path'), name)}`")
        return out

    lines = [f"### `{name}`", "", note, ""]
    split = REPORT_SPLIT.get(name)
    if split and split in df.columns and not df.empty:
        for value in pd.unique(df[split]):
            d = df[df[split] == value].drop(columns=split)
            if name in REPORT_DROP_EMPTY_COLUMNS:
                subset = REPORT_DROP_EMPTY_COLUMNS[name]
                d = d.drop(columns=[c for c in (d.columns if subset is None else subset)
                                    if c in d.columns and d[c].isna().all()])
            labels = REPORT_SPLIT_LABELS.get(name, {})
            shown = f"{labels[value]} (`{value}`)" if value in labels else _fmt_cell(value, split)
            lines += [f"#### `{split}` = {shown}", ""]
            lines += part(d) + [""]
    else:
        lines += part(df) + [""]
    return lines


def _timing_block(timing: dict, cfg: dict) -> list[str]:
    est = cfg["estimation"]
    labels = {
        "skill_code": "결과변수로 쓴 숙련 코드",
        "n_train": "학습에 쓴 비교 공고 수",
        "n_features": "설명 변수 수",
        "seconds": "학습 한 번에 걸린 시간(초)",
        "n_skills_kept": "희소 필터(설정의 기준)를 통과한 숙련 수",
        "est_total_hours": "전체 추정 시간(시간) = 초 × 폴드 수 × 통과한 숙련 수 / 3600",
    }
    rows = pd.DataFrame({"key": list(labels), "설명": list(labels.values()),
                         "값": [timing.get(k) for k in labels]})
    lines = [
        "교차적합(자료를 여러 묶음으로 나누어, 한 묶음의 예측값을 나머지 묶음으로 학습한 모형으로 구하는 방법)에서 "
        "나눈 묶음 하나를 폴드라고 한다.",
        "",
        "대상: " + _ALIVE_NOTE + " 가운데 설정의 처치 정의(`" + str(cfg["treatment"]["definition"]) + "`)로 처치 공고나 "
        "비교 공고인 공고. 기업(`bzno`)으로 나눈 폴드 " + str(est["n_folds"]) + "개 가운데 첫 번째 폴드를 뺀 나머지(학습 "
        "쪽)의 비교 공고만으로 LightGBM 분류 모형을 한 번 학습했다. 설정: `" + str(est["lgbm"]) + "`, 난수 "
        + str(est["seed"]) + ". "
        + _arm_note(cfg) + " "
        "`n_skills_kept`는 3절 `t03b_sparse_filter_grid`에서 `config.yaml`의 기준(`skills.sparse_filter`)에 해당하는 "
        "행의 `n_skills_kept`와 같다(두 곳이 같은 공고와 같은 처치 공고를 쓴다).",
        "",
        _md_table(rows),
        "",
    ]
    if timing.get("seconds") is None:
        lines += ["학습 시간을 재지 않았다. 처치 공고가 없거나, 비교 공고의 기업 수가 폴드 수보다 적거나, 학습할 공고가 "
                  "없거나, 결과변수가 한 값뿐인 경우이다.", ""]
    return lines


def _scan_meta_lines(scan_meta: dict | None) -> list[str]:
    """보고서 머리에 쓰는 1단계(scan) 실행 정보 두 줄."""
    if scan_meta is None:
        return ["- 중간 자료를 만든 1단계(`scan`) 실행: 알 수 없음(1단계 기록을 받지 못함)"]
    changed = scan_meta.get("git_uncommitted_changes")
    note = (" (실행할 때 git에 기록된 파일에 커밋하지 않은 변경이 있었다)" if changed is True
            else " (커밋하지 않은 변경이 있었는지 알 수 없음)" if changed is None else "")
    return [
        f"- 중간 자료를 만든 1단계(`scan`) 실행: {scan_meta.get('scan_started')} ~ {scan_meta.get('scan_finished')}",
        f"- 1단계의 코드 시점(git 커밋): {scan_meta.get('git_commit')}{note}",
    ]


def _note_from_attrs(df: pd.DataFrame) -> str:
    """표를 만들 때 정해지는 수를 설명 끝에 붙일 문장으로 돌려준다(지금은 `t15_aiie_coverage`의 소분류 수만)."""
    if "n_sub_total" not in df.attrs:
        return ""
    return (f" 산업분류표(`data.ksic`)의 소분류 {df.attrs['n_sub_total']:,}개 가운데 AIIE 값이 붙은 소분류는 "
            f"{df.attrs['n_sub_with_aiie']:,}개이다.")


def write_report(tables: dict, timing: dict, cfg: dict, report_path: Path, commit: str,
                 scan_meta: dict | None = None) -> Path:
    """집계표와 학습 시간 측정 결과로 자료 점검 보고서(마크다운)를 쓰고 경로를 돌려준다. 숫자만 싣는다.

    `commit`은 보고서를 만드는 지금의 코드 시점이고, `scan_meta`는 중간 자료를 만든 1단계의 기록(`scan_meta.json`)이다.
    """
    report_path = rawio._check_out_path(report_path)
    notes = _table_notes(cfg)
    k = cfg["treatment"]["keywords"]
    definition = str(cfg["treatment"]["definition"])
    fill_on = bool(cfg["period"]["fill"]["enabled"])
    unit = str(cfg["period"]["unit"])
    unit_text = f"{PERIOD_UNIT_LABELS[unit]} (`{unit}`)" if unit in PERIOD_UNIT_LABELS else f"`{unit}`"
    lines = [
        "# 자료 점검(S0) 보고서",
        "",
        f"- 보고서를 만든 날(2단계 `report` 실행): {dt.date.today().isoformat()}",
        f"- 보고서를 만든 코드 시점(git 커밋): {commit}",
        *_scan_meta_lines(scan_meta),
        f"- 분석 기간: {cfg['period']['start']} ~ {cfg['period']['end']} (기간 단위: {unit_text})",
        f"- 분석 대상 세분류 수: {len(cfg['occupations']['keco4'])}",
        f"- 처치 정의(`treatment.definition`): `{definition}`. {DEFINITION_TEXT.get(definition, '')} "
        + _COMPARISON_TEXT,
        f"- 키워드 사전은 2026-11-13에 확정했다(`docs/decisions.md` 결정 32, 36). 생성형 AI 활용 표현은 2026-11-14에 "
        f"넓혔다(결정 41). 엄격한 사전 {len(k['strict'])}개, "
        f"넓은 사전에 더한 표현 {len(k['broad_extra'])}개, 제외 표현 {len(k['exclude'])}개, 생성형 AI 활용 표현 "
        f"{len(k['use'])}개(`config.yaml`의 `treatment.keywords`).",
        "- 게시일: `post_start`, 없으면 `post_end`. "
        + ("두 날짜가 모두 없으면 같은 연도 파일에서 행 순서로 가장 가까운 앞 공고의 날짜를 쓴다(`period.fill.enabled`, "
           "결정 38)." if fill_on else "두 날짜가 모두 없는 공고는 분석 기간 단계에서 뺀다(`period.fill.enabled` = false)."),
        "- 이 보고서는 `experiments/scripts/s00_data_audit.py report`가 만든다. 모든 표의 전체 CSV는 "
        "`experiments/outputs/s0/tables/`에 있다.",
        "",
    ]
    for number, title, names in REPORT_SECTIONS:
        lines += [f"## {number}. {title}", ""]
        if number in REPORT_SECTION_INTROS:
            lines += [REPORT_SECTION_INTROS[number], ""]
        if number == 7:
            lines += _timing_block(timing, cfg)
        for name in names:
            if name in FILL_TABLES and name not in tables:
                lines += [f"설정 `period.fill.enabled`가 false여서 날짜를 채우지 않았으므로 `{name}`은 만들지 않았다.", ""]
                continue
            df = tables[name]
            if MISSING_EXTERNAL_ATTR in df.attrs:
                lines += [f"외부 자료가 없어 만들지 않았다(`{name}`). 없는 파일: "
                          + ", ".join(df.attrs[MISSING_EXTERNAL_ATTR]) + ".", ""]
                continue
            lines += _table_block(name, df, notes[name] + _note_from_attrs(df))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return report_path


def _git_commit() -> str:
    """저장소의 현재 커밋(짧은 형식)을 돌려준다. git을 쓸 수 없으면 정해진 안내 문자열을 돌려준다."""
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=paths.ROOT, capture_output=True,
                           text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return GIT_UNKNOWN
    commit = r.stdout.strip()
    return commit if r.returncode == 0 and commit else GIT_UNKNOWN


def _git_uncommitted_changes() -> bool | None:
    """git에 기록된 파일에 커밋하지 않은 변경이 있으면 True를 돌려준다(git에 없는 새 파일은 보지 않는다).
    git을 쓸 수 없으면 None을 돌려준다."""
    try:
        r = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=paths.ROOT,
                           capture_output=True, text=True, errors="replace", timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(r.stdout.strip()) if r.returncode == 0 else None


def run_report(cfg: dict, root: Path, s0_dir: Path, report_path: Path, commit: str | None = None) -> Path:
    """중간 자료로 집계표를 만들고, 학습 시간을 재고, 보고서를 쓴다."""
    root, s0_dir = Path(root), Path(s0_dir)
    tables = build_tables(cfg, root, s0_dir)
    codes = treatment.agent_skill_codes(skills.load_codebook(root / cfg["data"]["codebook"]), cfg)
    base = pd.read_parquet(s0_dir / "base.parquet", columns=_TIMING_BASE_COLUMNS)
    treat = pd.read_parquet(s0_dir / "treat.parquet")
    long = pq.read_table(s0_dir / "skills_long.parquet", columns=["posting_id", "skill_code"],
                         read_dictionary=["skill_code"]).to_pandas()
    long = skills.drop_codes(long, codes["dev"] | codes["use"])  # AI 에이전트 숙련코드는 결과변수가 아니다
    timing = time_outcome_model(base, treat, long, cfg)
    del base, treat, long
    path = write_report(tables, timing, cfg, report_path, commit or _git_commit(), _read_scan_meta(s0_dir))
    print(f"보고서를 썼습니다: {path}", flush=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="자료 점검(S0)")
    sub = parser.add_subparsers(dest="command", required=True, metavar="명령")
    sub.add_parser("scan", help="원자료를 한 번 훑어 중간 자료를 experiments/outputs/s0/에 쓴다")
    sub.add_parser("report", help="중간 자료로 집계표, 학습 시간 측정, 보고서(docs/s0-data-audit.md)를 만든다")
    sub.add_parser("all", help="scan 다음에 report를 실행한다")
    args = parser.parse_args(argv)
    cfg = paths.load_config()
    if args.command == "all":   # 오래 걸리는 scan 전에, 중간 자료 없이 확인할 수 있는 report의 입력을 확인한다
        _check_report_inputs(cfg, paths.ROOT)
    if args.command in ("scan", "all"):
        scan(cfg, paths.ROOT, paths.S0)
    if args.command in ("report", "all"):
        run_report(cfg, paths.ROOT, paths.S0, paths.ROOT / "docs" / "s0-data-audit.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
