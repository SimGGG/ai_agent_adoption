"""자료 점검(S0) 스크립트.

1단계 `scan`: 연도별 공고 파일을 행 묶음(row group) 단위로 한 번 훑어, 공고 단위 중간 자료와
작은 집계 자료, 1단계 기록(`scan_meta.json`: 실행 시각, 코드 시점, 1단계의 결과를 바꾸는 설정 값)을
`out_dir`에 쓴다. 공고 원문 열은 처치 판별과 키워드 표현별 공고 수를 세는 데에만 메모리에서 쓰고,
어떤 파일과 화면 출력에도 남기지 않는다.

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
from aabundle import classif, ked, rawio, sample, skills, treatment, validate

OUTPUT_NAMES = ("occ_year_all", "base", "treat", "skills_long", "scan_stats", "kw_pattern_hits",
                "post_start_missing", "sido_other_words")
# 1단계를 실행한 때, 코드 시점, 1단계의 결과를 바꾸는 설정 항목을 적은 기록(JSON). 위 여덟 파일과 함께 쓰고 지운다.
META_NAME = "scan_meta"
# 1단계 기록에 남기는 설정 항목. `scan`이 읽어 결과가 달라지는 항목만이다. 나머지 항목(`data.keco`, `period.unit`,
# `occupations.min_use_postings`, `sample`, `treatment.definition`, `treatment.keyword_dictionary`, `skills`,
# `estimation` 등)은 2단계(`report`)에서만 쓰므로, 바꾼 뒤 `report`만 다시 실행하면 된다.
SCAN_CONFIG_KEYS = ("data.years", "data.codebook", "data.ksic", "data.ked_firm", "period.start", "period.end",
                    "occupations.keco4", "treatment.text_columns", "treatment.keywords", "treatment.agent_skills",
                    "salary_band_edges")
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
    "year": _INT, "month": _INT, "quarter": _STR, "bzno": _STR, "ksic5": _STR, "ksic_sub": _STR,
    "firm_emp_log": pa.float64(), "keco4": _STR, "keco2": _STR, "salary_band": _STR, "sido": _STR,
    "ked_matched": _BOOL, "firm_age": _INT, "ked_size": _STR, "ked_ipo": _STR, "ked_venture": _STR,
    "in_period": _BOOL, "in_occ": _BOOL, "has_bzno": _BOOL, "n_skills": _INT,
}
_TREAT_FLAGS = ["kw_strict", "kw_broad", "has_dev", "has_use", "treat_code", "treat_all"]

OCC_SCHEMA = pa.schema([("year", _INT), ("keco4", _STR), ("n", _INT), ("n_skill", _INT),
                        ("n_use", _INT), ("n_dev", _INT)])
STATS_SCHEMA = pa.schema([("year", _INT), ("n_raw", _INT), ("n_in_period", _INT), ("n_in_occ", _INT),
                          ("n_tokens", _INT), ("n_unmatched", _INT), ("n_length_mismatch_rows", _INT)])
KW_SCHEMA = pa.schema([("year", _INT), ("dictionary", _STR), ("pattern", _STR), ("n", _INT)])
_PSM_COUNTS = ["n_raw", "n_no_post_start", "n_in_occ", "n_in_occ_no_post_start", "n_in_occ_use",
               "n_in_occ_use_no_post_start"]
PSM_SCHEMA = pa.schema([("year", _INT), ("source", _STR)] + [(c, _INT) for c in _PSM_COUNTS])
WORDS_SCHEMA = pa.schema([("word", _STR), ("n", _INT)])

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
            for c in [*sample.BASE_COLUMNS, "n_skills"]]
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


def _post_start_counts(raw: pd.DataFrame, in_occ: np.ndarray, has_use: np.ndarray) -> pd.DataFrame:
    """행 묶음 하나의 모든 공고를 수집 사이트별로 세어, 게시일(`post_start`)이 없는 공고 수를 함께 돌려준다.

    `in_occ`는 분석 대상 세분류 여부, `has_use`는 생성형 AI 활용 숙련 보유 여부이며 `raw`와 위치로 맞춘다.
    """
    no_date = raw["post_start"].isna().to_numpy(dtype=bool)
    occ_use = in_occ & has_use
    df = pd.DataFrame({
        "source": raw["source"].astype("string").fillna(MISSING_LABEL).to_numpy(),
        "n_raw": np.ones(len(raw), dtype="int64"),
        "n_no_post_start": no_date.astype("int64"),
        "n_in_occ": in_occ.astype("int64"),
        "n_in_occ_no_post_start": (in_occ & no_date).astype("int64"),
        "n_in_occ_use": occ_use.astype("int64"),
        "n_in_occ_use_no_post_start": (occ_use & no_date).astype("int64"),
    })
    return df.groupby("source", as_index=False).sum()


def scan(cfg: dict, root: Path, out_dir: Path) -> dict:
    """원자료를 한 번 훑어 중간 자료 여덟 파일(`OUTPUT_NAMES`)과 1단계 기록(`scan_meta.json`)을 `out_dir`에 쓰고
    {이름: 경로}를 돌려준다.

    `post_start_missing`은 공고 파일의 모든 공고를 파일 연도 × 수집 사이트별로 세어 게시일이 없는 공고 수를
    적은 표이고, `sido_other_words`는 `base`에 들어간 공고 가운데 시도가 `기타`인 공고의 근무지 첫 낱말
    (`sample.region_first_word`, 20자까지)별 공고 수를 많은 순으로 200개까지 적은 표이다. 근무지 열 자체는
    `base`에 쓰지 않는다. `scan_meta.json`에는 실행을 시작하고 끝낸 시각, 실행을 시작할 때의 git 커밋과 git에 기록된
    파일에 커밋하지 않은 변경이 있었는지, 설정 가운데 `SCAN_CONFIG_KEYS` 항목의 값만 적는다(공고 단위 값은 적지 않는다).

    설정을 확인하고(`validate.validate_config`) 입력을 확인한 뒤 훑기 전에, `out_dir`에 있는 이 함수의 결과 파일
    아홉 개와 그 `.part` 파일을 지운다(폴더의 다른 파일은 건드리지 않는다). 각 파일은 먼저 `.part`가 붙은
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
    read_cols = list(dict.fromkeys(sample.RAW_COLUMNS + skills.raw_skill_columns()))
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
    labelled = [("strict", p) for p in kw_cfg["strict"]] + [("broad_extra", p) for p in kw_cfg["broad_extra"]]

    out = {name: out_dir / f"{name}.parquet" for name in OUTPUT_NAMES}
    out[META_NAME] = out_dir / f"{META_NAME}.json"
    parts = {name: _part_path(p) for name, p in out.items()}

    def remove_outputs():
        for p in [*out.values(), *parts.values()]:
            p.unlink(missing_ok=True)

    remove_outputs()  # 이전 실행의 결과 파일이 이번 실행의 결과로 보이지 않게 한다
    writers = {}
    try:
        for name in ("base", "treat", "skills_long"):
            writers[name] = rawio.parquet_writer(parts[name], schemas[name])

        occ_pieces, stats_rows, base_ids, psm_pieces = [], [], [], []
        hits = defaultdict(int)
        other_words = defaultdict(int)
        for year in years:
            t0 = time.perf_counter()
            pf = files[year]
            acc = {"base": [], "treat": [], "skills_long": []}
            occ_year, psm_year = [], []
            st = {"year": year, "n_raw": 0, "n_in_period": 0, "n_in_occ": 0, **{k: 0 for k in _STAT_KEYS}}

            for i in range(pf.num_row_groups):
                raw = pf.read_row_group(i, columns=read_cols).to_pandas()
                st["n_raw"] += len(raw)

                # 모든 공고: 숙련 긴 표와 연도 × 세분류 집계
                long, s = skills.explode_skills(raw, codebook)
                for k in _STAT_KEYS:
                    st[k] += s[k]
                flags = treatment.skill_flags(long, codes, raw["posting_id"])
                cal_year = pd.to_datetime(raw["post_start"]).dt.year.fillna(year).astype("int64")
                occ_year.append(pd.DataFrame({
                    "year": cal_year.to_numpy(),
                    "keco4": raw["keco_2025"].astype("string").fillna("NA").to_numpy(),
                    "n": np.ones(len(raw), dtype="int64"),
                    "n_skill": raw["posting_id"].isin(long["posting_id"]).to_numpy(dtype="int64"),
                    "n_use": flags["has_use"].to_numpy(dtype="int64"),
                    "n_dev": flags["has_dev"].to_numpy(dtype="int64"),
                }).groupby(["year", "keco4"], as_index=False).sum())

                # 분석 기간과 분석 대상 직종에 드는 공고만 남긴다
                b = sample.build_posting_base(raw, ked_firm, ksic_valid, cfg)
                in_occ = b["in_occ"].to_numpy(dtype=bool)
                # 모든 공고: 게시일이 없는 공고 수(분석 기간 단계에서 빠지는 공고)
                psm_year.append(_post_start_counts(raw, in_occ, flags["has_use"].to_numpy(dtype=bool)))
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

                # 남은 공고가 있는 행 묶음에서만 원문 열을 다시 읽는다. 원문은 여기서만 쓰고 저장하지 않는다.
                text = _joined_text(pf.read_row_group(i, columns=text_cols).to_pandas(), text_cols)[keep]
                text = text.reset_index(drop=True)
                tr = treatment.build_treat(b["posting_id"], text, long, codebook, cfg)
                broad = tr["kw_broad"].to_numpy(dtype=bool)
                if broad.any():
                    t_broad = text[broad].reset_index(drop=True)
                    y_broad = b["year"].astype("int64").to_numpy()[broad]
                    for y in np.unique(y_broad):
                        counts = treatment.pattern_counts(t_broad[y_broad == y], kws["broad"], kws["exclude"])
                        for p, n in counts.items():
                            hits[(int(y), p)] += n
                del text

                acc["base"].append(_to_table(b, schemas["base"]))
                acc["treat"].append(_to_table(tr, schemas["treat"]))
                acc["skills_long"].append(_to_table(long, schemas["skills_long"]))

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
                g.insert(0, "year", year)
                psm_pieces.append(g)
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
        kw_years = sorted(set(years) | {y for y, _ in hits})
        kw_hits = pd.DataFrame(
            [(y, d, p, hits.get((y, p), 0)) for y in kw_years for d, p in labelled],
            columns=KW_SCHEMA.names,
        )
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
        for name, df, schema in (("occ_year_all", occ, OCC_SCHEMA), ("scan_stats", stats, STATS_SCHEMA),
                                 ("kw_pattern_hits", kw_hits, KW_SCHEMA), ("post_start_missing", psm, PSM_SCHEMA),
                                 ("sido_other_words", words, WORDS_SCHEMA)):
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

KIND_LABELS = {"dev": "개발", "use": "생성형 AI 활용"}
OCC_STATUS = ("설정에만 있음", "자료에만 있음", "양쪽에 있음")
MISSING_LABEL = "결측"
GIT_UNKNOWN = "알 수 없음(git으로 커밋을 확인하지 못함)"

TIMING_FEATURES = ["firm_emp_log", "ked_size", "ked_ipo", "ked_venture", "firm_age", "ksic_sub", "keco4",
                   "career_type", "career_years_min", "edu_level", "salary_band", "emp_type", "sido", "quarter"]
_EXCLUSION_COLUMNS = ["year", "in_period", "in_occ", "has_bzno"]
# 집계표에 쓰는 base.parquet의 열
_TABLE_BASE_COLUMNS = ["posting_id", "year", "month", "quarter", "source", "keco4", "keco2", "ksic5", "ksic_sub",
                       "salary_annual_min", "sido", "ai_peripheral", "ked_matched", "in_period", "in_occ",
                       "has_bzno", "n_skills"]
# 학습 시간 측정에 쓰는 base.parquet의 열
_TIMING_BASE_COLUMNS = list(dict.fromkeys(["posting_id", "bzno", "n_skills", *_EXCLUSION_COLUMNS, *TIMING_FEATURES]))


def _require_interim(s0_dir: Path) -> None:
    """1단계의 중간 자료 여덟 파일과 1단계 기록이 모두 있는지 확인한다. 없으면 없는 파일 이름을 적은 예외를 낸다."""
    names = [f"{n}.parquet" for n in OUTPUT_NAMES] + [f"{META_NAME}.json"]
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
    slim = base[_EXCLUSION_COLUMNS].reset_index(drop=True)
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

def _t01_exclusions(stats: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    cols = ["year", "step", "n_before", "n_dropped", "n_after"]
    first = pd.concat([
        pd.DataFrame({"year": stats["year"], "step": "in_period",
                      "n_before": stats["n_raw"], "n_after": stats["n_in_period"]}),
        pd.DataFrame({"year": stats["year"], "step": "in_occ",
                      "n_before": stats["n_in_period"], "n_after": stats["n_in_occ"]}),
    ], ignore_index=True)
    first["n_dropped"] = first["n_before"] - first["n_after"]
    later = counts[counts["step"].isin(["has_bzno", "has_skill"])]
    out = pd.concat([first[cols], later[cols]], ignore_index=True)
    out = out.astype({c: "int64" for c in cols if c != "step"})
    order = out["step"].map({s: i for i, s in enumerate(sample.EXCLUSION_STEPS)})
    return out.assign(_o=order).sort_values(["year", "_o"], kind="stable").drop(columns="_o").reset_index(drop=True)


_T01B_COLUMNS = ["level", "year", "source", "n_raw", "n_no_post_start", "no_post_start_rate", "n_in_occ",
                 "n_in_occ_no_post_start", "n_in_occ_use", "n_in_occ_use_no_post_start", "in_occ_use_no_post_start_rate"]


def _t01b_post_start_missing(psm: pd.DataFrame) -> pd.DataFrame:
    """게시일이 없는 공고 수. 연도별 행(`level` = 연도)과, 게시일이 없는 공고가 있는 연도 × 수집 사이트별 행."""
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
    out["no_post_start_rate"] = _rate(out["n_no_post_start"], out["n_raw"])
    out["in_occ_use_no_post_start_rate"] = _rate(out["n_in_occ_use_no_post_start"], out["n_in_occ_use"])
    return out[_T01B_COLUMNS]


def _t02a_treated_by_period(base: pd.DataFrame, flags: pd.DataFrame, alive: np.ndarray) -> pd.DataFrame:
    b = base.loc[alive, ["year", "month", "quarter"]]
    f = flags.loc[alive].astype("int64")
    year = b["year"].astype("int64").astype("string")
    half = pd.Series(np.where(b["month"].to_numpy() <= 6, "H1", "H2"), index=b.index, dtype="string")
    pieces = []
    for unit, period in (("year", year), ("half", year + half), ("quarter", b["quarter"].astype("string"))):
        g = f.groupby(period).sum()
        g.insert(0, "n", f.groupby(period).size())
        pieces.append(g.rename_axis("period").reset_index().assign(unit=unit))
    out = pd.concat(pieces, ignore_index=True)
    return out[["unit", "period", "n", *_TREAT_FLAGS]].astype({"n": "int64", **{c: "int64" for c in _TREAT_FLAGS}})


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


def _t03a_skills_per_posting(base: pd.DataFrame) -> pd.DataFrame:
    s = base.loc[base["n_skills"] > 0, "n_skills"].astype("float64")
    row = {"n_postings": len(s)}
    row.update({k: (s.quantile(q) if len(s) else np.nan) for k, q in SKILL_COUNT_QUANTILES.items()})
    row["mean"] = s.mean() if len(s) else np.nan
    return pd.DataFrame([row])


def _sparse_filter_sample(alive: np.ndarray, treat: pd.DataFrame, pos: np.ndarray, skill_code: pd.Series,
                          definition: str, cfg: dict) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """희소 필터를 적용할 공고와 숙련을 정한다. `t03b_sparse_filter_grid`와 `time_outcome_model`이 함께 쓴다.

    대상 공고는 제외 규칙 네 단계를 통과한 공고(`alive`, base와 같은 길이)이다. `pos`와 `skill_code`는 숙련 긴 표의
    행별 base 위치와 숙련 코드이다. 돌려주는 값은 대상 공고의 행만 남긴 숙련 표(열 `posting_id`는 base에서의 위치,
    `skill_code`), 대상 공고의 base 위치, 대상 공고의 처치 정의 값(`select_arms`, 대상 공고와 같은 순서)이다.
    """
    alive_idx = np.flatnonzero(alive)
    arm = treatment.select_arms(treat.iloc[alive_idx], definition, _keyword_col(cfg))
    keep = alive[pos]
    lk = pd.DataFrame({"posting_id": pos[keep], "skill_code": skill_code[keep].reset_index(drop=True)})
    return lk, alive_idx, arm


def _t03b_sparse_filter_grid(long: pd.DataFrame, treat: pd.DataFrame, alive: np.ndarray,
                             agent_codes: set[str], cfg: dict) -> pd.DataFrame:
    outcome = skills.drop_codes(long[["pos", "skill_code"]], agent_codes)
    lk, alive_idx, arm = _sparse_filter_sample(alive, treat, outcome["pos"].to_numpy(), outcome["skill_code"],
                                               "use", cfg)
    treated = set(alive_idx[(arm == 1).fillna(False).to_numpy(dtype=bool)].tolist())
    n_postings = len(alive_idx)
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


def _t04_bzno_ked_by_treatment(base: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    has_bzno = base["has_bzno"].to_numpy(dtype=bool)
    df = pd.DataFrame({"has_use": flags["has_use"].to_numpy(dtype=bool), "has_bzno": has_bzno,
                       "ked": base["ked_matched"].to_numpy(dtype=bool) & has_bzno})
    g = df.groupby("has_use").agg(n=("has_bzno", "size"), n_has_bzno=("has_bzno", "sum"),
                                  n_ked_matched=("ked", "sum"))
    g = g.reindex([False, True], fill_value=0).astype("int64").rename_axis("has_use").reset_index()
    g["has_bzno_rate"] = _rate(g["n_has_bzno"], g["n"])
    g["ked_matched_rate"] = _rate(g["n_ked_matched"], g["n_has_bzno"])
    return g[["has_use", "n", "n_has_bzno", "has_bzno_rate", "n_ked_matched", "ked_matched_rate"]]


def _t03e_no_outcome_skill(long: pd.DataFrame, flags: pd.DataFrame, alive: np.ndarray,
                           agent_codes: set[str]) -> pd.DataFrame:
    """제외 규칙 네 단계를 통과한 공고 가운데, AI 에이전트 숙련코드를 빼면 숙련이 하나도 남지 않는 공고 수."""
    outcome = skills.drop_codes(long[["pos", "skill_code"]], agent_codes)
    has_outcome = np.zeros(len(alive), dtype=bool)
    has_outcome[outcome["pos"].to_numpy()] = True
    none_left = alive & ~has_outcome
    rows = []
    for group in ("전체", "has_use", "has_dev", "treat_code"):
        member = alive if group == "전체" else alive & flags[group].to_numpy(dtype=bool)
        rows.append({"group": group, "n": int(member.sum()), "n_no_outcome_skill": int((member & none_left).sum())})
    out = pd.DataFrame(rows)
    out["no_outcome_skill_rate"] = _rate(out["n_no_outcome_skill"], out["n"])
    return out


def _t04b_ked_lines(ked_firm: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"n_lines": [int(ked_firm.attrs["n_lines"])],
                         "n_dropped_lines": [int(ked_firm.attrs["n_dropped_lines"])],
                         "n_masked_values": [int(ked_firm.attrs["n_masked_values"])],
                         "n_firms": [len(ked_firm)]})


def _t05_source(base: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame({"year": base["year"].to_numpy(), "source": base["source"].astype("string").fillna(MISSING_LABEL),
                       "use": flags["has_use"].to_numpy(dtype=bool), "skill": base["n_skills"].to_numpy() > 0})
    g = df.groupby(["year", "source"]).agg(n=("use", "size"), n_use=("use", "sum"), n_skill=("skill", "sum"))
    g = g.reset_index()
    g["share_in_year"] = _rate(g["n"], g.groupby("year")["n"].transform("sum"))
    g["has_use_rate"] = _rate(g["n_use"], g["n"])
    g["has_skill_rate"] = _rate(g["n_skill"], g["n"])
    # 연도 안에서는 공고가 많은 사이트부터 싣는다(보고서에서 행이 잘려도 큰 사이트가 남게 하기 위해서다)
    g = g.sort_values(["year", "n", "source"], ascending=[True, False, True], ignore_index=True)
    return g[["year", "source", "n", "share_in_year", "has_use_rate", "has_skill_rate"]]


def _t06a_overlap(base: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    lab = {c: _flag_label(flags[c]) for c in ("kw_strict", "treat_code", "has_use", "has_dev")}
    lab["ai_peripheral"] = _flag_label(base["ai_peripheral"])
    sets = (["kw_strict", "treat_code", "ai_peripheral"], ["kw_strict", "has_use", "has_dev"])
    out = pd.concat([_crosstab(pd.DataFrame({c: lab[c] for c in s}), " × ".join(s)) for s in sets],
                    ignore_index=True)
    return out[["cross", "kw_strict", "treat_code", "ai_peripheral", "has_use", "has_dev", "n"]]


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
    k = cfg["treatment"]["keywords"]
    labelled = [("strict", p) for p in k["strict"]] + [("broad_extra", p) for p in k["broad_extra"]]
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


def _t09a_by_ksic_sub(base: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame({"ksic_sub": base["ksic_sub"].astype("string").fillna(MISSING_LABEL),
                       "use": flags["has_use"].to_numpy(dtype=bool)})
    g = df.groupby("ksic_sub").agg(n=("use", "size"), n_has_use=("use", "sum")).astype("int64").reset_index()
    return g.sort_values(["n", "ksic_sub"], ascending=[False, True], ignore_index=True)


def _t09b_ksic_not_in_table(base: pd.DataFrame) -> pd.DataFrame:
    m = (base["ksic_sub"] == classif.KSIC_NOT_IN_TABLE).fillna(False).to_numpy(dtype=bool)
    vc = base.loc[m, "ksic5"].astype("string").fillna(MISSING_LABEL).value_counts()
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
    o = occ[occ["keco4"] != "NA"].groupby("keco4")["n_use"].sum()
    in_data = set(o.index[o >= cfg["occupations"]["min_use_postings"]])
    in_cfg = set(cfg["occupations"]["keco4"])
    rows = sorted(in_data | in_cfg)
    status = [OCC_STATUS[2] if c in in_data and c in in_cfg else OCC_STATUS[0] if c in in_cfg else OCC_STATUS[1]
              for c in rows]
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


def build_tables(cfg: dict, root: Path, s0_dir: Path) -> dict[str, pd.DataFrame]:
    """1단계의 중간 자료로 자료 점검 집계표를 만들어 {표 이름: 표}로 돌려준다.

    각 표는 `s0_dir/tables/{표 이름}.csv`(utf-8-sig)로도 쓰고, 그 경로를 표의 `attrs["csv_path"]`에 둔다.
    표에는 집계값만 넣는다(공고 번호, 사업자번호, 공고 원문은 넣지 않는다).
    """
    validate.validate_config(cfg)
    root, s0_dir = Path(root), Path(s0_dir)
    _require_interim(s0_dir)
    _check_scan_config(cfg, _read_scan_meta(s0_dir))
    codebook = skills.load_codebook(root / cfg["data"]["codebook"])
    codes = treatment.agent_skill_codes(codebook, cfg)
    keco = classif.load_keco(root / cfg["data"]["keco"])
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
    other_words = pd.read_parquet(s0_dir / "sido_other_words.parquet")

    alive, counts = _exclusions(base, cfg)
    years = sorted({int(y) for y in cfg["data"]["years"]} | {int(y) for y in base["year"].unique()})

    tables = {
        "t01_exclusions": _t01_exclusions(stats, counts),
        "t01b_post_start_missing": _t01b_post_start_missing(psm),
        "t02a_treated_by_period": _t02a_treated_by_period(base, flags, alive),
        "t02b_treated_by_occ": _t02b_treated_by_occ(base, flags, keco, cfg),
        "t03a_skills_per_posting": _t03a_skills_per_posting(base),
        "t03b_sparse_filter_grid": _t03b_sparse_filter_grid(long, treat, alive, codes["dev"] | codes["use"], cfg),
        "t03c_skills_by_source_column": _t03c_skills_by_source_column(long, codebook),
        "t03d_match_stats": stats[["year", "n_tokens", "n_unmatched", "n_length_mismatch_rows"]].copy(),
        "t03e_no_outcome_skill": _t03e_no_outcome_skill(long, flags, alive, codes["dev"] | codes["use"]),
        "t04_bzno_ked_by_treatment": _t04_bzno_ked_by_treatment(base, flags),
        "t04b_ked_lines": ked_lines,
        "t05_source": _t05_source(base, flags),
        "t06a_overlap": _t06a_overlap(base, flags),
        "t06b_agent_skill_by_year": _t06b_agent_skill_by_year(long, base, codebook, codes, years),
        "t06c_keyword_pattern_hits": _t06c_keyword_pattern_hits(kw, cfg),
        "t08_2026_monthly": _t08_last_year_monthly(base, cfg),
        "t09a_by_ksic_sub": _t09a_by_ksic_sub(base, flags),
        "t09b_ksic_not_in_table": _t09b_ksic_not_in_table(base),
        "t10a_use_by_occ_year": _t10a_use_by_occ_year(base, flags, alive, keco, cfg, years),
        "t10b_occ_list_check": _t10b_occ_list_check(occ, cfg),
        "t11_salary": _t11_salary(base),
        "t12_sido_other": _t12_sido_other(base),
        "t12b_sido_other_words": _t12b_sido_other_words(other_words),
    }
    table_dir = rawio._check_out_path(s0_dir / "tables")
    table_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        path = table_dir / f"{name}.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        df.attrs["csv_path"] = str(path)
    return tables


def _timing_features(base: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:
    """학습 시간 측정용 설명 변수 표. 숫자 열은 float64로, 나머지 열은 category로 바꾼다."""
    X = base[TIMING_FEATURES].iloc[rows].reset_index(drop=True)
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
    lk, alive_idx, arm = _sparse_filter_sample(alive, treat, pos, long["skill_code"],
                                               cfg["treatment"]["definition"], cfg)
    is_treated = (arm == 1).fillna(False).to_numpy(dtype=bool)
    is_comparison = (arm == 0).fillna(False).to_numpy(dtype=bool)
    if lk.empty:
        return out
    kept_codes = skills.sparse_filter(lk, set(alive_idx[is_treated].tolist()), len(alive_idx),
                                      sf["min_rate"], sf["min_treated"])
    out["n_skills_kept"] = len(kept_codes)
    n_by_code = lk.groupby("skill_code", observed=True)["posting_id"].nunique()
    if kept_codes:
        n_by_code = n_by_code[n_by_code.index.isin(kept_codes)]
    n_by_code = n_by_code.sort_index().sort_values(ascending=False, kind="stable")
    top = str(n_by_code.index[0])
    out["skill_code"] = top

    model_rows = alive_idx[is_treated | is_comparison]
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

# (절 번호, 절 제목, 표 이름). 1~10절의 제목은 설계 문서 6.6절의 항목 내용 그대로이다. 다만 1절 제목에는
# 표에 있는 분석 기간 단계를 더했다.
REPORT_SECTIONS = [
    (1, "제외 규칙 단계별 공고 수 (연도별): 분석 기간, 분석 대상 직종 한정, 사업자번호 없음, 숙련 없음",
     ["t01_exclusions", "t01b_post_start_missing"]),
    (2, "기간별 처치 공고 수 (분기·반기·연 단위): 키워드(엄격한 사전, 넓은 사전), 개발 숙련, 생성형 AI 활용 숙련, "
        "숙련코드 전체, 키워드와 숙련코드를 합친 것 각각. 분석 대상 세분류별·중분류별 처치 공고 수(처치 정의별)",
     ["t02a_treated_by_period", "t02b_treated_by_occ"]),
    (3, "공고당 숙련 수의 분포와 숙련별 출현율 (희소 필터 기준별 남는 숙련 수), 숙련 열별 숙련 수, "
        "숙련명으로 최종판 코드를 찾지 못한 숙련 수",
     ["t03a_skills_per_posting", "t03b_sparse_filter_grid", "t03c_skills_by_source_column", "t03d_match_stats",
      "t03e_no_outcome_skill"]),
    (4, "처치 여부별 사업자번호 보유 비율과 KED 결합 비율", ["t04_bzno_ked_by_treatment", "t04b_ked_lines"]),
    (5, "연도별 수집 사이트 구성, 사이트별 처치 공고 비율과 숙련 보유 비율", ["t05_source"]),
    (6, "키워드 처치와 숙련코드 처치의 겹침, 두 처치와 `ai_peripheral`의 겹침, AI 에이전트 숙련코드 54개 각각의 "
        "기간별 공고 수, 키워드 표현별 공고 수",
     ["t06a_overlap", "t06b_agent_skill_by_year", "t06c_keyword_pattern_hits"]),
    (7, "숙련 1개에 대한 결과모형의 학습 시간 (미도입 공고를 전부 쓴 표본 기준)과 이로부터 계산한 전체 추정 시간", []),
    (8, "2026년의 월별 공고 수와 숙련 보유 비율", ["t08_2026_monthly"]),
    (9, "산업 소분류별·직종 세분류별 공고 수와 처치 공고 수, 소분류표에 없는 산업코드의 건수와 예",
     ["t09a_by_ksic_sub", "t09b_ksic_not_in_table"]),
    (10, "분석 대상 세분류 28개 각각의 생성형 AI 활용 숙련 공고 수와 비율(연도별, 제외 규칙 적용 뒤). "
         "전체 공고에서 세분류별 생성형 AI 활용 숙련 공고 수를 다시 세어 3.4절의 28개 목록과 같은지 확인",
     ["t10a_use_by_occ_year", "t10b_occ_list_check"]),
    (11, "연봉과 근무지", ["t11_salary", "t12_sido_other", "t12b_sido_other_words"]),
]

# 절 제목 바로 아래에 싣는 문장. 그 절에서 처음 나오는 용어의 정의이다.
REPORT_SECTION_INTROS = {
    3: "희소 필터는 출현율이 기준보다 낮거나 처치 공고에서 나온 횟수가 기준보다 적은 숙련을 결과변수에서 빼는 규칙이다.",
    7: ("결과모형은 공고의 특성으로 그 숙련이 공고에 들어 있을 확률을 예측하는 모형이다. 제목의 `미도입 공고`(설계 문서 "
        "6.6절의 표현)는 이 보고서의 비교 공고와 같은 공고이다."),
}
# 보고서 머리와 `t02a`의 소제목에 쓰는 기간 단위(`period.unit`)의 한국어 이름
PERIOD_UNIT_LABELS = {"quarter": "분기", "half": "반기", "year": "연"}
assert set(PERIOD_UNIT_LABELS) == set(validate.PERIOD_UNITS)

# 보고서에서 이 열의 값마다 따로 나눠 싣는 표. 행이 30개를 넘어 뒤쪽 행이 잘리지 않게 하기 위해서다.
REPORT_SPLIT = {"t01b_post_start_missing": "level", "t02a_treated_by_period": "unit",
                "t02b_treated_by_occ": "level", "t05_source": "year", "t06a_overlap": "cross",
                "t06b_agent_skill_by_year": "kind", "t10b_occ_list_check": "status"}
# 나눠 실을 때 소제목에 값 대신 쓰는 한국어 이름(값은 뒤에 코드 표시로 붙인다)
REPORT_SPLIT_LABELS = {"t02a_treated_by_period": PERIOD_UNIT_LABELS}
# 나눠 실을 때 그 부분에서 값이 하나도 없으면 빼는 열. None이면 모든 열이 대상이다.
# t06a는 두 교차표의 열이 서로 달라서, t01b는 연도 행에 수집 사이트가 없어서다.
REPORT_DROP_EMPTY_COLUMNS = {"t06a_overlap": None, "t01b_post_start_missing": ["source"]}
# 보고서에 싣는 행 수를 표마다 따로 정한 것. None이면 모든 행을 싣는다. 여기 없는 표는 REPORT_MAX_ROWS이다.
# t06b는 설계 문서 6.6절 6번이 AI 에이전트 숙련코드 54개 각각을 보이라고 해서다.
REPORT_MAX_ROWS_BY_TABLE = {"t06b_agent_skill_by_year": None}
# 보고서에서 정규표현식 패턴이 든 열. 값을 코드 표시(백틱)로 감싼다.
REPORT_CODE_COLUMNS = {"pattern"}
_PLAIN_INT_COLUMNS = {"year", "month", "min_treated"}

_BASE_NOTE = "`base.parquet`의 공고(분석 기간 안, 분석 대상 세분류. 사업자번호와 숙련 제외 전)"
_ALIVE_NOTE = "제외 규칙 네 단계(분석 기간, 분석 대상 세분류, 사업자번호, 숙련)를 모두 적용한 공고"
_POST_YEAR = "연도와 기간은 공고 게시일(`post_start`) 기준이다."
_TREAT_ALL_NOTE = ("`treat_all`은 숙련코드 처치(`treat_code`)와 엄격한 사전의 키워드 처치(`kw_strict`)를 합친 것이며, "
                   "`treatment.keyword_dictionary`를 무엇으로 정해도 엄격한 사전을 쓴다.")


def _table_notes(cfg: dict) -> dict[str, str]:
    """표마다 대상 공고와 연도 기준을 적은 설명."""
    last = max(int(y) for y in cfg["data"]["years"])
    return {
        "t01_exclusions": (
            "`in_period`, `in_occ` 두 단계는 `scan_stats.parquet`(공고 파일의 모든 공고)에서 왔고 `year`는 공고 파일의 "
            "연도이다. `has_bzno`, `has_skill` 두 단계는 " + _BASE_NOTE + "에 `sample.apply_exclusions`를 적용한 것이고 "
            "`year`는 공고 게시일(`post_start`)의 연도이다. 두 연도 기준이 다르므로, 같은 `year` 행이라도 `in_occ`의 "
            "`n_after`와 `has_bzno`의 `n_before`는 같지 않을 수 있다. `in_period` 단계에서는 게시일이 분석 기간 밖인 "
            "공고와 게시일이 없는 공고(`post_start` 결측)가 함께 빠진다. 게시일이 없는 공고의 수는 "
            "`t01b_post_start_missing`에 있다."),
        "t01b_post_start_missing": (
            "대상: 공고 파일의 모든 공고. `year`는 공고 파일의 연도이다. 게시일이 없는 공고는 `in_period` 단계에서 "
            "빠지므로 `base.parquet`에 없다. `n_raw`는 공고 수, `n_no_post_start`는 그 가운데 게시일(`post_start`)이 "
            "없는 공고 수, `n_in_occ`는 게시일과 관계없이 분석 대상 세분류(`occupations.keco4`)에 드는 공고 수, "
            "`n_in_occ_no_post_start`는 그 가운데 게시일이 없는 공고 수, `n_in_occ_use`는 분석 대상 세분류에 들고 "
            "생성형 AI 활용 숙련이 있는 공고 수, `n_in_occ_use_no_post_start`는 그 가운데 게시일이 없는 공고 수이다. "
            "`no_post_start_rate`는 `n_no_post_start / n_raw`, `in_occ_use_no_post_start_rate`는 "
            "`n_in_occ_use_no_post_start / n_in_occ_use`이다. `level` = 연도는 연도별 합계이고, `level` = "
            "연도·수집 사이트는 게시일이 없는 공고가 있는 사이트만 연도 안에서 그 공고가 많은 순으로 싣는다."),
        "t02a_treated_by_period": (
            "대상: " + _ALIVE_NOTE + ". " + _POST_YEAR + " 처치 정의 열은 그 기간에 해당 표시가 참인 공고 수이다. "
            + _TREAT_ALL_NOTE + " "
            "자료가 마지막 기간의 도중에 끝나면 마지막 기간의 행에는 그 기간의 일부 달만 들어 있다. 이때 마지막 반기의 "
            "행이 마지막 분기의 행과 같을 수 있다. 마지막 연도의 월별 공고 수는 8절의 `t08_2026_monthly`에 있다."),
        "t02b_treated_by_occ": ("대상: " + _BASE_NOTE + ". `n`은 전체 공고 수, 처치 정의 열은 해당 표시가 참인 공고 수이다. "
                                + _TREAT_ALL_NOTE),
        "t03a_skills_per_posting": (
            "대상: " + _BASE_NOTE + " 가운데 숙련이 하나 이상 있는 공고. 공고당 숙련 수는 `skills_long.parquet`의 "
            "공고별 행 수이다."),
        "t03b_sparse_filter_grid": (
            "대상: " + _ALIVE_NOTE + "(7절의 학습 시간 측정에서 희소 필터를 적용하는 공고와 같다). 처치 공고는 이 공고 "
            "가운데 생성형 AI 활용 처치 정의의 처치 공고(`select_arms(…, \"use\") == 1`)이다. "
            "AI 에이전트 숙련코드는 세기 전에 뺐다. `n_postings`는 출현율의 분모, `n_treated`는 처치 공고 수이다. "
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
            "대상: " + _ALIVE_NOTE + ". `전체`는 이 공고 모두이고, `has_use`, `has_dev`, `treat_code`는 그 가운데 해당 "
            "표시가 참인 공고이다. `n_no_outcome_skill`은 숙련이 있어 숙련 없음 단계를 통과했지만, 결과변수에서 빼는 "
            "AI 에이전트 숙련코드 54개를 빼면 숙련이 하나도 남지 않는 공고 수이고, `no_outcome_skill_rate`는 "
            "`n_no_outcome_skill / n`이다. 이 공고들은 지금 설정에서는 분석에 남아, 모든 숙련의 결과변수가 0인 공고가 "
            "된다. 이 공고를 분석에 남길지는 저자가 정한다(`docs/decisions.md` 6.7-17)."),
        "t04_bzno_ked_by_treatment": (
            "대상: " + _BASE_NOTE + ". `has_use`는 생성형 AI 활용 숙련 보유 여부이다. `ked_matched_rate`의 분모는 "
            "사업자번호가 있는 공고 수(`n_has_bzno`)이다."),
        "t04b_ked_lines": (
            "KED 기업 자료 파일의 전체 줄 수, 구분자 수가 맞지 않거나 줄 안에 캐리지 리턴 문자가 있어 버린 줄 수, "
            "남은 사업자번호 수이다. `n_masked_values`는 버리지 않은 줄에서 읽는 여섯 항목(사업자번호, 설립일, "
            "정보일자, 기업규모 구분, 기업공개 구분, 벤처 여부) 가운데 주민등록번호와 형태가 같은 값(숫자 13개, 여섯째 숫자 "
            "뒤에 `-`가 있을 수 있음)이 들어 있어 값을 읽지 않고 결측으로 둔 항목 수이다. 0이 아니면 항목이 밀린 "
            "줄이 있다는 뜻이다."),
        "t05_source": (
            "대상: " + _BASE_NOTE + ". " + _POST_YEAR + " `share_in_year`는 그 연도 공고 가운데 그 사이트 공고의 "
            "비율이다."),
        "t06a_overlap": "대상: " + _BASE_NOTE + ". 값 `1`은 참, `0`은 거짓, `결측`은 값이 없는 공고이다.",
        "t06b_agent_skill_by_year": "대상: " + _BASE_NOTE + ". " + _POST_YEAR,
        "t06c_keyword_pattern_hits": (
            "대상: " + _BASE_NOTE + " 가운데 넓은 사전에 걸린 공고. " + _POST_YEAR + " 제외 패턴을 지운 뒤 센다. "
            "한 공고가 여러 패턴에 걸리면 패턴마다 센다."),
        "t08_2026_monthly": (
            "대상: " + _BASE_NOTE + " 가운데 `config.yaml`의 `data.years`의 마지막 연도(" + str(last) + ")에 게시된 "
            "공고. 월은 공고 게시일 기준이다."),
        "t09a_by_ksic_sub": (
            "대상: " + _BASE_NOTE + ". `분류표에 없음`은 소분류표에 없는 코드, `결측`은 산업 코드가 없거나 형식을 "
            "맞출 수 없는 공고이다. `n_has_use`는 생성형 AI 활용 숙련 공고 수이다. 직종 세분류별 건수는 2절의 "
            "`t02b_treated_by_occ`에 있다."),
        "t09b_ksic_not_in_table": (
            "대상: " + _BASE_NOTE + " 가운데 소분류가 `분류표에 없음`인 공고. 5자리로 맞춘 산업 코드별 공고 수, 많은 "
            f"순으로 최대 {KSIC_NOT_IN_TABLE_TOP}개."),
        "t10a_use_by_occ_year": (
            "대상: " + _ALIVE_NOTE + ". " + _POST_YEAR + " `{연도}_n_use`는 생성형 AI 활용 숙련 공고 수, "
            "`{연도}_use_rate`는 그 연도 그 세분류 공고 가운데 그 비율이다."),
        "t10b_occ_list_check": (
            "대상: `occ_year_all.parquet`(공고 파일의 모든 공고, 분석 기간과 제외 규칙 적용 전). `n_use`는 모든 연도의 "
            "생성형 AI 활용 숙련 공고 수의 합이다. 자료 쪽 목록은 `n_use`가 `occupations.min_use_postings`("
            f"{cfg['occupations']['min_use_postings']}) 이상인 세분류이고, 설정 쪽 목록은 `occupations.keco4`이다."),
        "t11_salary": (
            "대상: " + _BASE_NOTE + ". " + _POST_YEAR + " `n_stated`는 연봉(`salary_annual_min`)이 기재된 공고 수이고, "
            "분위수는 기재된 공고의 `salary_annual_min`(원)이다."),
        "t12_sido_other": (
            "대상: " + _BASE_NOTE + ". `기타`는 근무지에서 시도를 알 수 없는 공고, `결측`은 근무지가 없는 공고이다. "
            "`전남광주`는 근무지 첫 낱말이 `전남광주`나 `전남광주통합특별시`인 공고로, `광주`, `전남`과 따로 센다."),
        "t12b_sido_other_words": (
            "대상: " + _BASE_NOTE + " 가운데 시도가 `기타`인 공고. 근무지(`region`)의 첫 낱말(첫 쉼표 앞에서 처음 나오는 "
            "공백, `>`, `&` 앞까지, 끝의 `전지역`과 `전체`를 뗀 것, 20자까지)별 공고 수를 많은 순으로 싣는다. "
            f"CSV에는 많은 순으로 최대 {SIDO_OTHER_WORDS_TOP}개가 있다."),
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
        "`n_skills_kept`는 3절 `t03b_sparse_filter_grid`에서 `config.yaml`의 기준(`skills.sparse_filter`)에 해당하는 "
        "행의 `n_skills_kept`이다(두 곳이 같은 공고와 같은 처치 공고를 쓴다. 단, `t03b`는 처치 정의를 생성형 AI 활용"
        "(`use`)으로 정해 두었으므로 설정의 처치 정의가 `use`가 아니면 두 값이 다를 수 있다).",
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


def write_report(tables: dict, timing: dict, cfg: dict, report_path: Path, commit: str,
                 scan_meta: dict | None = None) -> Path:
    """집계표와 학습 시간 측정 결과로 자료 점검 보고서(마크다운)를 쓰고 경로를 돌려준다. 숫자만 싣는다.

    `commit`은 보고서를 만드는 지금의 코드 시점이고, `scan_meta`는 중간 자료를 만든 1단계의 기록(`scan_meta.json`)이다.
    """
    report_path = rawio._check_out_path(report_path)
    notes = _table_notes(cfg)
    k = cfg["treatment"]["keywords"]
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
        f"- 키워드 사전은 임시값이다(확정 전). 엄격한 사전 {len(k['strict'])}개, 넓은 사전에 더한 표현 "
        f"{len(k['broad_extra'])}개, 제외 표현 {len(k['exclude'])}개 (`config.yaml`의 `treatment.keywords`).",
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
            lines += _table_block(name, tables[name], notes[name])
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
    if args.command in ("scan", "all"):
        scan(cfg, paths.ROOT, paths.S0)
    if args.command in ("report", "all"):
        run_report(cfg, paths.ROOT, paths.S0, paths.ROOT / "docs" / "s0-data-audit.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
