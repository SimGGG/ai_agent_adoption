"""자료 점검(S0) 스크립트.

`scan`: 연도별 공고 파일을 행 묶음(row group) 단위로 한 번 훑어, 공고 단위 중간 자료와
작은 집계 자료를 `out_dir`에 쓴다. 공고 원문 열은 처치 판별과 키워드 표현별 공고 수를
세는 데에만 메모리에서 쓰고, 어떤 파일과 화면 출력에도 남기지 않는다.

실행: 저장소 최상위 폴더에서 `python experiments/scripts/s00_data_audit.py scan`
"""
import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/ 폴더의 paths를 불러오기 위해서다

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

import paths
from aabundle import classif, ked, rawio, sample, skills, treatment

OUTPUT_NAMES = ("occ_year_all", "base", "treat", "skills_long", "scan_stats", "kw_pattern_hits")

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

_STAT_KEYS = ["n_tokens", "n_unmatched", "n_length_mismatch_rows"]


def _plain(t: pa.DataType) -> pa.DataType:
    """문자열 계열 형식은 모두 `string`으로 맞춘다."""
    if pa.types.is_string(t) or pa.types.is_large_string(t) or pa.types.is_string_view(t):
        return _STR
    return t


def _schemas(raw_schema: pa.Schema) -> dict[str, pa.Schema]:
    """공고 단위 세 파일의 열 구성을 정한다. 연도나 행 묶음에 따라 형식이 바뀌지 않게 하기 위해서다."""
    id_type = _plain(raw_schema.field("posting_id").type)
    base = [(c, _BASE_DERIVED_TYPES[c] if c in _BASE_DERIVED_TYPES else _plain(raw_schema.field(c).type))
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


def scan(cfg: dict, root: Path, out_dir: Path) -> dict:
    """원자료를 한 번 훑어 중간 자료 여섯 파일을 `out_dir`에 쓰고 {이름: 경로}를 돌려준다.

    각 파일은 먼저 `.part`가 붙은 이름으로 쓰고, 모든 연도를 끝까지 처리한 뒤에 정해진 이름으로 바꾼다.
    도중에 멈추면 `.part` 파일을 지우므로, 정해진 이름의 파일은 끝까지 처리된 실행의 결과만 남는다.
    """
    root, out_dir = Path(root), Path(out_dir)
    years = [int(y) for y in cfg["data"]["years"]]
    postings_dir = root / "data" / "postings"
    text_cols = list(cfg["treatment"]["text_columns"])
    read_cols = list(dict.fromkeys(sample.RAW_COLUMNS + skills.raw_skill_columns()))
    if not text_cols:
        raise ValueError("cfg['treatment']['text_columns']가 비어 있습니다")

    # 오래 훑기 전에 모든 연도 파일과 필요한 열이 있는지 확인한다.
    files = {y: rawio.open_postings(postings_dir, y, read_cols + text_cols) for y in years}
    schemas = _schemas(files[years[0]].schema_arrow)

    ked_firm = ked.extract_ked_firm(root / cfg["data"]["ked_firm"])
    ksic_valid = classif.load_ksic_sub(root / cfg["data"]["ksic"])
    codebook = skills.load_codebook(root / cfg["data"]["codebook"])
    codes = treatment.agent_skill_codes(codebook, cfg)
    kws = treatment.keyword_lists(cfg)
    kw_cfg = cfg["treatment"]["keywords"]
    labelled = [("strict", p) for p in kw_cfg["strict"]] + [("broad_extra", p) for p in kw_cfg["broad_extra"]]

    out = {name: out_dir / f"{name}.parquet" for name in OUTPUT_NAMES}
    parts = {name: _part_path(p) for name, p in out.items()}
    writers = {}
    try:
        for name in ("base", "treat", "skills_long"):
            writers[name] = rawio.parquet_writer(parts[name], schemas[name])

        occ_pieces, stats_rows, base_ids = [], [], []
        hits = defaultdict(int)
        for year in years:
            t0 = time.perf_counter()
            pf = files[year]
            acc = {"base": [], "treat": [], "skills_long": []}
            occ_year = []
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
                keep = b["in_period"].to_numpy(dtype=bool) & b["in_occ"].to_numpy(dtype=bool)
                st["n_in_period"] += int(b["in_period"].sum())
                st["n_in_occ"] += int(keep.sum())
                if not keep.any():
                    continue
                b = b[keep].reset_index(drop=True)
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
            occ = pd.DataFrame({f.name: pd.Series([], dtype=f.type.to_pandas_dtype()) for f in OCC_SCHEMA})
        stats = pd.DataFrame(stats_rows, columns=STATS_SCHEMA.names)
        kw_years = sorted(set(years) | {y for y, _ in hits})
        kw_hits = pd.DataFrame(
            [(y, d, p, hits.get((y, p), 0)) for y in kw_years for d, p in labelled],
            columns=KW_SCHEMA.names,
        )
        for name, df, schema in (("occ_year_all", occ, OCC_SCHEMA), ("scan_stats", stats, STATS_SCHEMA),
                                 ("kw_pattern_hits", kw_hits, KW_SCHEMA)):
            with rawio.parquet_writer(parts[name], schema) as w:
                w.write_table(_to_table(df, schema))
    except BaseException:
        for w in writers.values():
            w.close()
        for p in parts.values():
            p.unlink(missing_ok=True)
        raise

    for name in OUTPUT_NAMES:
        parts[name].replace(out[name])
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="자료 점검(S0)")
    sub = parser.add_subparsers(dest="command", required=True, metavar="명령")
    sub.add_parser("scan", help="원자료를 한 번 훑어 중간 자료를 experiments/outputs/s0/에 쓴다")
    args = parser.parse_args(argv)
    if args.command == "scan":
        scan(paths.load_config(), paths.ROOT, paths.S0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
