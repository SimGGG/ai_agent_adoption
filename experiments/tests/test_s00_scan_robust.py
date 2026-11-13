"""자료 점검 1단계의 연도 간 자료형 맞추기와 실패 시 결과 파일 정리를 확인한다. 수치는 가상 자료의 것이다."""
import datetime as dt
import pathlib

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from aabundle import rawio
from test_s00_scan import load_script

# post_start_missing, sido_other_words는 과제 10 수정에서, post_end_gap은 과제 4에서 더했다
OUT_NAMES = ["occ_year_all", "base", "treat", "skills_long", "scan_stats", "kw_pattern_hits",
             "post_start_missing", "sido_other_words", "post_end_gap"]


def _year_file(synth_data, year, prefix, changes):
    """2025년 가상 파일을 `year`년 파일로 옮겨 쓴다. 공고 번호 앞 글자와 날짜의 연도를 바꾸고 `changes`의 열을 바꾼다."""
    t = pq.read_table(rawio.posting_path(synth_data["postings_dir"], 2025))
    ids = [prefix + p[1:] for p in t.column("posting_id").to_pylist()]
    t = t.set_column(t.schema.get_field_index("posting_id"), "posting_id", pa.array(ids, pa.string()))
    # 과제 3: 마감일 열이 생기고 게시일이 없는 공고가 더해져, 두 날짜 열을 모두 옮기고 결측은 그대로 둔다
    # 과제 4: 날짜로 쓸 수 없는 마감일(p17의 9999-12-31)은 연도를 바꾸지 않고 그대로 둔다
    for col in ("post_start", "post_end"):
        dates = [d if d is None or d.year != 2025 else d.replace(year=year) for d in t.column(col).to_pylist()]
        t = t.set_column(t.schema.get_field_index(col), col, pa.array(dates, pa.date32()))
    for name, arr in changes.items():
        t = t.set_column(t.schema.get_field_index(name), name, arr)
    pq.write_table(t, rawio.posting_path(synth_data["postings_dir"], year), row_group_size=5)


def _two_year_cfg(synth_data):
    cfg = synth_data["cfg"]
    cfg["data"]["years"] = [2023, 2024]
    cfg["period"]["start"], cfg["period"]["end"] = "2023-01-01", "2024-12-31"
    cfg["period"]["groups"] = [{"name": "초기", "years": [2023, 2024]}]
    return cfg


def test_scan_promotes_column_types_across_years(synth_data, tmp_path):
    n = 17   # 가상 공고 수(과제 3에서 12에서 14로, 과제 3A에서 16으로, 과제 4에서 17로 바꿨다)
    # 2023년: career_type은 값이 하나도 없어 null 형식, dup_count는 정수
    _year_file(synth_data, 2023, "a", {"career_type": pa.nulls(n)})
    # 2024년: career_type은 문자열, dup_count는 결측과 소수가 있는 실수
    dup = [1.0] * n
    dup[0], dup[1] = 2.5, None
    _year_file(synth_data, 2024, "b", {"dup_count": pa.array(dup, pa.float64())})

    out = load_script().scan(_two_year_cfg(synth_data), synth_data["root"], tmp_path / "s0")

    schema = pq.read_schema(out["base"])
    assert schema.field("dup_count").type == pa.float64()
    assert schema.field("career_type").type == pa.string()
    base = pd.read_parquet(out["base"]).set_index("posting_id")
    assert len(base) == 26 and set(base["year"]) == {2023, 2024}   # 과제 3: 연도마다 p13이 더해져 20에서 22로, 과제 3A: p15, p16이 더해져 26으로 바꿨다
    assert base.loc["b01", "dup_count"] == 2.5 and pd.isna(base.loc["b02", "dup_count"])
    assert base.loc["a01", "dup_count"] == 1.0
    assert base.loc[base["year"] == 2023, "career_type"].isna().all()
    assert base.loc[base["year"] == 2024, "career_type"].notna().all()
    stats = pd.read_parquet(out["scan_stats"]).set_index("year")
    assert stats.loc[2023, "n_in_occ"] == 13 and stats.loc[2024, "n_in_occ"] == 13   # 과제 3: p13이 더해져 10에서 11로, 과제 3A: p15, p16이 더해져 13으로 바꿨다


def test_scan_stops_before_writing_when_types_cannot_be_matched(synth_data, tmp_path):
    n = 17   # 가상 공고 수(과제 3에서 12에서 14로, 과제 3A에서 16으로, 과제 4에서 17로 바꿨다)
    _year_file(synth_data, 2023, "a", {})
    _year_file(synth_data, 2024, "b", {"ai_peripheral": pa.array(["예"] * n, pa.string())})
    out_dir = tmp_path / "s0"
    with pytest.raises(ValueError, match=r"ai_peripheral.*2023년 bool.*2024년 string"):
        load_script().scan(_two_year_cfg(synth_data), synth_data["root"], out_dir)
    assert not out_dir.exists()

    # 모든 연도에서 값이 없는 열도 훑기 전에 멈춘다
    _year_file(synth_data, 2023, "a", {"career_type": pa.nulls(n)})
    _year_file(synth_data, 2024, "b", {"career_type": pa.nulls(n)})
    with pytest.raises(ValueError, match=r"career_type.*null"):
        load_script().scan(_two_year_cfg(synth_data), synth_data["root"], out_dir)
    assert not out_dir.exists()


def _stale(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in OUT_NAMES:
        pd.DataFrame({"old": [1]}).to_parquet(out_dir / f"{name}.parquet")
    (out_dir / "scan_meta.json").write_text("{}", encoding="utf-8")   # 전체 검토 수정에서 더한 1단계 기록
    (out_dir / "base.parquet.part").write_bytes(b"old")
    (out_dir / "keep.txt").write_text("다른 파일", encoding="utf-8")


def _left(out_dir):
    return sorted(p.name for p in out_dir.iterdir() if p.name != "keep.txt")


def test_failed_scan_leaves_no_stale_or_partial_files(synth_data, tmp_path, monkeypatch):
    mod = load_script()
    out_dir = tmp_path / "s0"

    # (a) 훑는 도중에 멈추는 경우
    _stale(out_dir)
    def fail(*args, **kwargs):
        raise RuntimeError("시험용 실패")
    with monkeypatch.context() as m:
        m.setattr(mod.treatment, "build_treat", fail)
        with pytest.raises(RuntimeError, match="시험용 실패"):
            mod.scan(synth_data["cfg"], synth_data["root"], out_dir)
    assert _left(out_dir) == []
    assert (out_dir / "keep.txt").exists()

    # 이름을 바꾸는 도중에 멈추는 경우. 전체 검토 수정에서 PermissionError가 나면 같은 파일을 몇 번 다시 바꿔 보게
    # 했으므로, 세 번째 파일의 이름 바꾸기가 매번 실패하게 한다.
    _stale(out_dir)
    real_replace = pathlib.Path.replace
    calls = []
    def replace_then_fail(self, target):
        calls.append(target)
        if target in calls[:-1] or len(set(calls)) == 3:
            raise PermissionError("시험용 이름 바꾸기 실패")
        return real_replace(self, target)
    with monkeypatch.context() as m:
        m.setattr(pathlib.Path, "replace", replace_then_fail)
        m.setattr(mod, "RENAME_WAIT_SECONDS", 0)
        with pytest.raises(PermissionError):
            mod.scan(synth_data["cfg"], synth_data["root"], out_dir)
    assert len(set(calls)) == 3 and len(calls) == 2 + mod.RENAME_ATTEMPTS
    assert _left(out_dir) == []
    assert (out_dir / "keep.txt").exists()

    # (b) 결과 파일이 이미 있는 폴더에서 다시 실행하면 정상으로 끝난다
    _stale(out_dir)
    out = mod.scan(synth_data["cfg"], synth_data["root"], out_dir)
    out = mod.scan(synth_data["cfg"], synth_data["root"], out_dir)
    assert _left(out_dir) == sorted([f"{name}.parquet" for name in OUT_NAMES] + ["scan_meta.json"])
    assert len(pd.read_parquet(out["base"])) == 13   # 과제 3: p13이 더해져 10에서 11로, 과제 3A: p15, p16이 더해져 13으로 바꿨다
