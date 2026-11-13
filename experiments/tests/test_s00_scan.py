import importlib.util, pandas as pd, paths
import pyarrow as pa
import pyarrow.parquet as pq
from aabundle import rawio

def load_script():
    spec = importlib.util.spec_from_file_location("s00", paths.ROOT / "experiments" / "scripts" / "s00_data_audit.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def test_scan_outputs(synth_data, tmp_path):
    out = load_script().scan(synth_data["cfg"], synth_data["root"], tmp_path / "s0")
    base = pd.read_parquet(out["base"]); treat = pd.read_parquet(out["treat"])
    long = pd.read_parquet(out["skills_long"]); occ = pd.read_parquet(out["occ_year_all"])
    stats = pd.read_parquet(out["scan_stats"]).set_index("year")

    # 13: 과제 3에서 더한 마감일만 있는 공고. 날짜가 없는 p14는 분석 기간 단계에서 빠진다
    # 15, 16: 과제 3A에서 더한 공고(생성형 AI 활용 키워드만 있는 공고, 본문에 Fulfillment가 있는 공고)
    assert sorted(base.posting_id) == [f"p{i:02d}" for i in (1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 15, 16)]
    assert sorted(treat.posting_id) == sorted(base.posting_id)
    assert set(long.posting_id) <= set(base.posting_id)
    assert base.set_index("posting_id").loc["p12", "n_skills"] == 2
    assert base.set_index("posting_id").loc["p10", "n_skills"] == 0

    o = occ.set_index("keco4")
    assert o.loc["0262", "n"] == 7 and o.loc["0262", "n_use"] == 2       # p02, p04 (n: 과제 3에서 p14를 더해 5에서 6으로, 과제 3A에서 p15를 더해 7로 바꿨다)
    assert o.loc["1332", "n_use"] == 1 and o.loc["1332", "n_dev"] == 2   # p05 / p01, p05
    assert o.loc["9052", "n"] == 1 and o.loc["NA", "n"] == 1
    assert stats.loc[2025, "n_raw"] == 16 and stats.loc[2025, "n_in_occ"] == 13   # 과제 3: p13, p14를 더해 12, 10에서 14, 11로, 과제 3A: p15, p16을 더해 16, 13으로 바꿨다
    assert stats.loc[2025, "n_unmatched"] == 1

    kw = pd.read_parquet(out["kw_pattern_hits"]).set_index("pattern")
    assert len(kw) == len(synth_data["cfg"]["treatment"]["keywords"]["strict"]) + \
        len(synth_data["cfg"]["treatment"]["keywords"]["broad_extra"])
    assert kw.loc["Claude", "n"] == 1 and kw.loc["Claude", "dictionary"] == "broad_extra"   # p11
    assert kw.loc["AI\\s*agent", "n"] == 0 and kw.loc["Gemini", "n"] == 0                  # p07은 분석 대상 직종이 아니다

# 과제 10 수정에서 더한 시험. 가상 자료 공고 12건에는 게시일이 없는 공고와 근무지가 `기타`인 공고가 없어,
# 시험 안에서만 아래 공고의 게시일과 근무지를 바꿔 쓴다. p02(분석 대상 직종, 생성형 AI 활용 숙련, 잡사이트B)와
# p07(분석 대상 직종 아님, 잡사이트A)은 게시일이 없다. p05, p09, p11, p12는 base에 남고 근무지가 `기타`이다.
# p07의 근무지도 `광산구`이지만 base에 없으므로 세지 않는다.
# 과제 3에서 가상 자료에 게시일이 없는 공고 p13(마감일 있음, 잡사이트A), p14(날짜 없음, 잡사이트B)가 더해졌고,
# 게시일이 없으면 마감일을 쓰므로(결정 25) 마감일이 있는 p02는 base에 남는다.
MISSING_POST_START = {"p02": None, "p07": None}
OTHER_REGIONS = {"p05": "광산구 어딘가", "p09": "광산구", "p11": "아시아·중동", "p12": "가" * 25, "p07": "광산구"}


def rewrite_postings(synth_data, post_start=MISSING_POST_START, region=OTHER_REGIONS):
    """2025년 가상 공고 파일에서 지정한 공고의 게시일과 근무지를 바꿔 다시 쓴다(시험 안에서만)."""
    path = rawio.posting_path(synth_data["postings_dir"], 2025)
    t = pq.read_table(path)
    ids = t.column("posting_id").to_pylist()
    for col, changes, typ in (("post_start", post_start, pa.date32()), ("region", region, pa.string())):
        values = [changes.get(i, v) for i, v in zip(ids, t.column(col).to_pylist())]
        t = t.set_column(t.schema.get_field_index(col), col, pa.array(values, typ))
    pq.write_table(t, path, row_group_size=5)


def test_scan_counts_postings_without_post_start_and_other_region_words(synth_data, tmp_path):
    rewrite_postings(synth_data)
    out = load_script().scan(synth_data["cfg"], synth_data["root"], tmp_path / "s0")
    stats = pd.read_parquet(out["scan_stats"]).set_index("year")
    m = pd.read_parquet(out["post_start_missing"])
    assert list(m.columns) == ["year", "source", "n_raw", "n_no_post_start", "n_in_occ", "n_in_occ_no_post_start",
                               "n_in_occ_use", "n_in_occ_use_no_post_start"]
    assert (m.groupby("year")["n_raw"].sum() == stats["n_raw"]).all()
    assert m["n_no_post_start"].sum() == len(MISSING_POST_START) + 2          # + p13, p14
    s = m.set_index("source")
    assert s.loc["잡사이트B"].drop("year").tolist() == [5, 2, 4, 2, 2, 1]   # p02, p05, p08, p11, p14
    assert s.loc["잡사이트A"].drop("year").tolist() == [6, 2, 5, 1, 1, 0]   # p01, p04, p07, p10, p13, p16(과제 3A)
    assert m["n_in_occ"].sum() == 14 and m["n_in_occ_use"].sum() == 3       # 게시일과 관계없이 센다(과제 3A: p15, p16을 더해 12에서 바꿨다)
    assert "p02" in set(pd.read_parquet(out["base"])["posting_id"])          # 마감일이 있어 남는다(과제 3)

    w = pd.read_parquet(out["sido_other_words"])
    assert list(w.columns) == ["word", "n"]
    assert w.set_index("word")["n"].to_dict() == {"광산구": 2, "아시아·중동": 1, "가" * 20: 1}
    assert w.iloc[0].tolist() == ["광산구", 2]


def test_scan_never_stores_posting_text(synth_data, tmp_path):
    out = load_script().scan(synth_data["cfg"], synth_data["root"], tmp_path / "s0")
    # 1단계 기록(scan_meta.json, 전체 검토 수정에서 더함)은 parquet가 아니어서 글자로 확인한다
    assert "에이전트 개발" not in out.pop("scan_meta").read_text(encoding="utf-8")
    for p in out.values():
        df = pd.read_parquet(p)
        assert "detail" not in df.columns and "title" not in df.columns
        assert not df.astype(str).apply(lambda c: c.str.contains("에이전트 개발")).any().any()
