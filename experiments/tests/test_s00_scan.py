import datetime as dt
import importlib.util, pandas as pd, paths
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import synth
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
    assert stats.loc[2025, "n_raw"] == 17 and stats.loc[2025, "n_in_occ"] == 13   # 과제 3: p13, p14를 더해 12, 10에서 14, 11로, 과제 3A: p15, p16을 더해 16, 13으로, 과제 4: p17을 더해 n_raw를 17로 바꿨다
    assert stats.loc[2025, "n_unmatched"] == 1

    kw = pd.read_parquet(out["kw_pattern_hits"])
    kw = kw[kw["dictionary"] != "use"].set_index("pattern")   # 과제 4에서 더한 생성형 AI 활용 사전의 행은 따로 시험한다
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


def rewrite_postings(synth_data, post_start=MISSING_POST_START, region=OTHER_REGIONS, post_end=None):
    """2025년 가상 공고 파일에서 지정한 공고의 게시일, 근무지, 마감일을 바꿔 다시 쓴다(시험 안에서만).

    `post_end`는 최종 검토 수정에서 더했다. 주지 않으면 마감일은 바꾸지 않는다."""
    path = rawio.posting_path(synth_data["postings_dir"], 2025)
    t = pq.read_table(path)
    ids = t.column("posting_id").to_pylist()
    for col, changes, typ in (("post_start", post_start, pa.date32()), ("post_end", post_end or {}, pa.date32()),
                              ("region", region, pa.string())):
        values = [changes.get(i, v) for i, v in zip(ids, t.column(col).to_pylist())]
        t = t.set_column(t.schema.get_field_index(col), col, pa.array(values, typ))
    pq.write_table(t, path, row_group_size=5)


def test_scan_counts_postings_without_post_start_and_other_region_words(synth_data, tmp_path):
    rewrite_postings(synth_data)
    out = load_script().scan(synth_data["cfg"], synth_data["root"], tmp_path / "s0")
    stats = pd.read_parquet(out["scan_stats"]).set_index("year")
    m = pd.read_parquet(out["post_start_missing"])
    # 과제 4: 게시일을 정할 수 없는 공고, 날짜로 쓸 수 없는 마감일, 마감일로 게시일을 정한 공고의 건수 열을 더했다
    assert list(m.columns) == ["year", "source", "n_raw", "n_no_post_start", "n_no_date", "n_end_unusable",
                               "n_in_occ", "n_in_occ_no_post_start", "n_in_occ_no_date", "n_in_occ_end_unusable",
                               "n_in_occ_from_end_in_period", "n_in_occ_use", "n_in_occ_use_no_post_start",
                               "n_in_occ_use_no_date", "n_in_occ_use_from_end_in_period"]
    assert (m.groupby("year")["n_raw"].sum() == stats["n_raw"]).all()
    assert m["n_no_post_start"].sum() == len(MISSING_POST_START) + synth.N_NO_POST_START   # + p13, p14, p17(과제 4)
    s = m.set_index("source").drop(columns="year")
    # 잡사이트B: p02, p05, p08, p11, p14, p17(과제 4). p02는 마감일로 게시일을 정하고, p17은 마감일이 9999-12-31
    assert s.loc["잡사이트B"].to_dict() == {
        "n_raw": 6, "n_no_post_start": 3, "n_no_date": 2, "n_end_unusable": 1,
        "n_in_occ": 5, "n_in_occ_no_post_start": 3, "n_in_occ_no_date": 2, "n_in_occ_end_unusable": 1,
        "n_in_occ_from_end_in_period": 1, "n_in_occ_use": 2, "n_in_occ_use_no_post_start": 1,
        "n_in_occ_use_no_date": 0, "n_in_occ_use_from_end_in_period": 1}
    # 잡사이트A: p01, p04, p07, p10, p13, p16(과제 3A). p13은 마감일로 게시일을 정한다
    assert s.loc["잡사이트A"].to_dict() == {
        "n_raw": 6, "n_no_post_start": 2, "n_no_date": 0, "n_end_unusable": 0,
        "n_in_occ": 5, "n_in_occ_no_post_start": 1, "n_in_occ_no_date": 0, "n_in_occ_end_unusable": 0,
        "n_in_occ_from_end_in_period": 1, "n_in_occ_use": 1, "n_in_occ_use_no_post_start": 0,
        "n_in_occ_use_no_date": 0, "n_in_occ_use_from_end_in_period": 0}
    assert m["n_in_occ"].sum() == 15 and m["n_in_occ_use"].sum() == 3       # 게시일과 관계없이 센다(과제 3A: p15, p16을 더해 12에서 14로, 과제 4: p17을 더해 15로 바꿨다)
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


# 과제 4에서 더한 시험. 가상 자료를 바꾸지 않고 1단계를 한 번 실행한 결과를 본다.
@pytest.fixture
def s00():
    return load_script()


@pytest.fixture
def scan_out(s00, synth_data, tmp_path):
    return s00.scan(synth_data["cfg"], synth_data["root"], tmp_path / "s0")


def test_scan_counts_dates_taken_from_post_end(scan_out):
    psm = pd.read_parquet(scan_out["post_start_missing"])
    total = psm.drop(columns=["year", "source"]).sum()
    # 가상 공고 ID_END_ONLY, ID_NO_DATE, ID_END_UNUSABLE(과제 4) / ID_NO_DATE, ID_END_UNUSABLE
    assert total["n_no_post_start"] == synth.N_NO_POST_START and total["n_no_date"] == synth.N_NO_DATE
    assert total["n_end_unusable"] == synth.N_END_UNUSABLE == 1 and total["n_in_occ_end_unusable"] == 1   # ID_END_UNUSABLE
    assert total["n_in_occ_from_end_in_period"] == 1
    assert total["n_in_occ_no_post_start"] - total["n_in_occ_no_date"] >= total["n_in_occ_from_end_in_period"]
    base = pd.read_parquet(scan_out["base"])
    assert synth.ID_END_ONLY in set(base["posting_id"]) and synth.ID_NO_DATE not in set(base["posting_id"])
    assert synth.ID_END_UNUSABLE not in set(base["posting_id"])
    assert set(base["date_source"]) == {"post_start", "post_end"} and base["period"].notna().all()


def test_scan_writes_post_end_gap(scan_out, s00):
    gap = pd.read_parquet(scan_out["post_end_gap"])
    assert list(gap.columns) == ["year", "bin", "n"]
    assert gap["bin"].nunique() == len(s00.POST_END_GAP_BINS)                 # 공고가 없는 구간도 0으로 있다
    assert gap.loc[gap["bin"] == "8~30일", "n"].sum() >= 1                      # 가상 공고의 마감일은 게시일 30일 뒤
    assert gap["n"].sum() == synth.N_IN_OCC_BOTH_DATES      # 분석 대상 세분류이고 두 날짜가 모두 있는 가상 공고 수


def test_post_end_gap_is_empty_with_one_date_column(synth_data, tmp_path, s00):
    synth_data["cfg"]["period"]["date_columns"] = ["post_start"]
    out = s00.scan(synth_data["cfg"], synth_data["root"], tmp_path / "s0")
    gap = pd.read_parquet(out["post_end_gap"])
    assert list(gap.columns) == ["year", "bin", "n"] and gap.empty
    psm = pd.read_parquet(out["post_start_missing"])
    assert psm["n_end_unusable"].sum() == 0 and psm["n_in_occ_from_end_in_period"].sum() == 0


def test_scan_stores_text_length(scan_out):
    base = pd.read_parquet(scan_out["base"]).set_index("posting_id")
    assert base["text_len"].dtype == "int64" and (base["text_len"] >= 0).all()
    detail = {p[0]: p[4] for p in synth._POSTINGS}       # 설정의 본문 열은 detail 하나이다
    assert base["text_len"].to_dict() == {i: len(detail[i] or "") for i in base.index}
    assert base.loc["p06", "text_len"] == 0               # 본문이 없는 공고
    # 본문 자체는 어느 결과에도 없다(test_scan_never_stores_posting_text는 새 열과 새 파일까지 모두 본다)
    texts = [p[4] for p in synth._POSTINGS if p[4]]
    for name in ("base", "treat", "skills_long"):
        df = pd.read_parquet(scan_out[name]).astype(str)
        for t in texts:
            assert not df.apply(lambda c: c.str.contains(t, regex=False)).any().any()


def test_scan_counts_use_dictionary_patterns(scan_out, synth_data):
    kw = pd.read_parquet(scan_out["kw_pattern_hits"])
    use = kw[kw["dictionary"] == "use"]
    patterns = synth_data["cfg"]["treatment"]["keywords"]["use"]
    assert use.groupby("year")["pattern"].apply(list).to_dict() == {y: patterns for y in synth_data["cfg"]["data"]["years"]}
    by_pattern = use.groupby("pattern")["n"].sum()
    assert by_pattern[r"Chat\s*GPT"] >= 1                                  # ID_USE_KEYWORD_ONLY
    # `Claude`는 넓은 사전과 생성형 AI 활용 사전에 모두 있다. 사전마다 따로 센다(p11)
    assert by_pattern["Claude"] == 1
    assert kw.loc[(kw["dictionary"] == "broad_extra") & (kw["pattern"] == "Claude"), "n"].sum() == 1
    treat = pd.read_parquet(scan_out["treat"])
    assert by_pattern.max() <= treat["kw_use"].sum()                         # kw_use가 참인 공고 안에서만 센다


# 아래는 최종 검토 수정에서 더한 시험이다.

def test_fallback_date_outside_the_period_is_dropped_but_counted(synth_data, tmp_path, s00):
    # 분석 기간을 2025~2026년으로 넓히고(기간 둘), 분석 대상 세분류(1332)의 두 공고에서 게시일을 지운다.
    # p12는 마감일이 공고 파일의 연도(2025)와 다른 해이지만 분석 기간 안(2026-01-15)이어서 남고,
    # p11은 마감일이 분석 기간 밖(2027-01-15)이어서 빠진다.
    cfg = synth_data["cfg"]
    cfg["period"]["end"] = "2026-12-31"
    cfg["period"]["groups"] = [{"name": synth.PERIOD_NAME, "years": [2025]}, {"name": "가상 기간 2", "years": [2026]}]
    rewrite_postings(synth_data, post_start={"p11": None, "p12": None}, region={},
                     post_end={"p11": dt.date(2027, 1, 15), "p12": dt.date(2026, 1, 15)})
    out = s00.scan(cfg, synth_data["root"], tmp_path / "s0")
    base = pd.read_parquet(out["base"]).set_index("posting_id")
    assert "p11" not in base.index
    assert base.loc["p12", "date_source"] == "post_end" and base.loc["p12", "year"] == 2026
    total = pd.read_parquet(out["post_start_missing"]).drop(columns=["year", "source"]).sum()
    assert total["n_in_occ_no_post_start"] == 5        # p11, p12, ID_END_ONLY, ID_NO_DATE, ID_END_UNUSABLE
    assert total["n_in_occ_no_date"] == 2              # ID_NO_DATE, ID_END_UNUSABLE
    assert total["n_in_occ_from_end_in_period"] == 2   # p12, ID_END_ONLY
    # 마감일로 정한 게시일이 분석 기간 밖인 공고(p11)가 있어 부등호가 등호가 아니다
    assert total["n_in_occ_no_post_start"] - total["n_in_occ_no_date"] > total["n_in_occ_from_end_in_period"]


def _add_date_column(synth_data, name: str, values: dict) -> None:
    """2025년 가상 공고 파일에 날짜 열 하나를 더해 다시 쓴다. `values`는 {공고 번호: 날짜}이고 나머지는 결측이다."""
    path = rawio.posting_path(synth_data["postings_dir"], 2025)
    t = pq.read_table(path)
    ids = t.column("posting_id").to_pylist()
    t = t.append_column(name, pa.array([values.get(i) for i in ids], pa.date32()))
    pq.write_table(t, path, row_group_size=5)


def test_scan_reads_a_configured_date_column_outside_the_raw_columns(synth_data, tmp_path, s00):
    # 게시일과 마감일이 모두 없는 ID_NO_DATE에만 셋째 날짜 열의 값을 준다
    _add_date_column(synth_data, "post_alt", {synth.ID_NO_DATE: dt.date(2025, 6, 1)})
    synth_data["cfg"]["period"]["date_columns"] = ["post_start", "post_end", "post_alt"]
    out = s00.scan(synth_data["cfg"], synth_data["root"], tmp_path / "s0")
    base = pd.read_parquet(out["base"]).set_index("posting_id")
    assert base.loc[synth.ID_NO_DATE, "date_source"] == "post_alt" and base.loc[synth.ID_NO_DATE, "month"] == 6


def test_scan_names_a_missing_date_column_before_reading(synth_data, tmp_path, s00):
    synth_data["cfg"]["period"]["date_columns"] = ["post_start", "post_end", "post_alt"]
    with pytest.raises(KeyError, match=r"없는 열: \['post_alt'\]"):
        s00.scan(synth_data["cfg"], synth_data["root"], tmp_path / "s0")
    assert not (tmp_path / "s0").exists()      # 훑기 전에 멈춰 결과 폴더도 만들지 않는다
