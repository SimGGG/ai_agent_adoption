import pandas as pd, pytest
from aabundle import classif, ked, rawio, sample

@pytest.fixture
def base(synth_data):
    raw = pd.concat(rawio.iter_postings(synth_data["postings_dir"], 2025, sample.RAW_COLUMNS))
    return sample.build_posting_base(raw, ked.extract_ked_firm(synth_data["ked_firm"]),
                                     classif.load_ksic_sub(synth_data["ksic"]), synth_data["cfg"]).set_index("posting_id")

def test_to_sido_variants():
    s = pd.Series(["서울 마포구, 서울전체, 영등포구", "충청남도 천안시", "세종특별자치시", "서울전체,",
                   "광산구", "아시아·중동", "전남광주 어딘가", None])
    out = sample.to_sido(s)
    # 7번째 값: 과제 10 수정에서 전남광주를 별도 범주로 두기로 해(설계 문서 6.7절 16번) `기타`에서 바꿨다
    assert out.tolist()[:7] == ["서울", "충남", "세종", "서울", "기타", "기타", "전남광주"]
    assert pd.isna(out[7])

def test_to_sido_real_forms():
    s = pd.Series(["서울시", "인천시 연수구", "대전시", "세종전지역", "서울>서울전지역", "서울>강남구",
                   "서울&gt;강남구", "서울전지역", "경기전지역", "전남광주", "전남광주통합특별시 광산구",
                   "아시아·중동", "전국", "글로벌", "서울특별시 강남구, 경기도 성남시", None])
    out = sample.to_sido(s)
    assert out.iloc[:11].tolist() == ["서울", "인천", "대전", "세종", "서울", "서울", "서울", "서울", "경기",
                                      "전남광주", "전남광주"]
    assert out.iloc[11:14].tolist() == ["기타", "기타", "기타"]
    assert out.iloc[14] == "서울"
    assert pd.isna(out.iloc[15])


def test_to_sido_does_not_match_by_prefix():
    out = sample.to_sido(pd.Series(["서울대입구", "경기장", "부산물", "시"]))
    assert out.tolist() == ["기타", "기타", "기타", "기타"]


def test_region_first_word():
    out = sample.region_first_word(pd.Series(["서울>강남구", "세종전지역", " 서울시 강남구", "서울전체, 부산", "", None]))
    assert out.iloc[:5].tolist() == ["서울", "세종", "서울시", "서울", ""]
    assert pd.isna(out.iloc[5])


def test_salary_band():
    s = pd.Series([None, 25_000_000, 40_000_000])
    assert sample.salary_band(s, []).tolist() == ["미기재", "기재", "기재"]
    out = sample.salary_band(s, [30_000_000])
    assert out[0] == "미기재" and out[1] != out[2]

def test_base_keeps_every_row_and_flags(base):
    assert len(base) == 12
    assert sorted(base.index[~base.in_occ]) == ["p07", "p08"]
    assert sorted(base.index[~base.has_bzno]) == ["p04"]
    assert base.in_period.all()
    assert base.loc["p05", "quarter"] == "2025Q3" and base.loc["p01", "keco2"] == "13"

def test_ked_join(base):
    assert base.loc["p01", "ked_matched"] and base.loc["p01", "firm_age"] == 15   # 2025 - 2010
    assert not base.loc["p09", "ked_matched"]          # 1000000005는 KED 가상 파일에서 버려진 행
    assert pd.isna(base.loc["p05", "firm_age"])         # 결합됐지만 설립일 없음

def test_ksic_columns(base):
    assert set(base["ksic_sub"].dropna()) <= {"620", "862", "012", classif.KSIC_NOT_IN_TABLE}

def test_exclusions_in_order(base, synth_data):
    has_skill = pd.Series(True, index=base.index)
    has_skill[["p06", "p10"]] = False
    kept, counts = sample.apply_exclusions(base.reset_index(), synth_data["cfg"], has_skill.reset_index(drop=True))
    assert sorted(kept.posting_id) == ["p01", "p02", "p03", "p05", "p09", "p11", "p12"]
    c = counts.set_index("step")
    assert c.loc["in_occ", "n_dropped"] == 2 and c.loc["has_bzno", "n_dropped"] == 1 and c.loc["has_skill", "n_dropped"] == 2
    assert c.loc["has_skill", "n_after"] == 7

def test_period_filter(synth_data):
    cfg = synth_data["cfg"]
    cfg["period"] = {"start": "2025-07-01", "end": "2025-12-31", "unit": "quarter"}
    raw = pd.concat(rawio.iter_postings(synth_data["postings_dir"], 2025, sample.RAW_COLUMNS))
    b = sample.build_posting_base(raw, ked.extract_ked_firm(synth_data["ked_firm"]), set(), cfg)
    assert (~b.in_period).sum() == 4                     # p01~p04

def test_missing_bzno_never_joins(synth_data):
    raw = pd.concat(rawio.iter_postings(synth_data["postings_dir"], 2025, sample.RAW_COLUMNS))
    k = ked.extract_ked_firm(synth_data["ked_firm"])
    bad = pd.DataFrame({"bzno": pd.array([None], dtype="string"), "ked_estab_year": pd.array([2000], dtype="Int16"),
                        "ked_size": pd.array(["BAD"], dtype="string"), "ked_ipo": pd.array(["Y"], dtype="string"),
                        "ked_venture": pd.array(["Y"], dtype="string")})
    b = sample.build_posting_base(raw, pd.concat([k, bad], ignore_index=True), set(), synth_data["cfg"]).set_index("posting_id")
    assert len(b) == 12
    assert not b.loc["p04", "ked_matched"]
    assert pd.isna(b.loc["p04", "ked_size"]) and pd.isna(b.loc["p04", "firm_age"])
    assert b.loc["p01", "ked_matched"] and b.loc["p01", "firm_age"] == 15
    assert not (b["ked_size"] == "BAD").any()
