import pandas as pd, pytest
from aabundle import classif, ked, rawio, sample

import synth

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


def test_posting_date_uses_columns_in_order():
    raw = pd.DataFrame({"post_start": ["2025-03-02", None, None, "날짜아님"],
                        "post_end": ["2025-04-01", "2025-05-10", None, "9999-12-31"]})
    date, source = sample.posting_date(raw, ["post_start", "post_end"])
    assert date.dt.strftime("%Y-%m-%d").tolist()[:2] == ["2025-03-02", "2025-05-10"]
    assert date.isna().tolist() == [False, False, True, True]          # 읽을 수 없는 값과 범위 밖의 날짜는 결측
    assert source.tolist()[:2] == ["post_start", "post_end"]
    assert source.isna().tolist() == [False, False, True, True]


def test_posting_date_accepts_date_objects():
    import datetime as dt
    raw = pd.DataFrame({"post_start": [dt.date(2025, 3, 2), None], "post_end": [None, dt.date(9999, 12, 31)]})
    date, source = sample.posting_date(raw, ["post_start", "post_end"])
    assert date.isna().tolist() == [False, True] and source.isna().tolist() == [False, True]


def test_posting_date_names_missing_columns():
    raw = pd.DataFrame({"post_start": ["2025-03-02"]})
    with pytest.raises(KeyError, match="post_end"):
        sample.posting_date(raw, ["post_start", "post_end"])


def test_period_name():
    groups = [{"name": "초기", "years": [2023, 2024]}, {"name": "1 도입기", "years": [2025]}]
    out = sample.period_name(pd.Series([2023, 2024, 2025, 2026, pd.NA], dtype="Int64"), groups)
    assert out.tolist()[:3] == ["초기", "초기", "1 도입기"]
    assert out.isna().tolist() == [False, False, False, True, True]


def test_merge_sido():
    s = pd.Series(["광주", "전남", "전남광주", "서울", "기타", pd.NA], dtype="string")
    out = sample.merge_sido(s, {"광주·전남": ["광주", "전남", "전남광주"]})
    assert out.tolist()[:5] == ["광주·전남", "광주·전남", "광주·전남", "서울", "기타"]
    assert pd.isna(out.iloc[5])


def test_posting_date_strips_whitespace():
    raw = pd.DataFrame({"post_start": ["2025-05-10 ", "2025-05-10\t", " 2025-05-10"], "post_end": [None, None, None]})
    date, source = sample.posting_date(raw, ["post_start", "post_end"])
    assert date.dt.strftime("%Y-%m-%d").tolist() == ["2025-05-10"] * 3 and source.tolist() == ["post_start"] * 3


def test_unusable_dates_marks_present_but_unusable_values():
    import datetime as dt
    raw = pd.DataFrame({
        "post_start": ["2025-03-02", None, "", "   ", "날짜아님", "9999-12-31"],
        "post_end": [dt.date(2025, 4, 1), dt.date(9999, 12, 31), dt.date(1, 1, 1), None, None, dt.date(2400, 3, 19)],
    })
    bad = sample.unusable_dates(raw, ["post_start", "post_end"])
    assert bad["post_start"].tolist() == [False, False, False, False, True, True]
    assert bad["post_end"].tolist() == [False, True, True, False, False, True]
    assert bad.index.equals(raw.index)


def test_unusable_dates_names_missing_columns():
    with pytest.raises(KeyError, match="post_end"):
        sample.unusable_dates(pd.DataFrame({"post_start": ["2025-03-02"]}), ["post_start", "post_end"])


def test_base_uses_post_end_when_post_start_is_missing(base, synth_data):       # base: 이 파일의 기존 고정물
    """가상 공고 가운데 게시일이 없고 마감일만 있는 공고는 마감일의 연도·월·기간을 갖고 분석 기간 안에 든다.
    둘 다 없는 공고는 연도가 결측이고 분석 기간 밖이다."""
    by_id = base   # 이 파일의 base 고정물은 이미 posting_id를 번호표로 쓴다
    assert by_id.loc[synth.ID_END_ONLY, "date_source"] == "post_end"
    assert by_id.loc[synth.ID_END_ONLY, "in_period"] and by_id.loc[synth.ID_END_ONLY, "period"] == synth.PERIOD_NAME
    # 과제 3 수정 1차: 월, 분기, 업력도 마감일로 계산한다(업력 = 마감일의 연도 - KED 설립 연도)
    end = pd.Timestamp(synth._POST_END_WITHOUT_START[synth.ID_END_ONLY])
    bzno = next(p[2] for p in synth._POSTINGS if p[0] == synth.ID_END_ONLY)
    k = ked.extract_ked_firm(synth_data["ked_firm"]).set_index("bzno")
    assert by_id.loc[synth.ID_END_ONLY, "month"] == end.month
    assert by_id.loc[synth.ID_END_ONLY, "quarter"] == f"{end.year}Q{end.quarter}"
    assert by_id.loc[synth.ID_END_ONLY, "firm_age"] == end.year - k.loc[bzno, "ked_estab_year"]
    assert pd.isna(by_id.loc[synth.ID_NO_DATE, "date_source"]) and pd.isna(by_id.loc[synth.ID_NO_DATE, "year"])
    assert not by_id.loc[synth.ID_NO_DATE, "in_period"]
    assert base.loc[base["in_period"], "period"].notna().all()


def test_base_merges_sido_and_blanks_ksic_not_in_table(base):
    assert not base["sido"].isin(["광주", "전남", "전남광주"]).any()
    assert (base["sido"] == "광주·전남").any()          # 가상 공고의 근무지에 광주와 전남광주가 있다
    assert not (base["ksic_sub"] == classif.KSIC_NOT_IN_TABLE).any()
    # 분류표에 없는 코드는 5자리 코드는 남고 소분류만 결측이다(보고서가 건수를 셀 수 있게)
    assert (base["ksic5"].notna() & base["ksic_sub"].isna()).any()

def test_base_keeps_every_row_and_flags(base):
    assert len(base) == 17                               # 과제 3에서 p13, p14를 더해 12에서 14로, 과제 3A에서 p15, p16을 더해 16으로 바꿨다(과제 4: p17을 더해 17)
    assert sorted(base.index[~base.in_occ]) == ["p07", "p08"]
    assert sorted(base.index[~base.has_bzno]) == ["p04"]
    # 과제 3: 게시일과 마감일이 모두 없는 공고. 과제 4: 마감일이 날짜로 쓸 수 없는 값인 p17을 더했다
    assert sorted(base.index[~base.in_period]) == [synth.ID_NO_DATE, synth.ID_END_UNUSABLE]
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
    # p15, p16: 과제 3A에서 더한 공고
    assert sorted(kept.posting_id) == ["p01", "p02", "p03", "p05", "p09", "p11", "p12", "p13", "p15", "p16"]
    # 과제 3: 연도가 결측인 공고(p14)가 생겨 건수 표에 연도가 결측인 행이 있으므로 단계별로 더해 본다
    c = counts.groupby("step")[["n_dropped", "n_after"]].sum()
    assert c.loc["in_period", "n_dropped"] == 2   # p14, p17(과제 4에서 p17을 더해 1에서 바꿨다)
    assert c.loc["in_occ", "n_dropped"] == 2 and c.loc["has_bzno", "n_dropped"] == 1 and c.loc["has_skill", "n_dropped"] == 2
    assert c.loc["has_skill", "n_after"] == 10   # 과제 3A에서 p15, p16을 더해 8에서 바꿨다

def test_period_filter(synth_data):
    cfg = synth_data["cfg"]
    cfg["period"].update({"start": "2025-07-01", "end": "2025-12-31"})
    raw = pd.concat(rawio.iter_postings(synth_data["postings_dir"], 2025, sample.RAW_COLUMNS))
    b = sample.build_posting_base(raw, ked.extract_ked_firm(synth_data["ked_firm"]), set(), cfg)
    assert (~b.in_period).sum() == 6                     # p01~p04와 날짜가 없는 p14(과제 3에서 4에서 바꿨다), p17(과제 4에서 6으로 바꿨다)

def test_period_compares_dates_not_times(synth_data):
    # 전체 검토 수정에서 더한 시험. 게시일에 시각이 있어도 날짜로 분석 기간을 판단한다.
    cfg = synth_data["cfg"]
    cfg["period"].update({"start": "2025-07-01", "end": "2025-12-31"})
    raw = pd.concat(rawio.iter_postings(synth_data["postings_dir"], 2025, sample.RAW_COLUMNS)).reset_index(drop=True)
    raw = raw.iloc[:4].copy()
    raw["post_start"] = pd.to_datetime(["2025-12-31 23:59:00", "2025-07-01 00:00:00", "2025-06-30 23:59:59",
                                        "2026-01-01 00:00:00"])
    b = sample.build_posting_base(raw, ked.extract_ked_firm(synth_data["ked_firm"]), set(), cfg)
    assert b["in_period"].tolist() == [True, True, False, False]

def test_apply_exclusions_keeps_input_index_labels(base, synth_data):
    # 전체 검토 수정에서 더한 시험. 스크립트의 `_exclusions`는 남은 행의 번호표로 위치를 찾는다.
    b = base.reset_index()
    b.index = b.index * 10 + 3
    kept, _ = sample.apply_exclusions(b, synth_data["cfg"])
    assert kept.index.tolist() == b.index[b["in_period"] & b["in_occ"] & b["has_bzno"]].tolist()
    # 123: 과제 3에서 더한 p13. 143, 153: 과제 3A에서 더한 p15, p16
    assert kept.index.tolist() == [3, 13, 23, 43, 53, 83, 93, 103, 113, 123, 143, 153]

def test_missing_bzno_never_joins(synth_data):
    raw = pd.concat(rawio.iter_postings(synth_data["postings_dir"], 2025, sample.RAW_COLUMNS))
    k = ked.extract_ked_firm(synth_data["ked_firm"])
    bad = pd.DataFrame({"bzno": pd.array([None], dtype="string"), "ked_estab_year": pd.array([2000], dtype="Int16"),
                        "ked_size": pd.array(["BAD"], dtype="string"), "ked_ipo": pd.array(["Y"], dtype="string"),
                        "ked_venture": pd.array(["Y"], dtype="string")})
    b = sample.build_posting_base(raw, pd.concat([k, bad], ignore_index=True), set(), synth_data["cfg"]).set_index("posting_id")
    assert len(b) == 17                                  # 과제 3에서 p13, p14를 더해 12에서 14로, 과제 3A에서 p15, p16을 더해 16으로 바꿨다(과제 4: p17을 더해 17)
    assert not b.loc["p04", "ked_matched"]
    assert pd.isna(b.loc["p04", "ked_size"]) and pd.isna(b.loc["p04", "firm_age"])
    assert b.loc["p01", "ked_matched"] and b.loc["p01", "firm_age"] == 15
    assert not (b["ked_size"] == "BAD").any()
