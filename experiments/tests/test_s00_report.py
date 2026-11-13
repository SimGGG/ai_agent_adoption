import copy
import re

import numpy as np
import pandas as pd
import pytest
import synth
from aabundle import skills, treatment
from test_s00_scan import load_script, rewrite_postings

def run(synth_data, tmp_path):
    s00 = load_script()
    s00.scan(synth_data["cfg"], synth_data["root"], tmp_path / "s0")
    return s00, s00.build_tables(synth_data["cfg"], synth_data["root"], tmp_path / "s0")

def test_exclusion_table_matches_fixture(synth_data, tmp_path):
    _, t = run(synth_data, tmp_path)
    c = t["t01_exclusions"].query("year == 2025").set_index("step")
    assert c.loc["in_occ", "n_dropped"] == 2 and c.loc["has_bzno", "n_dropped"] == 1
    assert c.loc["has_skill", "n_dropped"] == 2 and c.loc["has_skill", "n_after"] == 10   # 과제 3: p13을 더해 7에서 8로, 과제 3A: p15, p16을 더해 10으로 바꿨다

def test_treated_counts_by_period(synth_data, tmp_path):
    _, t = run(synth_data, tmp_path)
    q = t["t02a_treated_by_period"].query("unit == 'quarter'").set_index("period")
    assert q.loc["2025Q1", "has_use"] == 1 and q.loc["2025Q3", "has_use"] == 1   # p02 / p05 (p04는 사업자번호 없음)
    assert q.loc["2025Q4", "has_use"] == 0                                        # 처치 공고가 없는 기간도 0으로 나온다

def test_occupation_list_check(synth_data, tmp_path):
    cfg = synth_data["cfg"]; cfg["occupations"]["min_use_postings"] = 2
    s00 = load_script(); s00.scan(cfg, synth_data["root"], tmp_path / "s0")
    chk = s00.build_tables(cfg, synth_data["root"], tmp_path / "s0")["t10b_occ_list_check"].set_index("keco4")
    assert chk.loc["0262", "status"] == "양쪽에 있음" and chk.loc["1332", "status"] == "설정에만 있음"

def test_agent_skill_table_lists_every_code(synth_data, tmp_path):
    _, t = run(synth_data, tmp_path)
    a = t["t06b_agent_skill_by_year"].set_index("skill_code")
    assert len(a) == 4 and a.loc["IT-05-0410", "kind"] == "생성형 AI 활용" and a.loc["IT-05-0413", "kind"] == "개발"
    # 과제 5: `use` 사전의 행이 더해져 Claude, Gemini가 두 사전에 한 행씩 있으므로 넓은 사전의 행만 본다
    k = t["t06c_keyword_pattern_hits"].query("dictionary == 'broad_extra'").set_index("pattern")
    assert k.loc["Claude", "total"] == 1 and k.loc["Gemini", "total"] == 0

def test_timing_skips_when_too_small(synth_data, tmp_path):
    s00, _ = run(synth_data, tmp_path)
    d = tmp_path / "s0"
    timing = s00.time_outcome_model(pd.read_parquet(d / "base.parquet"), pd.read_parquet(d / "treat.parquet"),
                                    pd.read_parquet(d / "skills_long.parquet"), synth_data["cfg"])
    assert set(timing) == {"skill_code", "n_train", "n_features", "seconds", "n_skills_kept", "est_total_hours"}

def test_report_has_all_sections_and_no_text(synth_data, tmp_path):
    s00, t = run(synth_data, tmp_path)
    timing = {"skill_code": None, "n_train": 0, "n_features": 14, "seconds": None, "n_skills_kept": 0, "est_total_hours": None}
    p = s00.write_report(t, timing, synth_data["cfg"], tmp_path / "report.md", commit="testcommit")
    text = p.read_text(encoding="utf-8")
    assert all(f"## {i}." in text for i in range(1, 12))
    assert "testcommit" in text and "임시" in text and "에이전트 개발" not in text


# 아래는 과제 9에서 더한 시험이다.

def test_ked_line_counts_and_csv(synth_data, tmp_path):
    _, t = run(synth_data, tmp_path)
    k = t["t04b_ked_lines"]
    # n_masked_values는 전체 검토 수정에서 더한 열이다
    assert list(k.columns) == ["n_lines", "n_dropped_lines", "n_masked_values", "n_firms"] and len(k) == 1
    assert k.loc[0, "n_lines"] == 10 and k.loc[0, "n_dropped_lines"] == 4 and k.loc[0, "n_firms"] == 4
    assert k.loc[0, "n_masked_values"] == 0
    for name in t:
        csv = pd.read_csv(tmp_path / "s0" / "tables" / f"{name}.csv", encoding="utf-8-sig", dtype=str)
        cells = csv.fillna("").to_numpy().ravel().tolist()
        assert "posting_id" not in csv.columns and "bzno" not in csv.columns
        assert not any(re.fullmatch(r"p\d{2}|\d{10}", c) for c in cells)   # 공고 번호와 사업자번호가 없다


def _larger_sample(n=400, n_firms=40, n_use=60, seed=0):
    """학습 시간을 실제로 재 볼 만큼 큰 가상 자료(시험 안에서만 쓴다)."""
    rng = np.random.default_rng(seed)
    ids = [f"q{i:04d}" for i in range(n)]
    pick = lambda values: rng.choice(values, size=n)
    base = pd.DataFrame({
        "posting_id": ids, "year": 2025, "month": pick([1, 4, 7, 10]),
        "bzno": [f"{2000000000 + i % n_firms}" for i in range(n)],
        "firm_emp_log": rng.normal(3, 1, n), "ked_size": pick(["S1", "M1", None]), "ked_ipo": pick(["N", "Y"]),
        "ked_venture": pick(["Y", "N"]), "firm_age": rng.integers(0, 30, n).astype("float64"),
        "ksic_sub": pick(["620", "582"]), "keco4": pick(["1332", "0262"]), "career_type": pick(["ENTRY", "IRRELEVANT"]),
        "career_years_min": rng.integers(0, 5, n).astype("float64"), "edu_level": pick(["BACHELOR", "HIGH_SCHOOL"]),
        "salary_band": pick(["기재", "미기재"]), "emp_type": pick(["regular", "part_time"]), "sido": pick(["서울", "경기"]),
        "in_period": True, "in_occ": True, "has_bzno": True,
        # 과제 5에서 학습 시간 측정의 설명 변수에 더한 열(text_len은 text_len_log로 바꿔 쓴다)
        "source": pick(["잡사이트A", "잡사이트B"]), "text_len": rng.integers(0, 3000, n),
    })
    base["quarter"] = "2025Q" + ((base["month"] - 1) // 3 + 1).astype(str)
    has_use = np.arange(n) < n_use
    # kw_use: 과제 3A에서 더한 열(처치 정의를 고르려면 있어야 한다)
    treat = pd.DataFrame({"posting_id": ids, "kw_strict": False, "kw_broad": False, "kw_use": False, "has_dev": False,
                          "has_use": has_use, "treat_code": has_use, "treat_all": has_use})
    # S1은 짝수 번째, S2는 3의 배수 번째, S3는 둘 다 아닌 공고에 둔다(모든 공고에 숙련이 하나 이상 있다).
    rows = [(pid, code, "occupation") for i, pid in enumerate(ids)
            for code, hit in (("S1", i % 2 == 0), ("S2", i % 3 == 0), ("S3", i % 2 and i % 3)) if hit]
    long = pd.DataFrame(rows, columns=["posting_id", "skill_code", "src"])
    base["n_skills"] = long["posting_id"].value_counts().reindex(base["posting_id"], fill_value=0).to_numpy()
    return base, treat, long


def test_timing_measures_one_model_without_output(synth_data, capsys):
    s00 = load_script()
    cfg = synth_data["cfg"]; cfg["skills"]["sparse_filter"]["min_treated"] = 5
    base, treat, long = _larger_sample()   # 처치 공고 60건 가운데 S1 30건, S2 20건, S3 20건
    timing = s00.time_outcome_model(base, treat, long, cfg)
    # 과제 5: 설명 변수에 source와 text_len_log를 더해 14에서 16으로 바꿨다
    assert timing["skill_code"] == "S1" and timing["n_features"] == 16 and timing["n_train"] > 0
    assert isinstance(timing["seconds"], float) and timing["seconds"] >= 0
    assert timing["n_skills_kept"] == 3
    n_folds = synth_data["cfg"]["estimation"]["n_folds"]
    assert timing["est_total_hours"] == pytest.approx(timing["seconds"] * n_folds * 3 / 3600)
    assert capsys.readouterr().out == ""


def test_command_line_report_and_all(synth_data, tmp_path, monkeypatch):
    s00 = load_script()
    monkeypatch.setattr(s00.paths, "ROOT", synth_data["root"])
    monkeypatch.setattr(s00.paths, "S0", tmp_path / "s0cli")
    monkeypatch.setattr(s00.paths, "load_config", lambda: synth_data["cfg"])
    with pytest.raises(FileNotFoundError, match="base.parquet"):
        s00.main(["report"])
    assert s00.main(["all"]) == 0
    text = (synth_data["root"] / "docs" / "s0-data-audit.md").read_text(encoding="utf-8")
    assert all(f"## {i}." in text for i in range(1, 12))
    assert "`Claude`" in text and "`AI\\s*agent`" in text
    # 공고 번호(p10은 연봉 10% 분위수의 열 이름과 같아 뺀다)와 사업자번호가 보고서에 없다.
    # 과제 5: 가상 공고 p13~p17까지 넓혔다(p25 등은 분위수 열 이름이라 `\bp\d{2}\b`로는 넓힐 수 없다)
    assert not re.search(r"\bp(0[1-9]|1[1-7])\b", text) and not re.search(r"\b1000000\d{3}\b", text)
    assert (tmp_path / "s0cli" / "tables" / "t01_exclusions.csv").exists()


def test_report_numbers_keep_small_rates_and_csv_keeps_full_precision(synth_data, tmp_path):
    s00, t = run(synth_data, tmp_path)
    f = lambda v: s00._fmt_cell(v, "has_use_rate")
    assert f(4e-05) == "0.00004" and f(0.00012) == "0.00012" and f(0.00049) == "0.00049"
    assert f(0.5) == "0.5" and f(0.0) == "0" and f(1.625) == "1.625" and f(45_000_000.0) == "45,000,000"
    assert all("e" not in f(v).lower() for v in (4e-05, 1.234e-07, 0.00049))
    # CSV에는 보고서용으로 줄인 값이 아니라 계산한 값 그대로가 들어간다
    csv = pd.read_csv(tmp_path / "s0" / "tables" / "t04_bzno_ked_by_treatment.csv", encoding="utf-8-sig")
    # 과제 3: KED와 결합되는 비교 공고 p13을 더해 5 / 7에서 6 / 8로 바꿨다. 과제 3A: KED와 결합되는 p15와
    # 결합되지 않는 p16(1000000005는 KED 가상 파일에서 버려진 행)을 더해 7 / 10으로 바꿨다
    assert csv.loc[csv["has_use"] == False, "ked_matched_rate"].iloc[0] == pytest.approx(7 / 10, rel=1e-12)


# 아래는 과제 10 수정(첫 실제 자료 실행에서 드러난 문제)에서 더한 시험이다.

_NO_TIMING = {"skill_code": None, "n_train": 0, "n_features": 14, "seconds": None, "n_skills_kept": 0,
              "est_total_hours": None}


def test_post_start_missing_and_other_region_words_in_report(synth_data, tmp_path):
    rewrite_postings(synth_data)   # p02, p07은 게시일 없음. p05, p09, p11, p12는 근무지가 `기타`
    s00, t = run(synth_data, tmp_path)
    m = t["t01b_post_start_missing"]
    y = m[m["level"] == "연도"].set_index("year")
    assert len(y) == 1 and pd.isna(y.loc[2025, "source"])
    # 과제 3: 게시일이 없는 가상 공고 p13, p14를 더해 12, 2, 2 / 12에서 바꿨다
    # 과제 3A: 게시일이 있는 공고 p15, p16을 더해 14, 4 / 14에서 16, 4 / 16으로 바꿨다
    # 과제 4: 게시일이 없고 마감일을 날짜로 쓸 수 없는 p17(잡사이트B)을 더해 17, 5 / 17로 바꿨다
    assert y.loc[2025, "n_raw"] == 17 and y.loc[2025, "n_no_post_start"] == 5
    # 과제 5: 비율 열을 게시일을 정할 수 없는 공고의 비율로 바꿨다(p14, p17 / 17. p02, p07은 마감일로 정한다,
    # 생성형 AI 활용 숙련이 있는 분석 대상 세분류 공고 p02, p04, p05에는 게시일을 정할 수 없는 공고가 없다)
    assert y.loc[2025, "no_date_rate"] == pytest.approx(2 / 17)
    assert y.loc[2025, "in_occ_use_no_date_rate"] == 0
    s = m[m["level"] == "연도·수집 사이트"]
    # 게시일 없는 공고가 있는 사이트만, 건수가 많은 사이트부터. 과제 4에서 p17이 더해져 잡사이트B가 3건,
    # 잡사이트A가 2건이 되어 순서를 바꿨다(이름 순이었다면 A가 앞에 온다)
    assert s["source"].tolist() == ["잡사이트B", "잡사이트A"]
    w = t["t12b_sido_other_words"]
    assert list(w.columns) == ["word", "n"] and w.iloc[0].tolist() == ["광산구", 2]

    text = s00.write_report(t, _NO_TIMING, synth_data["cfg"], tmp_path / "report.md", commit="c").read_text(encoding="utf-8")
    sec = lambda i: text.index(f"## {i}. ")
    assert sec(1) < text.index("### `t01_exclusions`") < text.index("### `t01b_post_start_missing`") < sec(2)
    assert sec(11) < text.index("### `t12_sido_other`") < text.index("### `t12b_sido_other_words`")
    assert "#### `level` = 연도\n" in text and "#### `level` = 연도·수집 사이트" in text
    notes = s00._table_notes(synth_data["cfg"])
    assert "`post_start` 결측" in notes["t01_exclusions"] and "t01b_post_start_missing" in notes["t01_exclusions"]


def test_report_requires_the_new_interim_files(synth_data, tmp_path):
    s00 = load_script()
    s00.scan(synth_data["cfg"], synth_data["root"], tmp_path / "s0")
    for name in ("post_start_missing", "sido_other_words"):
        (tmp_path / "s0" / f"{name}.parquet").unlink()
    with pytest.raises(FileNotFoundError, match=r"post_start_missing\.parquet, sido_other_words\.parquet"):
        s00.build_tables(synth_data["cfg"], synth_data["root"], tmp_path / "s0")


def test_report_row_limit_per_table():
    s00 = load_script()
    df = pd.DataFrame({"skill_code": [f"X{i:02d}" for i in range(40)], "kind": "개발", "2025": range(40)})
    agent = "\n".join(s00._table_block("t06b_agent_skill_by_year", df, "설명"))
    assert all(f"| X{i:02d} |" in agent for i in range(40)) and "위 30행만" not in agent
    other = "\n".join(s00._table_block("t09a_by_ksic_sub", df.rename(columns={"skill_code": "ksic_sub"}), "설명"))
    assert "| X29 |" in other and "| X30 |" not in other and "위 30행만" in other


def test_source_rows_sorted_by_count_within_year(synth_data, tmp_path):
    _, t = run(synth_data, tmp_path)
    s = t["t05_source"]
    # 과제 3에서 잡사이트A의 공고 p13이 더해져 잡사이트A와 C가 4건으로 같아졌다. 건수가 같으면 이름 순이고,
    # 이름 순이었다면 3건인 잡사이트B가 C보다 앞에 온다. 과제 3A에서 잡사이트C의 p15와 잡사이트A의 p16이 더해져
    # 두 사이트가 5건으로 같다.
    s25 = s[s["year"] == 2025]
    assert s25["source"].tolist() == ["잡사이트A", "잡사이트C", "잡사이트B"] and s25["n"].tolist() == [5, 5, 3]
    csv = pd.read_csv(tmp_path / "s0" / "tables" / "t05_source.csv", encoding="utf-8-sig")
    assert csv["source"].tolist() == ["잡사이트A", "잡사이트C", "잡사이트B"]


def test_sparse_grid_uses_the_analysis_sample(synth_data, tmp_path):
    cfg = synth_data["cfg"]; cfg["skills"]["sparse_filter"]["min_treated"] = 1
    s00, d = load_script(), tmp_path / "s0"
    s00.scan(cfg, synth_data["root"], d)
    g = s00.build_tables(cfg, synth_data["root"], d)["t03b_sparse_filter_grid"]
    row = g[(g["min_rate"] == cfg["skills"]["sparse_filter"]["min_rate"]) & (g["min_treated"] == 1)]
    assert len(row) == 1
    codes = treatment.agent_skill_codes(skills.load_codebook(synth_data["codebook"]), cfg)
    long = skills.drop_codes(pd.read_parquet(d / "skills_long.parquet"), codes["dev"] | codes["use"])
    timing = s00.time_outcome_model(pd.read_parquet(d / "base.parquet"), pd.read_parquet(d / "treat.parquet"), long, cfg)
    # SM-01-0001과 synth.PLACEBO_CANDIDATE(둘 다 처치 공고 p02). 과제 5에서 p02에 PLACEBO_CANDIDATE를 더해 1에서 2로 바꿨다
    assert row["n_skills_kept"].iloc[0] == timing["n_skills_kept"] == 2
    assert row["n_postings"].iloc[0] == 10     # 네 단계를 모두 통과한 공고: p01 p02 p03 p05 p09 p11 p12 p13(과제 3) p15 p16(과제 3A)
    assert row["n_treated"].iloc[0] == 2       # p02, p05 (p04는 사업자번호가 없어 빠진다)


def test_report_wording(synth_data, tmp_path):
    s00, t = run(synth_data, tmp_path)
    text = s00.write_report(t, _NO_TIMING, synth_data["cfg"], tmp_path / "report.md", commit="c").read_text(encoding="utf-8")
    assert "## 1. 제외 규칙 단계별 공고 수 (연도별): 분석 기간, 분석 대상 직종 한정" in text
    assert "(기간 단위: 연 (`year`))" in text
    assert text.index("희소 필터는 출현율이") < text.index("### `t03b_sparse_filter_grid`")
    assert text.index("## 7. ") < text.index("결과모형은 공고의 특성으로") < text.index("## 8. ")
    notes = s00._table_notes(synth_data["cfg"])
    assert "8절" in notes["t02a_treated_by_period"] and "n_skills_kept" in "".join(s00._timing_block(_NO_TIMING, synth_data["cfg"]))


# 아래는 전체 검토 수정에서 더한 시험이다.

def test_postings_with_only_agent_skill_codes(synth_data, tmp_path):
    # 네 단계를 통과한 공고 p01 p02 p03 p05 p09 p11 p12 p13 가운데 p05는 숙련이 AI 에이전트 숙련코드(IT-05-0413,
    # IT-05-0410)뿐이어서, 54개 코드를 빼면 결과변수로 쓰는 숙련이 하나도 없다.
    # 과제 3: p13을 더해 전체 공고 수를 7에서 8로, 비율을 1 / 7에서 1 / 8로 바꿨다.
    # 과제 3A: p15, p16을 더해 전체 공고 수를 10으로, 비율을 1 / 10으로 바꿨다.
    s00, t = run(synth_data, tmp_path)
    e = t["t03e_no_outcome_skill"]
    assert list(e.columns) == ["group", "n", "n_no_outcome_skill", "no_outcome_skill_rate"]
    g = e.set_index("group")
    assert g.index.tolist() == ["전체", "has_use", "has_dev", "treat_code"]
    assert g["n"].tolist() == [10, 2, 2, 3] and g["n_no_outcome_skill"].tolist() == [1, 1, 1, 1]
    assert g.loc["전체", "no_outcome_skill_rate"] == pytest.approx(1 / 10)
    text = s00.write_report(t, _NO_TIMING, synth_data["cfg"], tmp_path / "report.md", commit="c").read_text(encoding="utf-8")
    assert text.index("### `t03d_match_stats`") < text.index("### `t03e_no_outcome_skill`") < text.index("## 4. ")
    assert "저자가 정한다" in s00._table_notes(synth_data["cfg"])["t03e_no_outcome_skill"]


def test_report_wording_after_final_review(synth_data, tmp_path):
    s00, t = run(synth_data, tmp_path)
    text = s00.write_report(t, _NO_TIMING, synth_data["cfg"], tmp_path / "report.md", commit="c").read_text(encoding="utf-8")
    assert "겹" not in text.replace("겹치", "").replace("겹침", "")
    sec7 = text[text.index("## 7. "):text.index("## 8. ")]
    assert "폴드" in sec7 and sec7.index("교차적합(") < sec7.index("폴드")
    assert "미도입 공고" not in sec7.split("\n", 1)[1].replace("제목의 `미도입 공고`", "")
    assert "#### `unit` = 연 (`year`)" in text and "#### `unit` = 분기 (`quarter`)" in text
    assert "#### `unit` = 반기 (`half`)" in text
    notes = s00._table_notes(synth_data["cfg"])
    assert "토큰" not in notes["t03d_match_stats"]
    assert "근무지 첫 낱말이 `전남광주`나 `전남광주통합특별시`인 공고" in notes["t12_sido_other"]
    for name in ("t02a_treated_by_period", "t02b_treated_by_occ"):
        assert "keyword_dictionary" in notes[name] and "treat_all" in notes[name]
    assert "n_masked_values" in notes["t04b_ked_lines"]


# 아래는 과제 5에서 더한 시험이다. 가상 자료로 1단계(scan)와 build_tables를 실행한 결과를 본다.

@pytest.fixture
def synth_scan(synth_data, tmp_path):
    """scan까지 끝낸 고정물: 스크립트 모듈, 설정, 가상 자료 폴더, 중간 자료 폴더."""
    s00 = load_script()
    s0_dir = tmp_path / "s0"
    s00.scan(synth_data["cfg"], synth_data["root"], s0_dir)
    return {"s00": s00, "cfg": synth_data["cfg"], "root": synth_data["root"], "s0_dir": s0_dir}


@pytest.fixture
def tables(synth_scan):
    return synth_scan["s00"].build_tables(synth_scan["cfg"], synth_scan["root"], synth_scan["s0_dir"])


def test_t01b_and_t01c(tables):
    t = tables["t01b_post_start_missing"]
    assert {"n_no_date", "n_in_occ_from_end_in_period", "no_date_rate"} <= set(t.columns)
    year = t[t["level"] == "연도"]
    # 브리프의 기대값 1은 과제 4에서 p17(마감일이 9999-12-31)이 더해지기 전의 값이어서 synth의 상수로 쓴다
    assert year["n_no_date"].sum() == synth.N_NO_DATE and year["n_in_occ_from_end_in_period"].sum() == 1
    assert year["n_end_unusable"].sum() == synth.N_END_UNUSABLE and year["n_in_occ_end_unusable"].sum() == 1
    assert year["no_date_rate"].iloc[0] == pytest.approx(synth.N_NO_DATE / year["n_raw"].iloc[0])
    gap = tables["t01c_post_end_gap"]
    assert gap.groupby("year")["share_in_year"].sum().round(6).eq(1).all()
    assert list(gap.columns) == ["year", "bin", "n", "share_in_year"]


def test_t02a_has_configured_periods(tables, synth_data):
    t = tables["t02a_treated_by_period"]
    names = [g["name"] for g in synth_data["cfg"]["period"]["groups"]]
    assert t.loc[t["unit"] == "period", "period"].tolist() == [n for n in names if n in set(t["period"])]
    assert "kw_use" in t.columns and synth.PERIOD_NAME in set(t["period"])


def test_t02c_marks_groups_below_the_minimum(tables, synth_data):
    t = tables["t02c_occ_group_treated"]
    assert (t["meets_min"] == (t["n_treated"] >= synth_data["cfg"]["occupations"]["group_min_treated"])).all()
    assert (t["n_treated"] + t["n_control"] <= t["n"]).all()
    assert list(t.columns) == ["keco2", "name", "n", "n_treated", "n_control", "meets_min"]
    assert set(t["keco2"]) == {c[:2] for c in synth.OCCUPATIONS}


def test_t09b_counts_codes_missing_from_the_table(tables):
    assert tables["t09b_ksic_not_in_table"]["n"].sum() >= 1


def test_t10b_lists_codes_in_excluded_groups_separately(synth_data, tmp_path):
    cfg = synth_data["cfg"]
    cfg["occupations"]["min_use_postings"] = 1
    # 1332(p05에 생성형 AI 활용 숙련)가 속한 중분류를 빼면 설정의 목록에도 1332를 둘 수 없다(validate가 막는다)
    cfg["occupations"]["keco4"] = ["0262"]
    cfg["occupations"]["exclude_keco2"] = ["13"]
    s00 = load_script(); s00.scan(cfg, synth_data["root"], tmp_path / "s0")
    chk = s00.build_tables(cfg, synth_data["root"], tmp_path / "s0")["t10b_occ_list_check"].set_index("keco4")
    assert chk.loc["1332", "status"] == "설정에서 뺀 중분류" and chk.loc["0262", "status"] == "양쪽에 있음"


def test_t13_lists_every_candidate_including_absent_ones(tables):
    t = tables["t13_adjacent_candidates"]
    assert set(t["skill_code"]) == {synth.ADJ_PRESENT, synth.ADJ_ABSENT}
    assert t.set_index("skill_code").loc[synth.ADJ_ABSENT, "n"] == 0
    assert not set(t["skill_code"]) & synth.AGENT_CODES            # AI 에이전트 숙련코드는 후보가 아니다
    assert list(t.columns) == ["skill_code", "skill_name", "detail", "n", "rate", "n_treated", "treated_rate",
                               "passes_sparse_filter"]
    assert t["skill_code"].iloc[0] == synth.ADJ_PRESENT              # 공고가 많은 순


def test_t14_has_no_treated_columns(tables):
    t = tables["t14_placebo_candidates"]
    assert list(t.columns) == ["skill_code", "skill_name", "domain", "sub", "n", "rate"]
    assert synth.PLACEBO_CANDIDATE in set(t["skill_code"])


@pytest.mark.parametrize("key", ["adjacent_candidate_details", "placebo_candidate_domains"])
def test_unknown_candidate_name_is_an_error(synth_scan, key):        # synth_scan: scan까지 끝낸 고정물
    s00 = synth_scan["s00"]
    cfg = copy.deepcopy(synth_scan["cfg"])
    cfg["skills"][key] = ["없는이름"]
    with pytest.raises(ValueError, match="없는이름"):
        s00.build_tables(cfg, synth_scan["root"], synth_scan["s0_dir"])


def test_t16_t17_t18(tables):
    t16 = tables["t16_detail_by_arm"].set_index("arm")
    assert {"처치 공고", "비교 공고", "같은 세분류·연도 안의 차이"} <= set(t16.index)
    assert t16.loc["처치 공고", "n"] + t16.loc["비교 공고", "n"] <= tables["t03b_sparse_filter_grid"]["n_postings"].iloc[0]
    t17 = tables["t17_firm_treated"].set_index("key")["값"]
    assert t17["n_treated_same_firm_year"] <= t17["n_treated"] and t17["n_treated_same_firm_year_occ"] <= t17["n_treated_same_firm_year"]
    assert not tables["t17_firm_treated"].astype(str).apply(lambda c: c.str.fullmatch(r"\d{10}")).any().any()   # 사업자번호가 없다
    t18 = tables["t18_first_treated_firms"]
    assert (t18["n_firms_with_earlier"] <= t18["n_firms"]).all()


def test_t16_t17_t18_values_on_synthetic_postings(tables):
    # 설정의 처치 정의는 use: 처치 공고 p02(기업 1000000001), p05(1000000003). 비교 공고는 네 단계를 통과한 공고
    # 가운데 엄격한 사전의 키워드, 생성형 AI 활용 표현, AI 에이전트 숙련이 모두 없는 p03 p12 p13 p16이다
    # (p01, p09는 키워드, p11, p15는 생성형 AI 활용 표현에 걸려 어느 쪽도 아니다)
    t16 = tables["t16_detail_by_arm"].set_index("arm")
    assert t16.loc["처치 공고", "n"] == 2 and t16.loc["비교 공고", "n"] == 4
    # AI 에이전트 숙련코드를 뺀 숙련 수: p02 2개(SM-01-0001, PLACEBO_CANDIDATE), p05 0개
    assert t16.loc["처치 공고", "n_skills_mean"] == pytest.approx(1.0)
    t17 = tables["t17_firm_treated"].set_index("key")["값"]
    assert t17["n_treated"] == 2 and t17["n_firms_treated"] == 2
    # p02와 같은 기업·연도에 비교 공고 p13이 있다. 세분류는 다르다(p02는 0262, p13은 1332)
    assert t17["n_treated_same_firm_year"] == 1 and t17["n_treated_same_firm_year_occ"] == 0
    t18 = tables["t18_first_treated_firms"]
    assert t18["n_firms"].sum() == 2 and t18["n_firms_with_earlier"].sum() == 0   # 가상 자료는 2025년 한 해뿐이다


def test_overlap_and_keyword_tables_show_the_use_dictionary(tables, synth_data):
    o = tables["t06a_overlap"]
    part = o[o["cross"] == "kw_use × has_use × has_dev"]
    assert len(part) == 8 and part["n"].sum() == tables["t05_source"]["n"].sum()   # 둘 다 base.parquet의 모든 공고
    assert part.set_index(["kw_use", "has_use", "has_dev"]).loc[("1", "0", "0"), "n"] >= 1   # p11, p15
    k = tables["t06c_keyword_pattern_hits"]
    use = k[k["dictionary"] == "use"]
    assert use["pattern"].tolist() == synth_data["cfg"]["treatment"]["keywords"]["use"]
    assert use.set_index("pattern").loc["Claude", "total"] == 1


def test_timing_features_include_source_and_text_length():
    s00 = load_script()
    assert {"source", "text_len_log", "quarter"} <= set(s00.TIMING_FEATURES)
    base = pd.DataFrame({c: ["a", "b"] for c in s00.TIMING_FEATURES if c != "text_len_log"})
    base["text_len"] = [0, 99]
    X = s00._timing_features(base, np.array([0, 1]))
    assert list(X.columns) == s00.TIMING_FEATURES
    assert X["text_len_log"].tolist() == pytest.approx([0.0, np.log1p(99)])


def test_report_new_sections_and_notes(tables, synth_data, tmp_path):
    s00 = load_script()
    cfg = synth_data["cfg"]
    text = s00.write_report(tables, _NO_TIMING, cfg, tmp_path / "report.md", commit="c").read_text(encoding="utf-8")
    assert "28개" not in text and "25개" not in text
    for name in ("t01c_post_end_gap", "t02c_occ_group_treated", "t13_adjacent_candidates", "t14_placebo_candidates",
                 "t16_detail_by_arm", "t17_firm_treated", "t18_first_treated_firms"):
        assert f"### `{name}`" in text
    assert "## 12. 인접 숙련 후보(6.7절 3번)와 placebo 숙련 후보의 숙련별 공고 수와 출현율" in text
    assert "## 14. 처치 여부별 공고당 숙련 수와 본문 글자 수" in text and "## 13." not in text
    assert "게시일과 마감일이 모두 있는 공고의 두 날짜 간격 분포" in text.split("\n## 2.")[0]
    assert text.index("## 12. ") < text.index("### `t13_adjacent_candidates`") < text.index("### `t14_placebo_candidates`")
    assert text.index("## 14. ") < text.index("### `t16_detail_by_arm`")
    assert "#### `unit` = 설정의 기간 (`period`)" in text
    assert "위약" not in text
    notes = s00._table_notes(cfg)
    for name in ("t02a_treated_by_period", "t02b_treated_by_occ"):
        assert "kw_use" in notes[name] and "use_keyword" in notes[name]
    assert "`Claude`" in notes["t06c_keyword_pattern_hits"] and "`Gemini`" in notes["t06c_keyword_pattern_hits"]
    assert "9999-12-31" in notes["t01b_post_start_missing"] and "n_in_occ_end_unusable" in notes["t01c_post_end_gap"]
    assert "placebo 숙련" in notes["t14_placebo_candidates"] and "추정 결과" in notes["t14_placebo_candidates"]
    assert "설정에서 뺀 중분류" in notes["t10b_occ_list_check"] and "data.years" in notes["t10b_occ_list_check"]
    assert "`광주·전남`" in notes["t12_sido_other"]
    assert "post_start`, 없으면 `post_end`" in s00._POST_YEAR
