import re

import numpy as np
import pandas as pd
import pytest
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
    assert c.loc["has_skill", "n_dropped"] == 2 and c.loc["has_skill", "n_after"] == 7

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
    k = t["t06c_keyword_pattern_hits"].set_index("pattern")
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
    })
    base["quarter"] = "2025Q" + ((base["month"] - 1) // 3 + 1).astype(str)
    has_use = np.arange(n) < n_use
    treat = pd.DataFrame({"posting_id": ids, "kw_strict": False, "kw_broad": False, "has_dev": False,
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
    assert timing["skill_code"] == "S1" and timing["n_features"] == 14 and timing["n_train"] > 0
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
    # 공고 번호(p10은 연봉 10% 분위수의 열 이름과 같아 뺀다)와 사업자번호가 보고서에 없다
    assert not re.search(r"\bp(0[1-9]|11|12)\b", text) and not re.search(r"\b1000000\d{3}\b", text)
    assert (tmp_path / "s0cli" / "tables" / "t01_exclusions.csv").exists()


def test_report_numbers_keep_small_rates_and_csv_keeps_full_precision(synth_data, tmp_path):
    s00, t = run(synth_data, tmp_path)
    f = lambda v: s00._fmt_cell(v, "has_use_rate")
    assert f(4e-05) == "0.00004" and f(0.00012) == "0.00012" and f(0.00049) == "0.00049"
    assert f(0.5) == "0.5" and f(0.0) == "0" and f(1.625) == "1.625" and f(45_000_000.0) == "45,000,000"
    assert all("e" not in f(v).lower() for v in (4e-05, 1.234e-07, 0.00049))
    # CSV에는 보고서용으로 줄인 값이 아니라 계산한 값 그대로가 들어간다
    csv = pd.read_csv(tmp_path / "s0" / "tables" / "t04_bzno_ked_by_treatment.csv", encoding="utf-8-sig")
    assert csv.loc[csv["has_use"] == False, "ked_matched_rate"].iloc[0] == pytest.approx(5 / 7, rel=1e-12)


# 아래는 과제 10 수정(첫 실제 자료 실행에서 드러난 문제)에서 더한 시험이다.

_NO_TIMING = {"skill_code": None, "n_train": 0, "n_features": 14, "seconds": None, "n_skills_kept": 0,
              "est_total_hours": None}


def test_post_start_missing_and_other_region_words_in_report(synth_data, tmp_path):
    rewrite_postings(synth_data)   # p02, p07은 게시일 없음. p05, p09, p11, p12는 근무지가 `기타`
    s00, t = run(synth_data, tmp_path)
    m = t["t01b_post_start_missing"]
    y = m[m["level"] == "연도"].set_index("year")
    assert len(y) == 1 and pd.isna(y.loc[2025, "source"])
    assert y.loc[2025, "n_raw"] == 12 and y.loc[2025, "n_no_post_start"] == 2
    assert y.loc[2025, "no_post_start_rate"] == pytest.approx(2 / 12)
    assert y.loc[2025, "in_occ_use_no_post_start_rate"] == pytest.approx(1 / 3)
    s = m[m["level"] == "연도·수집 사이트"]
    assert s["source"].tolist() == ["잡사이트A", "잡사이트B"]   # 게시일 없는 공고가 있는 사이트만, 건수가 같으면 이름 순
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
    first = s[s["year"] == 2025].iloc[0]
    assert first["source"] == "잡사이트C" and first["n"] == s.loc[s["year"] == 2025, "n"].max()   # 잡사이트C 4건
    csv = pd.read_csv(tmp_path / "s0" / "tables" / "t05_source.csv", encoding="utf-8-sig")
    assert csv["source"].iloc[0] == "잡사이트C"


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
    assert row["n_skills_kept"].iloc[0] == timing["n_skills_kept"] == 1   # SM-01-0001 (처치 공고 p02)
    assert row["n_postings"].iloc[0] == 7      # 네 단계를 모두 통과한 공고: p01 p02 p03 p05 p09 p11 p12
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
    # 네 단계를 통과한 공고 p01 p02 p03 p05 p09 p11 p12 가운데 p05는 숙련이 AI 에이전트 숙련코드(IT-05-0413,
    # IT-05-0410)뿐이어서, 54개 코드를 빼면 결과변수로 쓰는 숙련이 하나도 없다.
    s00, t = run(synth_data, tmp_path)
    e = t["t03e_no_outcome_skill"]
    assert list(e.columns) == ["group", "n", "n_no_outcome_skill", "no_outcome_skill_rate"]
    g = e.set_index("group")
    assert g.index.tolist() == ["전체", "has_use", "has_dev", "treat_code"]
    assert g["n"].tolist() == [7, 2, 2, 3] and g["n_no_outcome_skill"].tolist() == [1, 1, 1, 1]
    assert g.loc["전체", "no_outcome_skill_rate"] == pytest.approx(1 / 7)
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
