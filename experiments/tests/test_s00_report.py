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
    assert c.loc["has_skill", "n_dropped"] == 2 and c.loc["has_skill", "n_after"] == 12   # 과제 3: p13을 더해 7에서 8로, 과제 3A: p15, p16을 더해 10으로 바꿨다. infra-2b 과제 1: 날짜를 받은 p14, p17이 남아 12로 바꿨다

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
    # 과제 6: 키워드 사전을 2026-11-13에 확정해(결정 32, 36) 보고서 머리의 `임시값` 문장을 바꿨다
    assert "testcommit" in text and "키워드 사전은 2026-11-13에 확정했다" in text and "임시" not in text
    assert "에이전트 개발" not in text


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
    # 결합되지 않는 p16(1000000005는 KED 가상 파일에서 버려진 행)을 더해 7 / 10으로 바꿨다.
    # infra-2b 과제 1: 날짜를 받아 base에 남는 p14(1000000002), p17(1000000006)이 모두 KED와 결합되어 9 / 12로 바꿨다.
    # infra-2b 과제 1 수정 1차: 행을 has_use에서 설정의 처치 정의(use_any)로 나눈 공고로 바꿨다. 비교 공고 8건(p03 p06
    # p10 p12 p13 p14 p16 p17) 모두 사업자번호가 있고, p10, p16(1000000005)만 KED와 결합되지 않아 6 / 8이다
    assert csv.loc[csv["arm"] == "비교 공고", "ked_matched_rate"].iloc[0] == pytest.approx(6 / 8, rel=1e-12)


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
    # 최종 검토 수정: "`post_start` 결측이면"을 "게시일(`post_start`)을 날짜로 쓸 수 없으면"으로 바꿨다
    assert "게시일(`post_start`)을 날짜로 쓸 수 없으면" in notes["t01_exclusions"]
    assert "t01b_post_start_missing" in notes["t01_exclusions"]


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
    # infra-2b 과제 1: 날짜를 채우는 설정이 기본값이 되면 잡사이트B의 p14, p17이 base에 남아 세 사이트가 모두 5건이
    # 되므로, 건수로 정렬하는지 보는 이 시험은 날짜를 채우지 않는 설정으로 실행한다(기대값은 그대로)
    synth_data["cfg"]["period"]["fill"]["enabled"] = False
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
    assert cfg["treatment"]["definition"] == "use_any"   # infra-2b 과제 1 수정 1차: 설정의 처치 정의 그대로 센다
    s00, d = load_script(), tmp_path / "s0"
    s00.scan(cfg, synth_data["root"], d)
    g = s00.build_tables(cfg, synth_data["root"], d)["t03b_sparse_filter_grid"]
    row = g[(g["min_rate"] == cfg["skills"]["sparse_filter"]["min_rate"]) & (g["min_treated"] == 1)]
    assert len(row) == 1
    codes = treatment.agent_skill_codes(skills.load_codebook(synth_data["codebook"]), cfg)
    long = skills.drop_codes(pd.read_parquet(d / "skills_long.parquet"), codes["dev"] | codes["use"])
    timing = s00.time_outcome_model(pd.read_parquet(d / "base.parquet"), pd.read_parquet(d / "treat.parquet"), long, cfg)
    # infra-2b 과제 1 수정 1차: 희소 필터는 설정의 처치 정의(use_any)의 처치 공고와 비교 공고만 대상으로 한다.
    # 네 단계를 모두 통과한 공고 12건 가운데 처치 공고 p02 p05(생성형 AI 활용 숙련), p11 p15(키워드) 4건과
    # 비교 공고 p03 p12 p13 p14 p16 p17 6건. p01, p09는 엄격한 사전의 키워드에 걸려 어느 쪽도 아니어서 뺀다
    # (바꾸기 전: 처치 정의 use, 대상 공고 12건, 처치 공고 p02 p05 2건, 남는 숙련 2개)
    assert row["n_postings"].iloc[0] == 10 and row["n_treated"].iloc[0] == 4
    # 처치 공고에 한 번 이상 나온 숙련: SM-01-0001(p02, p15), PLACEBO_CANDIDATE(p02), IT-01-0001과 학사 학위(p11).
    # p05의 숙련은 모두 AI 에이전트 숙련코드여서 세지 않는다
    assert row["n_skills_kept"].iloc[0] == timing["n_skills_kept"] == 4


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
    # infra-2b 과제 1: 날짜를 받은 p14, p17이 남아 전체 공고 수를 12로, 비율을 1 / 12로 바꿨다.
    # infra-2b 과제 1 수정 1차: 행을 has_use, has_dev, treat_code에서 설정의 처치 정의(use_any)의 처치 공고(p02 p05
    # p11 p15)와 비교 공고(p03 p12 p13 p14 p16 p17)로 바꿨다(바꾸기 전: n [12, 2, 2, 3], n_no_outcome_skill [1, 1, 1, 1])
    s00, t = run(synth_data, tmp_path)
    e = t["t03e_no_outcome_skill"]
    assert list(e.columns) == ["group", "n", "n_no_outcome_skill", "no_outcome_skill_rate"]
    g = e.set_index("group")
    assert g.index.tolist() == ["전체", "처치 공고", "비교 공고"]
    assert g["n"].tolist() == [12, 4, 6] and g["n_no_outcome_skill"].tolist() == [1, 1, 0]
    assert g["no_outcome_skill_rate"].tolist() == pytest.approx([1 / 12, 1 / 4, 0.0])
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
    # infra-2b 과제 1 수정 1차: 설정의 처치 정의(use_any)의 처치 공고 4건과 비교 공고 6건이 대상이다.
    # ADJ_PRESENT는 비교 공고 p03에만 있다
    a = t.set_index("skill_code").loc[synth.ADJ_PRESENT]
    assert (a["n"], a["n_treated"], a["passes_sparse_filter"]) == (1, 0, False)
    assert a["rate"] == pytest.approx(1 / 10) and a["treated_rate"] == 0


def test_t14_counts_on_the_treated_and_comparison_postings(tables):
    # infra-2b 과제 1 수정 1차. PLACEBO_CANDIDATE는 use_any의 처치 공고 p02에 있어 희소 필터(min_treated 1)를 통과한다.
    # 출현율의 분모는 처치 공고와 비교 공고의 수(10)이다
    p = tables["t14_placebo_candidates"].set_index("skill_code").loc[synth.PLACEBO_CANDIDATE]
    assert p["n"] == 1 and p["rate"] == pytest.approx(1 / 10)


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


def test_t16_t17_t18_values_on_synthetic_postings(synth_scan):
    # infra-2b 과제 1: 설정의 처치 정의가 use_any로 바뀌어, 표의 계산을 보는 이 시험은 처치 정의를 use로 두고
    # build_tables를 실행한다(처치 정의는 2단계에서만 쓴다)
    cfg = copy.deepcopy(synth_scan["cfg"])
    cfg["treatment"]["definition"] = "use"
    tables = synth_scan["s00"].build_tables(cfg, synth_scan["root"], synth_scan["s0_dir"])
    # 처치 정의 use: 처치 공고 p02(기업 1000000001), p05(1000000003). 비교 공고는 네 단계를 통과한 공고
    # 가운데 엄격한 사전의 키워드, 생성형 AI 활용 표현, AI 에이전트 숙련이 모두 없는 p03 p12 p13 p16이다
    # (p01, p09는 키워드, p11, p15는 생성형 AI 활용 표현에 걸려 어느 쪽도 아니다).
    # infra-2b 과제 1: 날짜를 받아 남는 p14, p17도 비교 공고여서 비교 공고 수를 4에서 6으로 바꿨다
    t16 = tables["t16_detail_by_arm"].set_index("arm")
    assert t16.loc["처치 공고", "n"] == 2 and t16.loc["비교 공고", "n"] == 6
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
    # 과제 6에서 t15_aiie_coverage를 더했다
    for name in ("t01c_post_end_gap", "t02c_occ_group_treated", "t13_adjacent_candidates", "t14_placebo_candidates",
                 "t15_aiie_coverage", "t16_detail_by_arm", "t17_firm_treated", "t18_first_treated_firms"):
        assert f"### `{name}`" in text
    assert "## 12. 인접 숙련 후보(6.7절 3번)와 placebo 숙련 후보의 숙련별 공고 수와 출현율" in text
    # 과제 6: 13절이 생겨 `"## 13." not in text`를 13절의 제목과 위치로 바꿨다
    assert "## 14. 처치 여부별 공고당 숙련 수와 본문 글자 수" in text
    assert "## 13. 산업별 AI 노출 지수가 붙은 공고의 비율과 값의 분포" in text
    assert text.index("## 12. ") < text.index("## 13. ") < text.index("### `t15_aiie_coverage`") < text.index("## 14. ")
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


# 아래는 과제 6에서 더한 시험이다(산업 소분류별 AI 노출 지수의 결합 비율).

def test_t15_aiie_coverage(tables, synth_scan, synth_data, tmp_path):
    t = tables["t15_aiie_coverage"]
    # infra-2b 과제 1 수정 1차: 행을 has_use(1, 0)에서 설정의 처치 정의(use_any)로 나눈 처치 공고와 비교 공고로 바꿨다
    assert list(t.columns) == ["arm", "n", "n_with_aiie", "with_aiie_rate", "p10", "p25", "p50", "p75", "p90"]
    assert t["arm"].tolist() == ["전체", "처치 공고", "비교 공고"]
    assert (t["n_with_aiie"] <= t["n"]).all()
    g = t.set_index("arm")
    # 전체 행은 제외 규칙 네 단계를 적용한 공고 12건, 처치 공고 4건(p02 p05 p11 p15), 비교 공고 6건이다.
    # 처치 공고와 비교 공고를 더하면 t03b의 n_postings이다(어느 쪽도 아닌 p01, p09는 두 행에 없다)
    assert g["n"].tolist() == [12, 4, 6]
    assert g.loc["처치 공고", "n"] + g.loc["비교 공고", "n"] == tables["t03b_sparse_filter_grid"]["n_postings"].iloc[0]
    # 소분류 620(1.5): p01, p13 / 862(0.5): p02, p14. 처치 공고 가운데 값이 붙는 공고는 p02뿐이고, 비교 공고 가운데는
    # p13, p14이다
    assert g["n_with_aiie"].tolist() == [4, 1, 2]
    # 산업코드가 없는 가상 공고(p05, p11)는 값이 붙지 않은 쪽에 센다
    s00 = synth_scan["s00"]
    base = pd.read_parquet(synth_scan["s0_dir"] / "base.parquet")
    alive, _ = s00._exclusions(base, synth_scan["cfg"])
    sub = base["ksic_sub"][alive]
    assert sub.isna().sum() >= 1
    assert g.loc["전체", "n_with_aiie"] == sub.isin(synth.SUBS_WITH_AIIE).sum()
    assert g.loc["전체", "n"] - g.loc["전체", "n_with_aiie"] >= sub.isna().sum()
    # 값이 붙은 소분류 수와 전체 소분류 수(가상 산업분류표의 620, 862, 011, 012, 981 가운데 620, 862)를 설명에 적는다.
    # 최종 검토 수정: 가상 산업분류표에 011, 981을 더해 "소분류 3개"에서 "소분류 5개"로 바꿨다
    text = s00.write_report(tables, _NO_TIMING, synth_data["cfg"], tmp_path / "report.md", commit="c").read_text(
        encoding="utf-8")
    sec13 = text[text.index("## 13. "):text.index("## 14. ")]
    assert "소분류 5개 가운데 AIIE 값이 붙은 소분류는 2개" in sec13
    assert "한국표준산업분류(KSIC)" in sec13 and "국제표준산업분류(ISIC Rev.4)" in sec13 and "미국 산업분류(NAICS)" in sec13


@pytest.mark.parametrize("how", ["no_key", "no_file"])
def test_t15_is_skipped_without_external_files(synth_scan, tmp_path, how):
    s00, root = synth_scan["s00"], synth_scan["root"]
    cfg = copy.deepcopy(synth_scan["cfg"])
    if how == "no_key":
        del cfg["data"]["aiie"]
        shown = "`data.aiie`"
    else:
        (root / cfg["data"]["naics_isic"]).unlink()
        shown = cfg["data"]["naics_isic"]
    t = s00.build_tables(cfg, root, synth_scan["s0_dir"])
    assert "t16_detail_by_arm" in t and not t["t01_exclusions"].empty      # 다른 표는 그대로 만든다
    assert not (synth_scan["s0_dir"] / "tables" / "t15_aiie_coverage.csv").exists()
    text = s00.write_report(t, _NO_TIMING, cfg, tmp_path / "report.md", commit="c").read_text(encoding="utf-8")
    sec13 = text[text.index("## 13. "):text.index("## 14. ")]
    assert "외부 자료가 없어 만들지 않았다" in sec13 and shown in sec13
    assert "### `t15_aiie_coverage`" not in text


# 아래는 최종 검토 수정에서 더한 시험이다.

def test_kw_use_note_names_every_definition_that_treats_it(synth_data):
    s00 = load_script()
    assert "`use_keyword`나 `use_any`일 때 처치 공고" in s00._KW_USE_NOTE
    assert "어떤 정의에서도 비교 공고는 아니다" in s00._KW_USE_NOTE and "에서만" not in s00._KW_USE_NOTE
    notes = s00._table_notes(synth_data["cfg"])
    assert s00._KW_USE_NOTE in notes["t02a_treated_by_period"] and s00._KW_USE_NOTE in notes["t02b_treated_by_occ"]


# 손으로 만든 공고 9건: (사업자번호, 연도, 세분류, 처치 정의로 나눈 값, AI 에이전트 숙련코드를 뺀 숙련 수, 본문 글자 수).
# 처치 정의로 나눈 값은 1이 처치 공고, 0이 비교 공고, -1이 어느 쪽도 아닌 공고(키워드에만 걸린 공고 등)이다.
# 사업자번호는 가상 값이다.
_FIRM_A, _FIRM_B = "1000000001", "1000000002"
_HAND_ROWS = [
    (_FIRM_A, 2024, "1332", 0, 1, 100),   # 0: A가 처치 공고를 처음 낸 해(2025) 전의 비교 공고
    (_FIRM_A, 2024, "0262", 0, 2, 200),   # 1: 위와 같다
    (_FIRM_A, 2025, "1332", 1, 4, 400),   # 2: A의 첫 처치 공고
    (_FIRM_A, 2025, "1332", 0, 2, 100),   # 3: 2와 같은 기업·연도·세분류의 비교 공고
    (_FIRM_B, 2024, "0262", 1, 3, 300),   # 4: B의 첫 처치 공고. 같은 해에 B의 비교 공고는 없다
    (_FIRM_B, 2025, "0262", 1, 5, 500),   # 5: B의 처치 공고
    (_FIRM_B, 2025, "0262", -1, 9, 900),  # 6: 5와 같은 기업·연도·세분류이지만 비교 공고가 아니다
    (_FIRM_B, 2025, "1332", 0, 1, 50),    # 7: 5와 같은 기업·연도, 다른 세분류의 비교 공고
    (None, 2025, "1332", 1, 6, 600),      # 8: 사업자번호가 없는 처치 공고(기업별 집계에서 빠진다)
]


def _hand_built():
    rows = pd.DataFrame(_HAND_ROWS, columns=["bzno", "year", "keco4", "arm", "n_skills", "text_len"])
    base = rows[["bzno", "year", "keco4", "text_len"]].copy()
    outcome = pd.DataFrame({"pos": np.repeat(np.arange(len(rows)), rows["n_skills"].to_numpy()), "skill_code": "S1"})
    return base, outcome, rows["arm"].to_numpy(dtype="int64")


def test_t16_on_hand_built_postings():
    s00 = load_script()
    base, outcome, arm = _hand_built()
    t = s00._t16_detail_by_arm(base, outcome, arm).set_index("arm")
    # 처치 공고 2, 4, 5, 8: 숙련 수 4, 3, 5, 6 / 글자 수 400, 300, 500, 600
    assert t.loc["처치 공고", "n"] == 4 and t.loc["처치 공고", "n_skills_mean"] == pytest.approx(4.5)
    assert t.loc["처치 공고", ["n_skills_p50", "n_skills_p75", "n_skills_p90"]].tolist() == pytest.approx([4.5, 5.25, 5.7])
    assert t.loc["처치 공고", ["text_len_mean", "text_len_p50"]].tolist() == pytest.approx([450.0, 450.0])
    # 비교 공고 0, 1, 3, 7: 숙련 수 1, 2, 2, 1 / 글자 수 100, 200, 100, 50
    assert t.loc["비교 공고", "n"] == 4 and t.loc["비교 공고", "n_skills_mean"] == pytest.approx(1.5)
    assert t.loc["비교 공고", ["text_len_mean", "text_len_p50"]].tolist() == pytest.approx([112.5, 100.0])
    # 두 집단이 모두 있는 칸: (0262, 2024) 처치 1건(3, 300) − 비교 1건(2, 200) = (1, 100), 가중치 1.
    # (1332, 2025) 처치 2건 평균(5, 500) − 비교 2건 평균(1.5, 75) = (3.5, 425), 가중치 2.
    # (1332, 2024)는 비교 공고만, (0262, 2025)는 처치 공고만 있다(6은 비교 공고가 아니다).
    w = t.loc["같은 세분류·연도 안의 차이"]
    assert w["n"] == 3
    assert w["n_skills_mean"] == pytest.approx((1 * 1 + 3.5 * 2) / 3)
    assert w["text_len_mean"] == pytest.approx((100 * 1 + 425 * 2) / 3)


def test_t17_on_hand_built_postings():
    s00 = load_script()
    base, _, arm = _hand_built()
    v = s00._t17_firm_treated(base, arm).set_index("key")["값"]
    # 사업자번호가 있는 처치 공고 2, 4, 5: A 1건, B 2건
    assert v["n_treated"] == 3 and v["n_firms_treated"] == 2
    assert v["treated_per_firm_p50"] == pytest.approx(1.5) and v["treated_per_firm_max"] == 2
    assert v["top_firms_share"] == pytest.approx(1.0)
    # 같은 기업·연도에 비교 공고가 있는 처치 공고: 2(3이 있다), 5(7이 있다). 4는 없다
    assert v["n_treated_same_firm_year"] == 2 and v["same_firm_year_rate"] == pytest.approx(2 / 3)
    assert v["n_firms_same_firm_year"] == 2
    # 같은 세분류까지 같은 비교 공고: 2만(3). 5는 같은 칸에 비교 공고가 아닌 공고 6만 있어 세지 않는다.
    # 비교 공고 대신 처치 공고가 아닌 모든 공고와 비교했다면 2가 된다
    assert v["n_treated_same_firm_year_occ"] == 1 and v["same_firm_year_occ_rate"] == pytest.approx(1 / 3)


def test_t18_on_hand_built_postings():
    s00 = load_script()
    base, _, arm = _hand_built()
    t = s00._t18_first_treated_firms(base, arm)
    assert list(t.columns) == ["first_year", "n_firms", "n_firms_with_earlier", "n_earlier_postings"]
    # B는 2024년에 처음 처치 공고를 냈고 그 전 비교 공고가 없다. A는 2025년에 처음 냈고 2024년 비교 공고가 2건(0, 1)이다.
    # A의 2025년 비교 공고 3은 그 전 연도가 아니므로 세지 않는다
    assert t.to_dict("list") == {"first_year": [2024, 2025], "n_firms": [1, 1], "n_firms_with_earlier": [0, 1],
                                 "n_earlier_postings": [0, 2]}


def test_t02d_date_source_by_period(tables, synth_data, tmp_path):
    t = tables["t02d_date_source_by_period"]
    assert list(t.columns) == ["period", "date_source", "n", "n_treated", "share_in_period", "treated_share_in_period"]
    # infra-2b 과제 1: 설정이 날짜를 채우고(결정 38) 처치 정의가 use_any(결정 40)여서 기대값을 바꿨다.
    # 네 단계를 통과한 공고 12건 가운데 ID_END_ONLY(p13)는 마감일로 게시일을 정했고, ID_NO_DATE(p14)와
    # ID_END_UNUSABLE(p17)은 앞 공고의 날짜를 받았다. use_any의 처치 공고 p02, p05(생성형 AI 활용 숙련),
    # p11, p15(생성형 AI 활용 키워드)는 모두 게시일이 있다
    # (바꾸기 전: 두 행, n [9, 1], n_treated [2, 0], share [0.9, 0.1], treated share [1.0, 0.0])
    assert t["period"].tolist() == [synth.PERIOD_NAME] * 3
    assert t["date_source"].tolist() == ["post_start", "post_end", "fill"]
    assert t["n"].tolist() == [9, 1, 2] and t["n_treated"].tolist() == [4, 0, 0]
    assert t["share_in_period"].tolist() == pytest.approx([9 / 12, 1 / 12, 2 / 12])
    assert t["treated_share_in_period"].tolist() == pytest.approx([1.0, 0.0, 0.0])
    s00 = load_script()
    text = s00.write_report(tables, _NO_TIMING, synth_data["cfg"], tmp_path / "report.md", commit="c").read_text(
        encoding="utf-8")
    assert text.index("### `t02c_occ_group_treated`") < text.index("### `t02d_date_source_by_period`") < text.index("## 3. ")
    note = s00._table_notes(synth_data["cfg"])["t02d_date_source_by_period"]
    assert "마감일은 실제 게시일보다 늦으므로" in note and "기간 구분이 덜 정확하다" in note


def test_t01b_note_and_all_rows_in_report(synth_data):
    s00 = load_script()
    note = s00._table_notes(synth_data["cfg"])["t01b_post_start_missing"]
    assert "`n_in_occ_no_post_start − n_in_occ_no_date − n_in_occ_from_end_in_period`" in note
    assert s00.REPORT_MAX_ROWS_BY_TABLE["t01b_post_start_missing"] is None
    sites = pd.DataFrame({"level": "연도·수집 사이트", "year": 2023, "source": [f"사이트{i:02d}" for i in range(40)],
                          "n_no_post_start": 1})
    block = "\n".join(s00._table_block("t01b_post_start_missing", sites, "설명"))
    assert all(f"| 사이트{i:02d} |" in block for i in range(40)) and "위 30행만" not in block


def test_report_words_after_final_review(tables, synth_data, tmp_path):
    s00 = load_script()
    text = s00.write_report(tables, _NO_TIMING, synth_data["cfg"], tmp_path / "report.md", commit="c").read_text(
        encoding="utf-8")
    assert "글자열" not in text and "문자열" in s00._table_notes(synth_data["cfg"])["t16_detail_by_arm"]
    assert "`post_start` 결측" not in text


@pytest.mark.parametrize("change", ["candidates", "keco"])
def test_all_checks_report_inputs_before_the_scan(synth_data, tmp_path, monkeypatch, change):
    s00 = load_script()
    cfg = copy.deepcopy(synth_data["cfg"])
    if change == "candidates":
        cfg["skills"]["adjacent_candidate_details"] = ["없는세부분류"]
        cfg["skills"]["placebo_candidate_domains"] = ["없는대분류"]
        expected, message = ValueError, None
    else:
        (synth_data["root"] / cfg["data"]["keco"]).unlink()
        expected, message = FileNotFoundError, "KECO2025.xlsx"
    monkeypatch.setattr(s00.paths, "ROOT", synth_data["root"])
    monkeypatch.setattr(s00.paths, "S0", tmp_path / "s0cli")
    monkeypatch.setattr(s00.paths, "load_config", lambda: cfg)
    with pytest.raises(expected, match=message) as e:
        s00.main(["all"])
    if change == "candidates":   # 두 설정 항목의 없는 이름을 한 번에 적는다
        assert "없는세부분류" in str(e.value) and "없는대분류" in str(e.value)
    s0 = tmp_path / "s0cli"
    assert not s0.exists() or not any(s0.iterdir())      # 1단계(scan)가 중간 자료를 쓰기 전에 멈춘다


def test_all_still_skips_t15_without_external_files(synth_data, tmp_path, monkeypatch):
    s00 = load_script()
    (synth_data["root"] / synth_data["cfg"]["data"]["aiie"]).unlink()
    monkeypatch.setattr(s00.paths, "ROOT", synth_data["root"])
    monkeypatch.setattr(s00.paths, "S0", tmp_path / "s0cli")
    monkeypatch.setattr(s00.paths, "load_config", lambda: synth_data["cfg"])
    assert s00.main(["all"]) == 0
    assert "외부 자료가 없어 만들지 않았다" in (synth_data["root"] / "docs" / "s0-data-audit.md").read_text(encoding="utf-8")


def test_crosswalk_and_ksic_table_must_list_the_same_sub_classes(synth_scan):
    s00, root = synth_scan["s00"], synth_scan["root"]
    path = root / synth_scan["cfg"]["data"]["ksic_isic"]
    lines = path.read_text(encoding="utf-8").splitlines()
    # 산업분류표에 있는 862를 빼고, 산업분류표에 없는 999를 더한다
    lines = [ln for ln in lines if not ln.startswith("862,")] + ["999,가상 소분류,620,"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="862") as e:
        s00.build_tables(synth_scan["cfg"], root, synth_scan["s0_dir"])
    assert "999" in str(e.value)


# 아래는 infra-2b 과제 1(결정 38~41)에서 더한 시험이다. 수치는 가상 자료의 것이다.

@pytest.fixture
def use_any_tables(synth_data, tmp_path):
    """설정 그대로(처치 정의 use_any, 날짜를 채움) 두 공고의 본문만 바꿔 scan과 build_tables를 실행한다.

    p01(개발 숙련 IT-05-0398)의 본문을 `ChatGPT 연동 개발`로 바꿔 엄격한 사전에는 걸리지 않고 생성형 AI 활용 키워드
    사전에만 걸리게 한다(개발 숙련과 키워드가 함께 있는 공고). p17(두 날짜가 모두 없어 p16의 2025-09-01을 받는 공고)의
    본문을 `생성형 AI 활용 보고서`로 바꿔 날짜를 받은 처치 공고를 만든다.
    """
    assert synth_data["cfg"]["treatment"]["definition"] == "use_any" and synth_data["cfg"]["period"]["fill"]["enabled"]
    rewrite_postings(synth_data, post_start={}, region={},
                     detail={"p01": "ChatGPT 연동 개발", synth.ID_END_UNUSABLE: "생성형 AI 활용 보고서"})
    return run(synth_data, tmp_path)[1]


def test_treated_and_control_under_use_any(use_any_tables):
    # 네 단계를 통과한 공고 12건: p01 p02 p03 p05 p09 p11 p12 p13 p14 p15 p16 p17.
    # use_any의 처치 공고: p01(개발 숙련과 키워드), p02 p05(생성형 AI 활용 숙련), p11 p15 p17(키워드만) = 6건.
    # 비교 공고: p03 p12 p13 p14 p16 = 5건. p09는 엄격한 사전의 키워드(`multi-agent`)에 걸려 어느 쪽도 아니다.
    a = use_any_tables["t02a_treated_by_period"]
    assert list(a.columns[:5]) == ["unit", "period", "n", "n_treated", "n_control"]
    q = a[a["unit"] == "quarter"].set_index("period")
    assert q.index.tolist() == ["2025Q1", "2025Q2", "2025Q3", "2025Q4"]
    assert q["n"].tolist() == [2, 1, 4, 5]                   # Q3: p05 p15 p16 p17(받은 날짜 09-01), Q4: p09 p11 p12 p13 p14
    assert q["n_treated"].tolist() == [2, 0, 3, 1] and q["n_control"].tolist() == [0, 1, 1, 3]
    assert q["kw_use"].tolist() == [1, 0, 2, 1] and q["has_use"].tolist() == [1, 0, 1, 0]
    assert q["has_dev"].tolist() == [1, 0, 1, 0] and q["kw_strict"].tolist() == [0, 0, 0, 1]
    p = a[a["unit"] == "period"].set_index("period").loc[synth.PERIOD_NAME]
    assert (p["n"], p["n_treated"], p["n_control"], p["kw_use"], p["has_use"]) == (12, 6, 5, 4, 2)

    c = use_any_tables["t02c_occ_group_treated"].set_index("keco2")
    assert c.index.tolist() == ["02", "13"]
    assert c["name"].tolist() == ["경영·행정·사무직", "정보통신 연구개발직 및 공학기술직"]
    assert c["n"].tolist() == [5, 7]                         # 02: p02 p03 p09 p14 p15, 13: p01 p05 p11 p12 p13 p16 p17
    assert c["n_treated"].tolist() == [2, 4] and c["n_control"].tolist() == [2, 3]
    assert not c["meets_min"].any()

    d = use_any_tables["t02d_date_source_by_period"]
    assert d["date_source"].tolist() == ["post_start", "post_end", "fill"]
    assert d["n"].tolist() == [9, 1, 2] and d["n_treated"].tolist() == [5, 0, 1]   # 날짜를 받은 처치 공고 p17
    assert d["share_in_period"].tolist() == pytest.approx([9 / 12, 1 / 12, 2 / 12])
    assert d["treated_share_in_period"].tolist() == pytest.approx([5 / 6, 0.0, 1 / 6])

    e = use_any_tables["t02e_fill_gap_by_period"]
    assert list(e.columns) == ["period", "gap_bin", "n", "n_treated", "share_in_period"]
    assert e["period"].tolist() == [synth.PERIOD_NAME] * 6
    assert e["gap_bin"].tolist() == ["7일 이내", "8~30일", "31~90일", "91~365일", "365일 초과", "뒤 날짜 없음"]
    # p14: p13의 2025-12-20을 받고 뒤의 p15(2025-08-01)까지 141일. p17: 뒤에 날짜 있는 공고가 없다
    assert e["n"].tolist() == [0, 0, 0, 1, 0, 1] and e["n_treated"].tolist() == [0, 0, 0, 0, 0, 1]
    assert e["share_in_period"].tolist() == pytest.approx([0, 0, 0, 1 / 12, 0, 1 / 12])


def test_use_definition_leaves_keyword_postings_out_of_both_groups(synth_data, tmp_path):
    # 같은 공고를 처치 정의 use로 나누면 p01(개발 숙련), p11 p15 p17(키워드)은 처치 공고도 비교 공고도 아니다
    rewrite_postings(synth_data, post_start={}, region={},
                     detail={"p01": "ChatGPT 연동 개발", synth.ID_END_UNUSABLE: "생성형 AI 활용 보고서"})
    synth_data["cfg"]["treatment"]["definition"] = "use"
    t = run(synth_data, tmp_path)[1]
    p = t["t02a_treated_by_period"].query("unit == 'period'").iloc[0]
    assert (p["n"], p["n_treated"], p["n_control"]) == (12, 2, 5)
    assert t["t02d_date_source_by_period"]["n_treated"].tolist() == [2, 0, 0]


def test_t01d_date_fill_on_synthetic_postings(tables):
    t = tables["t01d_date_fill"]
    assert list(t.columns) == ["year", "source", "gap_bin", "n", "n_in_occ", "n_in_occ_in_period"]
    # p14(잡사이트B, 141일), p17(잡사이트B, 뒤 날짜 없음)
    assert t.values.tolist() == [[2025, "잡사이트B", "91~365일", 1, 1, 1], [2025, "잡사이트B", "뒤 날짜 없음", 1, 1, 1]]


def test_t01d_sorts_sites_by_filled_postings():
    s00 = load_script()
    fill = pd.DataFrame({"year": [2025, 2025, 2025, 2024], "source": ["가", "나", "나", "다"],
                         "gap_bin": ["7일 이내", "뒤 날짜 없음", "8~30일", "365일 초과"],
                         "n": [1, 2, 3, 9], "n_in_occ": [1, 2, 3, 9], "n_in_occ_in_period": [1, 2, 3, 0]})
    t = s00._t01d_date_fill(fill)
    # 연도 순, 연도 안에서는 받은 공고가 많은 사이트(나 5건)부터, 사이트 안에서는 구간 순
    assert t[["year", "source", "gap_bin"]].values.tolist() == [
        [2024, "다", "365일 초과"], [2025, "나", "8~30일"], [2025, "나", "뒤 날짜 없음"], [2025, "가", "7일 이내"]]
    assert s00.REPORT_MAX_ROWS_BY_TABLE["t01d_date_fill"] is None


def test_max_gap_days_removes_far_filled_postings_in_the_report(synth_scan):
    s00 = synth_scan["s00"]
    cfg = copy.deepcopy(synth_scan["cfg"])
    cfg["period"]["fill"]["max_gap_days"] = 100              # 2단계에서만 쓰므로 scan을 다시 하지 않는다
    t = s00.build_tables(cfg, synth_scan["root"], synth_scan["s0_dir"])
    one = t["t01_exclusions"].query("year == 2025")
    assert one["step"].tolist() == ["in_period", "in_occ", "fill_gap", "has_bzno", "has_skill"]
    # p14(141일)와 p17(뒤 날짜 없음)이 빠진다
    assert one.set_index("step").loc["fill_gap", ["n_before", "n_dropped", "n_after"]].tolist() == [15, 2, 13]
    assert t["t02d_date_source_by_period"]["n"].tolist() == [9, 1, 0]
    assert t["t02e_fill_gap_by_period"]["n"].sum() == 0
    assert "`period.fill.max_gap_days` = 100" in s00._table_notes(cfg)["t01_exclusions"]
    # 값이 없으면 fill_gap 행이 없다
    assert "fill_gap" not in set(s00.build_tables(synth_scan["cfg"], synth_scan["root"],
                                                  synth_scan["s0_dir"])["t01_exclusions"]["step"])


def test_report_without_fill_skips_the_fill_tables(synth_data, tmp_path):
    synth_data["cfg"]["period"]["fill"]["enabled"] = False
    s00, t = run(synth_data, tmp_path)
    assert not set(s00.FILL_TABLES) & set(t)
    assert not any((tmp_path / "s0" / "tables" / f"{n}.csv").exists() for n in s00.FILL_TABLES)
    assert t["t02d_date_source_by_period"]["date_source"].tolist() == ["post_start", "post_end"]
    text = s00.write_report(t, _NO_TIMING, synth_data["cfg"], tmp_path / "report.md", commit="c").read_text(encoding="utf-8")
    for name in s00.FILL_TABLES:
        assert f"`period.fill.enabled`가 false여서 날짜를 채우지 않았으므로 `{name}`은 만들지 않았다." in text
        assert f"### `{name}`" not in text
    assert "같은 연도 파일에서 행 순서로 가장 가까운 앞 공고의 날짜" not in s00._table_notes(synth_data["cfg"])["t02d_date_source_by_period"]


def test_report_states_the_configured_definition_and_fill(tables, synth_data, tmp_path):
    s00 = load_script()
    cfg = synth_data["cfg"]
    text = s00.write_report(tables, _NO_TIMING, cfg, tmp_path / "report.md", commit="c").read_text(encoding="utf-8")
    head = text[:text.index("## 1.")]
    assert ("처치 정의(`treatment.definition`): `use_any`. 생성형 AI 활용 숙련(숙련코드)이 있거나 본문이 생성형 AI "
            "활용 키워드 사전(`treatment.keywords.use`)에 걸린 공고를 처치 공고로 본다.") in head
    assert "비교 공고는 어떤 처치 정의에서도 같다" in head and "결정 41" in head
    assert "두 날짜가 모두 없으면 같은 연도 파일에서 행 순서로 가장 가까운 앞 공고의 날짜를 쓴다" in head
    # 생성형 AI 활용 처치를 숙련코드만으로 정한다고 읽힐 문장이 없다
    assert "생성형 AI 활용 처치 정의의 처치 공고" not in text
    assert "처치 정의를 생성형 AI 활용(`use`)으로" not in text
    sec1, sec2 = text[text.index("## 1. "):text.index("## 2. ")], text[text.index("## 2. "):text.index("## 3. ")]
    assert "### `t01d_date_fill`" in sec1 and sec1.index("### `t01c_post_end_gap`") < sec1.index("### `t01d_date_fill`")
    assert sec2.index("### `t02d_date_source_by_period`") < sec2.index("### `t02e_fill_gap_by_period`")
    notes = s00._table_notes(cfg)
    assert ("두 날짜가 모두 없는 공고는 같은 연도 파일에서 행 순서로 가장 가까운 앞 공고의 날짜를 받는다"
            in notes["t01b_post_start_missing"])
    assert "n_in_occ_fill_dates" in notes["t01b_post_start_missing"] and "n_filled <= n_no_date" in notes["t01b_post_start_missing"]
    assert ("앞뒤 날짜의 간격은 날짜를 채운 공고의 바로 앞과 바로 뒤에 있는 날짜 있는 공고 사이의 일수이다. 간격이 짧으면 "
            "채운 날짜가 실제 게시일과 가까울 가능성이 높고, 간격이 길거나 뒤에 날짜가 없으면 한 수집 사이트의 공고가 날짜 "
            "없이 길게 이어진 구간이어서 채운 날짜가 실제 게시일과 다를 수 있다.") in notes["t01d_date_fill"]
    assert "`fill`은 두 날짜가 모두 없어" in notes["t02d_date_source_by_period"]
    assert "period.fill.max_gap_days" in notes["t02e_fill_gap_by_period"]
    for name in ("t02a_treated_by_period", "t02c_occ_group_treated", "t02e_fill_gap_by_period", "t16_detail_by_arm"):
        assert "`use_any`" in notes[name]
    # infra-2b 과제 1 수정 1차: 처치 공고를 나누는 표는 모두 설정의 처치 정의를 따른다고 한 문장으로 적는다
    follows = "처치 공고와 비교 공고는 설정의 처치 정의(`treatment.definition` = `use_any`)를 따른다."
    for name in ("t03b_sparse_filter_grid", "t03e_no_outcome_skill", "t04_bzno_ked_by_treatment", "t05_source",
                 "t09a_by_ksic_sub", "t13_adjacent_candidates", "t14_placebo_candidates", "t15_aiie_coverage"):
        assert follows in notes[name], name
    assert follows in "".join(s00._timing_block(_NO_TIMING, cfg))
    for removed in ("숙련코드)이 있는지만 본 표시", "설정의 처치 정의와 다르므로", "관계없이 이 정의를 쓴다",
                    "설정의 처치 정의와 관계없다", "설정의 처치 정의가 `use`일 때만"):
        assert removed not in text and removed not in "".join(notes.values())
    assert "## 4. 처치 여부별 사업자번호 보유 비율과 KED 결합 비율" in text
    assert "## 5. 연도별 수집 사이트 구성, 사이트별 처치 공고 비율과 숙련 보유 비율" in text
    assert set(s00.DEFINITION_TEXT) == set(treatment.DEFINITIONS)


def test_tables_split_by_the_configured_definition(tables):
    # infra-2b 과제 1 수정 1차. t04, t05, t09a는 base.parquet의 공고 15건(사업자번호와 숙련 제외 전)을 설정의 처치
    # 정의(use_any)로 나눈다: 처치 공고 p02 p04 p05 p11 p15(p04는 사업자번호가 없다), 비교 공고 p03 p06 p10 p12 p13
    # p14 p16 p17, 어느 쪽도 아닌 공고 p01 p09
    t04 = tables["t04_bzno_ked_by_treatment"]
    assert list(t04.columns) == ["arm", "n", "n_has_bzno", "has_bzno_rate", "n_ked_matched", "ked_matched_rate"]
    assert t04["arm"].tolist() == ["전체", "처치 공고", "비교 공고"]
    assert t04["n"].tolist() == [15, 5, 8] and t04["n_has_bzno"].tolist() == [14, 4, 8]
    # KED와 결합되지 않는 공고: 사업자번호 1000000005의 p09, p10, p16
    assert t04["n_ked_matched"].tolist() == [11, 4, 6]
    assert t04["ked_matched_rate"].tolist() == pytest.approx([11 / 14, 1.0, 6 / 8])
    # t05: 잡사이트A p01 p04 p10 p13 p16(처치 p04), 잡사이트B p02 p05 p11 p14 p17(처치 p02 p05 p11),
    # 잡사이트C p03 p06 p09 p12 p15(처치 p15)
    t05 = tables["t05_source"].set_index("source")
    assert "treated_rate" in t05.columns and "has_use_rate" in t05.columns
    assert t05.loc[["잡사이트A", "잡사이트B", "잡사이트C"], "treated_rate"].tolist() == pytest.approx([0.2, 0.6, 0.2])
    assert t05.loc[["잡사이트A", "잡사이트B", "잡사이트C"], "has_use_rate"].tolist() == pytest.approx([0.2, 0.4, 0.0])
    # t09a: 620은 p01 p13(처치 공고 없음), 862는 p02 p14(처치 p02), 012는 p03 p09 p15(처치 p15)
    t09 = tables["t09a_by_ksic_sub"].set_index("ksic_sub")
    assert list(t09.columns) == ["n", "n_has_use", "n_treated"]
    assert t09.loc[["620", "862", "012"], "n"].tolist() == [2, 2, 3]
    assert t09.loc[["620", "862", "012"], "n_treated"].tolist() == [0, 1, 1]
    assert t09["n_treated"].sum() == 5 and t09["n_has_use"].sum() == 3


def test_t10b_marks_the_occupations_removed_by_decision_39():
    import paths
    s00 = load_script()
    cfg = paths.load_config()
    removed = ["2143", "2151", "4154", "4155", "4166"]
    codes = list(cfg["occupations"]["keco4"]) + removed + ["0122"]
    occ = pd.DataFrame({"year": 2025, "keco4": codes, "n": 100, "n_skill": 100, "n_use": 40, "n_dev": 0})
    t = s00._t10b_occ_list_check(occ, cfg).set_index("keco4")
    assert (t.loc[removed + ["0122"], "status"] == "설정에서 뺀 중분류").all()
    assert (t.loc[cfg["occupations"]["keco4"], "status"] == "양쪽에 있음").all() and len(t) == 26
