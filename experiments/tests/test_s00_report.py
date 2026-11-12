import re

import numpy as np
import pandas as pd
import pytest
from test_s00_scan import load_script

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
    assert list(k.columns) == ["n_lines", "n_dropped_lines", "n_firms"] and len(k) == 1
    assert k.loc[0, "n_lines"] == 10 and k.loc[0, "n_dropped_lines"] == 4 and k.loc[0, "n_firms"] == 4
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
