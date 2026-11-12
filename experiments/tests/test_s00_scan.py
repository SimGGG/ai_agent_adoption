import importlib.util, pandas as pd, paths

def load_script():
    spec = importlib.util.spec_from_file_location("s00", paths.ROOT / "experiments" / "scripts" / "s00_data_audit.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def test_scan_outputs(synth_data, tmp_path):
    out = load_script().scan(synth_data["cfg"], synth_data["root"], tmp_path / "s0")
    base = pd.read_parquet(out["base"]); treat = pd.read_parquet(out["treat"])
    long = pd.read_parquet(out["skills_long"]); occ = pd.read_parquet(out["occ_year_all"])
    stats = pd.read_parquet(out["scan_stats"]).set_index("year")

    assert sorted(base.posting_id) == [f"p{i:02d}" for i in (1, 2, 3, 4, 5, 6, 9, 10, 11, 12)]
    assert sorted(treat.posting_id) == sorted(base.posting_id)
    assert set(long.posting_id) <= set(base.posting_id)
    assert base.set_index("posting_id").loc["p12", "n_skills"] == 2
    assert base.set_index("posting_id").loc["p10", "n_skills"] == 0

    o = occ.set_index("keco4")
    assert o.loc["0262", "n"] == 5 and o.loc["0262", "n_use"] == 2       # p02, p04
    assert o.loc["1332", "n_use"] == 1 and o.loc["1332", "n_dev"] == 2   # p05 / p01, p05
    assert o.loc["9052", "n"] == 1 and o.loc["NA", "n"] == 1
    assert stats.loc[2025, "n_raw"] == 12 and stats.loc[2025, "n_in_occ"] == 10
    assert stats.loc[2025, "n_unmatched"] == 1

    kw = pd.read_parquet(out["kw_pattern_hits"]).set_index("pattern")
    assert len(kw) == len(synth_data["cfg"]["treatment"]["keywords"]["strict"]) + \
        len(synth_data["cfg"]["treatment"]["keywords"]["broad_extra"])
    assert kw.loc["Claude", "n"] == 1 and kw.loc["Claude", "dictionary"] == "broad_extra"   # p11
    assert kw.loc["AI\\s*agent", "n"] == 0 and kw.loc["Gemini", "n"] == 0                  # p07은 분석 대상 직종이 아니다

def test_scan_never_stores_posting_text(synth_data, tmp_path):
    out = load_script().scan(synth_data["cfg"], synth_data["root"], tmp_path / "s0")
    for p in out.values():
        df = pd.read_parquet(p)
        assert "detail" not in df.columns and "title" not in df.columns
        assert not df.astype(str).apply(lambda c: c.str.contains("에이전트 개발")).any().any()
