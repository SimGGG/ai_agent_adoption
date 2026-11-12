import pandas as pd, pytest
from aabundle import rawio, skills

@pytest.fixture
def exploded(synth_data):
    cb = skills.load_codebook(synth_data["codebook"])
    raw = pd.concat(rawio.iter_postings(synth_data["postings_dir"], 2025, ["posting_id"] + skills.raw_skill_columns()))
    return cb, *skills.explode_skills(raw, cb)

def codes_of(long, pid):
    return sorted(long.loc[long.posting_id == pid, "skill_code"])

def test_code_in_codebook_is_kept(exploded):
    _, long, _ = exploded
    assert codes_of(long, "p01") == ["IT-01-0001", "IT-05-0398"]

def test_old_format_code_is_mapped_by_name(exploded):
    _, long, _ = exploded
    assert codes_of(long, "p09") == ["SS-02-0009"]
    assert codes_of(long, "p11") == ["IT-01-0001", "XT-01-0001"]
    assert long.loc[(long.posting_id == "p11") & (long.skill_code == "XT-01-0001"), "src"].item() == "qualification"

def test_unknown_skill_is_dropped_and_counted(exploded):
    _, long, stats = exploded
    assert codes_of(long, "p10") == [] and stats["n_unmatched"] == 1

def test_length_mismatch_falls_back_to_codes(exploded):
    _, long, stats = exploded
    assert codes_of(long, "p12") == ["IT-01-0001", "SM-01-0001"]
    assert stats["n_length_mismatch_rows"] == 1

def test_posting_without_skills_has_no_rows(exploded):
    _, long, _ = exploded
    assert codes_of(long, "p06") == []
    assert not long.duplicated(["posting_id", "skill_code"]).any()

def test_same_skill_in_two_columns_is_one_row():
    cb = pd.DataFrame({"skill_code": ["IT-01-0001"], "skill_code_v31": ["x"], "skill_name": ["숙련A"],
                       "domain": ["d"], "sub": ["s"], "detail": ["t"]})
    raw = pd.DataFrame({c: [None] for c in skills.raw_skill_columns()} | {"posting_id": ["q1"]})
    raw["skills_occupation"], raw["skills_occupation_name"] = "IT-01-0001", "숙련A"
    raw["skills_soft"], raw["skills_soft_name"] = "IT-01-0001", "숙련A"
    long, _ = skills.explode_skills(raw, cb)
    assert len(long) == 1 and long["src"].item() == "occupation+soft"

def _one_cell(codes, names):
    """숙련 열 하나(`occupation`)에만 값이 있는 공고 한 건(시험 안에서만 쓴다)."""
    raw = pd.DataFrame({c: [None] for c in skills.raw_skill_columns()} | {"posting_id": ["q1"]})
    raw["skills_occupation"], raw["skills_occupation_name"] = codes, names
    return raw

def test_codes_and_names_pair_by_position_before_empty_tokens_are_dropped(synth_data):
    # 전체 검토 수정에서 더한 시험. OLD-A, OLD-C는 설명 파일에 없는 옛 코드이다.
    # 빈 토큰을 버린 뒤 순서를 매기면 OLD-C가 `숙련B`(SM-01-0001)와 짝지어진다. 원래 위치로는 OLD-C의 숙련명이 비어 있다.
    cb = skills.load_codebook(synth_data["codebook"])
    long, stats = skills.explode_skills(_one_cell("OLD-A||OLD-C", "숙련A|숙련B|"), cb)
    assert long["skill_code"].tolist() == ["IT-01-0001"]
    assert stats == {"n_tokens": 2, "n_unmatched": 1, "n_length_mismatch_rows": 0}

def test_leading_empty_code_token_keeps_name_position(synth_data):
    cb = skills.load_codebook(synth_data["codebook"])
    long, stats = skills.explode_skills(_one_cell("|OLD-B", "숙련A|숙련B"), cb)
    assert long["skill_code"].tolist() == ["SM-01-0001"]
    assert stats == {"n_tokens": 1, "n_unmatched": 0, "n_length_mismatch_rows": 0}

def test_raw_token_counts_decide_length_mismatch(synth_data):
    cb = skills.load_codebook(synth_data["codebook"])
    # 숙련명 쪽에만 끝의 빈 토큰이 있으면 나눈 조각 수가 달라 개수가 다른 칸으로 센다(코드만 쓴다)
    long, stats = skills.explode_skills(_one_cell("OLD-A|OLD-B", "숙련A|숙련B|"), cb)
    assert long.empty and stats["n_length_mismatch_rows"] == 1 and stats["n_unmatched"] == 2
    # 코드가 없고 숙련명만 있는 칸도 개수가 다른 칸이다
    _, stats = skills.explode_skills(_one_cell(None, "숙련A"), cb)
    assert stats["n_length_mismatch_rows"] == 1

def test_hierarchy_and_drop(exploded):
    cb, long, _ = exploded
    h = skills.attach_hierarchy(long, cb)
    assert h.loc[h.skill_code == "SS-02-0009", "domain"].iloc[0] == "의사소통"
    assert "IT-05-0410" not in set(skills.drop_codes(long, {"IT-05-0410"})["skill_code"])

def test_sparse_filter_thresholds(exploded):
    _, long, _ = exploded
    kept = skills.sparse_filter(long, treated_ids={"p02", "p05"}, n_postings=10, min_rate=0.2, min_treated=1)
    assert kept == ["IT-05-0410", "SM-01-0001"]   # 0410: 2건 모두 처치. SM-01-0001: 4건 중 p02가 처치. IT-01-0001: 4건이지만 처치 공고에 없음
    assert skills.sparse_filter(long, treated_ids=set(), n_postings=10, min_rate=0.0, min_treated=1) == []

def test_codebook_with_duplicate_name_is_rejected(tmp_path, synth_data):
    df = pd.read_csv(synth_data["codebook"], dtype=str, encoding="utf-8-sig")
    pd.concat([df, df.iloc[[0]].assign(skill_code_v4="ZZ-00-0000")]).to_csv(tmp_path / "cb.csv", index=False, encoding="utf-8-sig")
    with pytest.raises(ValueError):
        skills.load_codebook(tmp_path / "cb.csv")
