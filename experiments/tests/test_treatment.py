import pandas as pd, pytest
from aabundle import rawio, skills, treatment

@pytest.fixture
def treat(synth_data):
    cfg = synth_data["cfg"]
    cb = skills.load_codebook(synth_data["codebook"])
    raw = pd.concat(rawio.iter_postings(synth_data["postings_dir"], 2025,
                                        ["posting_id", "detail"] + skills.raw_skill_columns()))
    long, _ = skills.explode_skills(raw, cb)
    return treatment.build_treat(raw["posting_id"], raw["detail"], long, cb, cfg).set_index("posting_id")

def ids(mask):
    return sorted(mask[mask].index)

def test_agent_codes_split_with_exceptions(synth_data):
    cb = skills.load_codebook(synth_data["codebook"])
    codes = treatment.agent_skill_codes(cb, synth_data["cfg"])
    assert codes["use"] == {"IT-05-0410", "IT-05-0396"}       # 이름 규칙 + 옮긴 것
    assert codes["dev"] == {"IT-05-0398", "IT-05-0413"}       # 나머지 + 옮긴 것
    assert "IT-05-0200" not in codes["use"] | codes["dev"]    # 같은 중분류의 다른 세부 분류

def test_exception_outside_agent_codes_is_rejected(synth_data):
    cfg = synth_data["cfg"]
    cfg["treatment"]["agent_skills"]["move_to_use"] = ["IT-01-0001"]
    with pytest.raises(ValueError):
        treatment.agent_skill_codes(skills.load_codebook(synth_data["codebook"]), cfg)

def test_keyword_flags(treat):
    assert ids(treat.kw_strict) == ["p01", "p07", "p09"]      # p03은 제외 패턴, p06은 본문 결측
    assert ids(treat.kw_broad) == ["p01", "p05", "p07", "p09", "p11"]   # p11은 저자가 더한 표현(Claude)
    s = pd.Series(["Claude Code 사용", "claude와 Gemini", "보험 에이전트", None])
    assert treatment.pattern_counts(s, ["Claude", "Gemini", "에이전트"], ["보험\\s*에이전트"]) == \
        {"Claude": 2, "Gemini": 1, "에이전트": 0}

def test_skill_code_flags(treat):
    assert ids(treat.has_use) == ["p02", "p04", "p05"]
    assert ids(treat.has_dev) == ["p01", "p05"]
    assert ids(treat.treat_code) == ["p01", "p02", "p04", "p05"]
    assert ids(treat.treat_all) == ["p01", "p02", "p04", "p05", "p07", "p09"]
    assert len(treat) == 12

def test_select_arms_use_definition(treat):
    arm = treatment.select_arms(treat.reset_index(), "use")
    arm.index = treat.index
    assert ids(arm == 1) == ["p02", "p04", "p05"]
    assert ids(arm == 0) == ["p03", "p06", "p08", "p10", "p11", "p12"]
    assert ids(arm.isna()) == ["p01", "p07", "p09"]           # 개발 숙련만 있거나 키워드에만 걸린 공고

# 전체 검토 수정에서 더한 시험: 다섯 처치 정의 각각의 처치 공고와 비교 공고(가상 자료 공고 12건)
_STRICT_COMPARISON = ["p03", "p06", "p08", "p10", "p11", "p12"]
_BROAD_COMPARISON = ["p03", "p06", "p08", "p10", "p12"]          # p11은 넓은 사전(Claude)에 걸린다
_ARMS = [
    ("kw_strict", "all", ["p01", "p02", "p04", "p05", "p07", "p09"], _STRICT_COMPARISON),
    ("kw_strict", "dev", ["p01", "p05"], _STRICT_COMPARISON),
    ("kw_strict", "use", ["p02", "p04", "p05"], _STRICT_COMPARISON),
    ("kw_strict", "keyword", ["p01", "p07", "p09"], _STRICT_COMPARISON),
    ("kw_strict", "code", ["p01", "p02", "p04", "p05"], _STRICT_COMPARISON),
    ("kw_broad", "all", ["p01", "p02", "p04", "p05", "p07", "p09", "p11"], _BROAD_COMPARISON),
    ("kw_broad", "dev", ["p01", "p05"], _BROAD_COMPARISON),
    ("kw_broad", "use", ["p02", "p04", "p05"], _BROAD_COMPARISON),
    ("kw_broad", "keyword", ["p01", "p05", "p07", "p09", "p11"], _BROAD_COMPARISON),
    ("kw_broad", "code", ["p01", "p02", "p04", "p05"], _BROAD_COMPARISON),
]

@pytest.mark.parametrize("keyword_col, definition, treated, comparison", _ARMS)
def test_select_arms_every_definition(treat, keyword_col, definition, treated, comparison):
    arm = treatment.select_arms(treat.reset_index(), definition, keyword_col)
    arm.index = treat.index
    assert ids(arm == 1) == treated
    assert ids(arm == 0) == comparison
    assert ids(arm.isna()) == sorted(set(treat.index) - set(treated) - set(comparison))

def test_select_arms_comparison_is_same_for_every_definition(treat):
    t = treat.reset_index()
    comps = [set(t.posting_id[(treatment.select_arms(t, d) == 0).fillna(False).to_numpy(dtype=bool)]) for d in treatment.DEFINITIONS]
    assert all(c == comps[0] for c in comps)
    with pytest.raises(ValueError):
        treatment.select_arms(t, "unknown")
