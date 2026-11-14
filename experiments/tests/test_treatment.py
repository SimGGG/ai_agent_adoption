import pandas as pd, pytest
from aabundle import rawio, skills, treatment

import synth

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
    assert ids(treat.kw_use) == ["p11", "p15"]                # p11은 Claude, p15는 ChatGPT 활용(과제 3A)
    s = pd.Series(["Claude Code 사용", "claude와 Gemini", "보험 에이전트", None])
    assert treatment.pattern_counts(s, ["Claude", "Gemini", "에이전트"], ["보험\\s*에이전트"]) == \
        {"Claude": 2, "Gemini": 1, "에이전트": 0}

def test_skill_code_flags(treat):
    assert ids(treat.has_use) == ["p02", "p04", "p05"]
    assert ids(treat.has_dev) == ["p01", "p05"]
    assert ids(treat.treat_code) == ["p01", "p02", "p04", "p05"]
    assert ids(treat.treat_all) == ["p01", "p02", "p04", "p05", "p07", "p09"]
    assert len(treat) == 17   # 과제 3에서 p13, p14를 더해 12에서 14로, 과제 3A에서 p15, p16을 더해 16으로 바꿨다(과제 4: p17을 더해 17)

def test_select_arms_use_definition(treat):
    arm = treatment.select_arms(treat.reset_index(), "use")
    arm.index = treat.index
    assert ids(arm == 1) == ["p02", "p04", "p05"]
    # p13, p14: 과제 3. 과제 3A에서 생성형 AI 활용 키워드에 걸린 p11을 빼고 p16을 더했다. p17: 과제 4
    assert ids(arm == 0) == ["p03", "p06", "p08", "p10", "p12", "p13", "p14", "p16", "p17"]
    # 개발 숙련만 있거나 키워드에만 걸린 공고(p11, p15는 생성형 AI 활용 키워드에 걸린 공고, 과제 3A)
    assert ids(arm.isna()) == ["p01", "p07", "p09", "p11", "p15"]

# 전체 검토 수정에서 더한 시험: 처치 정의 각각의 처치 공고와 비교 공고(가상 자료 공고 16건,
# 과제 3에서 비교 공고 p13, p14를 더했다). 과제 3A에서 생성형 AI 활용 키워드에 걸린 공고(p11, p15)를
# 비교 공고에서 빼고 비교 공고 p16을 더했다. 그래서 엄격한 사전과 넓은 사전의 비교 공고가 같아졌다
# (p11은 넓은 사전에도 걸린다). 과제 4에서 비교 공고 p17을 더했다.
_COMPARISON = ["p03", "p06", "p08", "p10", "p12", "p13", "p14", "p16", "p17"]
_ARMS = [
    ("kw_strict", "all", ["p01", "p02", "p04", "p05", "p07", "p09"], _COMPARISON),
    ("kw_strict", "dev", ["p01", "p05"], _COMPARISON),
    ("kw_strict", "use", ["p02", "p04", "p05"], _COMPARISON),
    ("kw_strict", "keyword", ["p01", "p07", "p09"], _COMPARISON),
    ("kw_strict", "code", ["p01", "p02", "p04", "p05"], _COMPARISON),
    ("kw_strict", "use_keyword", ["p11", "p15"], _COMPARISON),
    ("kw_strict", "use_any", ["p02", "p04", "p05", "p11", "p15"], _COMPARISON),
    ("kw_broad", "all", ["p01", "p02", "p04", "p05", "p07", "p09", "p11"], _COMPARISON),
    ("kw_broad", "dev", ["p01", "p05"], _COMPARISON),
    ("kw_broad", "use", ["p02", "p04", "p05"], _COMPARISON),
    ("kw_broad", "keyword", ["p01", "p05", "p07", "p09", "p11"], _COMPARISON),
    ("kw_broad", "code", ["p01", "p02", "p04", "p05"], _COMPARISON),
    ("kw_broad", "use_keyword", ["p11", "p15"], _COMPARISON),
    ("kw_broad", "use_any", ["p02", "p04", "p05", "p11", "p15"], _COMPARISON),
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


def _treat(rows):
    cols = ["kw_strict", "kw_broad", "kw_use", "has_dev", "has_use"]
    return pd.DataFrame(rows, columns=cols).astype(bool)


def test_use_keyword_postings_leave_the_comparison_group():
    #            strict broad  use   dev   use(code)
    t = _treat([(0, 0, 0, 0, 0),      # 아무 것도 아님 → 비교 공고
                (0, 0, 1, 0, 0),      # 생성형 AI 활용 키워드만 → 어느 쪽도 아님
                (0, 0, 0, 0, 1),      # 생성형 AI 활용 숙련만 → 처치 공고
                (0, 0, 1, 0, 1)])     # 둘 다 → 처치 공고
    assert treatment.select_arms(t, "use").tolist() == [0, pd.NA, 1, 1]
    assert treatment.select_arms(t, "use_keyword").tolist() == [0, 1, pd.NA, 1]
    assert treatment.select_arms(t, "use_any").tolist() == [0, 1, 1, 1]
    assert treatment.select_arms(t, "dev").tolist() == [0, pd.NA, pd.NA, pd.NA]      # 비교 공고는 어떤 정의에서도 같다


def test_comparison_group_follows_keyword_col():
    """넓은 사전에만 걸린 공고는 키워드 열이 `kw_strict`이면 비교 공고이고, `kw_broad`이면 비교 공고가 아니다.
    과제 3A 검토 수정에서 더한 시험이다. 가상 공고 가운데 넓은 사전에 걸린 공고는 모두 다른 이유로도 비교 공고에서
    빠지므로, 비교 공고가 `keyword_col`을 따르는지는 여기서 확인한다."""
    #            strict broad  use   dev   use(code)
    t = _treat([(0, 1, 0, 0, 0)])     # 넓은 사전에만 걸린 공고
    assert treatment.select_arms(t, "use").tolist() == [0]
    assert treatment.select_arms(t, "use", keyword_col="kw_broad").tolist() == [pd.NA]
    assert treatment.select_arms(t, "keyword", keyword_col="kw_broad").tolist() == [1]
    assert treatment.select_arms(t, "use_any", keyword_col="kw_broad").tolist() == [pd.NA]


def test_select_arms_requires_kw_use():
    t = _treat([(0, 0, 0, 0, 0)]).drop(columns="kw_use")
    with pytest.raises(KeyError, match="kw_use"):
        treatment.select_arms(t, "use")


def test_build_treat_flags_use_keywords(treat):
    """가상 공고 본문에 `ChatGPT 활용`이 있고 숙련코드는 없는 공고(`synth.ID_USE_KEYWORD_ONLY`)는 `kw_use`만 참이다.
    본문에 `Fulfillment`만 있는 공고(`synth.ID_FULFILLMENT`)는 `kw_use`가 거짓이다."""
    tr = treat                                                # 고정물 `treat`는 이미 posting_id를 색인으로 둔다
    assert tr.loc[synth.ID_USE_KEYWORD_ONLY, ["kw_use", "kw_strict", "has_use"]].tolist() == [True, False, False]
    assert not tr.loc[synth.ID_FULFILLMENT, "kw_use"]
    assert list(tr.columns)[:3] == ["kw_strict", "kw_broad", "kw_use"]
