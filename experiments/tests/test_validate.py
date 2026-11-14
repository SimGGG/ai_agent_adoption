"""설정 확인(`validate.validate_config`) 시험. 전체 검토 수정에서 더했다."""
import copy
import re

import pytest

import paths
from aabundle import validate
from test_s00_scan import load_script


def test_shipped_config_passes():
    validate.validate_config(paths.load_config())


def _set(cfg: dict, key: str, value) -> dict:
    out = copy.deepcopy(cfg)
    *parents, last = key.split(".")
    d = out
    for p in parents:
        d = d[p]
    d[last] = value
    return out


# (항목, 바꿀 값, 예외 메시지에 들어 있어야 할 글)
_BAD = [
    ("data.years", [], "data.years"),
    ("data.years", ["2025"], "data.years"),
    ("occupations.keco4", [], "occupations.keco4"),
    ("occupations.keco4", ["0122", 82], "따옴표"),
    ("occupations.keco4", ["122"], "occupations.keco4"),
    ("period.start", "2027-01-01", "period.start"),
    ("period.end", "2026-13-01", "period.end"),
    ("period.unit", "month", "period.unit"),
    ("treatment.keywords.strict", [], "treatment.keywords.strict"),
    ("treatment.keywords.strict", ["AI\\s*agent", ""], "treatment.keywords.strict"),
    ("treatment.keywords.broad_extra", ["Claude", "("], "treatment.keywords.broad_extra"),
    ("treatment.keywords.exclude", [""], "treatment.keywords.exclude"),
    ("treatment.keyword_dictionary", "wide", "treatment.keyword_dictionary"),
    ("treatment.definition", "both", "treatment.definition"),
    ("treatment.agent_skills.move_to_use", ["IT-05-0396", "IT-05-0382"], "IT-05-0382"),
    ("treatment.text_columns", [], "treatment.text_columns"),
    ("salary_band_edges", [30_000_000, 30_000_000], "salary_band_edges"),
    ("salary_band_edges", [40_000_000, 30_000_000], "salary_band_edges"),
    ("estimation.n_folds", 1, "estimation.n_folds"),
    ("period.date_columns", [], "period.date_columns"),
    ("period.groups", [{"name": "초기", "years": [2023, 2024]}, {"name": "1 도입기", "years": [2025]}], "2026"),          # 기간에 드는 연도가 빠짐
    ("period.groups", [{"name": "초기", "years": [2023, 2024, 2025]}, {"name": "1 도입기", "years": [2025]},
                       {"name": "2 도입기", "years": [2026]}], "2025"),                                                   # 한 연도가 두 기간에 듦
    ("period.groups", [{"name": "초기", "years": [2022, 2023, 2024]}, {"name": "1 도입기", "years": [2025]},
                       {"name": "2 도입기", "years": [2026]}], "2022"),                                                   # 분석 기간 밖의 연도
    ("period.groups", [{"name": "초기", "years": [2023, 2024]}, {"name": "초기", "years": [2025]},
                       {"name": "2 도입기", "years": [2026]}], "초기"),                                                    # 이름이 겹침
    ("occupations.exclude_keco2", ["13"], "1332"),                                                                      # 뺀 중분류의 세분류가 목록에 남음
    ("occupations.exclude_keco2", [1], "occupations.exclude_keco2"),
    ("occupations.group_min_treated", 0, "occupations.group_min_treated"),
    ("industry.not_in_table", "drop", "industry.not_in_table"),
    ("region.merge_sido", {"광주·전남": ["광주", "전남"], "호남": ["전남", "전북"]}, "전남"),                              # 한 시도가 두 묶음에 듦
    ("skills.adjacent_candidate_details", "MLOps및AI인프라", "skills.adjacent_candidate_details"),                       # 목록이 아님
    ("skills.placebo_candidate_domains", [""], "skills.placebo_candidate_domains"),
    ("treatment.keywords.use", [], "treatment.keywords.use"),
    ("treatment.keywords.use", ["GPT", "("], "treatment.keywords.use"),
    ("data.aiie", "", "data.aiie"),
    ("period.date_columns", ["post_start", "post_end", "post_start"], r"period\.date_columns`에 같은 열 이름"),   # 최종 검토 수정
    # infra-2b 과제 1(결정 38): 날짜를 채우는 규칙
    ("period.fill", None, r"period\.fill`"),
    ("period.fill", {"enabled": True}, r"period\.fill`"),                                                   # 항목이 빠짐
    ("period.fill", {"enabled": True, "max_gap_days": None, "limit": 3}, r"period\.fill`"),                 # 모르는 항목
    ("period.fill.enabled", "yes", r"period\.fill\.enabled"),
    ("period.fill.enabled", 1, r"period\.fill\.enabled"),
    ("period.fill.max_gap_days", -1, r"period\.fill\.max_gap_days"),
    ("period.fill.max_gap_days", 30.5, r"period\.fill\.max_gap_days"),
    ("period.fill.max_gap_days", True, r"period\.fill\.max_gap_days"),
    ("period.fill.max_gap_days", "30", r"period\.fill\.max_gap_days"),
]


def test_use_keywords_shipped():
    kw = paths.load_config()["treatment"]["keywords"]["use"]
    assert len(kw) == 70 and len(set(kw)) == 70          # 결정 41에서 34에서 71로 바꿨고, 결정 44에서 `에이닷`을 빼 70이다
    joined = re.compile("|".join(kw), re.IGNORECASE)
    for hit in ["ChatGPT 활용 경험", "챗GPT", "생성형 AI 툴", "프롬프트 엔지니어링", "LLM 기반", "LLMs", "AI 활용 능력", "GenAI",
                # 결정 41에서 더한 표현
                "클로드 활용", "Codex 사용 경험", "코덱스", "나노바나나", "sLLM 구축", "Grok", "DALL-E", "Veo 3", "클링 AI",
                "Cursor AI", "NotebookLM", "Llama 3", "DeepSeek"]:
        assert joined.search(hit), hit
    for miss in ["Fulfillment 센터 운영", "풀필먼트", "명령 프롬프트", "KAI 활용", "달리 정한 바",
                 # 결정 41에서 더한 표현이 다른 뜻의 낱말에 걸리지 않는지
                 "ngrok 터널", "리사이클링 AI 기술", "필수노하우", "CODEX 규격", "코덱스 기준", "windsurfing", "LlamaIndex",
                 "런웨이 패션쇼", "cursor 위치", "에이닷컴", "gamma 보정",
                 "에이닷 서비스 운영"]:                      # 결정 44에서 뺀 표현
        assert not joined.search(miss), miss


def test_groups_require_year_unit():
    cfg = paths.load_config()
    cfg["period"]["unit"] = "quarter"
    with pytest.raises(ValueError, match="period.unit"):
        validate.validate_config(cfg)


def test_shipped_values_of_2026_11_13():
    cfg = paths.load_config()
    assert cfg["data"]["years"] == [2023, 2024, 2025, 2026]
    assert (cfg["period"]["start"], cfg["period"]["end"], cfg["period"]["unit"]) == ("2023-01-01", "2026-08-31", "year")
    assert [(g["name"], g["years"]) for g in cfg["period"]["groups"]] == [
        ("초기", [2023, 2024]), ("1 도입기", [2025]), ("2 도입기", [2026])]
    assert cfg["period"]["date_columns"] == ["post_start", "post_end"]
    assert len(cfg["occupations"]["keco4"]) == 20          # 결정 39에서 25에서 20으로 바꿨다
    assert not {"0122", "6121", "6154"} & set(cfg["occupations"]["keco4"])
    assert not {"2143", "2151", "4154", "4155", "4166"} & set(cfg["occupations"]["keco4"])
    assert {c[:2] for c in cfg["occupations"]["keco4"]} == {"02", "13"}
    # 결정 39에서 ["01", "61"]에서 바꿨다
    assert cfg["occupations"]["exclude_keco2"] == ["01", "61", "21", "41"] and cfg["occupations"]["group_min_treated"] == 100
    assert cfg["treatment"]["definition"] == "use_any"
    assert cfg["period"]["fill"] == {"enabled": True, "max_gap_days": None}
    assert cfg["industry"]["not_in_table"] == "missing"
    assert cfg["region"]["merge_sido"] == {"광주·전남": ["광주", "전남", "전남광주"]}
    assert cfg["salary_band_edges"] == [26000000, 30000000, 35000000, 40000000]
    assert cfg["skills"]["sparse_filter"] == {"min_rate": 0.001, "min_treated": 30}


@pytest.mark.parametrize("key, value, message", _BAD)
def test_each_rule_names_the_key(key, value, message):
    cfg = _set(paths.load_config(), key, value)
    with pytest.raises(ValueError, match=message):
        validate.validate_config(cfg)


def test_allowed_edge_values_pass():
    cfg = paths.load_config()
    for key, value in (("salary_band_edges", []), ("salary_band_edges", [30_000_000, 45_000_000.5]),
                       ("treatment.keywords.broad_extra", []), ("treatment.keywords.exclude", []),
                       ("estimation.n_folds", 2), ("occupations.exclude_keco2", []),
                       ("skills.adjacent_candidate_details", []), ("region.merge_sido", {}),
                       ("industry.not_in_table", "category"), ("period.fill.enabled", False),
                       ("period.fill.max_gap_days", 0), ("period.fill.max_gap_days", 30)):
        validate.validate_config(_set(cfg, key, value))
    # 분석 기간이 하루여도 된다(기간 묶음은 그 연도만 담는다). 2021-01-01은 기본 시작일 앞이므로 시작일도 함께 바꾼다
    one_day = _set(_set(_set(cfg, "period.start", "2021-01-01"), "period.end", "2021-01-01"),
                   "period.groups", [{"name": "하루", "years": [2021]}])
    validate.validate_config(one_day)


# 재검토 뒤 더한 시험. 표현은 `|`로 이어 붙여 쓰므로(`treatment.flag_treatment`), 하나씩은 읽히지만 이어 붙이면
# 읽히지 않는 표현도 막는다. 이 파이썬(3.12)에서는 `(?i)` 같은 전체 적용 표시가 이어 붙인 식의 맨 앞이 아니면 오류이다.
@pytest.mark.parametrize("key, value", [
    ("treatment.keywords.strict", ["AI\\s*agent", "(?i)llm"]),
    ("treatment.keywords.broad_extra", ["Claude", "(?i)llm"]),
    ("treatment.keywords.broad_extra", ["(?i)llm"]),          # 넓은 사전은 엄격한 사전 뒤에 이어 붙인다
    ("treatment.keywords.exclude", ["에이전시", "(?i)보험\\s*에이전트"]),
])
def test_patterns_are_also_checked_joined(key, value):
    with pytest.raises(ValueError, match=key.replace(".", r"\.")) as e:
        validate.validate_config(_set(paths.load_config(), key, value))
    assert "이어 붙인" in str(e.value)


def test_empty_use_dictionary_explains_the_comparison_group():    # 최종 검토 수정에서 더했다
    with pytest.raises(ValueError, match="treatment.keywords.use") as e:
        validate.validate_config(_set(paths.load_config(), "treatment.keywords.use", []))
    assert "본문이 있는 모든 공고" in str(e.value) and "비교 공고" in str(e.value)


def test_inline_flag_at_the_start_of_a_joined_list_passes():
    validate.validate_config(_set(paths.load_config(), "treatment.keywords.strict", ["(?i)llm", "AI\\s*agent"]))


def test_missing_key_is_named():
    cfg = paths.load_config()
    del cfg["estimation"]["n_folds"]
    with pytest.raises(ValueError, match="estimation.n_folds"):
        validate.validate_config(cfg)


def test_scan_and_build_tables_check_the_config_first(synth_data, tmp_path):
    s00 = load_script()
    out_dir = tmp_path / "s0"
    bad = _set(synth_data["cfg"], "occupations.keco4", [1332, "0262"])
    with pytest.raises(ValueError, match="따옴표"):
        s00.scan(bad, synth_data["root"], out_dir)
    assert not out_dir.exists()

    s00.scan(synth_data["cfg"], synth_data["root"], out_dir)
    with pytest.raises(ValueError, match="treatment.keyword_dictionary"):
        s00.build_tables(_set(synth_data["cfg"], "treatment.keyword_dictionary", "wide"), synth_data["root"], out_dir)
    assert not (out_dir / "tables").exists()
