"""설정 확인(`validate.validate_config`) 시험. 전체 검토 수정에서 더했다."""
import copy

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
]


@pytest.mark.parametrize("key, value, message", _BAD)
def test_each_rule_names_the_key(key, value, message):
    cfg = _set(paths.load_config(), key, value)
    with pytest.raises(ValueError, match=message):
        validate.validate_config(cfg)


def test_allowed_edge_values_pass():
    cfg = paths.load_config()
    for key, value in (("salary_band_edges", []), ("salary_band_edges", [30_000_000, 45_000_000.5]),
                       ("treatment.keywords.broad_extra", []), ("treatment.keywords.exclude", []),
                       ("period.end", "2021-01-01"), ("estimation.n_folds", 2)):
        validate.validate_config(_set(cfg, key, value))


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
