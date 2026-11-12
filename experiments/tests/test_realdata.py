"""실제 자료(`data/`)가 설계 문서 3장의 수치와 맞는지 확인하는 시험.

기본 실행에서는 빠진다(`pyproject.toml`의 `addopts`). `-m realdata`로 실행한다.
`data/` 폴더가 없으면 모두 건너뛴다. 실패 메시지에는 건수와 비율만 적고 공고 번호나 사업자번호는 적지 않는다.
KED 원자료는 읽지 않는다. 마지막 시험은 열만 있는 빈 KED 표로 기본 자료를 만든다.
"""
import pandas as pd
import pytest

import paths
from aabundle import classif, rawio, sample, skills, treatment

pytestmark = pytest.mark.skipif(not paths.DATA.exists(), reason="data/ 폴더가 없습니다")

_KED_FIRM_COLUMNS = ["bzno", "ked_estab_year", "ked_size", "ked_ipo", "ked_venture"]


@pytest.fixture(scope="module")
def cfg():
    return paths.load_config()


@pytest.fixture(scope="module")
def cb(cfg):
    return skills.load_codebook(paths.ROOT / cfg["data"]["codebook"])


@pytest.mark.realdata
def test_codebook_size(cfg):
    cb = skills.load_codebook(paths.ROOT / cfg["data"]["codebook"])
    assert len(cb) == 8550 and cb["domain"].nunique() == 17 and cb["sub"].nunique() == 65


@pytest.mark.realdata
def test_agent_skill_codes_match_decisions(cfg, cb):
    codes = treatment.agent_skill_codes(cb, cfg)
    assert len(codes["use"]) == 11 and len(codes["dev"]) == 43
    assert codes["use"] | codes["dev"] == {f"IT-05-{i:04d}" for i in range(376, 430)}
    assert {"IT-05-0410", "IT-05-0396"} <= codes["use"] and "IT-05-0413" in codes["dev"]


@pytest.mark.realdata
def test_classification_tables(cfg):
    assert len(classif.load_ksic_sub(paths.ROOT / cfg["data"]["ksic"])) == 232
    keco = classif.load_keco(paths.ROOT / cfg["data"]["keco"])
    assert len(keco) == 495 and set(cfg["occupations"]["keco4"]) <= set(keco["keco4"])


@pytest.mark.realdata
def test_first_row_group_of_each_year_is_readable(cfg, cb):
    # 연도마다 첫 행 묶음을 읽어 build_posting_base와 explode_skills가 예외 없이 끝나는지 확인
    columns = sample.RAW_COLUMNS + skills.raw_skill_columns()
    ksic_valid = classif.load_ksic_sub(paths.ROOT / cfg["data"]["ksic"])
    empty_ked = pd.DataFrame({c: pd.Series([], dtype=object) for c in _KED_FIRM_COLUMNS})
    problems = []
    for year in cfg["data"]["years"]:
        raw = next(rawio.iter_postings(paths.POSTINGS, year, columns))
        _, stats = skills.explode_skills(raw, cb)
        if stats["n_tokens"] == 0:
            problems.append(f"{year}년: 숙련 토큰이 없습니다 (행 {len(raw)}개)")
        else:
            ratio = stats["n_unmatched"] / stats["n_tokens"]
            if not ratio < 0.01:
                problems.append(f"{year}년: 찾지 못한 숙련 비율 {ratio:.4f} "
                                f"({stats['n_unmatched']} / {stats['n_tokens']})")
        base = sample.build_posting_base(raw, empty_ked, ksic_valid, cfg)
        if len(base) != len(raw):
            problems.append(f"{year}년: 기본 자료 행 수 {len(base)}, 원자료 행 수 {len(raw)}")
    assert not problems, "; ".join(problems)
