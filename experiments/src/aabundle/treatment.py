"""처치 공고를 판별하는 함수.

처치 공고는 두 가지 근거로 표시한다. 하나는 공고 본문에 AI 에이전트 관련 키워드가 있는 경우이고,
다른 하나는 AI 에이전트 숙련 코드가 붙은 경우이다. 숙련 코드는 개발 숙련과 생성형 AI 활용 숙련으로 나눈다.
키워드 목록과 숙련 코드 선택 규칙은 모두 `config.yaml`의 `treatment` 항목에서 읽는다.
공고 본문은 판별에만 쓰고 결과에 담지 않는다.
"""
from collections.abc import Sequence

import numpy as np
import pandas as pd

DEFINITIONS = ("all", "dev", "use", "keyword", "code")


def flag_treatment(text, include, exclude=()):
    """제외 패턴을 지운 뒤 포함 사전에 걸리는 공고를 1로 표시"""
    if exclude:
        text = text.str.replace("|".join(exclude), " ", case=False, regex=True)
    return text.str.contains("|".join(include), case=False, regex=True, na=False).astype(int)


def keyword_lists(cfg: dict) -> dict:
    """엄격 목록, 넓은 목록(엄격 목록 + 추가 표현), 제외 목록을 돌려준다."""
    kw = cfg["treatment"]["keywords"]
    return {
        "strict": list(kw["strict"]),
        "broad": list(kw["strict"]) + list(kw["broad_extra"]),
        "exclude": list(kw["exclude"]),
    }


def pattern_counts(text: pd.Series, patterns: list[str], exclude: Sequence[str] = ()) -> dict[str, int]:
    """제외 패턴을 한 번 지운 뒤, 패턴마다 걸리는 공고 수를 센다. 대소문자는 구분하지 않는다."""
    s = text.astype("string")
    if exclude and len(s):
        s = s.str.replace("|".join(exclude), " ", case=False, regex=True)
    out = {}
    for p in patterns:
        if len(s) == 0:
            out[p] = 0
        else:
            out[p] = int(s.str.contains(p, case=False, regex=True, na=False).sum())
    return out


def agent_skill_codes(codebook: pd.DataFrame, cfg: dict) -> dict[str, set[str]]:
    """AI 에이전트 숙련 코드를 개발(`dev`)과 생성형 AI 활용(`use`)으로 나눠 돌려준다.

    설명 파일에서 `skill_code_v31`이 `v31_prefix`로 시작하는 숙련이 대상이다.
    숙련명이 `use_suffix`로 끝나면 `use`, 아니면 `dev`이다. 그 뒤 설정의 예외 목록대로 옮긴다.
    예외 목록에 대상이 아닌 코드가 있으면 `ValueError`를 낸다.
    """
    a = cfg["treatment"]["agent_skills"]
    target = codebook[codebook["skill_code_v31"].str.startswith(a["v31_prefix"])]
    is_use = target["skill_name"].str.endswith(a["use_suffix"])
    dev = set(target.loc[~is_use, "skill_code"])
    use = set(target.loc[is_use, "skill_code"])
    allowed = dev | use
    for key in ("move_to_dev", "move_to_use"):
        outside = sorted(set(a[key]) - allowed)
        if outside:
            raise ValueError(f"{key}에 AI 에이전트 숙련이 아닌 코드가 있습니다: {outside}")
    move_dev, move_use = set(a["move_to_dev"]), set(a["move_to_use"])
    dev = (dev - move_use) | move_dev
    use = (use - move_dev) | move_use
    return {"dev": dev, "use": use}


def skill_flags(long: pd.DataFrame, codes: dict[str, set[str]], posting_ids: pd.Series) -> pd.DataFrame:
    """공고마다 개발 숙련(`has_dev`)과 생성형 AI 활용 숙련(`has_use`)이 있는지 표시한다.

    숙련이 하나도 없는 공고는 두 열 모두 False이다. `posting_ids`의 모든 공고가 한 행씩 나온다.
    """
    ids = posting_ids.reset_index(drop=True)
    out = pd.DataFrame({"posting_id": ids})
    for name, col in (("dev", "has_dev"), ("use", "has_use")):
        hit = set(long.loc[long["skill_code"].isin(codes[name]), "posting_id"])
        out[col] = ids.isin(hit).to_numpy()
    return out


def build_treat(posting_ids: pd.Series, text: pd.Series, long: pd.DataFrame,
                codebook: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """공고마다 처치 표시 열을 만든다(모두 bool, 입력 순서 그대로, 새 RangeIndex).

    `posting_ids`와 `text`는 위치로 맞춘다. 본문이 결측인 공고는 키워드에 걸리지 않는다.
    """
    if len(posting_ids) != len(text):
        raise ValueError(f"공고 번호와 본문의 길이가 다릅니다: {len(posting_ids)} != {len(text)}")
    ids = posting_ids.reset_index(drop=True)
    txt = text.reset_index(drop=True).astype("string")
    kws = keyword_lists(cfg)
    out = pd.DataFrame({"posting_id": ids})
    out["kw_strict"] = flag_treatment(txt, kws["strict"], kws["exclude"]).astype(bool).to_numpy()
    out["kw_broad"] = flag_treatment(txt, kws["broad"], kws["exclude"]).astype(bool).to_numpy()
    flags = skill_flags(long, agent_skill_codes(codebook, cfg), ids)
    out["has_dev"] = flags["has_dev"].to_numpy()
    out["has_use"] = flags["has_use"].to_numpy()
    out["treat_code"] = out["has_dev"] | out["has_use"]
    out["treat_all"] = out["kw_strict"] | out["treat_code"]
    return out


def select_arms(treat: pd.DataFrame, definition: str, keyword_col: str = "kw_strict") -> pd.Series:
    """처치 공고는 1, 비교 공고는 0, 어느 쪽도 아니면 결측인 `Int8` 열을 돌려준다.

    비교 공고는 어떤 정의에서도 같다: 키워드에 걸리지 않고 두 종류의 AI 에이전트 숙련도 없는 공고.
    """
    if definition not in DEFINITIONS:
        raise ValueError(f"알 수 없는 처치 정의입니다: {definition!r} (가능: {DEFINITIONS})")
    kw = treat[keyword_col].to_numpy(dtype=bool)
    dev = treat["has_dev"].to_numpy(dtype=bool)
    use = treat["has_use"].to_numpy(dtype=bool)
    treated = {"all": kw | dev | use, "dev": dev, "use": use, "keyword": kw, "code": dev | use}[definition]
    comparison = ~kw & ~dev & ~use
    values = np.where(treated, 1, np.where(comparison, 0, -1))
    arr = pd.array(values, dtype="Int8")
    arr[values == -1] = pd.NA
    return pd.Series(arr, index=treat.index)
