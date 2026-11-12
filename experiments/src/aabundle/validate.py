"""설정(`config.yaml`)의 값이 쓸 수 있는 형태인지 일을 시작하기 전에 확인한다.

`paths.load_config`는 이 확인을 하지 않는다. 자료 점검 스크립트의 `scan`과 `build_tables`가 처음에 부른다.
값이 잘못되면 오래 훑은 뒤에야 멈추거나, 멈추지 않고 틀린 결과를 내기 때문이다(예: 따옴표 없이 적은 직종
코드 `0122`는 YAML에서 정수가 되어 어떤 공고와도 맞지 않는다).
"""
import datetime as dt
import re

from aabundle import treatment

# 보고서가 아는 기간 단위(`period.unit`)
PERIOD_UNITS = ("quarter", "half", "year")
KEYWORD_DICTIONARIES = ("strict", "broad")

_MISSING = object()


def _get(cfg: dict, key: str):
    d = cfg
    for part in key.split("."):
        if not isinstance(d, dict) or part not in d:
            return _MISSING
        d = d[part]
    return d


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _as_date(v):
    """날짜로 읽을 수 있으면 `datetime.date`를, 아니면 None을 돌려준다. 문자열은 `YYYY-MM-DD` 형식만 받는다."""
    if isinstance(v, dt.datetime):
        return v.date()
    if isinstance(v, dt.date):
        return v
    if isinstance(v, str):
        try:
            return dt.date.fromisoformat(v)
        except ValueError:
            return None
    return None


def _check_patterns(cfg: dict, key: str, required: bool, problems: list[str]) -> bool:
    """표현을 하나씩 확인한다. 문제가 없으면 True를 돌려준다."""
    n_before = len(problems)
    patterns = _get(cfg, key)
    if patterns is _MISSING:
        problems.append(f"`{key}` 항목이 없습니다")
        return False
    if not isinstance(patterns, list):
        problems.append(f"`{key}`는 목록이어야 합니다")
        return False
    if required and not patterns:
        problems.append(f"`{key}`가 비어 있습니다. 비어 있으면 처치 공고를 정할 수 없습니다")
    for i, p in enumerate(patterns):
        if not isinstance(p, str) or not p.strip():
            problems.append(f"`{key}`의 {i + 1}번째 표현이 비어 있거나 문자열이 아닙니다. "
                            "빈 표현은 본문이 있는 모든 공고에 걸립니다")
            continue
        try:
            re.compile(p, re.IGNORECASE)
        except re.error as e:
            problems.append(f"`{key}`의 {i + 1}번째 표현 {p!r}를 정규표현식으로 읽을 수 없습니다: {e}")
    return len(problems) == n_before


def _check_joined(patterns: list[str], key: str, what: str, problems: list[str]) -> bool:
    """처치 판별에서처럼 표현들을 `|`로 이어 붙인 식을 확인한다(`treatment.flag_treatment`, `pattern_counts`).
    문제가 없으면 True를 돌려준다.

    하나씩은 읽히지만 이어 붙이면 읽히지 않는 표현이 있기 때문이다. 예: `(?i)` 같은 전체 적용 표시는 식의 맨 앞이
    아니면 오류이다.
    """
    if not patterns:
        return True
    try:
        re.compile("|".join(patterns), re.IGNORECASE)
    except re.error as e:
        problems.append(f"`{key}`의 표현을 {what} `|`로 이어 붙인 식을 정규표현식으로 읽을 수 없습니다: {e}. "
                        "처치 판별에서는 표현을 이어 붙여 쓰므로, 표현 가운데에 `(?i)` 같은 표시를 두지 않아야 합니다")
        return False
    return True


def validate_config(cfg: dict) -> None:
    """설정을 확인하고, 잘못된 항목이 있으면 항목 이름과 문제를 모두 적은 ValueError를 낸다."""
    problems: list[str] = []

    def need(key: str):
        v = _get(cfg, key)
        if v is _MISSING:
            problems.append(f"`{key}` 항목이 없습니다")
        return v

    years = need("data.years")
    if years is not _MISSING and (not isinstance(years, list) or not years or not all(_is_int(y) for y in years)):
        problems.append(f"`data.years`는 정수 연도가 하나 이상 든 목록이어야 합니다: {years!r}")

    occs = need("occupations.keco4")
    if occs is not _MISSING:
        if not isinstance(occs, list) or not occs:
            problems.append(f"`occupations.keco4`는 세분류 코드가 하나 이상 든 목록이어야 합니다: {occs!r}")
        else:
            ints = [c for c in occs if _is_int(c)]
            if ints:
                problems.append(f"`occupations.keco4`에 정수로 읽힌 코드가 있습니다: {ints}. 세분류 코드는 따옴표로 "
                                "감싸 문자열로 적어야 합니다(따옴표가 없으면 YAML이 `0122`를 정수로 읽어 앞자리 0이 "
                                "사라지고 어떤 공고와도 맞지 않습니다)")
            bad = [c for c in occs if not _is_int(c) and not (isinstance(c, str) and re.fullmatch(r"\d{4}", c))]
            if bad:
                problems.append(f"`occupations.keco4`의 코드는 숫자 4자리 문자열이어야 합니다: {bad}")

    start, end = need("period.start"), need("period.end")
    d_start = _as_date(start) if start is not _MISSING else None
    d_end = _as_date(end) if end is not _MISSING else None
    if start is not _MISSING and d_start is None:
        problems.append(f"`period.start`를 날짜(YYYY-MM-DD)로 읽을 수 없습니다: {start!r}")
    if end is not _MISSING and d_end is None:
        problems.append(f"`period.end`를 날짜(YYYY-MM-DD)로 읽을 수 없습니다: {end!r}")
    if d_start is not None and d_end is not None and d_start > d_end:
        problems.append(f"`period.start`({d_start})가 `period.end`({d_end})보다 늦습니다")
    unit = need("period.unit")
    if unit is not _MISSING and unit not in PERIOD_UNITS:
        problems.append(f"`period.unit`은 {PERIOD_UNITS} 가운데 하나여야 합니다: {unit!r}")

    strict_ok = _check_patterns(cfg, "treatment.keywords.strict", True, problems)
    extra_ok = _check_patterns(cfg, "treatment.keywords.broad_extra", False, problems)
    exclude_ok = _check_patterns(cfg, "treatment.keywords.exclude", False, problems)
    # 하나씩 확인해 문제가 없는 목록만, 코드가 쓰는 대로 이어 붙여 다시 확인한다(`treatment.keyword_lists`)
    strict = _get(cfg, "treatment.keywords.strict")
    if strict_ok and _check_joined(strict, "treatment.keywords.strict", "모두", problems) and extra_ok:
        _check_joined(strict + _get(cfg, "treatment.keywords.broad_extra"), "treatment.keywords.broad_extra",
                      "엄격한 사전(`treatment.keywords.strict`) 뒤에", problems)
    if exclude_ok:
        _check_joined(_get(cfg, "treatment.keywords.exclude"), "treatment.keywords.exclude", "모두", problems)

    kd = need("treatment.keyword_dictionary")
    if kd is not _MISSING and kd not in KEYWORD_DICTIONARIES:
        problems.append(f"`treatment.keyword_dictionary`는 {KEYWORD_DICTIONARIES} 가운데 하나여야 합니다: {kd!r}")
    definition = need("treatment.definition")
    if definition is not _MISSING and definition not in treatment.DEFINITIONS:
        problems.append(f"`treatment.definition`은 {treatment.DEFINITIONS} 가운데 하나여야 합니다: {definition!r}")

    to_dev, to_use = need("treatment.agent_skills.move_to_dev"), need("treatment.agent_skills.move_to_use")
    lists_ok = True
    for key, v in (("treatment.agent_skills.move_to_dev", to_dev), ("treatment.agent_skills.move_to_use", to_use)):
        if v is not _MISSING and not isinstance(v, list):
            problems.append(f"`{key}`는 목록이어야 합니다: {v!r}")
            lists_ok = False
    if lists_ok and to_dev is not _MISSING and to_use is not _MISSING:
        both = sorted(set(to_dev) & set(to_use))
        if both:
            problems.append("`treatment.agent_skills.move_to_dev`와 `treatment.agent_skills.move_to_use`에 함께 든 "
                            f"코드가 있습니다: {both}. 한 코드는 한쪽으로만 옮길 수 있습니다")

    text_cols = need("treatment.text_columns")
    if text_cols is not _MISSING and (not isinstance(text_cols, list) or not text_cols
                                      or not all(isinstance(c, str) and c for c in text_cols)):
        problems.append(f"`treatment.text_columns`는 열 이름이 하나 이상 든 목록이어야 합니다: {text_cols!r}")

    edges = need("salary_band_edges")
    if edges is not _MISSING:
        if not isinstance(edges, list) or not all(_is_number(e) for e in edges):
            problems.append(f"`salary_band_edges`는 숫자 목록이어야 합니다: {edges!r}")
        elif any(a >= b for a, b in zip(edges, edges[1:])):
            problems.append(f"`salary_band_edges`는 앞의 값보다 큰 값이 이어지는 목록이어야 합니다: {edges!r}")

    n_folds = need("estimation.n_folds")
    if n_folds is not _MISSING and (not _is_int(n_folds) or n_folds < 2):
        problems.append(f"`estimation.n_folds`는 2 이상의 정수여야 합니다: {n_folds!r}")

    if problems:
        raise ValueError("설정(config.yaml)에 쓸 수 없는 값이 있습니다:\n- " + "\n- ".join(problems))
