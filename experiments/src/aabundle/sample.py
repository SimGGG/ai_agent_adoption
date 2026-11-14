"""공고 한 건을 한 행으로 하는 기본 자료를 만들고, 정해진 순서로 제외 규칙을 적용한다."""
import numpy as np
import pandas as pd

from aabundle import classif

RAW_COLUMNS = ["posting_id", "source", "post_start", "post_end", "bzno", "ksic10", "firm_emp", "keco_2025",
               "salary_annual_min", "emp_type", "edu_level", "career_type", "career_years_min",
               "region", "dup_count", "ai_peripheral"]

EXCLUSION_STEPS = ("in_period", "in_occ", "has_bzno", "has_skill")

_KED_COLUMNS = ["ked_estab_year", "ked_size", "ked_ipo", "ked_venture"]

# 게시일과 마감일이 모두 없어, 같은 연도 파일에서 행 순서로 가장 가까운 앞 공고의 날짜를 게시일로 쓴 공고의
# `date_source` 값(설계 문서 결정 38)
FILL_SOURCE = "fill"
FILL_COLUMNS = ["fill_date", "fill_gap_days"]

BASE_COLUMNS = ["posting_id", "source", "year", "month", "quarter", "period", "date_source", "fill_gap_days", "bzno", "ksic5",
                "ksic_sub", "firm_emp_log", "keco4", "keco2", "career_type", "career_years_min", "edu_level",
                "salary_annual_min", "salary_band", "emp_type", "sido", "dup_count", "ai_peripheral",
                "ked_matched", "firm_age", "ked_size", "ked_ipo", "ked_venture",
                "in_period", "in_occ", "has_bzno"]


def _build_sido_table() -> dict:
    table = {}
    for short in ["서울"]:
        table.update({short: short, short + "특별시": short})
    for short in ["부산", "대구", "인천", "광주", "대전", "울산"]:
        table.update({short: short, short + "광역시": short})
    table.update({"세종": "세종", "세종특별자치시": "세종"})
    table.update({"경기": "경기", "경기도": "경기"})
    table.update({"강원": "강원", "강원도": "강원", "강원특별자치도": "강원"})
    for short, long in [("충북", "충청북도"), ("충남", "충청남도"), ("전남", "전라남도"),
                        ("경북", "경상북도"), ("경남", "경상남도")]:
        table.update({short: short, long: short})
    table.update({"전북": "전북", "전라북도": "전북", "전북특별자치도": "전북"})
    table.update({"제주": "제주", "제주도": "제주", "제주특별자치도": "제주"})
    return table


_SIDO = _build_sido_table()
# 전남과 광주를 합친 통합특별시. `to_sido`는 `전남광주`로 돌려주고, 광주, 전남과 한 범주로 합치는 일은
# 설정 `region.merge_sido`에 따라 `merge_sido`가 한다(설계 문서 결정 29).
_SIDO.update({"전남광주": "전남광주", "전남광주통합특별시": "전남광주"})

SIDO_OTHER = "기타"


def region_first_word(region: pd.Series) -> pd.Series:
    """근무지 문자열의 첫 낱말을 돌려준다. 결측은 결측으로 둔다.

    첫 쉼표 앞의 글자에서 처음 나오는 공백, `>`, `&` 앞까지를 첫 낱말로 하고(`&`는 `&gt;` 같은 HTML
    문자 표기의 시작이다), 끝에 붙은 `전지역`이나 `전체`를 뗀다.
    """
    r = region.astype("string")
    word = r.str.extract(r"^\s*([^,\s>&]*)")[0].astype("string")
    return word.str.replace(r"(?:전지역|전체)$", "", regex=True)


def to_sido(region: pd.Series) -> pd.Series:
    """근무지 문자열을 시도 이름으로 바꾼다.

    `region_first_word`로 얻은 첫 낱말을 시도 표에서 찾는다. 표에 없고 `시`로 끝나면 끝의 `시`를 뗀
    낱말을 다시 찾는다(`서울시` → `서울`). 그래도 없으면 `기타`, 원래 결측이면 결측이다.
    시도 이름으로 시작하기만 하는 낱말(예: `서울대입구`)은 시도로 보지 않는다.
    """
    r = region.astype("string")
    word = region_first_word(r)
    out = word.map(_SIDO).astype("string")
    without_si = word.str.replace(r"시$", "", regex=True).map(_SIDO).astype("string")
    out = out.fillna(without_si.where(word.str.endswith("시").fillna(False)))
    return out.fillna(SIDO_OTHER).where(r.notna(), pd.NA)


def merge_sido(sido: pd.Series, merge: dict[str, list[str]]) -> pd.Series:
    """시도 이름을 묶음 이름으로 바꾼다. `merge`는 {묶음 이름: [시도 이름, ...]}이다.

    묶음에 들지 않는 시도 이름과 결측은 그대로 둔다.
    """
    s = sido.astype("string")
    to_group = {member: name for name, members in merge.items() for member in members}
    return s.map(to_group).astype("string").fillna(s)


def _require_columns(raw: pd.DataFrame, columns: list[str]) -> None:
    absent = [c for c in columns if c not in raw.columns]
    if absent:
        raise KeyError(f"게시일을 정할 열이 공고 자료에 없습니다: {', '.join(absent)}")


def _parse_dates(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    """열 하나를 날짜로 읽어 (날짜, 값이 있는지)를 돌려준다. `posting_date`와 `unusable_dates`가 함께 쓴다.

    문자열은 앞뒤 공백을 지운 뒤 ISO 8601 형식으로 읽고, 지운 뒤 빈 문자열은 값이 없는 것으로 본다.
    날짜는 시각을 버리고 `datetime64[ns]`로 맞추며, 읽을 수 없는 값과 그 범위(1677~2262년) 밖의 날짜는 결측이다.
    """
    s = values
    if pd.api.types.infer_dtype(s, skipna=True) in ("string", "mixed"):
        s = s.astype(object).map(lambda v: (v.strip() or None) if isinstance(v, str) else v)
    present = s.notna()
    d = pd.to_datetime(s, errors="coerce", format="ISO8601")
    d = d.where((d >= pd.Timestamp.min) & (d <= pd.Timestamp.max)).astype("datetime64[ns]").dt.normalize()
    return d, present


def posting_date(raw: pd.DataFrame, columns: list[str]) -> tuple[pd.Series, pd.Series]:
    """공고마다 게시일로 쓸 날짜와 그 날짜를 가져온 열 이름을 돌려준다(설계 문서 결정 25).

    `columns`의 순서대로 처음으로 날짜가 있는 열의 날짜를 쓴다. 시각은 버리고 날짜만 남긴다.
    날짜로 쓰는 값: 날짜 객체와 날짜·시각 값, 그리고 앞뒤 공백을 지운 뒤 ISO 8601 형식으로 읽히는 문자열
    (예: `2025-03-02`, `2025-03-02 13:00:00`, `20250302`). 그 밖의 값과 `datetime64[ns]`로 나타낼 수 있는
    범위(1677~2262년) 밖의 날짜(예: 0001-01-01, 9999-12-31)는 날짜가 없는 것으로 본다. 값은 있지만 이렇게
    날짜로 쓰지 못한 칸은 `unusable_dates`로 셀 수 있다. 어느 열에도 날짜가 없으면 날짜와 열 이름 모두 결측이다.
    `columns`의 열이 `raw`에 없으면 없는 열 이름을 적은 KeyError를 낸다.
    """
    _require_columns(raw, columns)
    date = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
    source = pd.Series(pd.NA, index=raw.index, dtype="string")
    for c in columns:
        d, _ = _parse_dates(raw[c])
        take = date.isna() & d.notna()
        date = date.where(~take, d)
        source = source.where(~take, c)
    return date, source


def unusable_dates(raw: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """`columns`의 열마다, 값은 있지만 `posting_date`가 날짜로 쓰지 못하는 칸을 참으로 표시한 표를 돌려준다.

    값이 있다는 것은 결측이 아니고 빈 문자열이나 공백뿐인 문자열도 아니라는 뜻이다. 날짜로 쓰지 못하는 값은
    읽을 수 없는 글자와 `posting_date`가 받는 범위 밖의 날짜이다. 색인은 `raw`와 같다.
    """
    _require_columns(raw, columns)
    out = {}
    for c in columns:
        d, present = _parse_dates(raw[c])
        out[c] = (present & d.isna()).to_numpy(dtype=bool)
    return pd.DataFrame(out, index=raw.index)


def fill_posting_dates(date: pd.Series) -> tuple[pd.Series, pd.Series]:
    """날짜가 없는 공고에 채울 날짜와 앞뒤 날짜의 간격(일)을 돌려준다(설계 문서 결정 38).

    `date`는 연도 파일 하나의 모든 공고의 게시일(`posting_date`의 첫 값)을 파일의 행 순서대로 담은 열이다.
    날짜가 없는 공고의 `fill_date`는 행 순서로 가장 가까운 앞 공고 가운데 날짜가 있는 공고의 날짜이고, 앞에 날짜가
    있는 공고가 없으면 결측이다. 날짜가 있는 공고의 `fill_date`는 결측이다. `fill_gap_days`는 `fill_date`가 있는
    공고에서 그 날짜와, 행 순서로 가장 가까운 뒤 공고 가운데 날짜가 있는 공고의 날짜 사이의 일수(절댓값)이다.
    뒤에 날짜가 있는 공고가 없거나 `fill_date`가 없으면 결측이다. 두 열 모두 색인이 `date`와 같다.
    """
    d = pd.Series(date, copy=True).astype("datetime64[ns]")
    before = d.ffill()
    after = d.bfill()
    fill_date = before.where(d.isna())
    days = (after - fill_date).abs().dt.days
    gap = days.astype("Float64").astype("Int64")
    return fill_date, gap


def period_name(year: pd.Series, groups: list[dict]) -> pd.Series:
    """연도마다 그 연도가 드는 기간의 이름을 돌려준다(설계 문서 결정 24). 어느 기간에도 들지 않으면 결측."""
    to_name = {int(y): g["name"] for g in groups for y in g["years"]}
    return year.astype("Int64").map(to_name).astype("string")


def _fmt_edge(e: float) -> str:
    return str(int(e)) if float(e).is_integer() else str(e)


def salary_band(salary: pd.Series, edges: list[float]) -> pd.Series:
    """연봉(원) 구간 이름을 돌려준다. 결측은 `미기재`, 구분 경계가 없으면 값이 있는 공고는 `기재`."""
    s = pd.to_numeric(salary).astype("float64")
    missing = s.isna()
    if not edges:
        return pd.Series(np.where(missing, "미기재", "기재"), index=salary.index, dtype="string")
    e = [_fmt_edge(x) for x in edges]
    labels = [f"~{e[0]}"] + [f"{e[i]}~{e[i + 1]}" for i in range(len(e) - 1)] + [f"{e[-1]}~"]
    band = pd.cut(s, bins=[-np.inf, *edges, np.inf], right=False, labels=labels)
    return band.astype("string").fillna("미기재")


def _check_fill(fill: pd.DataFrame, n: int) -> None:
    absent = [c for c in FILL_COLUMNS if c not in fill.columns]
    if absent:
        raise ValueError(f"채울 날짜 자료(fill)에 없는 열이 있습니다: {', '.join(absent)}")
    if len(fill) != n:
        raise ValueError(f"채울 날짜 자료(fill)의 행 수({len(fill)})가 공고 자료의 행 수({n})와 다릅니다")


def build_posting_base(raw: pd.DataFrame, ked_firm: pd.DataFrame, ksic_valid: set[str], cfg: dict,
                       fill: pd.DataFrame | None = None) -> pd.DataFrame:
    """공고 원자료와 KED 기업 자료를 합쳐 공고 한 건당 한 행의 기본 자료를 만든다.

    입력과 행 수와 순서가 같다. 결합 뒤 행 수가 달라지면 RuntimeError를 낸다.

    `fill`은 None이거나 열 `fill_date`, `fill_gap_days`(`fill_posting_dates`의 결과)가 있고 `raw`와 행 수가 같은
    표이며, `raw`와 위치로 맞춘다. 행 수가 다르거나 열이 없으면 ValueError를 낸다. 게시일(`posting_date`)이 없고
    `fill_date`가 있는 공고는 `fill_date`를 게시일로 쓰고, `date_source`는 `fill`, `fill_gap_days`는 `fill`의 값이다.
    그 밖의 공고와 `fill`이 None일 때의 `fill_gap_days`는 결측이다. 연도, 월, 분기, 기간, 분석 기간 여부, 업력은
    최종 게시일로 계산한다.
    """
    raw = raw.reset_index(drop=True)
    n = len(raw)

    # 게시일은 설정 `period.date_columns`의 순서대로 정한다(`post_start`, 없으면 `post_end`).
    # 시각을 버린 날짜로 비교하므로, 끝 날의 자정 뒤에 게시된 공고도 분석 기간 안이다.
    days, date_source = posting_date(raw, cfg["period"]["date_columns"])
    fill_gap = pd.Series(pd.NA, index=raw.index, dtype="Int64")
    if fill is not None:
        _check_fill(fill, n)
        fill_date = pd.Series(pd.to_datetime(fill["fill_date"]).to_numpy(), index=raw.index).astype("datetime64[ns]")
        gap = pd.Series(fill["fill_gap_days"].astype("Int64").to_numpy(), index=raw.index, dtype="Int64")
        take = days.isna() & fill_date.notna()
        days = days.where(~take, fill_date)
        date_source = date_source.where(~take, FILL_SOURCE)
        fill_gap = gap.where(take, pd.NA)
    year = days.dt.year.astype("Int64")
    month = days.dt.month.astype("Int64")
    quarter = year.astype("string") + "Q" + days.dt.quarter.astype("Int64").astype("string")
    start = pd.Timestamp(cfg["period"]["start"])
    end = pd.Timestamp(cfg["period"]["end"])
    in_period = ((days >= start) & (days <= end)).to_numpy(dtype=bool)

    bzno = raw["bzno"].astype("string")
    left = pd.DataFrame({"_key": bzno})
    right = ked_firm[["bzno", *_KED_COLUMNS]].copy()
    right["bzno"] = right["bzno"].astype("string")
    right = right[right["bzno"].notna()]  # 사업자번호가 없는 KED 행은 어떤 공고와도 결합하지 않는다
    merged = left.merge(right, how="left", left_on="_key", right_on="bzno", indicator=True)
    if len(merged) != n:
        raise RuntimeError(f"KED 결합 뒤 행 수가 달라졌습니다: 결합 전 {n}, 결합 뒤 {len(merged)}")
    ked_matched = (merged["_merge"] == "both").to_numpy(dtype=bool)
    estab = pd.to_numeric(merged["ked_estab_year"]).astype("Int64")
    age = year - estab
    age = age.where((age >= 0).fillna(False).to_numpy(dtype=bool))

    keco4 = raw["keco_2025"].astype("string")
    out = pd.DataFrame({
        "posting_id": raw["posting_id"],
        "source": raw["source"],
        "year": year,
        "month": month,
        "quarter": quarter,
        "period": period_name(year, cfg["period"]["groups"]),
        "date_source": date_source,
        "fill_gap_days": fill_gap,
        "bzno": bzno,
        "ksic5": classif.normalize_ksic(raw["ksic10"]),
        "ksic_sub": classif.ksic_sub(raw["ksic10"], ksic_valid, cfg["industry"]["not_in_table"]),
        "firm_emp_log": np.log1p(pd.to_numeric(raw["firm_emp"]).astype("float64")),
        "keco4": keco4,
        "keco2": classif.keco_mid(keco4),
        "career_type": raw["career_type"],
        "career_years_min": raw["career_years_min"],
        "edu_level": raw["edu_level"],
        "salary_annual_min": raw["salary_annual_min"],
        "salary_band": salary_band(raw["salary_annual_min"], cfg["salary_band_edges"]),
        "emp_type": raw["emp_type"],
        "sido": merge_sido(to_sido(raw["region"]), cfg["region"]["merge_sido"]),
        "dup_count": raw["dup_count"],
        "ai_peripheral": raw["ai_peripheral"],
        "ked_matched": ked_matched,
        "firm_age": age,
        "ked_size": merged["ked_size"].astype("string"),
        "ked_ipo": merged["ked_ipo"].astype("string"),
        "ked_venture": merged["ked_venture"].astype("string"),
        "in_period": in_period,
        "in_occ": keco4.isin(cfg["occupations"]["keco4"]).to_numpy(dtype=bool),
        "has_bzno": bzno.str.fullmatch(r"\d{10}").fillna(False).to_numpy(dtype=bool),
    })
    return out[BASE_COLUMNS]


def apply_exclusions(base: pd.DataFrame, cfg: dict, has_skill: pd.Series | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """정해진 순서로 공고를 빼고, 남은 공고와 연도·단계별 건수 표를 돌려준다.

    건수는 앞 단계를 통과한 공고만 대상으로 센다. 건너뛴 단계는 표에 행이 없다.
    게시일을 정할 수 없는 공고는 `year`가 결측이며, 건수 표에도 `year`가 결측인 행으로 나온다.
    `has_skill`은 `base`와 위치로 맞춘다(길이가 같아야 한다).

    설정 `period.fill.max_gap_days`에 값이 있으면, 날짜를 채운 공고(`date_source` = `fill`)는 앞뒤 날짜의 간격
    (`fill_gap_days`)이 있고 그 값 이하일 때만 분석 기간 단계를 통과한다. 이때 `base`에 `date_source`와
    `fill_gap_days` 열이 있어야 한다. 값이 없으면(null) 이 규칙은 적용하지 않는다.
    """
    if has_skill is not None and len(has_skill) != len(base):
        raise ValueError(f"has_skill의 길이({len(has_skill)})가 기본 자료의 행 수({len(base)})와 다릅니다")
    in_period = base["in_period"].to_numpy(dtype=bool)
    max_gap = cfg["period"]["fill"]["max_gap_days"]
    if max_gap is not None:
        absent = [c for c in ("date_source", "fill_gap_days") if c not in base.columns]
        if absent:
            raise KeyError(f"`period.fill.max_gap_days`를 적용할 열이 기본 자료에 없습니다: {', '.join(absent)}")
        filled = (base["date_source"].astype("string") == FILL_SOURCE).fillna(False).to_numpy(dtype=bool)
        near = base["fill_gap_days"].astype("Float64").le(max_gap).fillna(False).to_numpy(dtype=bool)
        in_period = in_period & (~filled | near)
    conditions = {
        "in_period": in_period,
        "in_occ": base["in_occ"].to_numpy(dtype=bool),
    }
    if cfg["sample"]["require_bzno"]:
        conditions["has_bzno"] = base["has_bzno"].to_numpy(dtype=bool)
    if has_skill is not None:
        conditions["has_skill"] = np.asarray(has_skill, dtype=bool)

    alive = np.ones(len(base), dtype=bool)
    year = base["year"].reset_index(drop=True)
    pieces = []
    for step in EXCLUSION_STEPS:
        if step not in conditions:
            continue
        after = alive & conditions[step]
        g = pd.DataFrame({"year": year, "n_before": alive.astype("int64"), "n_after": after.astype("int64")}) \
            .groupby("year", dropna=False, as_index=False)[["n_before", "n_after"]].sum()
        g.insert(1, "step", step)
        g["n_dropped"] = g["n_before"] - g["n_after"]
        pieces.append(g[["year", "step", "n_before", "n_dropped", "n_after"]])
        alive = after
    if pieces:
        counts = pd.concat(pieces, ignore_index=True)
    else:
        counts = pd.DataFrame(columns=["year", "step", "n_before", "n_dropped", "n_after"])
    return base[alive], counts
