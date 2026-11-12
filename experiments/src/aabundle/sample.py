"""공고 한 건을 한 행으로 하는 기본 자료를 만들고, 정해진 순서로 제외 규칙을 적용한다."""
import numpy as np
import pandas as pd

from aabundle import classif

RAW_COLUMNS = ["posting_id", "source", "post_start", "bzno", "ksic10", "firm_emp", "keco_2025",
               "salary_annual_min", "emp_type", "edu_level", "career_type", "career_years_min",
               "region", "dup_count", "ai_peripheral"]

EXCLUSION_STEPS = ("in_period", "in_occ", "has_bzno", "has_skill")

_KED_COLUMNS = ["ked_estab_year", "ked_size", "ked_ipo", "ked_venture"]

BASE_COLUMNS = ["posting_id", "source", "year", "month", "quarter", "bzno", "ksic5", "ksic_sub",
                "firm_emp_log", "keco4", "keco2", "career_type", "career_years_min", "edu_level",
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
# 전남과 광주를 합친 통합특별시. 광주, 전남과 합칠지 저자가 정할 수 있도록 따로 둔다(설계 문서 6.7절 16번).
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


def build_posting_base(raw: pd.DataFrame, ked_firm: pd.DataFrame, ksic_valid: set[str], cfg: dict) -> pd.DataFrame:
    """공고 원자료와 KED 기업 자료를 합쳐 공고 한 건당 한 행의 기본 자료를 만든다.

    입력과 행 수와 순서가 같다. 결합 뒤 행 수가 달라지면 RuntimeError를 낸다.
    """
    raw = raw.reset_index(drop=True)
    n = len(raw)

    dts = pd.to_datetime(raw["post_start"])
    year = dts.dt.year.astype("Int64")
    month = dts.dt.month.astype("Int64")
    quarter = year.astype("string") + "Q" + dts.dt.quarter.astype("Int64").astype("string")
    start = pd.Timestamp(cfg["period"]["start"])
    end = pd.Timestamp(cfg["period"]["end"])
    in_period = ((dts >= start) & (dts <= end)).to_numpy(dtype=bool)

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
        "bzno": bzno,
        "ksic5": classif.normalize_ksic(raw["ksic10"]),
        "ksic_sub": classif.ksic_sub(raw["ksic10"], ksic_valid),
        "firm_emp_log": np.log1p(pd.to_numeric(raw["firm_emp"]).astype("float64")),
        "keco4": keco4,
        "keco2": classif.keco_mid(keco4),
        "career_type": raw["career_type"],
        "career_years_min": raw["career_years_min"],
        "edu_level": raw["edu_level"],
        "salary_annual_min": raw["salary_annual_min"],
        "salary_band": salary_band(raw["salary_annual_min"], cfg["salary_band_edges"]),
        "emp_type": raw["emp_type"],
        "sido": to_sido(raw["region"]),
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
    `post_start`가 결측인 공고는 `year`가 결측이며, 건수 표에도 `year`가 결측인 행으로 나온다.
    `has_skill`은 `base`와 위치로 맞춘다(길이가 같아야 한다).
    """
    if has_skill is not None and len(has_skill) != len(base):
        raise ValueError(f"has_skill의 길이({len(has_skill)})가 기본 자료의 행 수({len(base)})와 다릅니다")
    conditions = {
        "in_period": base["in_period"].to_numpy(dtype=bool),
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
