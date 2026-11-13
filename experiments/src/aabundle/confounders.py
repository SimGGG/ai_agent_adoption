"""산업 소분류별 AI 노출 지수를 만들고 공고에 붙이는 함수(설계 문서 결정 30).

AI 노출 지수는 Felten, Raj, Seamans의 산업별 AI 노출 지수(AIIE) 가운데 언어 모형 판이며, 미국 산업분류(NAICS)
4자리 산업마다 값이 있다. 이를 한국표준산업분류(KSIC) 10차의 소분류(3자리)로 옮기려고 세 표를 차례로 잇는다.

1. KSIC 소분류 → 국제표준산업분류(ISIC Rev.4) 코드의 앞자리: 판단으로 만든 표(`ksic10_sub_to_isic4.csv`)
2. ISIC Rev.4 4자리 → NAICS 6자리: 미국 통계국 연계표(`2017_NAICS_to_ISIC_4.xlsx`)
3. NAICS 6자리의 앞 4자리 → AIIE

소분류의 값은 그 소분류에 이어진 서로 다른 NAICS 4자리 산업 가운데 AIIE 값이 있는 산업의 단순 평균이다.
"""
import re
import warnings
from pathlib import Path

import pandas as pd

AIIE_SHEET = "LM AIIE"
AIIE_CODE = "NAICS"
AIIE_DESCRIPTION = "NAICS Description"
AIIE_VALUE = "Language Modeling AIIE"
NAICS_ISIC_ISIC_HEADER = "ISIC 4.0"
KSIC_ISIC_COLUMNS = ["ksic_sub", "ksic_name", "isic", "note"]


def _require_file(path: Path, what: str) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{what} 파일이 없습니다: {path.name} (찾은 위치: {path.parent})")
    return path


def _clean(s: pd.Series) -> pd.Series:
    """값을 앞뒤 공백을 지운 문자열로 바꾼다. 빈 값은 결측으로 둔다."""
    t = s.astype("string").str.strip()
    return t.mask(t == "")


def load_naics_isic(path: Path) -> pd.DataFrame:
    """미국 통계국의 NAICS 2017 → ISIC Rev.4 연계표를 읽어 열 `naics6`, `isic4`(둘 다 문자열)로 돌려준다.

    첫 시트의 첫 줄이 머리글이다. NAICS 코드 열은 머리글에 `NAICS`가 들고 `Part of`로 시작하지 않으며 `TITLE`이
    없는 첫 열이다(실제 파일의 머리글은 줄바꿈이 든 `2017 NAICS US`이고, 그 앞에 `Part of NAICS US` 열이 있다).
    ISIC 코드 열은 머리글이 `ISIC 4.0`인 열이다. NAICS가 숫자 6자리인 행만 남기고(실제 파일 첫머리의 NAICS `0`
    행은 빠진다), ISIC 코드는 숫자로 저장되어 앞자리 0이 빠져 있으므로 4자리가 되게 앞에 0을 채운다.
    """
    path = _require_file(path, "NAICS-ISIC 연계표")
    with warnings.catch_warnings():
        # 실제 파일에는 openpyxl이 읽지 못하는 엑셀 확장 정보가 있어, 읽을 때마다 "그 정보를 버린다"는 경고가 난다.
        # 칸의 값과는 관계가 없는 경고이므로 이 경고만 끈다.
        warnings.filterwarnings("ignore", message="Unknown extension is not supported", category=UserWarning)
        raw = pd.read_excel(path, sheet_name=0, header=0, dtype=str)
    headers = [str(c) for c in raw.columns]
    norm = [" ".join(h.split()) for h in headers]
    naics_cols = [h for h, n in zip(headers, norm)
                  if "NAICS" in n and not n.startswith("Part of") and "TITLE" not in n.upper()]
    isic_cols = [h for h, n in zip(headers, norm) if n == NAICS_ISIC_ISIC_HEADER]
    if not naics_cols or not isic_cols:
        missing = ([] if naics_cols else ["NAICS 코드 열"]) + ([] if isic_cols else [f"`{NAICS_ISIC_ISIC_HEADER}` 열"])
        raise KeyError(f"{path.name}의 첫 시트에 {', '.join(missing)}이 없습니다. 머리글: {norm}")
    out = pd.DataFrame({"naics6": _clean(raw[naics_cols[0]]), "isic4": _clean(raw[isic_cols[0]])})
    out = out[out["naics6"].str.fullmatch(r"\d{6}").fillna(False).astype(bool)]
    bad = out["isic4"].isna() | ~out["isic4"].str.fullmatch(r"\d{1,4}").fillna(False).astype(bool)
    if bad.any():
        raise ValueError(f"{path.name}에 ISIC 코드가 숫자 1~4자리가 아닌 행이 있습니다: "
                         f"NAICS {out.loc[bad, 'naics6'].head(10).tolist()}")
    out["isic4"] = out["isic4"].str.zfill(4)
    return out.astype(object).reset_index(drop=True)


def load_aiie(path: Path, naics4_all: set[str]) -> pd.DataFrame:
    """AIIE 파일의 `LM AIIE` 시트를 읽어 NAICS 4자리 산업별 값(열 `naics4`, `aiie`)으로 편다.

    - 숫자 코드가 `000`으로 끝나면 3자리 수준의 항목이며, `naics4_all`(연계표에 나오는 모든 4자리 산업) 가운데 앞
      3자리가 같은 산업 모두에 값을 준다.
    - 그 밖의 숫자 코드는 앞 네 글자가 4자리 산업이다.
    - 숫자가 아닌 글자가 든 코드(예: `3250A1`)는 여러 산업을 묶은 항목이며, 설명에 나오는 4자리 숫자 모두에 값을
      준다(예: `Chemical Manufacturing (3251, 3252, 3253, and 3259 only)`).

    한 산업이 4자리 항목(또는 묶은 항목)과 3자리 항목 양쪽에서 값을 받으면 앞의 것을 쓴다. 4자리 항목이나 묶은
    항목끼리 같은 산업에 값을 주면 그 산업 코드를 적은 `ValueError`를 낸다.
    """
    path = _require_file(path, "AIIE")
    try:
        raw = pd.read_excel(path, sheet_name=AIIE_SHEET, dtype={AIIE_CODE: str, AIIE_DESCRIPTION: str})
    except ValueError as e:
        if AIIE_SHEET in str(e):
            raise KeyError(f"{path.name}에 `{AIIE_SHEET}` 시트가 없습니다") from e
        raise
    need = [AIIE_CODE, AIIE_DESCRIPTION, AIIE_VALUE]
    missing = [c for c in need if c not in raw.columns]
    if missing:
        raise KeyError(f"{path.name}의 `{AIIE_SHEET}` 시트에 열이 없습니다: {missing}")
    codes = _clean(raw[AIIE_CODE])
    values = pd.to_numeric(raw[AIIE_VALUE], errors="coerce")
    bad = codes.isna() | values.isna()
    if bad.any():
        raise ValueError(f"{path.name}의 `{AIIE_SHEET}` 시트에 코드나 값이 빈 행이 있습니다: {bad.sum()}행")

    specific: list[tuple[str, float, str]] = []      # (4자리 산업, 값, 원래 코드)
    three_digit: list[tuple[str, float]] = []
    for code, desc, value in zip(codes, raw[AIIE_DESCRIPTION].astype("string").fillna(""), values):
        if code.isdigit():
            if code.endswith("000"):
                three_digit.append((code[:3], float(value)))
            else:
                specific.append((code[:4], float(value), code))
        else:
            members = re.findall(r"(?<!\d)\d{4}(?!\d)", str(desc))
            if not members:
                raise ValueError(f"{path.name}의 묶은 항목 `{code}`의 설명에 4자리 산업 코드가 없습니다: {desc!r}")
            specific += [(m, float(value), code) for m in members]

    spec = pd.DataFrame(specific, columns=["naics4", "aiie", "code"])
    dup = spec[spec["naics4"].duplicated(keep=False)]
    if not dup.empty:
        pairs = dup.groupby("naics4")["code"].apply(lambda c: "/".join(c)).to_dict()
        raise ValueError(f"{path.name}에서 여러 항목이 같은 4자리 산업에 값을 줍니다: {pairs}")
    values_by_naics4 = dict(zip(spec["naics4"], spec["aiie"]))
    for prefix, value in three_digit:
        for naics4 in sorted(naics4_all):
            if naics4.startswith(prefix) and naics4 not in values_by_naics4:
                values_by_naics4[naics4] = value
    out = pd.DataFrame({"naics4": list(values_by_naics4), "aiie": list(values_by_naics4.values())})
    return out.sort_values("naics4").astype({"naics4": object, "aiie": "float64"}).reset_index(drop=True)


def load_ksic_isic(path: Path) -> pd.DataFrame:
    """판단으로 만든 KSIC 소분류 → ISIC Rev.4 표를 읽어 한 줄에 앞자리 하나가 되게 편다(열 `ksic_sub`, `isic_prefix`).

    표의 머리글은 `ksic_sub,ksic_name,isic,note`이고 `isic` 열은 ISIC 코드의 앞자리(2~4자리)를 `;`로 이은 것이다.
    `isic`이 빈 소분류(ISIC에 이을 곳이 없는 소분류)는 줄이 없다. `ksic_sub`가 겹치거나 숫자 3자리가 아니면
    `ValueError`를 낸다.
    """
    path = _require_file(path, "KSIC-ISIC 연계표")
    table = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if list(table.columns) != KSIC_ISIC_COLUMNS:
        raise KeyError(f"{path.name}의 머리글이 {KSIC_ISIC_COLUMNS}가 아닙니다: {list(table.columns)}")
    sub = table["ksic_sub"].str.strip()
    bad = sorted(set(sub[~sub.str.fullmatch(r"\d{3}")]))
    if bad:
        raise ValueError(f"{path.name}의 `ksic_sub`에 숫자 3자리가 아닌 값이 있습니다: {bad}")
    dup = sorted(set(sub[sub.duplicated()]))
    if dup:
        raise ValueError(f"{path.name}의 `ksic_sub`가 겹칩니다: {dup}")
    long = pd.DataFrame({"ksic_sub": sub, "isic_prefix": table["isic"].str.split(";")}).explode("isic_prefix")
    long["isic_prefix"] = long["isic_prefix"].str.strip()
    long = long[long["isic_prefix"] != ""]
    bad = long[~long["isic_prefix"].str.fullmatch(r"\d{2,4}")]
    if not bad.empty:
        raise ValueError(f"{path.name}의 `isic`에 숫자 2~4자리가 아닌 앞자리가 있습니다: "
                         f"{list(zip(bad['ksic_sub'], bad['isic_prefix']))}")
    return long.astype(object).reset_index(drop=True)


def aiie_by_ksic_sub(aiie: pd.DataFrame, naics_isic: pd.DataFrame, ksic_isic: pd.DataFrame) -> pd.DataFrame:
    """세 표를 이어 소분류별 AIIE(열 `ksic_sub`, `aiie`, `n_naics4`)를 만든다.

    소분류마다 그 소분류의 앞자리로 시작하는 ISIC 4자리 코드, 그 ISIC 코드에 이어진 NAICS 6자리의 앞 4자리 가운데
    AIIE 값이 있는 서로 다른 산업을 모아 단순 평균한다. `n_naics4`는 평균에 쓴 산업 수이다. 이어진 산업이 없는
    소분류는 행이 없다. 연계표의 어느 ISIC 코드와도 맞지 않는 앞자리가 있으면 표를 잘못 적은 것이므로 그 소분류와
    앞자리를 적은 `ValueError`를 낸다.
    """
    isic_codes = sorted(set(naics_isic["isic4"]))
    pairs = []          # (ksic_sub, isic4)
    unmatched = []
    for sub, prefix in zip(ksic_isic["ksic_sub"], ksic_isic["isic_prefix"]):
        hits = [c for c in isic_codes if c.startswith(prefix)]
        if not hits:
            unmatched.append(f"{sub}: {prefix}")
        pairs += [(sub, c) for c in hits]
    if unmatched:
        raise ValueError("NAICS-ISIC 연계표의 어느 ISIC 코드와도 맞지 않는 앞자리가 있습니다(소분류: 앞자리): "
                         + ", ".join(unmatched))
    linked = pd.DataFrame(pairs, columns=["ksic_sub", "isic4"]).merge(naics_isic[["naics6", "isic4"]], on="isic4")
    linked["naics4"] = linked["naics6"].str[:4]
    linked = linked[["ksic_sub", "naics4"]].drop_duplicates().merge(aiie[["naics4", "aiie"]], on="naics4")
    out = linked.groupby("ksic_sub").agg(aiie=("aiie", "mean"), n_naics4=("naics4", "size")).reset_index()
    return out.astype({"ksic_sub": object, "aiie": "float64", "n_naics4": "int64"})


def attach_aiie(ksic_sub: pd.Series, table: pd.DataFrame) -> pd.Series:
    """공고의 소분류(`ksic_sub`)로 `aiie_by_ksic_sub`의 값을 찾는다. 없으면 결측이다. 입력과 길이와 순서가 같다."""
    if not table["ksic_sub"].is_unique:
        raise ValueError("소분류별 AIIE 표의 `ksic_sub`가 겹칩니다")
    lookup = dict(zip(table["ksic_sub"], table["aiie"]))
    values = [lookup.get(s) if isinstance(s, str) else None for s in ksic_sub.astype(object)]
    return pd.Series(values, index=ksic_sub.index, dtype="float64", name="aiie")
