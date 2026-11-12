"""공고의 숙련 열 여섯 벌을 공고 × 숙련 긴 표로 바꾸는 함수.

원자료의 숙련 코드는 일부가 옛 형식이라 설명 파일(코드북)의 `skill_code`에 없다.
그래서 코드가 없으면 숙련명으로 최종판 코드를 찾는다. 행 단위 반복문 없이
`str.split`, `explode`, 병합으로만 처리한다(실제 자료에서 1,200번쯤 불린다).
"""
from pathlib import Path

import numpy as np
import pandas as pd

SKILL_COLUMNS = ["occupation", "transversal", "qualification", "competency", "soft", "physical"]

_CODEBOOK_SOURCE_COLUMNS = {
    "skill_code_v4": "skill_code",
    "skill_code_v31": "skill_code_v31",
    "skill_name": "skill_name",
    "domain_v4": "domain",
    "sub_v4": "sub",
    "detail_v4": "detail",
}
_LONG_COLUMNS = ["posting_id", "skill_code", "src"]


def raw_skill_columns() -> list[str]:
    """원자료의 숙련 열 12개(`skills_{c}`와 `skills_{c}_name`)를 돌려준다."""
    cols = []
    for c in SKILL_COLUMNS:
        cols += [f"skills_{c}", f"skills_{c}_name"]
    return cols


def load_codebook(path: Path) -> pd.DataFrame:
    """숙련 설명 파일을 읽어 열 `skill_code, skill_code_v31, skill_name, domain, sub, detail`로 돌려준다.

    `skill_code`나 `skill_name`이 겹치거나 필요한 열이 없으면 이유를 적은 예외를 낸다.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"숙련 설명 파일이 없습니다: {path}")
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    missing = [c for c in _CODEBOOK_SOURCE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"{path.name}에 없는 열: {missing}")
    df = df[list(_CODEBOOK_SOURCE_COLUMNS)].rename(columns=_CODEBOOK_SOURCE_COLUMNS)
    for col in ("skill_code", "skill_name"):
        dup = df.loc[df[col].duplicated(keep=False), col].unique().tolist()
        if dup:
            raise ValueError(f"숙련 설명 파일에서 {col}이 겹칩니다: {dup[:5]}")
    return df.reset_index(drop=True)


def _tokens(raw: pd.DataFrame, column: str) -> pd.DataFrame:
    """한 열을 `|`로 나눠 공백을 떼고 빈 토큰을 버린 표 `rid, pos, tok, n_raw`를 돌려준다.

    `rid`는 입력 행의 위치(0부터), `pos`는 빈 토큰을 버리기 전 그 칸 안에서의 순서(0부터)이다. 그래서 숙련 코드와
    숙련명을 원래 위치로 짝지을 수 있다. `n_raw`는 그 칸을 `|`로 나눈 조각 수(빈 조각 포함)이다. 빈 토큰만 있는
    칸은 행이 없다.
    """
    values = raw[column]
    s = pd.Series(values.to_numpy(dtype=object), dtype=object)
    s = s.where(values.notna().to_numpy(), "")
    s = s.astype(str).str.split("|").explode().str.strip()
    rid = s.index.to_numpy()
    pos = s.groupby(level=0, sort=False).cumcount().to_numpy()
    n_raw = s.groupby(level=0, sort=False).transform("size").to_numpy()
    keep = (s != "").to_numpy()
    return pd.DataFrame({"rid": rid[keep], "pos": pos[keep], "tok": s.to_numpy()[keep], "n_raw": n_raw[keep]})


def _empty_long(raw: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "posting_id": pd.Series([], dtype=raw["posting_id"].dtype if "posting_id" in raw else object),
        "skill_code": pd.Series([], dtype=object),
        "src": pd.Series([], dtype=object),
    })


def explode_skills(raw: pd.DataFrame, codebook: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """여섯 숙련 열을 공고 × 숙련 긴 표로 바꾼다.

    한 숙련에 대해 ① 코드가 설명 파일의 `skill_code`에 있으면 그 코드를 쓰고,
    ② 없으면 숙련명이 `skill_name`에 있을 때 그 행의 `skill_code`를 쓰고,
    ③ 둘 다 아니면 버리고 `n_unmatched`에 센다. 코드와 숙련명은 칸을 `|`로 나눈 원래 위치로 짝짓는다
    (빈 토큰을 버리기 전의 위치). 한 공고의 한 열에서 코드 칸과 숙련명 칸을 나눈 조각 수(빈 조각 포함)가
    다르면 그 열은 ①만 적용하고 `n_length_mismatch_rows`에 센다.
    같은 숙련이 여러 열에 있으면 한 행으로 합치고 `src`에 열 이름을 `+`로 이어 적는다.
    """
    needed = ["posting_id"] + raw_skill_columns()
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        raise KeyError(f"원자료에 없는 열: {missing}")

    known_codes = set(codebook["skill_code"])
    name_to_code = pd.Series(codebook["skill_code"].to_numpy(), index=codebook["skill_name"].to_numpy())

    n_tokens = n_unmatched = n_mismatch = 0
    pieces = []
    for kind in SKILL_COLUMNS:
        codes = _tokens(raw, f"skills_{kind}")
        if codes.empty:
            # 코드가 없는 칸은 숙련명만 있어도 개수가 다른 칸으로 센다.
            names_only = _tokens(raw, f"skills_{kind}_name")
            n_mismatch += int(names_only["rid"].nunique())
            continue
        names = _tokens(raw, f"skills_{kind}_name")
        # 개수는 빈 조각까지 센 나눔 조각 수로 비교한다(값이 없거나 빈 토큰만 있는 칸은 0개)
        n_code = codes.groupby("rid")["n_raw"].first()
        n_name = names.groupby("rid")["n_raw"].first()
        both = pd.concat([n_code.rename("c"), n_name.rename("n")], axis=1).fillna(0)
        differing = both.index[both["c"] != both["n"]]
        n_mismatch += len(differing)

        usable_names = names.loc[~names["rid"].isin(differing), ["rid", "pos", "tok"]]
        merged = codes[["rid", "pos", "tok"]].merge(
            usable_names.rename(columns={"tok": "name"}), on=["rid", "pos"], how="left"
        )
        final = merged["tok"].where(
            merged["tok"].isin(known_codes), merged["name"].map(name_to_code)
        )
        n_tokens += len(merged)
        n_unmatched += int(final.isna().sum())
        kept = pd.DataFrame({"rid": merged["rid"], "skill_code": final}).dropna()
        kept = kept.drop_duplicates()
        kept["kind"] = kind
        pieces.append(kept)

    stats = {"n_tokens": int(n_tokens), "n_unmatched": int(n_unmatched),
             "n_length_mismatch_rows": int(n_mismatch)}
    if not pieces or all(p.empty for p in pieces):
        return _empty_long(raw), stats

    allp = pd.concat(pieces, ignore_index=True)
    keys = allp[["rid", "skill_code"]].drop_duplicates().sort_values(["rid", "skill_code"], ignore_index=True)
    key_index = pd.MultiIndex.from_frame(keys)
    src = pd.Series("", index=keys.index, dtype=object)
    for kind in SKILL_COLUMNS:
        sub = allp[allp["kind"] == kind]
        if sub.empty:
            continue
        flag = key_index.isin(pd.MultiIndex.from_frame(sub[["rid", "skill_code"]]))
        src = src + np.where(flag, kind + "+", "")
    long = pd.DataFrame({
        "posting_id": raw["posting_id"].to_numpy()[keys["rid"].to_numpy()],
        "skill_code": keys["skill_code"].to_numpy(),
        "src": src.str[:-1].to_numpy(),
    })
    return long[_LONG_COLUMNS], stats


def attach_hierarchy(long: pd.DataFrame, codebook: pd.DataFrame) -> pd.DataFrame:
    """긴 표에 설명 파일의 `domain`, `sub` 열을 붙인다. 행 수와 순서는 그대로이다."""
    out = long.merge(codebook[["skill_code", "domain", "sub"]], on="skill_code", how="left")
    if len(out) != len(long):
        raise ValueError(f"계층 결합 뒤 행 수가 달라졌습니다: {len(long)} -> {len(out)}")
    return out


def drop_codes(long: pd.DataFrame, codes: set[str]) -> pd.DataFrame:
    """`codes`에 든 숙련 코드의 행을 뺀다."""
    return long[~long["skill_code"].isin(codes)].reset_index(drop=True)


def sparse_filter(long: pd.DataFrame, treated_ids: set[str], n_postings: int,
                  min_rate: float, min_treated: int) -> list[str]:
    """출현율이 `min_rate` 이상이고 처치 공고 안에서 `min_treated`건 이상 나온 숙련 코드를 정렬해 돌려준다.

    출현율은 그 숙련이 나온 서로 다른 공고 수를 `n_postings`로 나눈 값이다.
    """
    if long.empty:
        return []
    if n_postings <= 0:
        raise ValueError(f"n_postings는 1 이상이어야 합니다: {n_postings}")
    n_all = long.groupby("skill_code")["posting_id"].nunique()
    in_treated = long[long["posting_id"].isin(treated_ids)]
    n_treated = in_treated.groupby("skill_code")["posting_id"].nunique().reindex(n_all.index, fill_value=0)
    keep = (n_all / n_postings >= min_rate) & (n_treated >= min_treated)
    return sorted(n_all.index[keep].tolist())
