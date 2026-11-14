import pandas as pd
import pytest
import synth

import paths
from aabundle import confounders


def test_aiie_by_ksic_sub_links_three_tables():
    aiie = pd.DataFrame({"naics4": ["5112", "5415", "6111", "5182", "5191"], "aiie": [1.0, 2.0, 3.0, 4.0, 6.0]})
    naics_isic = pd.DataFrame({
        "naics6": ["511210", "541511", "541512", "611110", "999999", "518210", "519130"],
        "isic4": ["5820", "6201", "6201", "8510", "6201", "6311", "6312"]})
    ksic_isic = pd.DataFrame({
        "ksic_sub": ["582", "620", "851", "631", "011"],
        "isic_prefix": ["582", "620", "851", "631", "011"]})      # 631은 앞자리 하나가 ISIC 6311과 6312 둘에 이어진다
    naics_isic = pd.concat([naics_isic, pd.DataFrame({"naics6": ["111110"], "isic4": ["0111"]})], ignore_index=True)
    out = confounders.aiie_by_ksic_sub(aiie, naics_isic, ksic_isic).set_index("ksic_sub")
    assert out.loc["582", "aiie"] == 1.0
    assert out.loc["620", "aiie"] == 2.0 and out.loc["620", "n_naics4"] == 1     # 같은 산업은 한 번만, AIIE에 없는 9999는 빠짐
    assert out.loc["631", "aiie"] == 5.0 and out.loc["631", "n_naics4"] == 2     # 두 산업의 단순 평균
    assert "011" not in out.index                                               # ISIC에는 이어지지만 AIIE 값이 없는 소분류


def test_prefix_that_matches_no_isic_code_is_an_error():
    aiie = pd.DataFrame({"naics4": ["5415"], "aiie": [2.0]})
    naics_isic = pd.DataFrame({"naics6": ["541511"], "isic4": ["6201"]})
    ksic_isic = pd.DataFrame({"ksic_sub": ["620", "999"], "isic_prefix": ["620", "777"]})
    with pytest.raises(ValueError, match="777"):
        confounders.aiie_by_ksic_sub(aiie, naics_isic, ksic_isic)


def test_load_aiie_expands_three_digit_and_combined_entries(tmp_path):
    """가상 AIIE 시트: `611100`(4자리) 1.0, `531000`(3자리 수준) 2.0, `3250A1`(설명에 `(3251 and 3259 only)`) 3.0,
    `531100`(4자리) 9.0. 3자리 항목은 연계표의 5311·5312·5313에 퍼지되, 4자리 항목이 따로 있는 5311은 9.0이다."""
    path = synth.write_external(tmp_path)["aiie"]
    a = confounders.load_aiie(path, {"6111", "5311", "5312", "5313", "3251", "3259", "3252"}).set_index("naics4")["aiie"]
    assert a["6111"] == 1.0 and a["5312"] == 2.0 and a["5313"] == 2.0 and a["5311"] == 9.0
    assert a["3251"] == 3.0 and a["3259"] == 3.0 and "3252" not in a.index
    assert a.index.is_unique


def test_attach_aiie_keeps_rows_and_leaves_missing():
    table = pd.DataFrame({"ksic_sub": ["620"], "aiie": [2.0], "n_naics4": [1]})
    out = confounders.attach_aiie(pd.Series(["620", "999", pd.NA, "620"], dtype="string"), table)
    assert out.tolist()[0] == 2.0 and out.tolist()[3] == 2.0 and out.isna().tolist() == [False, True, True, False]


def test_loaders_pad_codes_and_drop_unusable_rows(tmp_path):
    """가상 엑셀과 CSV로 읽기 함수를 시험한다.
    연계표: 숫자로 저장된 ISIC `111` → `0111`, NAICS가 `0`인 행은 버림.
    CSV: `isic`이 `011;012`인 줄은 두 줄로 펴지고, `isic`이 빈 소분류는 줄이 없다."""
    paths_ = synth.write_external(tmp_path)          # {"aiie": …, "naics_isic": …, "ksic_isic": …}
    n = confounders.load_naics_isic(paths_["naics_isic"])
    assert n["naics6"].str.fullmatch(r"\d{6}").all() and n["isic4"].str.fullmatch(r"\d{4}").all()
    assert "0111" in set(n["isic4"])
    k = confounders.load_ksic_isic(paths_["ksic_isic"])
    assert k[k["ksic_sub"] == "011"]["isic_prefix"].tolist() == ["011", "012"]
    assert "981" not in set(k["ksic_sub"])


def test_shipped_table_covers_every_sub_class_once():
    k = pd.read_csv(paths.ROOT / "experiments" / "crosswalks" / "ksic10_sub_to_isic4.csv", dtype=str, keep_default_na=False)
    assert list(k.columns) == ["ksic_sub", "ksic_name", "isic", "note"]
    assert len(k) == 232 and k["ksic_sub"].is_unique and k["ksic_sub"].str.fullmatch(r"\d{3}").all()
    assert k.loc[k["isic"] != "", "isic"].str.fullmatch(r"\d{2,4}(;\d{2,4})*").all()


def test_ksic_isic_subs_lists_every_row_including_empty_isic(tmp_path):   # 최종 검토 수정에서 더했다
    path = synth.write_external(tmp_path)["ksic_isic"]
    assert confounders.ksic_isic_subs(path) == {"011", "012", "620", "862", "981"}   # 981은 `isic`이 비었다
    assert "981" not in set(confounders.load_ksic_isic(path)["ksic_sub"])           # load_ksic_isic은 그대로이다


def test_missing_file_names_the_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="ksic10_sub_to_isic4.csv"):
        confounders.load_ksic_isic(tmp_path / "ksic10_sub_to_isic4.csv")


# 아래는 브리프의 시험에 더한 것이다(브리프 4단계의 오류 규칙).

def test_overlapping_four_digit_entries_are_an_error(tmp_path):
    path = synth.write_external(tmp_path, aiie_extra=[("611110", "Elementary Schools again", 5.0)])["aiie"]
    with pytest.raises(ValueError, match="6111"):
        confounders.load_aiie(path, {"6111"})


def test_duplicate_sub_class_in_crosswalk_is_an_error(tmp_path):
    p = tmp_path / "ksic10_sub_to_isic4.csv"
    p.write_text("ksic_sub,ksic_name,isic,note\n620,가,620,\n620,나,631,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="620"):
        confounders.load_ksic_isic(p)
