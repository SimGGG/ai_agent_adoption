import pandas as pd
from aabundle import classif


def test_normalize_ksic_formats():
    s = pd.Series(["62010", "Q86201", "1220", "C30", "", None, "q86201", "123456"])
    out = classif.normalize_ksic(s)
    assert out.tolist()[:3] == ["62010", "86201", "01220"]
    assert out[3:].isna().all()


def test_ksic_sub_marks_codes_missing_from_table():
    out = classif.ksic_sub(pd.Series(["62010", "Q86201", "99999", None]), {"620", "862"})
    assert out.tolist()[:3] == ["620", "862", classif.KSIC_NOT_IN_TABLE]
    assert pd.isna(out[3])


def test_keco_mid():
    out = classif.keco_mid(pd.Series(["1332", "0262", None]))
    assert out.tolist()[:2] == ["13", "02"] and pd.isna(out[2])


def test_loaders_read_synthetic_tables(synth_data):
    assert classif.load_ksic_sub(synth_data["ksic"]) == {"620", "862", "012"}
    keco = classif.load_keco(synth_data["keco"])
    assert keco["keco4"].tolist() == ["1332", "0262", "9052"]
    assert set(keco.columns) == {"keco4", "keco4_name", "keco2", "keco2_name"}


def test_load_keco_fills_merged_cells(tmp_path):
    from openpyxl import Workbook
    wb = Workbook(); ws = wb.active
    for _ in range(3):
        ws.append(["머리"] * 8)
    ws.append(["1", "연구·공학", "13", "정보통신", "133", "소프트웨어", "1332", "응용 소프트웨어 개발자"])
    ws.append([None, None, None, None, None, None, "1333", "웹 개발자"])           # 병합된 칸: 위 줄의 중분류를 물려받는다
    ws.append([None, None, "14", "건설·채굴", "141", "건축", "1411", "건축가"])
    ws.append(["3항목", None, "2항목", None, "3항목", None, "3개", None])           # 합계 줄
    wb.save(tmp_path / "keco.xlsx")
    keco = classif.load_keco(tmp_path / "keco.xlsx")
    assert keco["keco4"].tolist() == ["1332", "1333", "1411"]
    assert keco["keco2"].tolist() == ["13", "13", "14"]
    assert keco["keco2_name"].tolist() == ["정보통신", "정보통신", "건설·채굴"]
