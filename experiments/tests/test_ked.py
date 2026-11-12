import pandas as pd
import pytest

import synth
from aabundle import ked


def test_forbidden_field_is_never_requested():
    assert ked.FORBIDDEN_INDEX == 18
    assert ked.FORBIDDEN_INDEX not in ked.FIELDS.values()


def test_extract_columns_and_types(synth_data):
    out = ked.extract_ked_firm(synth_data["ked_firm"])
    assert list(out.columns) == ["bzno", "ked_estab_year", "ked_size", "ked_ipo", "ked_venture"]
    assert out["bzno"].is_unique and out["bzno"].str.fullmatch(r"\d{10}").all()


def test_secret_marker_never_appears(synth_data):
    out = ked.extract_ked_firm(synth_data["ked_firm"])
    assert not out.astype(str).apply(lambda c: c.str.contains("SECRET18")).any().any()


def test_rows_with_extra_delimiter_are_dropped(synth_data):
    out = ked.extract_ked_firm(synth_data["ked_firm"])
    assert not {"1000000005", "1000000008"} & set(out["bzno"])


def test_non_date_establishment_becomes_missing(synth_data):
    out = ked.extract_ked_firm(synth_data["ked_firm"]).set_index("bzno")
    assert pd.isna(out.loc["1000000006", "ked_estab_year"])
    assert pd.isna(out.loc["1000000003", "ked_estab_year"])
    assert out.loc["1000000001", "ked_estab_year"] == 2010


def test_latest_info_date_wins_and_bad_bzno_dropped(synth_data):
    out = ked.extract_ked_firm(synth_data["ked_firm"])
    assert set(out["bzno"]) == {"1000000001", "1000000002", "1000000003", "1000000006"}
    # 1000000002는 info_date가 20230501인 행의 ked_size를 가진다 (synth.py에 정한 값과 비교)
    by = out.set_index("bzno")
    assert by.loc["1000000002", "ked_size"] == synth.KED_SIZE_LATEST


def test_empty_codes_become_missing(synth_data):
    by = ked.extract_ked_firm(synth_data["ked_firm"]).set_index("bzno")
    assert by.loc["1000000003", ["ked_size", "ked_ipo", "ked_venture"]].isna().all()


def test_small_chunks_give_same_result(synth_data):
    a = ked.extract_ked_firm(synth_data["ked_firm"])
    b = ked.extract_ked_firm(synth_data["ked_firm"], chunksize=2)
    pd.testing.assert_frame_equal(a, b)


def test_missing_file_names_the_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope.txt"):
        ked.extract_ked_firm(tmp_path / "nope.txt")


def test_shifted_and_split_rows_never_reach_output(synth_data):
    out = ked.extract_ked_firm(synth_data["ked_firm"])
    assert not out.astype(str).apply(lambda c: c.str.contains("SECRET18")).any().any()
    assert not {"1000000009", "1000000010", "9999999999"} & set(out["bzno"])


def test_line_filter_rejects_exactly_the_bad_rows(synth_data):
    with open(synth_data["ked_firm"], "rb") as f:
        accepted = [ked._split_line(raw) is not None for raw in f]
    # 행 6, 8(구분자 49개), 행 9(구분자 49개), 행 10(단독 줄바꿈 문자)만 버린다.
    assert accepted == [True, True, True, True, True, False, True, False, False, False]


def test_line_counts_are_recorded(synth_data):
    out = ked.extract_ked_firm(synth_data["ked_firm"])
    assert out.attrs["n_lines"] == 10
    assert out.attrs["n_dropped_lines"] == 4
