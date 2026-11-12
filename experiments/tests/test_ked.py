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
    assert out.attrs["n_masked_values"] == 0


# 아래는 전체 검토 수정에서 더한 시험이다. 18번 항목에는 주민등록번호와 형태만 같은, 0으로만 된 가짜 값을 쓴다.
FAKE_ID = "000000-0000000"


def _shifted_line_keeping_47_delimiters() -> bytes:
    """18번 앞(5번 항목 안)에 `|`가 둘 더 있고 끝의 두 항목이 없는 줄. 구분자는 47개 그대로이다.

    항목이 두 칸 밀려 18번 값이 `ked_ipo` 자리(20)로, 30번 값이 사업자번호 자리(32)로, 29번 값이 정보일자
    자리(31)로, 17번 값이 설립일자 자리(19)로 온다.
    """
    line = synth._ked_row("1", "20160101", "20230501", extra_at=5, extra_n=2,
                          overrides={18: FAKE_ID, 17: "20160101", 29: "20230501", 30: "1000000011"})
    fields = line.rstrip(b"\r\n").split(b"|")
    return b"|".join(fields[:-2]) + b"\r\n"


def test_selected_field_shaped_like_resident_number_is_masked(tmp_path):
    line = _shifted_line_keeping_47_delimiters()
    assert line.count(b"|") == 47
    rows = [synth._ked_row("1000000001", "20100315", "20230501"), line,
            synth._ked_row("1000000012", "20110101", "20230501", ipo="0000000000000")]
    path = tmp_path / "ked.txt"
    path.write_bytes(b"".join(rows))

    out = ked.extract_ked_firm(path)
    cells = [str(c) for c in out.to_numpy().ravel()]
    assert not any(FAKE_ID in c or "0000000000000" in c for c in cells)
    assert out.attrs["n_masked_values"] == 2
    assert out.attrs["n_dropped_lines"] == 0 and out.attrs["n_lines"] == 3
    by = out.set_index("bzno")
    assert pd.isna(by.loc["1000000011", "ked_ipo"]) and pd.isna(by.loc["1000000012", "ked_ipo"])
    # 10자리 사업자번호와 8자리 날짜는 가리지 않는다
    assert set(out["bzno"]) == {"1000000001", "1000000011", "1000000012"}
    assert by.loc["1000000001", "ked_estab_year"] == 2010 and by.loc["1000000011", "ked_estab_year"] == 2016
    assert by.loc["1000000001", "ked_ipo"] == "N"


def test_masking_searches_bytes_once_per_line(tmp_path, monkeypatch):
    path = tmp_path / "ked.txt"
    path.write_bytes(b"".join(synth._ked_row(f"10000000{i:02d}", "20100315", "20230501") for i in range(5)))
    calls = []
    real = ked._ID_SHAPE

    class Counting:
        def search(self, data):
            calls.append(data)
            return real.search(data)
    monkeypatch.setattr(ked, "_ID_SHAPE", Counting())
    ked.extract_ked_firm(path)
    assert len(calls) == 5 and all(isinstance(c, bytes) for c in calls)
