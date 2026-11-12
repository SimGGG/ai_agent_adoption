import pandas as pd, pytest
from aabundle import rawio

def test_iter_postings_yields_row_groups(synth_data):
    chunks = list(rawio.iter_postings(synth_data["postings_dir"], 2025, ["posting_id", "bzno"]))
    assert [len(c) for c in chunks] == [5, 5, 2]
    assert list(chunks[0].columns) == ["posting_id", "bzno"]

def test_missing_file_names_the_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="pfile_2030.parquet"):
        list(rawio.iter_postings(tmp_path, 2030, ["posting_id"]))

def test_missing_column_names_the_column(synth_data):
    with pytest.raises(KeyError, match="no_such_column"):
        list(rawio.iter_postings(synth_data["postings_dir"], 2025, ["posting_id", "no_such_column"]))

def test_write_refuses_raw_data_folder(tmp_path):
    with pytest.raises(ValueError):
        rawio.write_parquet(pd.DataFrame({"a": [1]}), tmp_path / "data" / "x.parquet")

def test_write_then_read_roundtrip(tmp_path):
    p = rawio.write_parquet(pd.DataFrame({"a": [1, 2]}), tmp_path / "outputs" / "x.parquet")
    assert rawio.read_parquet(p)["a"].tolist() == [1, 2]
