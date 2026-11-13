import pandas as pd, pyarrow as pa, pytest
from aabundle import rawio

def test_iter_postings_yields_row_groups(synth_data):
    chunks = list(rawio.iter_postings(synth_data["postings_dir"], 2025, ["posting_id", "bzno"]))
    assert [len(c) for c in chunks] == [5, 5, 5, 1]   # 과제 3에서 [5, 5, 2]에서 [5, 5, 4]로, 과제 3A에서 공고 두 건을 더해 바꿨다
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

def test_open_postings_reads_one_row_group(synth_data):
    pf = rawio.open_postings(synth_data["postings_dir"], 2025, ["posting_id", "detail"])
    assert pf.num_row_groups == 4   # 과제 3A에서 공고 두 건(p15, p16)을 더해 3에서 바꿨다
    assert pf.read_row_group(2, columns=["posting_id"]).column("posting_id").to_pylist() == ["p11", "p12", "p13", "p14", "p15"]   # 과제 3A: p15
    with pytest.raises(FileNotFoundError, match="pfile_2030.parquet"):
        rawio.open_postings(synth_data["postings_dir"], 2030, ["posting_id"])
    with pytest.raises(KeyError, match="no_such_column"):
        rawio.open_postings(synth_data["postings_dir"], 2025, ["posting_id", "no_such_column"])

def test_parquet_writer_appends_and_refuses_raw_data_folder(tmp_path):
    schema = pa.schema([("a", pa.int64())])
    with pytest.raises(ValueError):
        rawio.parquet_writer(tmp_path / "data" / "x.parquet", schema)
    assert not (tmp_path / "data").exists()
    p = tmp_path / "outputs" / "x.parquet"
    with rawio.parquet_writer(p, schema) as w:
        w.write_table(pa.table({"a": [1, 2]}, schema=schema))
        w.write_table(pa.table({"a": [3]}, schema=schema))
    assert pd.read_parquet(p)["a"].tolist() == [1, 2, 3]
