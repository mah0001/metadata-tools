from __future__ import annotations

import pandas as pd
import pytest

from metadata_tools import (
    CsvReader,
    MemoryBudgetError,
    ReadError,
    SchemaDriftError,
    UnsupportedFormatError,
    open_reader,
    read_metadata,
)


def write_csv(path, header, rows, encoding="utf-8", delimiter=","):
    lines = [delimiter.join(header)] + [delimiter.join(map(str, r)) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding=encoding)
    return path


@pytest.fixture
def mixed_csv(tmp_path):
    """A column that is integer for 50k rows, then decimals, then text."""
    rows = []
    for i in range(60_000):
        late = f"{i}.5" if i >= 50_000 else str(i)
        rows.append((i, f"{i % 900:04d}", late, "yes" if i == 59_999 else i % 7, "true" if i % 2 else "false", ""))
    return write_csv(tmp_path / "mixed.csv", ["id", "code", "late_float", "late_text", "flag", "empty"], rows)


def test_scan_infers_types_from_whole_file(mixed_csv):
    meta = read_metadata(mixed_csv)

    assert meta.number_rows == 60_000
    assert meta.storage_types == {
        "id": "int64",
        "code": "string",  # leading zeros kept
        "late_float": "double",
        "late_text": "string",
        "flag": "bool",
        "empty": "string",
    }


def test_scan_chunks_have_stable_dtypes(mixed_csv):
    with open_reader(mixed_csv, chunk_rows=7_000) as reader:
        chunks = list(reader.iter_chunks())

    assert sum(len(c) for c in chunks) == 60_000
    assert {tuple(c.dtypes.astype(str)) for c in chunks} == {tuple(reader.meta.dtypes.values())}
    combined = pd.concat(chunks)
    assert combined.index.equals(pd.RangeIndex(60_000))
    assert combined["code"].iloc[5] == "0005"


def test_chunk_rows_rebatches_exactly(mixed_csv):
    with open_reader(mixed_csv, chunk_rows=25_000) as reader:
        assert [len(c) for c in reader.iter_chunks()] == [25_000, 25_000, 10_000]


def test_sample_inference_raises_on_drift(mixed_csv):
    reader = open_reader(mixed_csv, csv_type_inference="sample", chunk_memory_bytes=1)
    assert reader.meta.number_rows is None
    with pytest.raises(SchemaDriftError):
        list(reader.iter_chunks())


def test_string_inference_and_dtype_override(mixed_csv):
    reader = open_reader(mixed_csv, csv_type_inference="string", dtypes={"id": "int32"})
    meta = reader.meta

    assert meta.storage_types["late_float"] == "string"
    assert meta.storage_types["id"] == "int32"
    rows = sum(len(c) for c in reader.iter_chunks())
    assert rows == 60_000
    assert reader.meta.number_rows == 60_000


def test_numpy_backend(mixed_csv):
    df = open_reader(mixed_csv, dtype_backend="numpy").read()

    assert str(df["id"].dtype) == "float64"
    assert df["code"].dtype == object


@pytest.mark.parametrize("encoding,expected", [("utf-8", "utf-8"), ("cp1252", "cp1252"), ("utf-8-sig", "utf-8")])
def test_encoding_detection(tmp_path, encoding, expected):
    path = write_csv(tmp_path / "enc.csv", ["name", "city"], [("José", "São Paulo"), ("Zoë", "Köln €")], encoding=encoding)

    with open_reader(path) as reader:
        df = reader.read()
    assert reader.meta.encoding == expected
    assert list(df.columns) == ["name", "city"]
    assert df["city"].tolist() == ["São Paulo", "Köln €"]


@pytest.mark.parametrize("delimiter", [";", "\t", "|"])
def test_delimiter_sniffing(tmp_path, delimiter):
    path = write_csv(tmp_path / "d.csv", ["a", "b", "c"], [(1, "x, y", 3), (4, "z", 6)], delimiter=delimiter)

    with open_reader(path) as reader:
        df = reader.read()
    assert reader.meta.delimiter == delimiter
    assert df["b"].tolist() == ["x, y", "z"]


def test_quoted_newlines(tmp_path):
    path = tmp_path / "q.csv"
    path.write_text('id,note\n1,"first\nsecond"\n2,plain\n', encoding="utf-8")

    df = open_reader(path).read()
    assert df["note"].tolist() == ["first\nsecond", "plain"]


def test_usecols(mixed_csv):
    reader = open_reader(mixed_csv, usecols=["flag", "id"])
    assert reader.meta.column_names == ["id", "flag"]
    assert list(next(reader.iter_chunks()).columns) == ["id", "flag"]

    with pytest.raises(ValueError, match="not in file"):
        open_reader(mixed_csv, usecols=["nope"]).meta


def test_read_refuses_over_budget(mixed_csv):
    with pytest.raises(MemoryBudgetError):
        open_reader(mixed_csv, max_full_read_bytes=1_000).read()


def test_empty_file(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_bytes(b"")
    with pytest.raises(ReadError):
        open_reader(path).meta


def test_format_lookup(tmp_path):
    path = tmp_path / "data.xyz"
    path.write_text("a\n1\n")
    with pytest.raises(UnsupportedFormatError):
        open_reader(path)
    assert isinstance(open_reader(path, format="csv"), CsvReader)
