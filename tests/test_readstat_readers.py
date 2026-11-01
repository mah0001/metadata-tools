from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pyarrow as pa
import pyreadstat
import pytest

from conftest import BRAZIL_DTA, SPSS_USER_MISSINGS, STATA_EXTENDED_MISSINGS
from metadata_tools import (
    IncompleteReadError,
    MemoryBudgetError,
    SpssReader,
    StataReader,
    open_reader,
    read_metadata,
)


def _read_chunks(path, **opts):
    with open_reader(path, **opts) as reader:
        chunks = list(reader.iter_chunks())
        return reader, chunks


@pytest.mark.parametrize("fixture", ["sav_path", "zsav_path", "dta_path"])
def test_chunks_cover_every_row_with_stable_dtypes(request, frame, fixture):
    path = request.getfixturevalue(fixture)
    reader, chunks = _read_chunks(path, chunk_rows=600)

    assert [len(c) for c in chunks] == [600, 600, 600, 600, 100]
    assert {tuple(c.dtypes.astype(str)) for c in chunks} == {tuple(reader.meta.dtypes.values())}
    assert reader.rows_read == len(frame)

    combined = pd.concat(chunks)
    assert combined.index.equals(pd.RangeIndex(len(frame)))
    assert combined["id"].astype(float).tolist() == frame["id"].tolist()
    assert combined["name"].tolist() == frame["name"].tolist()
    assert combined["when"].isna().sum() == frame["when"].isna().sum()


@pytest.mark.parametrize("fixture", ["sav_path", "dta_path"])
def test_read_matches_chunks(request, fixture):
    path = request.getfixturevalue(fixture)
    reader, chunks = _read_chunks(path, chunk_rows=1_000)
    full = open_reader(path).read()

    pd.testing.assert_frame_equal(full, pd.concat(chunks))


def test_pyarrow_backend_uses_storage_width(tmp_path):
    path = tmp_path / "ints.dta"
    pd.DataFrame(
        {"b": pd.Series([1, 2, 3], dtype="int8"), "f": pd.Series([0.5, 1.5, 2.5], dtype="float32")}
    ).to_stata(path, write_index=False)

    meta = read_metadata(path)
    assert meta.dtypes == {"b": "int8[pyarrow]", "f": "float[pyarrow]"}
    assert read_metadata(path, dtype_backend="numpy").dtypes == {"b": "float64", "f": "float64"}


def test_metadata(sav_path):
    meta = read_metadata(sav_path)

    assert meta.format == "spss"
    assert meta.number_rows == 2_500
    assert meta.column_names == ["id", "small", "ratio", "name", "when"]
    assert meta.column_labels["id"] == "Identifier"
    assert meta.value_labels["small"] == {0: "none", 1: "one"}
    assert meta.storage_types["name"] == "string"
    assert meta.raw is not None


def test_usecols_keeps_file_order(dta_path):
    reader, chunks = _read_chunks(dta_path, usecols=["name", "id"])

    assert reader.meta.column_names == ["id", "name"]
    assert list(chunks[0].columns) == ["id", "name"]


def test_chunk_rows_sized_from_memory_budget(sav_path):
    small = open_reader(sav_path, chunk_memory_bytes=10_000, min_chunk_rows=1)
    large = open_reader(sav_path, chunk_memory_bytes=10_000_000, min_chunk_rows=1)

    assert 0 < small.chunk_rows() < 1_000
    assert large.chunk_rows() > small.chunk_rows()


def test_read_refuses_over_budget(sav_path):
    reader = open_reader(sav_path, max_full_read_bytes=1_000)
    with pytest.raises(MemoryBudgetError):
        reader.read()


def test_incomplete_read_raises(sav_path):
    real = pyreadstat.read_sav

    def short_read(path, **kwargs):
        if kwargs.get("row_offset", 0) >= 1_000:
            kwargs["row_limit"] = 10
            df, meta = real(path, **kwargs)
            return df, meta
        return real(path, **kwargs)

    reader = open_reader(sav_path, chunk_rows=1_000)
    _ = reader.meta
    with patch.object(SpssReader, "_read_fn", staticmethod(short_read)):
        with pytest.raises(IncompleteReadError) as info:
            list(reader.iter_chunks())
    assert info.value.expected_rows == 2_500


def test_date_conversion_error_falls_back_to_raw_numbers(dta_path):
    real = pyreadstat.read_dta

    def fail_on_dates(path, **kwargs):
        if kwargs.get("dates_as_pandas_datetime"):
            raise OverflowError("date value out of range")
        return real(path, **kwargs)

    with patch.object(StataReader, "_read_fn", staticmethod(fail_on_dates)):
        reader, chunks = _read_chunks(dta_path)

    assert reader.meta.dtypes["when"] == "double[pyarrow]"
    assert any("raw numbers" in w for w in reader.warnings)


def test_encoding_switch_mid_file(dta_path):
    real = pyreadstat.read_dta

    def bad_bytes_late(path, **kwargs):
        if kwargs.get("row_offset", 0) >= 1_000 and "encoding" not in kwargs:
            raise pyreadstat.ReadstatError("Unable to convert string (invalid byte sequence)")
        return real(path, **kwargs)

    with patch.object(StataReader, "_read_fn", staticmethod(bad_bytes_late)):
        reader, chunks = _read_chunks(dta_path, chunk_rows=500)

    assert reader.rows_read == 2_500
    assert any("continuing with 'utf-8'" in w for w in reader.warnings)


def test_stata_pandas_fallback_when_pyreadstat_cannot_decode(dta_path):
    def always_fails(path, **kwargs):
        raise pyreadstat.ReadstatError("Unable to convert string (invalid byte sequence)")

    with patch.object(StataReader, "_read_fn", staticmethod(always_fails)):
        reader, chunks = _read_chunks(dta_path, chunk_rows=1_000)
        assert reader.meta.number_rows == 2_500
        assert reader.meta.column_labels["id"] == "Identifier"
        assert sum(len(c) for c in chunks) == 2_500
        assert {tuple(c.dtypes.astype(str)) for c in chunks} == {tuple(reader.meta.dtypes.values())}
        assert any("pandas StataReader" in w for w in reader.warnings)


def test_stata_pandas_fallback_mid_file(dta_path):
    real = pyreadstat.read_dta

    def fails_late(path, **kwargs):
        if kwargs.get("row_offset", 0) >= 1_000:
            raise pyreadstat.ReadstatError("Unable to convert string (invalid byte sequence)")
        return real(path, **kwargs)

    with patch.object(StataReader, "_read_fn", staticmethod(fails_late)):
        reader, chunks = _read_chunks(dta_path, chunk_rows=500)

    combined = pd.concat(chunks)
    assert combined.index.equals(pd.RangeIndex(2_500))
    assert {tuple(c.dtypes.astype(str)) for c in chunks} == {tuple(reader.meta.dtypes.values())}


def test_stata_user_missing_numeric_columns_are_object():
    reader, chunks = _read_chunks(STATA_EXTENDED_MISSINGS, user_missing=True, chunk_rows=60)

    assert reader.meta.dtypes["q_ext"] == "object"
    assert reader.meta.dtypes["str_letters"] == "string[pyarrow]"
    letters = pd.concat(chunks)["q_ext"]
    assert {"a", "b", "c"} <= {v for v in letters if isinstance(v, str)}


def test_stata_extended_missings_are_nan_by_default():
    reader, chunks = _read_chunks(STATA_EXTENDED_MISSINGS)
    q_ext = pd.concat(chunks)["q_ext"]

    assert q_ext.dtype == pd.ArrowDtype(pa.float64())
    assert q_ext.isna().any()


def test_spss_user_missing_values_kept():
    plain = pd.concat(_read_chunks(SPSS_USER_MISSINGS)[1])
    reader, chunks = _read_chunks(SPSS_USER_MISSINGS, user_missing=True)
    kept = pd.concat(chunks)

    assert reader.meta.missing_user_values or reader.meta.missing_ranges
    assert kept.isna().sum().sum() < plain.isna().sum().sum()


@pytest.mark.skipif(not BRAZIL_DTA.is_file(), reason="Brazil fixture not available")
def test_brazil_file_with_late_invalid_bytes():
    reader, chunks = _read_chunks(BRAZIL_DTA, chunk_rows=200)

    assert reader.meta.number_rows == 1531
    assert reader.meta.number_columns == 339
    assert reader.rows_read == 1531
