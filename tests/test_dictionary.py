from __future__ import annotations

import json
from unittest.mock import patch

import pandas as pd
import pyreadstat
import pytest

from conftest import SPSS_USER_MISSINGS, STATA_EXTENDED_MISSINGS
from metadata_tools import StataReader, get_dictionary, read_metadata


def by_name(dictionary):
    return {v.name: v for v in dictionary.variables}


def test_spss_dictionary():
    d = get_dictionary(SPSS_USER_MISSINGS)
    v = by_name(d)

    assert (d.format, d.rows, d.columns) == ("spss", 200, 5)
    assert d.file_label == "SPSS user missings fixture"
    assert v["q2"].label == "Nominal with user missings 8/9"
    assert v["q2"].measure == "nominal"
    assert v["q2"].decimals == 2
    assert v["q2"].missing_values == [8, 9]
    assert [(c.value, c.label, c.is_missing) for c in v["q2"].categories] == [
        (1, "Yes", False),
        (2, "No", False),
        (8, "Don't know", True),
        (9, "Refused", True),
    ]
    assert v["inc"].missing_ranges == [{"lo": -9, "hi": -1}]
    assert v["inc"].missing_values == [999]
    assert (v["name"].type, v["name"].format) == ("string", "A7")
    assert v["dob"].type == "date"


def test_stata_dictionary():
    d = get_dictionary(STATA_EXTENDED_MISSINGS)
    v = by_name(d)

    assert (d.format, d.rows, d.columns) == ("stata", 200, 9)
    assert [(c.value, c.is_missing) for c in v["q_ext"].categories] == [
        (1, False),
        (2, False),
        (".a", True),
        (".b", True),
    ]
    assert v["int_date"].type == "date"
    assert v["code_lz"].type == "string"
    assert v["q_ext"].measure is None
    assert v["q_ext"].decimals is None  # %10.0g is a general format


def test_stata_formats(tmp_path):
    path = tmp_path / "fmt.dta"
    df = pd.DataFrame({"amount": [1.5], "when": [pd.Timestamp("2024-01-02 03:04:05")]})
    pyreadstat.write_dta(df, str(path), variable_format={"amount": "%9.2f"})
    v = by_name(get_dictionary(path))

    assert v["amount"].decimals == 2
    assert v["when"].type == "datetime"


def test_dictionary_reads_no_data(dta_path):
    real = pyreadstat.read_dta

    def header_only(path, **kwargs):
        assert kwargs.get("metadataonly"), "dictionary must not read data"
        return real(path, **kwargs)

    with patch.object(StataReader, "_read_fn", staticmethod(header_only)):
        d = get_dictionary(dta_path)
    assert d.rows == 2_500


def test_stata_pandas_fallback(dta_path):
    def always_fails(path, **kwargs):
        raise pyreadstat.ReadstatError("Unable to convert string (invalid byte sequence)")

    with patch.object(StataReader, "_read_fn", staticmethod(always_fails)):
        d = get_dictionary(dta_path)
    assert by_name(d)["id"].label == "Identifier"
    assert d.rows == 2_500


def test_reader_meta_includes_declared_missings():
    # pyreadstat only reports missing ranges when asked for user missings.
    meta = read_metadata(SPSS_USER_MISSINGS, user_missing=False)
    assert meta.missing_ranges["q2"] == [{"lo": 8.0, "hi": 8.0}, {"lo": 9.0, "hi": 9.0}]


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("id;code;score;ok\n1;007;1.5;café\n2;008;2;false\n", encoding="cp1252")
    return path


def test_csv_has_names_only(csv_path):
    d = get_dictionary(csv_path)

    assert d.encoding == "cp1252"
    assert [v.name for v in d.variables] == ["id", "code", "score", "ok"]
    assert {(v.type, v.storage_type, v.label) for v in d.variables} == {(None, None, None)}
    assert d.rows is None


def test_csv_reads_header_only(tmp_path):
    path = tmp_path / "big.csv"
    path.write_text("a,b\n" + "1,x\n" * 200_000, encoding="utf-8")

    with patch("metadata_tools.readers.csv.pacsv.open_csv", side_effect=AssertionError("data read")):
        d = get_dictionary(path)
    assert [v.name for v in d.variables] == ["a", "b"]


def test_to_dict_is_json_serialisable():
    out = get_dictionary(SPSS_USER_MISSINGS).to_dict()
    text = json.dumps(out)
    assert out["created"].startswith("20")
    assert json.loads(text)["variables"][0]["categories"][2] == {
        "value": 8,
        "label": "Don't know",
        "is_missing": True,
    }
