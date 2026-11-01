"""Data dictionary from file headers: names, labels, types, categories, missing codes.

No data values are read, so this is fast on files of any size. There are no
frequencies or summary statistics. CSV files carry no metadata beyond the
header line, so a CSV dictionary has column names only.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Literal

from metadata_tools._encoding import detect_encoding
from metadata_tools.meta import FileFormat, FileMeta
from metadata_tools.options import ReadOptions
from metadata_tools.readers import CsvReader, format_for_path, open_reader
from metadata_tools.readers._readstat import ReadstatReader

VariableType = Literal["numeric", "string", "date", "datetime", "time"]

# Enough of a CSV file to decode the header line and sniff the delimiter.
_CSV_HEAD_BYTES = 64 * 1024

_SPSS_DATE = {"DATE", "ADATE", "EDATE", "JDATE", "SDATE", "QYR", "MOYR", "WKYR"}
_SPSS_DATETIME = {"DATETIME", "YMDHMS"}
_SPSS_TIME = {"TIME", "DTIME", "MTIME"}
_STATA_EXTENDED_MISSING = re.compile(r"^[a-z]$")


@dataclass
class Category:
    value: Any
    label: str
    is_missing: bool = False


@dataclass
class Variable:
    name: str
    label: str | None
    type: VariableType | None
    """None for CSV: types are not stored in the file, and guessing them
    means reading data."""
    storage_type: str | None
    """As stored: readstat type (int8, double, string...). None for CSV."""
    format: str | None
    """Display format (F8.2, %td, A7...). None for CSV."""
    decimals: int | None
    measure: str | None
    """nominal / ordinal / scale (SPSS only)."""
    categories: list[Category] = field(default_factory=list)
    missing_values: list[Any] = field(default_factory=list)
    """Declared discrete missing codes (SPSS)."""
    missing_ranges: list[dict[str, Any]] = field(default_factory=list)
    """Declared missing ranges as {"lo", "hi"} (SPSS)."""


@dataclass
class DataDictionary:
    path: str
    format: FileFormat
    encoding: str | None
    rows: int | None
    """None for CSV: counting rows means reading the whole file."""
    columns: int
    file_label: str | None
    notes: list[str]
    created: datetime | None
    modified: datetime | None
    variables: list[Variable]

    def to_dict(self) -> dict[str, Any]:
        """Plain dict, ready for ``json.dumps``."""
        out = asdict(self)
        for key in ("created", "modified"):
            if out[key] is not None:
                out[key] = out[key].isoformat()
        return out


def get_dictionary(
    path: str | os.PathLike[str],
    *,
    encoding: str | None = None,
    format: FileFormat | None = None,
) -> DataDictionary:
    """Build a data dictionary for ``path`` from its metadata only."""
    path = os.fspath(path)
    fmt = format or format_for_path(path)
    if fmt == "csv":
        meta = _csv_meta(path, encoding)
    else:
        reader = open_reader(path, format=fmt, encoding=encoding)
        assert isinstance(reader, ReadstatReader)
        meta = reader.read_header()

    raw = meta.raw
    return DataDictionary(
        path=path,
        format=fmt,
        encoding=meta.encoding,
        rows=meta.number_rows,
        columns=meta.number_columns,
        file_label=meta.file_label or None,
        notes=meta.notes,
        created=getattr(raw, "creation_time", None),
        modified=getattr(raw, "modification_time", None),
        variables=[_variable(meta, name) for name in meta.column_names],
    )


def _csv_meta(path: str, encoding: str | None) -> FileMeta:
    """Header line only: ``string`` inference reads no data rows."""
    options = ReadOptions(csv_type_inference="string")
    encoding = encoding or detect_encoding(path, options.encodings, max_bytes=_CSV_HEAD_BYTES)
    return CsvReader(path, options.with_(encoding=encoding)).meta


def _variable(meta: FileMeta, name: str) -> Variable:
    storage = meta.storage_types.get(name)
    fmt = meta.formats.get(name)
    missing_values, missing_ranges = _declared_missings(meta.missing_ranges.get(name, []))
    measure = meta.measure.get(name)
    return Variable(
        name=name,
        label=meta.column_labels.get(name) or None,
        type=None if meta.format == "csv" else _variable_type(meta.format, storage, fmt),
        storage_type=None if meta.format == "csv" else storage,
        format=fmt or None,
        decimals=_decimals(meta.format, storage, fmt),
        measure=measure if measure and measure != "unknown" else None,
        categories=_categories(
            meta.format, meta.value_labels.get(name, {}), missing_values, missing_ranges
        ),
        missing_values=missing_values,
        missing_ranges=missing_ranges,
    )


def _variable_type(file_format: str, storage: str | None, fmt: str | None) -> VariableType:
    if storage == "string":
        return "string"
    if fmt:
        if file_format == "stata":
            code = fmt.lstrip("%-")
            if code.startswith(("tc", "tC")):
                return "datetime"
            if code.startswith(("td", "d")):
                return "date"
        elif file_format == "spss":
            kind = re.match(r"[A-Z]+", fmt.upper())
            kind = kind.group(0) if kind else ""
            if kind in _SPSS_DATETIME:
                return "datetime"
            if kind in _SPSS_DATE:
                return "date"
            if kind in _SPSS_TIME:
                return "time"
    return "numeric"


def _decimals(file_format: str, storage: str | None, fmt: str | None) -> int | None:
    if not fmt or storage == "string":
        return None
    if file_format == "spss":
        m = re.search(r"\.(\d+)$", fmt)
    else:
        # Only fixed (%9.2f) and exponential (%10.3e) formats set decimals;
        # general formats (%10.0g) do not.
        m = re.match(r"^%-?\d*\.(\d+)[fe]c?$", fmt)
    return int(m.group(1)) if m else None


def _code(value: Any) -> Any:
    """1.0 -> 1, so codes read the way they were entered."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _declared_missings(ranges: list[dict[str, Any]]) -> tuple[list[Any], list[dict[str, Any]]]:
    """Split pyreadstat missing ranges into discrete codes (lo == hi) and ranges."""
    values, spans = [], []
    for r in ranges:
        lo, hi = _code(r["lo"]), _code(r["hi"])
        if lo == hi:
            values.append(lo)
        else:
            spans.append({"lo": lo, "hi": hi})
    return values, spans


def _categories(
    file_format: str,
    labels: dict[Any, str],
    missing_values: list[Any],
    missing_ranges: list[dict[str, Any]],
) -> list[Category]:
    out = []
    for raw_value, label in labels.items():
        value = _code(raw_value)
        if file_format == "stata" and isinstance(value, str) and _STATA_EXTENDED_MISSING.match(value):
            # pyreadstat gives labels for .a-.z under the bare letter.
            out.append(Category(value=f".{value}", label=label, is_missing=True))
            continue
        is_missing = value in missing_values or any(
            isinstance(value, (int, float)) and r["lo"] <= value <= r["hi"] for r in missing_ranges
        )
        out.append(Category(value=value, label=label, is_missing=is_missing))
    return out
