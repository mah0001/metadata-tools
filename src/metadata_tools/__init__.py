"""Memory-bounded chunked reading and data dictionaries for CSV, Stata and SPSS files."""

from __future__ import annotations

import os

from metadata_tools.dictionary import Category, DataDictionary, Variable, get_dictionary
from metadata_tools.errors import (
    IncompleteReadError,
    MemoryBudgetError,
    MetadataToolsError,
    ReadError,
    SchemaDriftError,
    UnsupportedFormatError,
)
from metadata_tools.meta import FileMeta
from metadata_tools.options import GiB, MiB, ReadOptions
from metadata_tools.readers import (
    CsvReader,
    Reader,
    SpssReader,
    StataReader,
    format_for_path,
    open_reader,
)

__version__ = "0.1.0"


def read_metadata(path: str | os.PathLike[str], **option_overrides) -> FileMeta:
    """Metadata for ``path`` without loading its data.

    For CSV this still streams the file once to find column types and the
    row count; pass ``csv_type_inference="sample"`` to skip that.
    """
    with open_reader(path, **option_overrides) as reader:
        return reader.meta


__all__ = [
    "Category",
    "CsvReader",
    "DataDictionary",
    "FileMeta",
    "GiB",
    "IncompleteReadError",
    "MemoryBudgetError",
    "MetadataToolsError",
    "MiB",
    "ReadError",
    "ReadOptions",
    "Reader",
    "SchemaDriftError",
    "SpssReader",
    "StataReader",
    "UnsupportedFormatError",
    "Variable",
    "format_for_path",
    "get_dictionary",
    "open_reader",
    "read_metadata",
]
