"""Format readers and extension lookup."""

from __future__ import annotations

import os

from metadata_tools.errors import UnsupportedFormatError
from metadata_tools.meta import FileFormat
from metadata_tools.options import ReadOptions
from metadata_tools.readers.base import Reader
from metadata_tools.readers.csv import CsvReader
from metadata_tools.readers.spss import SpssReader
from metadata_tools.readers.stata import StataReader

READERS: dict[FileFormat, type[Reader]] = {
    "csv": CsvReader,
    "stata": StataReader,
    "spss": SpssReader,
}

EXTENSIONS: dict[str, FileFormat] = {
    ".csv": "csv",
    ".tsv": "csv",
    ".txt": "csv",
    ".dta": "stata",
    ".sav": "spss",
    ".zsav": "spss",
}


def format_for_path(path: str | os.PathLike[str]) -> FileFormat:
    ext = os.path.splitext(os.fspath(path))[1].lower()
    try:
        return EXTENSIONS[ext]
    except KeyError:
        raise UnsupportedFormatError(f"No reader for {ext!r} files: {path}") from None


def open_reader(
    path: str | os.PathLike[str],
    options: ReadOptions | None = None,
    *,
    format: FileFormat | None = None,
    **option_overrides,
) -> Reader:
    """Return a reader for ``path``, chosen by extension unless ``format`` is given.

    Keyword arguments override fields of ``options``:

    >>> open_reader("big.csv", usecols=["id", "age"], chunk_memory_bytes=128 * MiB)
    """
    fmt = format or format_for_path(path)
    if fmt not in READERS:
        raise UnsupportedFormatError(f"Unknown format {fmt!r}")
    opts = options or ReadOptions()
    if option_overrides:
        opts = opts.with_(**option_overrides)
    return READERS[fmt](path, opts)


__all__ = [
    "CsvReader",
    "EXTENSIONS",
    "READERS",
    "Reader",
    "SpssReader",
    "StataReader",
    "format_for_path",
    "open_reader",
]
