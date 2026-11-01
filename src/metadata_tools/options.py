"""Read options shared by all readers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

MiB = 1024 * 1024
GiB = 1024 * MiB

DtypeBackend = Literal["pyarrow", "numpy"]
CsvTypeInference = Literal["scan", "sample", "string"]

DEFAULT_ENCODINGS: tuple[str | None, ...] = (
    None,
    "utf-8",
    "cp1252",
    "latin1",
    "cp850",
)


@dataclass(frozen=True)
class ReadOptions:
    """Options controlling how a file is read.

    Memory is controlled by two numbers: ``chunk_memory_bytes`` sizes each
    chunk yielded by ``iter_chunks()``, and ``max_full_read_bytes`` caps what
    ``read()`` will load in one go.
    """

    usecols: list[str] | None = None
    """Columns to read, or None for all. Order follows the file."""

    chunk_rows: int | None = None
    """Fixed rows per chunk. None sizes chunks from ``chunk_memory_bytes``."""

    chunk_memory_bytes: int = 256 * MiB
    """Target in-memory size of one chunk when ``chunk_rows`` is None."""

    min_chunk_rows: int = 1_000
    max_chunk_rows: int = 5_000_000

    max_full_read_bytes: int = 2 * GiB
    """``read()`` raises MemoryBudgetError when its estimate is above this."""

    dtype_backend: DtypeBackend = "pyarrow"
    """``pyarrow``: compact Arrow-backed dtypes sized to the file's storage
    types. ``numpy``: float64 for numbers, object for strings."""

    encoding: str | None = None
    """Force an encoding. None probes ``encodings`` in order."""

    encodings: tuple[str | None, ...] = DEFAULT_ENCODINGS

    # Stata / SPSS
    user_missing: bool = False
    """Keep user-defined missing values (SPSS) and extended missings .a-.z
    (Stata) instead of turning them into NaN. Stata numeric columns are then
    object dtype in every chunk."""

    convert_dates: bool = True
    """Convert date/datetime formatted numbers to datetime64. False keeps the
    raw numbers."""

    # CSV
    delimiter: str | None = None
    """CSV delimiter. None sniffs it from the first 64 KiB."""

    csv_type_inference: CsvTypeInference = "scan"
    """``scan``: a streaming pass over the whole file decides each column's
    type (also gives an exact row count). ``sample``: types come from the
    first block; a later mismatch raises SchemaDriftError. ``string``: every
    column is read as string."""

    dtypes: dict[str, str] = field(default_factory=dict)
    """CSV column type overrides as Arrow type names (``int64``, ``float64``,
    ``string``, ``bool``, ...)."""

    def with_(self, **changes) -> "ReadOptions":
        return replace(self, **changes)
