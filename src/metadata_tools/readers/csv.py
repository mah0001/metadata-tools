"""Streaming CSV reader built on pyarrow.csv."""

from __future__ import annotations

import csv
import logging
import os
from collections.abc import Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow import csv as pacsv

from metadata_tools._dtypes import arrow_to_pandas_dtype, conform
from metadata_tools._encoding import detect_encoding, python_codec, sniff_delimiter
from metadata_tools.errors import ReadError, SchemaDriftError
from metadata_tools.meta import FileMeta
from metadata_tools.options import MiB
from metadata_tools.readers.base import Reader

logger = logging.getLogger(__name__)

# Arrow's streaming CSV reader parses ~40 blocks ahead of the consumer
# whatever the block size (measured with pyarrow 26: 4 MiB blocks -> 162 MB
# pool peak, 16 MiB -> 648 MB), and small blocks are no slower. So blocks stay
# small and are combined into budget-sized chunks.
MIN_BLOCK = 1 * MiB
MAX_BLOCK = 8 * MiB
READAHEAD_BLOCKS = 40

# In-memory size of a chunk relative to its Arrow size.
_EXPANSION = {"pyarrow": 1, "numpy": 4}

_LEADING_ZERO = r"^[+-]?0\d"
_INTEGER_TEXT = r"^[+-]?\d+$"
_FLOAT_TEXT = r"(?i)^[+-]?((\d+\.?\d*|\.\d+)(e[+-]?\d+)?|nan|inf|infinity)$"
_BOOL_TEXT = pa.array(["true", "false"])


def _all_match(values: pa.Array, pattern: str) -> bool:
    return pc.all(pc.match_substring_regex(values, pattern)).as_py()


def _classify(values: pa.Array, previous: str | None) -> str:
    """Narrowest of bool/int/float/string that holds every (non-null) value.

    Only kinds still possible given ``previous`` are checked. A regex check
    comes before each cast because a failing Arrow cast is ~10x slower than
    the regex. Integers with leading zeros (ID and area codes) and integers
    too large for int64 are kept as strings rather than losing digits.
    """
    if previous in (None, "bool") and pc.all(pc.is_in(pc.utf8_lower(values), value_set=_BOOL_TEXT)).as_py():
        return "bool"
    if previous in (None, "int") and _all_match(values, _INTEGER_TEXT):
        if pc.any(pc.match_substring_regex(values, _LEADING_ZERO)).as_py():
            return "string"
        try:
            pc.cast(values, pa.int64())
        except pa.ArrowInvalid:
            return "string"
        return "int"
    if previous in (None, "int", "float") and _all_match(values, _FLOAT_TEXT):
        return "float"
    return "string"


def _merge(previous: str | None, current: str) -> str:
    if previous is None or previous == current:
        return current
    if {previous, current} == {"int", "float"}:
        return "float"
    return "string"


_KIND_TO_ARROW = {
    "bool": pa.bool_(),
    "int": pa.int64(),
    "float": pa.float64(),
    "string": pa.string(),
}


class CsvReader(Reader):
    """Reader for delimited text files.

    The encoding is found by streaming the file through a decoder, and the
    delimiter is sniffed from the first 64 KiB. Column types depend on
    ``ReadOptions.csv_type_inference``; with the default ``scan`` the reader
    makes one streaming pass to decide types before yielding any data, so
    every chunk has the same dtypes and ``meta.number_rows`` is exact.
    """

    format = "csv"

    def __init__(self, path, options=None) -> None:
        super().__init__(path, options)
        self._types: dict[str, pa.DataType] = {}
        self._dtypes: dict[str, object] = {}

    # -- metadata ---------------------------------------------------------

    def _load_meta(self) -> FileMeta:
        opts = self.options
        try:
            encoding = opts.encoding or detect_encoding(self.path, opts.encodings)
        except UnicodeError as exc:
            raise ReadError(str(exc)) from exc
        default_delim = "\t" if self.path.lower().endswith(".tsv") else ","
        delimiter = opts.delimiter or sniff_delimiter(self.path, encoding, default_delim)
        self._encoding = encoding
        self._delimiter = delimiter

        header = self._read_header()
        if opts.usecols is not None:
            unknown = [c for c in opts.usecols if c not in header]
            if unknown:
                raise ValueError(f"{self.path}: columns not in file: {unknown}")
            columns = [c for c in header if c in opts.usecols]
        else:
            columns = header
        self._columns = columns

        number_rows = None
        mode = opts.csv_type_inference
        if mode == "scan":
            self._types, number_rows = self._scan()
        elif mode == "sample":
            self._types = self._sample_types()
        else:
            self._types = {c: pa.string() for c in columns}
        for col, name in opts.dtypes.items():
            if col in self._types:
                self._types[col] = pa.type_for_alias(name)

        self._dtypes = {c: arrow_to_pandas_dtype(t, opts.dtype_backend) for c, t in self._types.items()}
        return FileMeta(
            path=self.path,
            format=self.format,
            file_size=os.path.getsize(self.path),
            encoding=encoding,
            number_rows=number_rows,
            column_names=list(columns),
            storage_types={c: str(t) for c, t in self._types.items()},
            delimiter=delimiter,
            dtypes={c: str(d) for c, d in self._dtypes.items()},
        )

    def _read_header(self) -> list[str]:
        with open(self.path, encoding=python_codec(self._encoding), newline="") as fh:
            try:
                return next(csv.reader(fh, delimiter=self._delimiter))
            except StopIteration:
                raise ReadError(f"{self.path}: file is empty") from None

    def _block_size(self) -> int:
        """Block size that keeps Arrow's readahead within a quarter of the chunk budget."""
        size = self.options.chunk_memory_bytes // (4 * READAHEAD_BLOCKS)
        return int(max(MIN_BLOCK, min(MAX_BLOCK, size)))

    def _chunk_arrow_bytes(self) -> int:
        """Arrow bytes to collect per chunk so the pandas chunk fits the budget."""
        return self.options.chunk_memory_bytes // _EXPANSION[self.options.dtype_backend]

    def _open(self, column_types: dict[str, pa.DataType] | None):
        return pacsv.open_csv(
            self.path,
            read_options=pacsv.ReadOptions(
                block_size=self._block_size(),
                encoding=self._encoding,
            ),
            parse_options=pacsv.ParseOptions(delimiter=self._delimiter, newlines_in_values=True),
            convert_options=pacsv.ConvertOptions(
                column_types=column_types,
                include_columns=self._columns,
                strings_can_be_null=True,
            ),
        )

    def _scan(self) -> tuple[dict[str, pa.DataType], int]:
        """Stream every column as text once and find the narrowest type for each."""
        kinds: dict[str, str | None] = {c: None for c in self._columns}
        rows = 0
        reader = self._open({c: pa.string() for c in self._columns})
        for batch in reader:
            rows += batch.num_rows
            for i, col in enumerate(batch.schema.names):
                if kinds[col] == "string":
                    continue
                values = pc.drop_null(batch.column(i))
                if len(values):
                    kinds[col] = _merge(kinds[col], _classify(values, kinds[col]))
        logger.debug("Scanned %s rows of %s: %s", rows, self.path, kinds)
        return {c: _KIND_TO_ARROW[k or "string"] for c, k in kinds.items()}, rows

    def _sample_types(self) -> dict[str, pa.DataType]:
        """Types Arrow infers from the first block; all-null columns become strings."""
        schema = self._open(None).schema
        return {
            f.name: pa.string() if pa.types.is_null(f.type) else f.type
            for f in schema
        }

    # -- data -------------------------------------------------------------

    def estimate_memory(self) -> int:
        # Arrow data is about the size of the text; pandas object strings are larger.
        return os.path.getsize(self.path) * _EXPANSION[self.options.dtype_backend]

    def _to_pandas(self, table: pa.Table | pa.RecordBatch) -> pd.DataFrame:
        if self.options.dtype_backend == "pyarrow":
            df = table.to_pandas(types_mapper=pd.ArrowDtype)
        else:
            df = table.to_pandas()
        return conform(df, self._dtypes, self.path)

    def _batches(self) -> Iterator[pa.RecordBatch]:
        try:
            yield from self._open(self._types)
        except pa.ArrowInvalid as exc:
            if "conversion error" in str(exc).lower():
                raise SchemaDriftError(
                    f"{self.path}: {exc}. Use csv_type_inference='scan' or set dtypes"
                ) from exc
            raise ReadError(f"{self.path}: {exc}") from exc

    def _iter_frames(self) -> Iterator[pd.DataFrame]:
        """Combine small Arrow blocks into chunks of ``chunk_rows`` rows, or of
        ``chunk_memory_bytes`` when no row count is set. Combining is zero-copy."""
        _ = self.meta
        rows = self.options.chunk_rows
        target_bytes = self._chunk_arrow_bytes()
        pending: list[pa.RecordBatch] = []
        pending_rows = 0
        pending_bytes = 0
        for batch in self._batches():
            pending.append(batch)
            pending_rows += batch.num_rows
            pending_bytes += batch.nbytes
            if rows:
                while pending_rows >= rows:
                    table = pa.Table.from_batches(pending)
                    yield self._to_pandas(table.slice(0, rows))
                    rest = table.slice(rows)
                    pending = rest.to_batches()
                    pending_rows = rest.num_rows
                    pending_bytes = rest.nbytes
            elif pending_bytes >= target_bytes:
                yield self._to_pandas(pa.Table.from_batches(pending))
                pending, pending_rows, pending_bytes = [], 0, 0
        if pending_rows:
            yield self._to_pandas(pa.Table.from_batches(pending))

    def _read_all(self) -> pd.DataFrame:
        _ = self.meta
        table = pa.Table.from_batches(list(self._batches()), schema=pa.schema(self._types))
        if self.meta.number_rows is None:
            self.meta.number_rows = table.num_rows
        return self._to_pandas(table)
