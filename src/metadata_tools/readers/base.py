"""Reader interface shared by every format."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import ClassVar

import pandas as pd

from metadata_tools.errors import IncompleteReadError, MemoryBudgetError
from metadata_tools.meta import FileFormat, FileMeta
from metadata_tools.options import ReadOptions

logger = logging.getLogger(__name__)


class Reader(ABC):
    """Reads one data file as a stream of pandas DataFrame chunks.

    Every chunk has the same columns and the same dtypes (``meta.dtypes``),
    and its index continues the row numbering of the previous chunk.

    >>> with open_reader("survey.sav") as reader:
    ...     for chunk in reader.iter_chunks():
    ...         ...
    """

    format: ClassVar[FileFormat]

    def __init__(self, path: str | os.PathLike[str], options: ReadOptions | None = None) -> None:
        self.path = os.fspath(path)
        if not os.path.isfile(self.path):
            raise FileNotFoundError(f"File {self.path} does not exist")
        self.options = options or ReadOptions()
        self.warnings: list[str] = []
        """Fallbacks taken while reading (encoding switches, disabled options)."""
        self.rows_read = 0
        """Rows yielded by the last completed or in-progress iteration."""
        self._meta: FileMeta | None = None

    @property
    def meta(self) -> FileMeta:
        if self._meta is None:
            self._meta = self._load_meta()
        return self._meta

    def iter_chunks(self) -> Iterator[pd.DataFrame]:
        """Yield the file in chunks sized by ``ReadOptions``.

        Raises IncompleteReadError at the end if fewer rows were read than the
        file declares.
        """
        meta = self.meta
        self.rows_read = 0
        for frame in self._iter_frames():
            if frame.empty:
                continue
            frame.index = pd.RangeIndex(self.rows_read, self.rows_read + len(frame))
            self.rows_read += len(frame)
            logger.debug(
                "Read chunk of %s rows from %s (%s/%s)",
                len(frame),
                self.path,
                self.rows_read,
                meta.number_rows,
            )
            yield frame

        if meta.number_rows is not None and self.rows_read != meta.number_rows:
            raise IncompleteReadError(self.path, self.rows_read, meta.number_rows)
        if meta.number_rows is None:
            meta.number_rows = self.rows_read

    def read(self) -> pd.DataFrame:
        """Read the whole file, refusing if it would exceed ``max_full_read_bytes``."""
        estimate = self.estimate_memory()
        limit = self.options.max_full_read_bytes
        if estimate > limit:
            raise MemoryBudgetError(self.path, estimate, limit)
        df = self._read_all()
        self.rows_read = len(df)
        expected = self.meta.number_rows
        if expected is not None and len(df) != expected:
            raise IncompleteReadError(self.path, len(df), expected)
        return df

    @abstractmethod
    def estimate_memory(self) -> int:
        """Estimated bytes for the whole file as one DataFrame."""

    @abstractmethod
    def _load_meta(self) -> FileMeta: ...

    @abstractmethod
    def _iter_frames(self) -> Iterator[pd.DataFrame]:
        """Yield chunks already converted to ``meta.dtypes``."""

    @abstractmethod
    def _read_all(self) -> pd.DataFrame: ...

    def _warn(self, message: str) -> None:
        logger.warning("%s: %s", self.path, message)
        self.warnings.append(message)

    def close(self) -> None:
        """Release resources. Readers open the file per read, so this is a no-op today."""

    def __enter__(self) -> "Reader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.path!r})"


def rows_for_budget(row_bytes: float, options: ReadOptions) -> int:
    """Rows per chunk: fixed ``chunk_rows``, or what fits in ``chunk_memory_bytes``."""
    if options.chunk_rows:
        return options.chunk_rows
    rows = int(options.chunk_memory_bytes // max(row_bytes, 1.0))
    return max(options.min_chunk_rows, min(options.max_chunk_rows, rows))
