"""Exceptions raised by metadata-tools readers."""

from __future__ import annotations


class MetadataToolsError(Exception):
    """Base class for all metadata-tools errors."""


class UnsupportedFormatError(MetadataToolsError):
    """The file extension or requested format has no reader."""


class ReadError(MetadataToolsError):
    """The file could not be decoded with any of the configured strategies."""


class IncompleteReadError(ReadError):
    """Fewer rows were read than the file header declares."""

    def __init__(self, path: str, rows_read: int, expected_rows: int) -> None:
        self.path = path
        self.rows_read = rows_read
        self.expected_rows = expected_rows
        super().__init__(
            f"Incomplete read of {path}: read {rows_read} rows, expected {expected_rows}"
        )


class SchemaDriftError(ReadError):
    """A chunk could not be converted to the dtypes pinned for the file."""


class MemoryBudgetError(MetadataToolsError):
    """A full in-memory read would exceed ``ReadOptions.max_full_read_bytes``."""

    def __init__(self, path: str, estimated_bytes: int, limit_bytes: int) -> None:
        self.path = path
        self.estimated_bytes = estimated_bytes
        self.limit_bytes = limit_bytes
        super().__init__(
            f"Reading {path} fully needs ~{estimated_bytes:,} bytes, over the "
            f"{limit_bytes:,} byte limit; use iter_chunks() instead"
        )
