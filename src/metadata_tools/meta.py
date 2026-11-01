"""Format-independent file metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

FileFormat = Literal["csv", "stata", "spss"]


@dataclass
class FileMeta:
    """Metadata for a data file, the same shape for every format.

    Fields a format cannot provide are left empty (CSV has no labels).
    ``number_rows`` is None when the row count is unknown until the file is
    read through (CSV without a scan, some SPSS files).
    """

    path: str
    format: FileFormat
    file_size: int
    encoding: str | None
    number_rows: int | None
    column_names: list[str]
    column_labels: dict[str, str | None] = field(default_factory=dict)
    value_labels: dict[str, dict[Any, str]] = field(default_factory=dict)
    storage_types: dict[str, str] = field(default_factory=dict)
    """Type as stored in the file: readstat types (int8, double, string...)
    for Stata/SPSS, Arrow type names for CSV."""
    formats: dict[str, str] = field(default_factory=dict)
    """Display formats (%td, F8.2, ...). Empty for CSV."""
    measure: dict[str, str] = field(default_factory=dict)
    display_width: dict[str, int] = field(default_factory=dict)
    missing_ranges: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    missing_user_values: dict[str, list[Any]] = field(default_factory=dict)
    file_label: str | None = None
    notes: list[str] = field(default_factory=list)
    delimiter: str | None = None
    dtypes: dict[str, str] = field(default_factory=dict)
    """pandas dtype of each column in every chunk this reader yields."""
    raw: Any = field(default=None, repr=False, compare=False)
    """The underlying library's metadata object, if any (pyreadstat)."""

    @property
    def number_columns(self) -> int:
        return len(self.column_names)
