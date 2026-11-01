"""Stata .dta reader: pyreadstat first, pandas StataReader as a last resort."""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import pandas as pd
import pyreadstat
from pandas.api.types import is_datetime64_any_dtype

from metadata_tools._dtypes import conform, readstat_numeric_dtype, string_dtype
from metadata_tools.errors import ReadError
from metadata_tools.meta import FileMeta
from metadata_tools.readers._readstat import SAMPLE_ROWS, ReadstatReader, is_chr_error

# pandas StataReader type codes -> readstat storage types. Ints are string widths.
_PANDAS_STATA_TYPES = {
    "b": "int8",
    "h": "int16",
    "l": "int32",
    "f": "float",
    "d": "double",
}


class StataReader(ReadstatReader):
    """Reader for Stata .dta files.

    With ``user_missing=True`` extended missings (.a-.z) come back as letters,
    so numeric columns are object dtype in every chunk. If pyreadstat cannot
    decode the file with any encoding, the pandas StataReader is used instead;
    it reads extended missings as NaN.
    """

    format = "stata"
    _read_fn = staticmethod(pyreadstat.read_dta)

    def __init__(self, path, options=None) -> None:
        super().__init__(path, options)
        self._pandas_mode = False
        self._pandas_columns: list[str] | None = None

    def _numeric_as_object(self) -> bool:
        return self.options.user_missing and not self._pandas_mode

    def _recover(self, kwargs: dict[str, Any], exc: BaseException) -> dict[str, Any] | None:
        if is_chr_error(exc) and kwargs.get("user_missing"):
            self._warn(f"extended missing codes could not be read ({exc}); reading them as NaN")
            return {**kwargs, "user_missing": False}
        return super()._recover(kwargs, exc)

    def _read_with_recovery(self, kwargs: dict[str, Any], offset: int, limit: int):
        try:
            return super()._read_with_recovery(kwargs, offset, limit)
        except ValueError as exc:
            if not (is_chr_error(exc) and kwargs.get("user_missing")):
                raise
            self._warn(
                f"extended missing codes could not be read in rows {offset}-{offset + limit} "
                f"({exc}); reading them as NaN in that range"
            )
            return super()._read_with_recovery({**kwargs, "user_missing": False}, offset, limit)

    # -- pandas fallback --------------------------------------------------

    def _open_pandas(self) -> pd.io.stata.StataReader:
        return pd.io.stata.StataReader(
            self.path,
            convert_dates=self.options.convert_dates,
            convert_categoricals=False,
            convert_missing=False,
        )

    def _fallback_meta(self) -> FileMeta:
        error = getattr(self, "_probe_error", None)
        self._warn(f"pyreadstat could not decode the file ({error}); using pandas StataReader")
        self._pandas_mode = True
        usecols = self.options.usecols
        with self._open_pandas() as reader:
            reader._ensure_open()
            varlist = list(reader._varlist)
            typlist = list(reader._typlist)
            fmtlist = list(reader._fmtlist)
            lbllist = list(getattr(reader, "_lbllist", varlist))
            nobs = reader._nobs
            var_labels = reader.variable_labels()
            label_sets = reader.value_labels()
            data_label = reader.data_label
            selected = [n for n in varlist if usecols is None or n in usecols]
            self._pandas_columns = selected
            sample = reader.read(SAMPLE_ROWS, columns=selected)

        idx = {name: i for i, name in enumerate(varlist)}
        storage = {
            n: "string" if isinstance(typlist[idx[n]], int) else _PANDAS_STATA_TYPES.get(typlist[idx[n]], "double")
            for n in selected
        }
        backend = self.options.dtype_backend
        self._dtypes = {}
        for n in selected:
            if storage[n] == "string":
                self._dtypes[n] = string_dtype(backend)
            elif is_datetime64_any_dtype(sample[n].dtype):
                self._dtypes[n] = sample[n].dtype
            else:
                self._dtypes[n] = readstat_numeric_dtype(storage[n], backend)
        sample = conform(sample, self._dtypes, self.path)
        if len(sample):
            self._row_bytes = sample.memory_usage(deep=True, index=False).sum() / len(sample)

        value_labels = {}
        for n in selected:
            labels = label_sets.get(lbllist[idx[n]]) if lbllist[idx[n]] else None
            if labels:
                value_labels[n] = {(int(k) if hasattr(k, "item") else k): v for k, v in labels.items()}

        return FileMeta(
            path=self.path,
            format=self.format,
            file_size=os.path.getsize(self.path),
            encoding=None,
            number_rows=nobs,
            column_names=selected,
            column_labels={n: var_labels.get(n, "") for n in selected},
            value_labels=value_labels,
            storage_types=storage,
            formats={n: fmtlist[idx[n]] for n in selected},
            file_label=data_label or None,
            dtypes={n: str(self._dtypes[n]) for n in selected},
        )

    def _pandas_frames(self, start: int, limit: int) -> Iterator[pd.DataFrame]:
        columns = self._pandas_columns or self.meta.column_names
        with self._open_pandas() as reader:
            skipped = 0
            while skipped < start:
                part = reader.read(min(limit, start - skipped), columns=columns)
                if part is None or part.empty:
                    raise ReadError(f"{self.path}: could not skip to row {start} with pandas")
                skipped += len(part)
            while True:
                try:
                    chunk = reader.read(limit, columns=columns)
                except StopIteration:
                    return
                if chunk is None or chunk.empty:
                    return
                yield conform(chunk[columns], self._dtypes, self.path)
                if len(chunk) < limit:
                    return

    def _iter_frames(self) -> Iterator[pd.DataFrame]:
        _ = self.meta
        if self._pandas_mode:
            yield from self._pandas_frames(0, self.chunk_rows())
        else:
            yield from super()._iter_frames()

    def _fallback_frames(self, offset: int, limit: int, exc: ReadError) -> Iterator[pd.DataFrame]:
        self._warn(f"pyreadstat failed at row {offset} ({exc}); reading the rest with pandas")
        self._pandas_columns = self.meta.column_names
        return self._pandas_frames(offset, limit)

    def _read_all(self) -> pd.DataFrame:
        _ = self.meta
        if self._pandas_mode:
            with self._open_pandas() as reader:
                df = reader.read(columns=self._pandas_columns)
            return conform(df, self._dtypes, self.path)
        return super()._read_all()
