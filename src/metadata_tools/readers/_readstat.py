"""Chunked reading of pyreadstat formats (SPSS, Stata) with decode fallbacks."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterator
from typing import Any

import pandas as pd
import pyreadstat
from pandas.api.types import is_datetime64_any_dtype

from metadata_tools._dtypes import conform, readstat_numeric_dtype, string_dtype
from metadata_tools.errors import ReadError
from metadata_tools.meta import FileMeta
from metadata_tools.readers.base import Reader, rows_for_budget

logger = logging.getLogger(__name__)

SAMPLE_ROWS = 1_000
_UNSET = object()


def is_encoding_error(exc: BaseException) -> bool:
    """Bytes that do not decode in the current encoding (a different one may work)."""
    if isinstance(exc, UnicodeError):
        return True
    return isinstance(exc, pyreadstat.ReadstatError) and "invalid byte sequence" in str(exc)


def is_date_error(exc: BaseException) -> bool:
    """pyreadstat cannot convert a date value (e.g. Stata date missings)."""
    if isinstance(exc, OverflowError):
        return True
    msg = str(exc)
    return isinstance(exc, ValueError) and "must have magnitude" in msg and "days=" in msg


def is_chr_error(exc: BaseException) -> bool:
    """pyreadstat cannot map a Stata extended-missing code to a letter."""
    return isinstance(exc, ValueError) and "chr() arg not in range" in str(exc)


_DECODE_ERRORS = (UnicodeError, OverflowError, ValueError, pyreadstat.ReadstatError)


class ReadstatReader(Reader):
    """Base for formats read with pyreadstat ``row_offset``/``row_limit``.

    Each chunk is a separate pyreadstat call. Skipping to an offset is cheap
    (about 1.0x for uncompressed and 1.3-1.6x for compressed SPSS at the end of
    a 500k-row file versus the start), so budget-sized chunks keep total
    overhead low.
    """

    _read_fn: Callable[..., tuple[pd.DataFrame, Any]]

    def __init__(self, path, options=None) -> None:
        super().__init__(path, options)
        self._kwargs: dict[str, Any] = {}
        self._dtypes: dict[str, Any] = {}
        self._row_bytes = 1.0

    # -- metadata ---------------------------------------------------------

    def _base_kwargs(self) -> dict[str, Any]:
        opts = self.options
        kwargs: dict[str, Any] = {"usecols": opts.usecols, "user_missing": opts.user_missing}
        if opts.convert_dates:
            kwargs["dates_as_pandas_datetime"] = True
        else:
            kwargs["disable_datetime_conversion"] = True
        return kwargs

    def _encoding_candidates(self, first: Any = _UNSET) -> list[str | None]:
        """Encodings to try in order, starting with ``first`` when given."""
        opts = self.options
        base = [opts.encoding] if opts.encoding else list(opts.encodings)
        ordered: list[str | None] = []
        for enc in ([] if first is _UNSET else [first]) + base:
            if enc not in ordered:
                ordered.append(enc)
        return ordered

    def _with_encoding(self, kwargs: dict[str, Any], encoding: str | None) -> dict[str, Any]:
        out = dict(kwargs)
        out.pop("encoding", None)
        if encoding is not None:
            out["encoding"] = encoding
        return out

    def _recover(self, kwargs: dict[str, Any], exc: BaseException) -> dict[str, Any] | None:
        """kwargs that avoid ``exc`` by giving up an option, or None."""
        if is_date_error(exc) and not kwargs.get("disable_datetime_conversion"):
            self._warn(f"date conversion failed ({exc}); dates are read as raw numbers")
            out = dict(kwargs)
            out.pop("dates_as_pandas_datetime", None)
            out["disable_datetime_conversion"] = True
            return out
        return None

    def _probe(self) -> tuple[dict[str, Any], Any, pd.DataFrame] | None:
        """Find kwargs that read the metadata and a sample; None if no encoding works."""
        last_error: BaseException | None = None
        for encoding in self._encoding_candidates():
            kwargs = self._with_encoding(self._base_kwargs(), encoding)
            while True:
                try:
                    _, raw = self._read_header(kwargs)
                    sample, _ = self._read_fn(self.path, row_limit=SAMPLE_ROWS, **kwargs)
                    if encoding != self._encoding_candidates()[0]:
                        self._warn(f"read with encoding {encoding!r}")
                    return kwargs, raw, sample
                except _DECODE_ERRORS as exc:
                    last_error = exc
                    recovered = self._recover(kwargs, exc)
                    if recovered is not None:
                        kwargs = recovered
                        continue
                    if not is_encoding_error(exc):
                        raise ReadError(f"{self.path}: {exc}") from exc
                    logger.debug("%s: encoding %r failed: %s", self.path, encoding, exc)
                    break
        self._probe_error = last_error
        return None

    def _read_header(self, kwargs: dict[str, Any]):
        """Metadata only. ``user_missing=True`` is what makes pyreadstat fill
        in declared missing values and ranges; it has no other effect here."""
        return self._read_fn(self.path, metadataonly=True, **{**kwargs, "user_missing": True})

    def read_header(self) -> FileMeta:
        """File metadata from the header alone, without reading any data.

        ``meta.dtypes`` is left empty since no data is sampled.
        """
        last_error: BaseException | None = None
        for encoding in self._encoding_candidates():
            kwargs = self._with_encoding(self._base_kwargs(), encoding)
            try:
                _, raw = self._read_header(kwargs)
            except _DECODE_ERRORS as exc:
                if not is_encoding_error(exc):
                    raise ReadError(f"{self.path}: {exc}") from exc
                last_error = exc
                continue
            self._kwargs = kwargs
            return self._meta_from_pyreadstat(raw)
        self._probe_error = last_error
        return self._fallback_meta()

    def _load_meta(self) -> FileMeta:
        probed = self._probe()
        if probed is None:
            return self._fallback_meta()
        self._kwargs, raw, sample = probed
        self._dtypes = self._plan_dtypes(raw, sample)
        sample = conform(sample, self._dtypes, self.path)
        if len(sample):
            self._row_bytes = sample.memory_usage(deep=True, index=False).sum() / len(sample)
        else:
            self._row_bytes = 8.0 * max(len(raw.column_names), 1)
        return self._meta_from_pyreadstat(raw)

    def _fallback_meta(self) -> FileMeta:
        """Metadata when pyreadstat cannot decode the file at all."""
        error = getattr(self, "_probe_error", None)
        raise ReadError(f"{self.path}: no encoding worked, last error: {error}") from error

    def _plan_dtypes(self, raw: Any, sample: pd.DataFrame) -> dict[str, Any]:
        backend = self.options.dtype_backend
        dtypes: dict[str, Any] = {}
        for col in raw.column_names:
            storage = raw.readstat_variable_types.get(col, "double")
            observed = sample[col].dtype if col in sample else None
            if storage == "string":
                dtypes[col] = string_dtype(backend)
            elif observed is not None and is_datetime64_any_dtype(observed):
                dtypes[col] = observed
            elif observed == object or self._numeric_as_object():
                # Times, dates with no values in the sample, or user-missing
                # codes mixed with numbers.
                dtypes[col] = pd.api.types.pandas_dtype(object)
            else:
                dtypes[col] = readstat_numeric_dtype(storage, backend)
        return dtypes

    def _numeric_as_object(self) -> bool:
        """True when numeric columns can hold non-numeric missing codes."""
        return False

    def _meta_from_pyreadstat(self, raw: Any) -> FileMeta:
        cols = list(raw.column_names)
        rows = raw.number_rows
        return FileMeta(
            path=self.path,
            format=self.format,
            file_size=os.path.getsize(self.path),
            encoding=self._kwargs.get("encoding") or getattr(raw, "file_encoding", None),
            number_rows=rows if rows is not None and rows >= 0 else None,
            column_names=cols,
            column_labels=dict(raw.column_names_to_labels),
            value_labels={k: dict(v) for k, v in (raw.variable_value_labels or {}).items()},
            storage_types=dict(raw.readstat_variable_types),
            formats=dict(raw.original_variable_types),
            measure=dict(getattr(raw, "variable_measure", {}) or {}),
            display_width=dict(getattr(raw, "variable_display_width", {}) or {}),
            missing_ranges=dict(getattr(raw, "missing_ranges", {}) or {}),
            missing_user_values=dict(getattr(raw, "missing_user_values", {}) or {}),
            file_label=getattr(raw, "file_label", None),
            notes=list(getattr(raw, "notes", None) or []),
            dtypes={c: str(self._dtypes[c]) for c in cols if c in self._dtypes},
            raw=raw,
        )

    # -- data -------------------------------------------------------------

    def chunk_rows(self) -> int:
        _ = self.meta  # sample-based row size is measured while loading metadata
        return rows_for_budget(self._row_bytes, self.options)

    def estimate_memory(self) -> int:
        meta = self.meta
        if meta.number_rows is None:
            # Row count unknown until read through: a deliberately high guess.
            return 4 * meta.file_size
        return int(meta.number_rows * self._row_bytes)

    def _read_chunk(self, offset: int, limit: int) -> pd.DataFrame:
        """One pyreadstat read, trying other encodings if this range fails to decode.

        The encoding probe only sees the first rows, so a later chunk can still
        contain bytes the chosen encoding rejects.
        """
        current = self._kwargs.get("encoding")
        last_error: BaseException | None = None
        for encoding in self._encoding_candidates(current):
            kwargs = self._with_encoding(self._kwargs, encoding)
            try:
                df, _ = self._read_with_recovery(kwargs, offset, limit)
            except _DECODE_ERRORS as exc:
                if not is_encoding_error(exc):
                    raise
                last_error = exc
                continue
            if encoding != current:
                self._warn(
                    f"encoding {current!r} failed at row {offset} ({last_error}); "
                    f"continuing with {encoding!r}"
                )
            self._kwargs = self._with_encoding(self._kwargs, encoding)
            return df
        raise ReadError(f"{self.path}: rows from {offset} do not decode: {last_error}") from last_error

    def _read_with_recovery(self, kwargs: dict[str, Any], offset: int, limit: int):
        return self._read_fn(self.path, row_offset=offset, row_limit=limit, **kwargs)

    def _iter_frames(self) -> Iterator[pd.DataFrame]:
        expected = self.meta.number_rows
        limit = self.chunk_rows()
        offset = 0
        while expected is None or offset < expected:
            try:
                df = self._read_chunk(offset, limit)
            except ReadError as exc:
                fallback = self._fallback_frames(offset, limit, exc)
                if fallback is None:
                    raise
                yield from fallback
                return
            if len(df) == 0:
                break
            yield conform(df, self._dtypes, self.path)
            offset += len(df)
            if len(df) < limit:
                break

    def _fallback_frames(self, offset: int, limit: int, exc: ReadError) -> Iterator[pd.DataFrame] | None:
        """Another way to read rows from ``offset`` on, or None to re-raise."""
        return None

    def _read_all(self) -> pd.DataFrame:
        df = self._read_chunk(0, 0)
        return conform(df, self._dtypes, self.path)
