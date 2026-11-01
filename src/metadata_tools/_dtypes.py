"""Pin one dtype per column and convert every chunk to it."""

from __future__ import annotations

import pandas as pd
import pyarrow as pa
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

from metadata_tools.errors import SchemaDriftError
from metadata_tools.options import DtypeBackend

# readstat storage type -> Arrow type used with the pyarrow backend.
_READSTAT_TO_ARROW = {
    "int8": pa.int8(),
    "int16": pa.int16(),
    "int32": pa.int32(),
    "int64": pa.int64(),
    "float": pa.float32(),
    "double": pa.float64(),
}


def string_dtype(backend: DtypeBackend):
    return pd.ArrowDtype(pa.string()) if backend == "pyarrow" else pd.api.types.pandas_dtype(object)


def readstat_numeric_dtype(storage_type: str, backend: DtypeBackend):
    if backend == "pyarrow":
        return pd.ArrowDtype(_READSTAT_TO_ARROW.get(storage_type, pa.float64()))
    return pd.api.types.pandas_dtype("float64")


def arrow_to_pandas_dtype(arrow_type: pa.DataType, backend: DtypeBackend):
    """Dtype for a CSV column of the given Arrow type."""
    if backend == "pyarrow":
        return pd.ArrowDtype(arrow_type)
    if pa.types.is_integer(arrow_type) or pa.types.is_floating(arrow_type):
        return pd.api.types.pandas_dtype("float64")
    if pa.types.is_timestamp(arrow_type):
        return pd.api.types.pandas_dtype(f"datetime64[{arrow_type.unit}]")
    return pd.api.types.pandas_dtype(object)


def conform(df: pd.DataFrame, dtypes: dict[str, object], path: str) -> pd.DataFrame:
    """Return ``df`` with every column converted to its pinned dtype."""
    if all(df[col].dtype == dtypes[col] for col in df.columns):
        return df
    columns = {}
    for col in df.columns:
        series = df[col]
        target = dtypes[col]
        if series.dtype != target:
            try:
                series = _convert(series, target)
            except (TypeError, ValueError, pa.ArrowException) as exc:
                raise SchemaDriftError(
                    f"{path}: column {col!r} is {series.dtype} in this chunk and cannot "
                    f"be converted to {target}: {exc}"
                ) from exc
        columns[col] = series
    return pd.DataFrame(columns, index=df.index, copy=False)


def _convert(series: pd.Series, target) -> pd.Series:
    if is_datetime64_any_dtype(target):
        # An all-missing date chunk comes back as object (None/NaT). Numbers
        # must not be read as epoch offsets.
        if is_numeric_dtype(series.dtype):
            if series.notna().any():
                raise TypeError("numeric values for a datetime column")
            return pd.Series(pd.NaT, index=series.index, dtype=target)
        return pd.to_datetime(series).astype(target)
    if isinstance(target, pd.ArrowDtype) and series.dtype == object:
        arr = pa.array(series.to_numpy(), type=target.pyarrow_dtype, from_pandas=True)
        return pd.Series(pd.arrays.ArrowExtensionArray(arr), index=series.index)
    return series.astype(target)
