from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat
import pytest

DATA_DIR = Path(__file__).resolve().parent / "data"
SPSS_USER_MISSINGS = DATA_DIR / "spss_user_missings.sav"
STATA_EXTENDED_MISSINGS = DATA_DIR / "stata_extended_missings.dta"

# Real survey file with latin-1 bytes in a UTF-8 declared file. Not copied
# into this repo; tests using it are skipped when it is not present.
BRAZIL_DTA = Path(os.environ.get("METADATA_TOOLS_BRAZIL_DTA", ""))


def make_frame(n: int = 2_500) -> pd.DataFrame:
    """Numbers, strings and dates, with missing values only in the second half.

    Chunked reads of the second half see NaN/NaT that the first half never
    does, which is what makes pyreadstat dtypes change between chunks.
    """
    rng = np.random.default_rng(0)
    small = rng.integers(0, 5, n).astype(float)
    small[n // 2 :: 7] = np.nan
    when = pd.Series([dt.datetime(2020, 1, 1) + dt.timedelta(days=int(d)) for d in rng.integers(0, 900, n)])
    when[n // 2 :] = pd.NaT
    return pd.DataFrame(
        {
            "id": np.arange(n, dtype=float),
            "small": small,
            "ratio": rng.random(n),
            "name": rng.choice(["alpha", "beta", "gamma"], n),
            "when": when,
        }
    )


@pytest.fixture
def frame() -> pd.DataFrame:
    return make_frame()


@pytest.fixture
def sav_path(tmp_path, frame) -> Path:
    path = tmp_path / "sample.sav"
    pyreadstat.write_sav(
        frame,
        str(path),
        column_labels={"id": "Identifier", "small": "Small codes"},
        variable_value_labels={"small": {0: "none", 1: "one"}},
    )
    return path


@pytest.fixture
def zsav_path(tmp_path, frame) -> Path:
    path = tmp_path / "sample.zsav"
    pyreadstat.write_sav(frame, str(path), compress=True)
    return path


@pytest.fixture
def dta_path(tmp_path, frame) -> Path:
    path = tmp_path / "sample.dta"
    pyreadstat.write_dta(frame, str(path), column_labels={"id": "Identifier"})
    return path
