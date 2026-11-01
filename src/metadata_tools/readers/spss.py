"""SPSS .sav / .zsav reader."""

from __future__ import annotations

import pyreadstat

from metadata_tools.readers._readstat import ReadstatReader


class SpssReader(ReadstatReader):
    """Reader for SPSS .sav and compressed .zsav files.

    With ``user_missing=True`` user-defined missing values stay as numbers and
    are listed in ``meta.missing_user_values`` / ``meta.missing_ranges``.
    """

    format = "spss"
    _read_fn = staticmethod(pyreadstat.read_sav)
