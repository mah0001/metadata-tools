"""Text encoding and delimiter detection for delimited files."""

from __future__ import annotations

import codecs
import csv
import logging
from collections.abc import Iterable

logger = logging.getLogger(__name__)

_BLOCK = 16 * 1024 * 1024
_SNIFF_BYTES = 64 * 1024

_BOMS = (
    (codecs.BOM_UTF32_LE, "utf-32"),
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF8, "utf-8"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)


def detect_encoding(
    path: str, candidates: Iterable[str | None], max_bytes: int | None = None
) -> str:
    """Return the first candidate that decodes the whole file, or its first
    ``max_bytes`` when given.

    A byte-order mark decides immediately. Otherwise each candidate is checked
    by streaming the file through an incremental decoder, so memory stays at
    one block regardless of file size. ``None`` in ``candidates`` means UTF-8.
    """
    with open(path, "rb") as fh:
        head = fh.read(4)
    for bom, name in _BOMS:
        if head.startswith(bom):
            return name

    tried: list[str] = []
    for candidate in candidates:
        name = codecs.lookup(candidate or "utf-8").name
        if name in tried:
            continue
        tried.append(name)
        if _decodes(path, name, max_bytes):
            return name
        logger.debug("%s is not valid %s", path, name)
    raise UnicodeError(f"{path} does not decode with any of {tried}")


def _decodes(path: str, encoding: str, max_bytes: int | None = None) -> bool:
    decoder = codecs.getincrementaldecoder(encoding)(errors="strict")
    remaining = max_bytes
    try:
        with open(path, "rb") as fh:
            while block := fh.read(_BLOCK if remaining is None else min(_BLOCK, remaining)):
                decoder.decode(block)
                if remaining is not None:
                    remaining -= len(block)
                    if remaining <= 0:
                        # A multi-byte character may be cut at the limit.
                        return True
        decoder.decode(b"", final=True)
    except UnicodeDecodeError:
        return False
    return True


def sniff_delimiter(path: str, encoding: str, default: str = ",") -> str:
    """Guess the delimiter from the first 64 KiB, falling back to ``default``."""
    with open(path, "rb") as fh:
        raw = fh.read(_SNIFF_BYTES)
    text = raw.decode(encoding, errors="replace")
    if len(raw) == _SNIFF_BYTES and "\n" in text:
        text = text[: text.rindex("\n")]
    try:
        return csv.Sniffer().sniff(text, delimiters=",;\t|").delimiter
    except csv.Error:
        return default


def python_codec(encoding: str) -> str:
    """Codec for Python's open(): strips a UTF-8 BOM the same way Arrow does."""
    return "utf-8-sig" if codecs.lookup(encoding).name == "utf-8" else encoding
