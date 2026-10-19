"""Fake Tesseract engine for the characterization tests.

``FakeTesseract.install`` replaces pytesseract's OCR call and version check
wherever they are bound (see ``seams.replace``). Each call records the
image's file name, size and mode and the config string, and returns text
derived from the page marker.

The Tesseract preprocessing (grayscale, Otsu binarization, border) turns a
uniform gray marker page into a white page, so the gray level is gone by the
time OCR runs. The page marker is therefore read from the preprocessed file
name, which carries the 1-based page or image number
(``page_0001_tess_preprocessed.png``).

``block_real_tesseract`` points pytesseract at a missing executable, so a
call that escapes the fake raises ``TesseractNotFoundError`` instead of
running an installed engine.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytesseract
import pytesseract.pytesseract as pytesseract_impl
import pytest

from tests.characterization.seams import replace

FAKE_VERSION = "5.3.0-fake"
MISSING_TESSERACT = Path("no-tesseract") / "tesseract.exe"
"""Executable path (under a test root) that never exists."""

_PAGE_NUMBER_RE = re.compile(r"(\d+)(?=\D*_tess_preprocessed)")


def ocr_text(page: int) -> str:
    """Return the fake OCR text of 1-based page *page*."""
    return f"OCR page {page}: wheat 12 sous, rye 9 sous in {1600 + page}."


def page_number(name: str) -> int | None:
    """Return the 1-based page number a preprocessed file name carries."""
    match = _PAGE_NUMBER_RE.search(name)
    return int(match.group(1)) if match else None


def block_real_tesseract(monkeypatch: pytest.MonkeyPatch, root: Path) -> Path:
    """Point pytesseract at a missing executable under *root*; return its path."""
    missing = root / MISSING_TESSERACT
    monkeypatch.setattr(pytesseract_impl, "tesseract_cmd", str(missing))
    return missing


def tesseract_cmd_config(missing: Path) -> dict[str, Any]:
    """Return an ``image_processing_config`` override naming *missing*."""
    return {"tesseract_image_processing": {"ocr": {"tesseract_cmd": str(missing)}}}


class FakeTesseract:
    """Stand-in for pytesseract's OCR call and availability check.

    ``texts`` overrides the answer for a 1-based page number; an empty
    string makes the page blank. ``available`` False makes the version
    check fail as a missing engine does.
    """

    def __init__(self, texts: Mapping[int, str] | None = None) -> None:
        self.texts: dict[int, str] = dict(texts or {})
        self.available = True
        self.calls: list[dict[str, Any]] = []
        self.version_checks = 0
        self._lock = threading.Lock()

    def install(self, monkeypatch: pytest.MonkeyPatch) -> FakeTesseract:
        """Replace the pytesseract entry points the tool calls."""
        replace(monkeypatch, "pytesseract", "image_to_string", self.image_to_string)
        replace(
            monkeypatch,
            "pytesseract",
            "get_tesseract_version",
            self.get_tesseract_version,
        )
        return self

    def reset(self) -> None:
        """Forget recorded calls and version checks."""
        with self._lock:
            self.calls.clear()
            self.version_checks = 0

    def get_tesseract_version(self, *args: Any, **kwargs: Any) -> str:
        """Return the fake version, or raise as a missing engine does."""
        with self._lock:
            self.version_checks += 1
        if not self.available:
            raise pytesseract.TesseractNotFoundError()
        return FAKE_VERSION

    def image_to_string(
        self, image: Any, lang: str | None = None, config: str = "", **kwargs: Any
    ) -> str:
        """Record the call and return the text of the image's page."""
        name = Path(str(getattr(image, "filename", "") or "")).name
        page = page_number(name)
        record: dict[str, Any] = {
            "image": name,
            "page": page,
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "config": config,
        }
        if lang is not None:
            record["lang"] = lang
        if kwargs:
            record["kwargs"] = sorted(kwargs)
        with self._lock:
            self.calls.append(record)
        if page is None:
            return "OCR text of an unnumbered image."
        return self.texts.get(page, ocr_text(page))

    def sorted_calls(self) -> list[dict[str, Any]]:
        """Return the recorded calls in file-name order."""
        with self._lock:
            return sorted(
                (dict(call) for call in self.calls), key=lambda call: call["image"]
            )


__all__ = [
    "FAKE_VERSION",
    "MISSING_TESSERACT",
    "FakeTesseract",
    "block_real_tesseract",
    "ocr_text",
    "page_number",
    "tesseract_cmd_config",
]
