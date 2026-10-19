"""Page failures: a failed page fails its unit and its item, and a stream error
from the LLM image-folder path propagates instead of counting as processed."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

import chronotranscriber.methods.llm as llm_method
from chronotranscriber.common.store import Store
from chronotranscriber.methods import Item
from chronotranscriber.methods.llm import LlmTranscriber
from chronotranscriber.methods.pages import transcribe_payload
from chronotranscriber.methods.units import (
    UnitsFailedError,
    run_unit_files,
    run_unit_stream,
)
from chronotranscriber.output.store import database_path, open_store
from tests.conftest import make_job, make_spec
from tests.methods.conftest import TOOL, open_handle, stream, units


@pytest.mark.unit
class TestPageFailuresFailTheItem:
    async def test_file_pipeline_raises_when_page_errors(self, tmp_path: Path) -> None:
        image = tmp_path / "page1.png"
        image.write_bytes(b"")
        db = tmp_path / "chronotranscriber.sqlite"

        async def boom(path: Path, idx: int) -> Any:
            raise RuntimeError("OCR exploded")

        async with Store(db, TOOL) as store:
            handle = await open_handle(store, pages=1, model="tesseract")
            with pytest.raises(UnitsFailedError) as excinfo:
                await run_unit_files(handle, [image], boom)

        assert excinfo.value.failed_pages == 1
        (row,) = units(db)
        assert row["status"] == "failed"
        assert row["content"] == "[transcription error: page1.png]"

    async def test_streaming_pipeline_raises_when_page_errors(
        self, tmp_path: Path
    ) -> None:
        db = tmp_path / "chronotranscriber.sqlite"
        payload = SimpleNamespace(
            image_name="page_0001.png",
            base64="Zm9v",
            mime_type="image/png",
            index=0,
            source_file="doc.pdf",
            page_index=0,
            provenance=lambda: {},
        )
        transcriber = MagicMock()
        transcriber.transcribe_image_from_base64 = MagicMock(
            side_effect=RuntimeError("API 404")
        )

        async with Store(db, TOOL) as store:
            handle = await open_handle(store, pages=1)
            with pytest.raises(UnitsFailedError):
                await run_unit_stream(
                    handle,
                    stream([payload]),
                    transcriber,
                    transcribe_payload,
                    concurrency=2,
                )

        assert [row["status"] for row in units(db)] == ["failed"]


class TestUnitStatusFollowsTheResult:
    @pytest.mark.parametrize(
        ("result", "status"),
        [
            (("p", "img.png", None, None, 0), "failed"),
            (("p", "img.png", "[transcription error: img.png]", None, 0), "failed"),
            (("p", "img.png", "real text", None, 0), "done"),
            (("a", "b"), "failed"),
            (None, "failed"),
        ],
    )
    async def test_status(self, tmp_path: Path, result: Any, status: str) -> None:
        async with open_store(tmp_path) as store:
            handle = await open_handle(store, pages=1)

            async def handler(_path: Path, _index: int) -> Any:
                return result

            try:
                await run_unit_files(handle, [Path("page_0001.png")], handler)
            except UnitsFailedError:
                assert status == "failed"
        (unit,) = units(database_path(tmp_path))
        assert unit["status"] == status


class _GuardError(RuntimeError):
    """A RuntimeError subclass, as the render failure-rate guard raises."""


@pytest.mark.unit
class TestImageFolderStreamErrors:
    async def test_runtime_error_from_the_page_stream_propagates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        folder = tmp_path / "scans"
        folder.mkdir()
        (folder / "page1.png").write_bytes(b"\x89PNG")
        spec = make_spec(folder, type="images", method="llm")
        method = LlmTranscriber(
            make_job(spec, output_dir=tmp_path / "out", input_root=tmp_path)
        )
        method._client = MagicMock()

        def failing_stream(_pages: Any, _indices: Any) -> Any:
            async def gen() -> Any:
                # A RuntimeError subclass must not be swallowed as a skip.
                raise _GuardError("render failure rate exceeded")
                yield

            return gen()

        monkeypatch.setattr(llm_method, "stream_pages", failing_stream)

        async with Store(tmp_path / "chronotranscriber.sqlite", TOOL) as store:
            handle = await open_handle(store, "scans", pages=0)
            item = Item(folder, "images", "llm", document=handle)
            with pytest.raises(_GuardError):
                await method.transcribe(item)
