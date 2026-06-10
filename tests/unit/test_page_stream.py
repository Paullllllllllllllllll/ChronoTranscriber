"""Offline tests for the streaming in-memory image pipeline.

Covers the page-payload producer (``modules.images.page_stream``), the
bounded queue runner (``run_streaming_transcription_tasks``), the
streaming transcription pipeline (lean JSONL records, provenance, resume
merge), and the repair re-render fallback. No API calls; PDFs are
generated with fitz in-memory.
"""

from __future__ import annotations

import asyncio
import base64
import json
import tracemalloc
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import fitz
import pytest
from PIL import Image

# Priming import to avoid a circular-import chain when this module is the
# first to touch modules.images.
import modules.transcribe.dual_mode  # noqa: F401
from modules.batch.repair import ImageEntry, Job, _resolve_repair_targets
from modules.images.page_stream import (
    PagePayload,
    compute_folder_skip_names,
    compute_pdf_skip_indices,
    folder_image_name,
    list_folder_images,
    load_image_payload,
    parse_pdf_page_index,
    pdf_page_image_name,
    render_single_pdf_page_payload,
    stream_folder_payloads,
    stream_pdf_payloads,
)
from modules.infra.concurrency import run_streaming_transcription_tasks
from modules.transcribe.pipeline import (
    build_file_provenance,
    run_streaming_transcription_pipeline,
)

IMG_CFG: dict[str, Any] = {
    "grayscale_conversion": True,
    "handle_transparency": True,
    "jpeg_quality": 90,
    "llm_detail": "high",
    "high_target_box": [64, 128],
    "resize_profile": "auto",
}


def _make_pdf(path: Path, pages: int = 3) -> Path:
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page(width=200, height=100)
        page.insert_text((20, 50), f"Page {i + 1}")
    doc.save(str(path))
    doc.close()
    return path


def _make_image_folder(folder: Path, count: int = 3) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        Image.new("RGB", (60, 40), (i * 20, i * 20, i * 20)).save(
            folder / f"scan_{i:02d}.png"
        )
    return folder


async def _collect(it: AsyncIterator[PagePayload]) -> list[PagePayload]:
    return [p async for p in it]


# ---------------------------------------------------------------------------
# Naming helpers
# ---------------------------------------------------------------------------


class TestNaming:
    @pytest.mark.unit
    def test_pdf_page_image_name_round_trip(self) -> None:
        assert pdf_page_image_name(0) == "page_0001_pre_processed.jpg"
        assert parse_pdf_page_index("page_0001_pre_processed.jpg") == 0
        assert parse_pdf_page_index("page_2130_pre_processed.jpg") == 2129
        assert parse_pdf_page_index("scan_01_pre_processed.jpg") is None

    @pytest.mark.unit
    def test_folder_image_name(self, tmp_path: Path) -> None:
        assert (
            folder_image_name(tmp_path / "scan_01.png") == "scan_01_pre_processed.jpg"
        )


# ---------------------------------------------------------------------------
# PDF producer
# ---------------------------------------------------------------------------


class TestStreamPdfPayloads:
    @pytest.mark.unit
    def test_yields_all_pages_with_provenance(self, tmp_path: Path) -> None:
        pdf = _make_pdf(tmp_path / "doc.pdf", pages=3)
        payloads = asyncio.run(
            _collect(
                stream_pdf_payloads(
                    pdf, target_dpi=72, img_cfg=IMG_CFG, model_type="openai"
                )
            )
        )
        assert [p.index for p in payloads] == [0, 1, 2]
        assert [p.image_name for p in payloads] == [
            "page_0001_pre_processed.jpg",
            "page_0002_pre_processed.jpg",
            "page_0003_pre_processed.jpg",
        ]
        for p in payloads:
            assert p.mime_type == "image/jpeg"
            assert p.source_file == str(pdf)
            assert p.page_index == p.index
            assert p.effective_dpi == 72
            assert p.byte_size > 0 and p.width > 0 and p.height > 0
            decoded = base64.b64decode(p.base64)
            assert len(decoded) == p.byte_size
            assert decoded[:3] == b"\xff\xd8\xff"  # JPEG bytes
            prov = p.provenance()
            assert prov["sha256"] == p.sha256 and len(p.sha256) == 64

    @pytest.mark.unit
    def test_respects_slice_and_skip(self, tmp_path: Path) -> None:
        pdf = _make_pdf(tmp_path / "doc.pdf", pages=5)
        payloads = asyncio.run(
            _collect(
                stream_pdf_payloads(
                    pdf,
                    target_dpi=72,
                    img_cfg=IMG_CFG,
                    model_type="openai",
                    page_indices=[1, 2, 3, 99],
                    skip_indices={2},
                )
            )
        )
        assert [p.index for p in payloads] == [1, 3]

    @pytest.mark.unit
    def test_sha256_stable_across_runs(self, tmp_path: Path) -> None:
        pdf = _make_pdf(tmp_path / "doc.pdf", pages=1)

        def first_sha() -> str:
            payloads = asyncio.run(
                _collect(
                    stream_pdf_payloads(
                        pdf, target_dpi=72, img_cfg=IMG_CFG, model_type="openai"
                    )
                )
            )
            return payloads[0].sha256

        assert first_sha() == first_sha()

    @pytest.mark.unit
    def test_memory_stays_bounded(self, tmp_path: Path) -> None:
        pdf = _make_pdf(tmp_path / "doc.pdf", pages=40)

        async def run() -> int:
            count = 0
            async for _payload in stream_pdf_payloads(
                pdf, target_dpi=150, img_cfg=IMG_CFG, model_type="openai"
            ):
                count += 1
            return count

        tracemalloc.start()
        count = asyncio.run(run())
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert count == 40
        # One raw page at a time: peak must stay far below an all-pages
        # accumulation (coarse structural bound).
        assert peak < 80 * 1024 * 1024


# ---------------------------------------------------------------------------
# Folder producer
# ---------------------------------------------------------------------------


class TestStreamFolderPayloads:
    @pytest.mark.unit
    def test_yields_sorted_sources_with_virtual_names(self, tmp_path: Path) -> None:
        folder = _make_image_folder(tmp_path / "scans", count=3)
        payloads = asyncio.run(
            _collect(
                stream_folder_payloads(folder, img_cfg=IMG_CFG, model_type="openai")
            )
        )
        assert [p.image_name for p in payloads] == [
            "scan_00_pre_processed.jpg",
            "scan_01_pre_processed.jpg",
            "scan_02_pre_processed.jpg",
        ]
        assert [p.index for p in payloads] == [0, 1, 2]
        assert all(p.page_index is None for p in payloads)
        assert payloads[0].source_file == str(folder / "scan_00.png")

    @pytest.mark.unit
    def test_skip_matches_virtual_and_raw_names(self, tmp_path: Path) -> None:
        folder = _make_image_folder(tmp_path / "scans", count=3)
        payloads = asyncio.run(
            _collect(
                stream_folder_payloads(
                    folder,
                    img_cfg=IMG_CFG,
                    model_type="openai",
                    skip_names={"scan_00_pre_processed.jpg", "scan_02.png"},
                )
            )
        )
        assert [p.image_name for p in payloads] == ["scan_01_pre_processed.jpg"]

    @pytest.mark.unit
    def test_list_folder_images_sorted(self, tmp_path: Path) -> None:
        folder = _make_image_folder(tmp_path / "scans", count=2)
        (folder / "notes.txt").write_text("not an image", encoding="utf-8")
        files = list_folder_images(folder)
        assert [f.name for f in files] == ["scan_00.png", "scan_01.png"]


# ---------------------------------------------------------------------------
# Skip-set helpers (resume interop with legacy JSONLs)
# ---------------------------------------------------------------------------


class TestSkipSets:
    @pytest.mark.unit
    def test_pdf_skip_indices_from_legacy_jsonl(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "doc.jsonl"
        records = [
            # Legacy disk-pipeline record (path-based)
            {
                "file_name": "doc.pdf",
                "pre_processed_image": "C:/out/page_0001_pre_processed.jpg",
                "image_name": "page_0001_pre_processed.jpg",
                "method": "gpt",
                "order_index": 0,
                "text_chunk": "text",
            },
            # New streaming record (lean)
            {
                "file_name": "doc.pdf",
                "pre_processed_image": None,
                "image_name": "page_0003_pre_processed.jpg",
                "method": "gpt",
                "order_index": 2,
                "text_chunk": "text",
            },
            # Metadata records must be ignored
            {"batch_session": {"status": "submitted"}},
            {"file_provenance": {"source_file": "C:/in/doc.pdf"}},
        ]
        jsonl.write_text(
            "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
        )
        assert compute_pdf_skip_indices(jsonl) == {0, 2}

    @pytest.mark.unit
    def test_folder_skip_names(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "scans.jsonl"
        record = {
            "folder_name": "scans",
            "image_name": "scan_00_pre_processed.jpg",
            "method": "gpt",
            "order_index": 0,
            "text_chunk": "text",
        }
        jsonl.write_text(json.dumps(record) + "\n", encoding="utf-8")
        assert compute_folder_skip_names(jsonl) == {"scan_00_pre_processed.jpg"}

    @pytest.mark.unit
    def test_missing_jsonl_yields_empty_sets(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.jsonl"
        assert compute_pdf_skip_indices(missing) == set()
        assert compute_folder_skip_names(missing) == set()


# ---------------------------------------------------------------------------
# Bounded queue runner
# ---------------------------------------------------------------------------


class TestRunStreamingTranscriptionTasks:
    @pytest.mark.unit
    def test_all_items_processed_and_streamed(self) -> None:
        async def run() -> tuple[list[Any], list[Any]]:
            async def producer() -> AsyncIterator[int]:
                for i in range(10):
                    yield i

            streamed: list[Any] = []

            async def on_result(r: Any) -> None:
                streamed.append(r)

            results = await run_streaming_transcription_tasks(
                producer(), lambda x: _double(x), 3, on_result=on_result
            )
            return results, streamed

        async def _double(x: int) -> int:
            return x * 2

        results, streamed = asyncio.run(run())
        assert sorted(results) == [i * 2 for i in range(10)]
        assert sorted(streamed) == sorted(results)

    @pytest.mark.unit
    def test_queue_stays_bounded(self) -> None:
        limit = 2

        async def run() -> int:
            produced = 0
            consumed = 0
            max_ahead = 0

            async def producer() -> AsyncIterator[int]:
                nonlocal produced
                for i in range(20):
                    produced += 1
                    yield i

            async def handler(x: int) -> int:
                nonlocal consumed, max_ahead
                max_ahead = max(max_ahead, produced - consumed)
                await asyncio.sleep(0.005)
                consumed += 1
                return x

            await run_streaming_transcription_tasks(producer(), handler, limit)
            return max_ahead

        max_ahead = asyncio.run(run())
        # Producer can be at most queue size (2*limit) + in-flight workers
        # (limit) + 1 (item awaiting put) ahead of consumption.
        assert max_ahead <= 2 * limit + limit + 1

    @pytest.mark.unit
    def test_producer_exception_propagates_after_drain(self) -> None:
        async def run() -> tuple[list[Any], Exception | None]:
            async def producer() -> AsyncIterator[int]:
                yield 1
                yield 2
                raise RuntimeError("render guard")

            handled: list[int] = []

            async def handler(x: int) -> int:
                handled.append(x)
                return x

            try:
                await run_streaming_transcription_tasks(producer(), handler, 2)
            except RuntimeError as e:
                return handled, e
            return handled, None

        handled, error = asyncio.run(run())
        assert error is not None and "render guard" in str(error)
        assert sorted(handled) == [1, 2]  # items before the failure survive

    @pytest.mark.unit
    def test_worker_error_recorded_as_none(self) -> None:
        async def run() -> list[Any]:
            async def producer() -> AsyncIterator[int]:
                for i in range(4):
                    yield i

            async def handler(x: int) -> int:
                if x == 2:
                    raise ValueError("boom")
                return x

            return await run_streaming_transcription_tasks(producer(), handler, 2)

        results = asyncio.run(run())
        assert None in results
        assert sorted(r for r in results if r is not None) == [0, 1, 3]


# ---------------------------------------------------------------------------
# Streaming transcription pipeline
# ---------------------------------------------------------------------------


class _FakeTranscriber:
    def __init__(self, fail_names: set[str] | None = None) -> None:
        self.fail_names = fail_names or set()
        self.calls: list[str] = []

    async def transcribe_image_from_base64(
        self, image_base64: str, mime_type: str
    ) -> dict[str, Any]:
        marker = base64.b64decode(image_base64).decode("utf-8")
        self.calls.append(marker)
        if marker in self.fail_names:
            return {"error": "simulated failure"}
        return {"output_text": f"Transcribed {marker}"}


def _fake_payload(index: int, marker: str) -> PagePayload:
    return PagePayload(
        index=index,
        image_name=pdf_page_image_name(index),
        base64=base64.b64encode(marker.encode("utf-8")).decode("utf-8"),
        sha256="ab" * 32,
        width=10,
        height=10,
        byte_size=len(marker),
        effective_dpi=300,
        source_file="C:/in/doc.pdf",
        page_index=index,
    )


async def _payload_gen(payloads: list[PagePayload]) -> AsyncIterator[PagePayload]:
    for p in payloads:
        yield p


class TestRunStreamingTranscriptionPipeline:
    @pytest.mark.unit
    def test_lean_jsonl_records_and_output(self, tmp_path: Path) -> None:
        temp_jsonl = tmp_path / "doc.jsonl"
        temp_jsonl.touch()
        output_txt = tmp_path / "doc.txt"
        payloads = [_fake_payload(0, "p1"), _fake_payload(1, "p2")]
        transcriber = _FakeTranscriber()

        asyncio.run(
            run_streaming_transcription_pipeline(
                _payload_gen(payloads),
                transcriber,
                temp_jsonl,
                output_txt,
                "doc.pdf",
                {"concurrency": {"transcription": {"concurrency_limit": 2}}},
                {"enabled": False},
                is_folder=False,
                output_format="txt",
                file_provenance={"file_provenance": {"source_file": "C:/in/doc.pdf"}},
            )
        )

        lines = [
            json.loads(line)
            for line in temp_jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        prov_lines = [r for r in lines if "file_provenance" in r]
        records = [r for r in lines if "image_name" in r]
        assert len(prov_lines) == 1
        assert len(records) == 2
        for rec in records:
            assert rec["pre_processed_image"] is None
            assert rec["source_file"] == "C:/in/doc.pdf"
            assert rec["image_provenance"]["sha256"] == "ab" * 32
            assert isinstance(rec["page_index"], int)
        # No base64 image payloads may leak into the temp JSONL
        raw = temp_jsonl.read_text(encoding="utf-8")
        for p in payloads:
            assert p.base64 not in raw

        out = output_txt.read_text(encoding="utf-8")
        assert "Transcribed p1" in out and "Transcribed p2" in out

    @pytest.mark.unit
    def test_resume_merge_includes_prior_records(self, tmp_path: Path) -> None:
        temp_jsonl = tmp_path / "doc.jsonl"
        prior = {
            "file_name": "doc.pdf",
            "pre_processed_image": "C:/old/page_0001_pre_processed.jpg",
            "image_name": "page_0001_pre_processed.jpg",
            "method": "gpt",
            "order_index": 0,
            "text_chunk": "Old page one text",
        }
        temp_jsonl.write_text(json.dumps(prior) + "\n", encoding="utf-8")
        output_txt = tmp_path / "doc.txt"

        asyncio.run(
            run_streaming_transcription_pipeline(
                _payload_gen([_fake_payload(1, "p2")]),
                _FakeTranscriber(),
                temp_jsonl,
                output_txt,
                "doc.pdf",
                {"concurrency": {"transcription": {"concurrency_limit": 2}}},
                {"enabled": False},
            )
        )

        out = output_txt.read_text(encoding="utf-8")
        assert "Old page one text" in out
        assert "Transcribed p2" in out
        # Absolute order: resumed page 1 before new page 2
        assert out.index("Old page one text") < out.index("Transcribed p2")


# ---------------------------------------------------------------------------
# File provenance
# ---------------------------------------------------------------------------


class TestBuildFileProvenance:
    @pytest.mark.unit
    def test_contains_versions_hash_and_config(self, tmp_path: Path) -> None:
        src = tmp_path / "doc.pdf"
        _make_pdf(src, pages=1)
        record = build_file_provenance(src, IMG_CFG, "openai", 24_000_000)
        prov = record["file_provenance"]
        assert prov["source_file"] == str(src)
        assert isinstance(prov["source_sha256"], str)
        assert len(prov["source_sha256"]) == 64
        assert prov["pillow_version"]
        assert prov["image_config"]["max_pixels_per_page"] == 24_000_000
        assert prov["image_config"]["jpeg_quality"] == 90


# ---------------------------------------------------------------------------
# Repair re-render fallback
# ---------------------------------------------------------------------------


class TestRepairRerenderFallback:
    @pytest.mark.unit
    def test_pdf_target_rerendered_from_source(self, tmp_path: Path) -> None:
        pdf = _make_pdf(tmp_path / "doc.pdf", pages=3)
        job = Job(
            parent_folder=tmp_path,
            identifier="doc",
            final_txt_path=tmp_path / "doc.txt",
            temp_jsonl_path=None,
            kind="PDF",
        )
        entries = [
            ImageEntry(
                order_index=1,
                image_name="page_0002_pre_processed.jpg",
                pre_processed_image=None,
                custom_id=None,
                source_file=str(pdf),
                page_index=1,
            )
        ]
        final_lines = ["page_0002_pre_processed.jpg: [transcription error]"]

        with patch(
            "modules.images.page_stream.resolve_image_settings",
            return_value=(IMG_CFG, "openai", 72, 0),
        ):
            targets = _resolve_repair_targets(
                job, entries, [0], final_lines, {"transcription_model": {}}
            )

        assert len(targets) == 1
        target = targets[0]
        assert target.image_path is None
        assert target.image_base64 is not None
        assert target.mime_type == "image/jpeg"
        assert base64.b64decode(target.image_base64)[:3] == b"\xff\xd8\xff"

    @pytest.mark.unit
    def test_unresolvable_target_skipped(self, tmp_path: Path) -> None:
        job = Job(
            parent_folder=tmp_path,
            identifier="doc",
            final_txt_path=tmp_path / "doc.txt",
            temp_jsonl_path=None,
            kind="PDF",
        )
        entries = [
            ImageEntry(
                order_index=0,
                image_name="page_0001_pre_processed.jpg",
                pre_processed_image=None,
                custom_id=None,
                source_file=None,
            )
        ]
        final_lines = ["page_0001_pre_processed.jpg: [transcription error]"]
        targets = _resolve_repair_targets(
            job, entries, [0], final_lines, {"transcription_model": {}}
        )
        assert targets == []

    @pytest.mark.unit
    def test_repair_helpers_render_single_page_and_image(self, tmp_path: Path) -> None:
        pdf = _make_pdf(tmp_path / "doc.pdf", pages=2)
        payload = render_single_pdf_page_payload(
            pdf, 1, target_dpi=72, img_cfg=IMG_CFG, model_type="openai"
        )
        assert payload.index == 1
        assert payload.image_name == "page_0002_pre_processed.jpg"

        img_path = tmp_path / "scan.png"
        Image.new("RGB", (30, 30), (255, 255, 255)).save(img_path)
        img_payload = load_image_payload(
            img_path, 4, img_cfg=IMG_CFG, model_type="openai"
        )
        assert img_payload.image_name == "scan_pre_processed.jpg"
        assert img_payload.index == 4
