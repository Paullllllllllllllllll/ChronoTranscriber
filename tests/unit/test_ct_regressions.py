"""Regression tests for the CT-1 .. CT-5 live-testing bug fixes.

CT-1: single-file --input must never produce a dot-prefixed output directory.
CT-2: page-level transcription failures must propagate to item status,
      exit code, and the --json summary.
CT-3: langchain OutputParserException ("Invalid json output") is retryable.
CT-4: --json emits a real one-line summary on all entry points.
CT-5: pydantic "Pydantic serializer warnings" are suppressed, narrowly.
"""

from __future__ import annotations

import json
import warnings
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import tenacity
from langchain_core.exceptions import OutputParserException

from modules.llm.providers.base import BaseProvider
from modules.transcribe.manager import _relative_key
from modules.transcribe.pipeline import (
    PageTranscriptionError,
    count_failed_page_results,
    run_streaming_transcription_pipeline,
    run_transcription_pipeline,
)


def _extract_json_line(output: str) -> dict[str, Any]:
    """Return the parsed JSON object from the last JSON line in *output*."""
    for line in reversed(output.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            result = json.loads(line)
            assert isinstance(result, dict)
            return result
    raise AssertionError(f"No JSON line found in output:\n{output}")


# ---------------------------------------------------------------------------
# CT-1: single-file input must not hash "." into a hidden output directory
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSingleFileOutputDirName:
    def test_relative_key_is_none_when_item_equals_input_root(
        self, tmp_path: Path
    ) -> None:
        pdf = tmp_path / "book.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        # Single-file input: item == input_root previously yielded "." and a
        # hidden ".-cdb4ee2a" output directory.
        assert _relative_key(pdf, pdf) is None

    def test_relative_key_for_directory_input_unchanged(self, tmp_path: Path) -> None:
        pdf = tmp_path / "book.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        assert _relative_key(pdf, tmp_path) == "book.pdf"

    def test_single_file_input_never_creates_dot_prefixed_dir(
        self, tmp_path: Path
    ) -> None:
        from modules.documents.pdf import PDFProcessor

        pdf = tmp_path / "acerbi_bentley_2014_p1-3.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        out_root = tmp_path / "out"

        rel_key = _relative_key(pdf, pdf)  # single-file input
        parent, _txt, _jsonl = PDFProcessor(pdf).prepare_output_folder(
            out_root, relative_key=rel_key
        )
        assert not parent.name.startswith(".")
        assert parent.name.startswith("acerbi_bentley_2014_p1-3")


# ---------------------------------------------------------------------------
# CT-2: page failures propagate as PageTranscriptionError (item failed)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPageFailurePropagation:
    def test_count_failed_page_results(self) -> None:
        ok = ("p", "a.png", "some text", None, 0)
        placeholder = ("p", "b.png", "[transcription error: b.png]", None, 1)
        none_text = ("p", "c.png", None, None, 2)
        malformed = None
        assert count_failed_page_results([ok]) == 0
        assert count_failed_page_results([ok, placeholder, none_text, malformed]) == 3

    @pytest.mark.asyncio
    async def test_file_pipeline_raises_when_page_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        img = tmp_path / "page1.png"
        img.write_bytes(b"")
        jsonl_path = tmp_path / "temp.jsonl"
        jsonl_path.write_text("", encoding="utf-8")
        output_path = tmp_path / "out.txt"

        def _boom(path: Path, cfg: str) -> str:
            raise RuntimeError("OCR exploded")

        monkeypatch.setattr("modules.transcribe.pipeline.perform_ocr", _boom)

        def fake_writer(pages: Any, out: Path, **kw: Any) -> Path:
            Path(out).write_text("partial", encoding="utf-8")
            return Path(out)

        monkeypatch.setattr(
            "modules.transcribe.pipeline.write_transcription_output", fake_writer
        )

        with pytest.raises(PageTranscriptionError) as excinfo:
            await run_transcription_pipeline(
                image_files=[img],
                method="tesseract",
                transcriber=None,
                temp_jsonl_path=jsonl_path,
                output_txt_path=output_path,
                source_name="doc.pdf",
                concurrency_config={"concurrency": {"transcription": {}}},
                image_processing_config={},
                postprocessing_config={},
            )
        assert excinfo.value.failed_pages == 1
        # The partial output was still written before the failure propagated.
        assert output_path.exists()

    @pytest.mark.asyncio
    async def test_file_pipeline_does_not_raise_when_all_pages_succeed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        img = tmp_path / "page1.png"
        img.write_bytes(b"")
        jsonl_path = tmp_path / "temp.jsonl"
        jsonl_path.write_text("", encoding="utf-8")
        output_path = tmp_path / "out.txt"

        monkeypatch.setattr(
            "modules.transcribe.pipeline.perform_ocr", lambda path, cfg: "fine text"
        )
        monkeypatch.setattr(
            "modules.transcribe.pipeline.write_transcription_output",
            lambda pages, out, **kw: Path(out),
        )

        await run_transcription_pipeline(
            image_files=[img],
            method="tesseract",
            transcriber=None,
            temp_jsonl_path=jsonl_path,
            output_txt_path=output_path,
            source_name="doc.pdf",
            concurrency_config={"concurrency": {"transcription": {}}},
            image_processing_config={},
            postprocessing_config={},
        )

    @pytest.mark.asyncio
    async def test_streaming_pipeline_raises_when_page_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        jsonl_path = tmp_path / "temp.jsonl"
        jsonl_path.write_text("", encoding="utf-8")
        output_path = tmp_path / "out.txt"

        payload = SimpleNamespace(
            image_name="page_0001.png",
            base64="Zm9v",
            mime_type="image/png",
            index=0,
            source_file="doc.pdf",
            page_index=0,
            provenance=lambda: {},
        )

        async def payload_source() -> Any:
            yield payload

        transcriber = MagicMock()
        transcriber.transcribe_image_from_base64 = MagicMock(
            side_effect=RuntimeError("API 404")
        )

        monkeypatch.setattr(
            "modules.transcribe.pipeline.write_transcription_output",
            lambda pages, out, **kw: Path(out),
        )

        with pytest.raises(PageTranscriptionError):
            await run_streaming_transcription_pipeline(
                payload_source(),
                transcriber,
                jsonl_path,
                output_path,
                "doc.pdf",
                {"concurrency": {"transcription": {"concurrency_limit": 2}}},
                {},
            )

    def test_json_summary_reports_failure_and_exit_1(self, capsys: Any) -> None:
        from main.unified_transcriber import _emit_json_summary
        from modules.transcribe.manager import ProcessingSummary

        _emit_json_summary(ProcessingSummary(processed=0, failed=1, total=1))
        payload = _extract_json_line(capsys.readouterr().out)
        assert payload["items_failed"] == 1
        assert payload["exit_code"] == 1
        # Backward-compatible keys
        assert set(payload) == {
            "tool",
            "dry_run",
            "items_total",
            "items_processed",
            "items_failed",
            "exit_code",
        }


# ---------------------------------------------------------------------------
# CT-6: mid-document daily-budget exhaustion must fail the item, withhold the
#       truncated output, and resume from the JSONL on the next run.
# ---------------------------------------------------------------------------


def _fake_pdf_stream(source_path: Any = None, *, page_indices: Any, **_kw: Any) -> Any:
    """Stand-in for stream_pdf_payloads: yield a lightweight payload per index."""

    async def gen() -> Any:
        for i in page_indices:
            yield SimpleNamespace(
                index=i,
                page_index=i,
                image_name=f"page_{i + 1:04d}_pre_processed.jpg",
                source_file=str(source_path),
            )

    return gen()


def _make_fake_pipeline(budget_per_pass: int) -> Any:
    """Fake run_streaming_transcription_pipeline mirroring the on-disk effects.

    Writes a JSONL record per admitted page, finalizes the output txt from the
    full JSONL (exactly as the real pipeline does), and sets ``exhausted`` once
    the per-pass page budget is spent so the caller's re-pass loop engages.
    """
    from modules.batch.jsonl import ensure_resume_marker, write_jsonl_record
    from modules.transcribe.pipeline import write_output_from_jsonl

    async def fake(
        payload_source: Any,
        transcriber: Any,
        temp_jsonl_path: Path,
        output_txt_path: Path,
        source_name: str,
        concurrency_config: Any,
        postprocessing_config: Any,
        *,
        is_folder: bool,
        output_format: str,
        file_provenance: Any,
        tracker: Any,
        exhausted: Any,
    ) -> None:
        ensure_resume_marker(temp_jsonl_path)
        written = 0
        async for p in payload_source:
            if written >= budget_per_pass:
                exhausted.set()
                break
            write_jsonl_record(
                temp_jsonl_path,
                {
                    "file_name": source_name,
                    "image_name": p.image_name,
                    "text_chunk": f"page {p.index} text",
                    "order_index": p.index,
                    "method": "gpt",
                },
            )
            written += 1
        write_output_from_jsonl(
            temp_jsonl_path,
            output_txt_path,
            postprocessing_config,
            output_format=output_format,
        )

    return fake


def _make_bare_manager(transient: Any = None) -> Any:
    """Build a WorkflowManager with only the attributes the GPT streaming flow
    touches, bypassing __init__ (which resolves output dirs and config)."""
    from modules.transcribe.manager import TransientFileTracker, WorkflowManager

    mgr = WorkflowManager.__new__(WorkflowManager)
    mgr.resume_mode = "skip"
    mgr.model_config = {"transcription_model": {"provider": "openai", "name": "gpt-4o"}}
    mgr.concurrency_config = {
        "concurrency": {"transcription": {"concurrency_limit": 2}},
        "daily_token_limit": {"enabled": True},
    }
    mgr.postprocessing_config = {}
    mgr.processing_settings = {"retain_temporary_jsonl": True}
    mgr.user_config = SimpleNamespace(
        use_batch_processing=False,
        retry_errors=False,
        output_format="txt",
    )
    mgr._transient_tracker = transient or TransientFileTracker()
    if transient is None:
        mgr._transient_tracker.configure({"retain_temporary_jsonl": True})
    return mgr


def _patch_streaming_env(
    monkeypatch: pytest.MonkeyPatch, pipeline: Any, wait_result: bool
) -> None:
    """Patch the module-level collaborators used by _process_gpt_streaming."""
    import modules.transcribe.manager as mm

    monkeypatch.setattr(
        mm,
        "resolve_image_settings",
        lambda p, m: ({}, "openai", 300, 1_000_000, "direct"),
    )
    monkeypatch.setattr(mm, "stream_pdf_payloads", _fake_pdf_stream)
    monkeypatch.setattr(mm, "get_token_tracker", lambda: MagicMock())
    monkeypatch.setattr(mm, "run_streaming_transcription_pipeline", pipeline)

    async def _wait(
        cfg: Any, reservation_aware: bool = False, stamp: Any = None
    ) -> bool:
        return wait_result

    monkeypatch.setattr(mm, "check_and_wait_for_token_limit", _wait)


async def _run_stream(mgr: Any, tmp_path: Path, jsonl: Path, out: Path) -> None:
    await mgr._process_gpt_streaming(
        source_path=tmp_path / "doc.pdf",
        source_name="doc.pdf",
        source_stem="doc",
        is_folder=False,
        total_units=4,
        page_indices=None,
        parent_folder=tmp_path,
        temp_jsonl_path=jsonl,
        output_txt_path=out,
        transcriber=MagicMock(),
    )


@pytest.mark.unit
class TestBudgetExhaustionContract:
    @pytest.mark.asyncio
    async def test_exhaustion_raises_and_withholds_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from modules.batch.jsonl import get_processed_image_names
        from modules.transcribe.pipeline import BudgetExhaustedError

        # 4-page doc, budget admits 2 pages, then the wait gives up (False).
        _patch_streaming_env(monkeypatch, _make_fake_pipeline(2), wait_result=False)
        mgr = _make_bare_manager()
        jsonl = tmp_path / "doc.jsonl"
        jsonl.touch()
        out = tmp_path / "doc.txt"

        with pytest.raises(BudgetExhaustedError) as excinfo:
            await _run_stream(mgr, tmp_path, jsonl, out)

        # (a) item counted failed via the exception channel.
        assert excinfo.value.deferred_pages == 2
        assert excinfo.value.completed_pages == 2
        # (b) the truncated finalized output is withheld.
        assert not out.exists()
        # ... but the JSONL retains the 2 completed pages plus its resume marker.
        assert len(get_processed_image_names(jsonl)) == 2

    @pytest.mark.asyncio
    async def test_resume_completes_file_from_jsonl(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import modules.transcribe.manager as mm
        from modules.batch.jsonl import get_processed_image_names
        from modules.transcribe.pipeline import BudgetExhaustedError

        jsonl = tmp_path / "doc.jsonl"
        jsonl.touch()
        out = tmp_path / "doc.txt"

        # First run: exhausts after 2 of 4 pages, withholds output, raises.
        _patch_streaming_env(monkeypatch, _make_fake_pipeline(2), wait_result=False)
        with pytest.raises(BudgetExhaustedError):
            await _run_stream(_make_bare_manager(), tmp_path, jsonl, out)
        assert not out.exists()

        # Second run after the daily reset: ample budget, no exhaustion. Page-
        # level resume reads the 2 cached pages from the JSONL and transcribes
        # only the remaining 2, then finalizes the complete output.
        monkeypatch.setattr(
            mm, "run_streaming_transcription_pipeline", _make_fake_pipeline(99)
        )
        await _run_stream(_make_bare_manager(), tmp_path, jsonl, out)

        assert out.exists()
        assert len(get_processed_image_names(jsonl)) == 4
        text = out.read_text(encoding="utf-8")
        for i in range(4):
            assert f"page {i} text" in text

    @pytest.mark.asyncio
    async def test_partial_jsonl_survives_aggressive_cleanup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from modules.batch.jsonl import get_processed_image_names
        from modules.transcribe.manager import TransientFileTracker
        from modules.transcribe.pipeline import BudgetExhaustedError

        jsonl = tmp_path / "doc.jsonl"
        jsonl.touch()
        out = tmp_path / "doc.txt"

        # retain_temporary_jsonl False makes cleanup_pending() delete tracked
        # JSONLs — the budget-partial artifact must nonetheless survive because
        # it is removed from the cleanup list before the item fails.
        transient = TransientFileTracker()
        transient.configure({"retain_temporary_jsonl": False})
        transient.register_jsonl(jsonl, "gpt")

        _patch_streaming_env(monkeypatch, _make_fake_pipeline(2), wait_result=False)
        mgr = _make_bare_manager(transient=transient)

        with pytest.raises(BudgetExhaustedError):
            await _run_stream(mgr, tmp_path, jsonl, out)

        transient.cleanup_pending()
        assert jsonl.exists()
        assert len(get_processed_image_names(jsonl)) == 2


# ---------------------------------------------------------------------------
# CT-7: a page failure on the SAME pass that exhausts the budget must not
#       bypass the withhold. The streaming pipeline finalizes the truncated
#       output then raises PageTranscriptionError; the manager must fold that
#       into the budget flow (withhold + JSONL protection) rather than let it
#       propagate and leave a truncated txt that resume files COMPLETE.
# ---------------------------------------------------------------------------


def _make_exhaust_and_fail_pipeline(budget_per_pass: int, fail_index: int) -> Any:
    """Fake pipeline whose exhausting pass ALSO fails a page.

    Mirrors the real pipeline's on-disk effects: writes a JSONL record per
    admitted page (an ``[transcription error]`` placeholder for ``fail_index``),
    finalizes the output txt, sets ``exhausted`` once the per-pass budget is
    spent, and — like the real pipeline — raises PageTranscriptionError AFTER
    finalizing when a page failed this pass.
    """
    from modules.batch.jsonl import ensure_resume_marker, write_jsonl_record
    from modules.transcribe.pipeline import (
        PageTranscriptionError,
        write_output_from_jsonl,
    )

    async def fake(
        payload_source: Any,
        transcriber: Any,
        temp_jsonl_path: Path,
        output_txt_path: Path,
        source_name: str,
        concurrency_config: Any,
        postprocessing_config: Any,
        *,
        is_folder: bool,
        output_format: str,
        file_provenance: Any,
        tracker: Any,
        exhausted: Any,
    ) -> None:
        ensure_resume_marker(temp_jsonl_path)
        written = 0
        failed = 0
        async for p in payload_source:
            if written >= budget_per_pass:
                exhausted.set()
                break
            is_fail = p.index == fail_index
            text = (
                f"[transcription error: {p.image_name}]"
                if is_fail
                else f"page {p.index} text"
            )
            write_jsonl_record(
                temp_jsonl_path,
                {
                    "file_name": source_name,
                    "image_name": p.image_name,
                    "text_chunk": text,
                    "order_index": p.index,
                    "method": "gpt",
                },
            )
            written += 1
            failed += int(is_fail)
        write_output_from_jsonl(
            temp_jsonl_path,
            output_txt_path,
            postprocessing_config,
            output_format=output_format,
        )
        if failed:
            raise PageTranscriptionError(source_name, failed, written)

    return fake


def _make_fail_first_pass_then_clean_pipeline(
    budget_first_pass: int, fail_index: int
) -> Any:
    """Stateful fake: pass 1 fails a page AND exhausts; pass 2 finishes clean.

    Models the daily reset freeing budget so the re-pass transcribes the
    remaining pages while the earlier page failure remains a placeholder in the
    JSONL. The final output is COMPLETE (with the placeholder), so it must NOT
    be withheld — only re-raised so the item counts failed.
    """
    from modules.batch.jsonl import ensure_resume_marker, write_jsonl_record
    from modules.transcribe.pipeline import (
        PageTranscriptionError,
        write_output_from_jsonl,
    )

    state = {"calls": 0}

    async def fake(
        payload_source: Any,
        transcriber: Any,
        temp_jsonl_path: Path,
        output_txt_path: Path,
        source_name: str,
        concurrency_config: Any,
        postprocessing_config: Any,
        *,
        is_folder: bool,
        output_format: str,
        file_provenance: Any,
        tracker: Any,
        exhausted: Any,
    ) -> None:
        state["calls"] += 1
        first = state["calls"] == 1
        ensure_resume_marker(temp_jsonl_path)
        written = 0
        failed = 0
        async for p in payload_source:
            if first and written >= budget_first_pass:
                exhausted.set()
                break
            is_fail = first and p.index == fail_index
            text = (
                f"[transcription error: {p.image_name}]"
                if is_fail
                else f"page {p.index} text"
            )
            write_jsonl_record(
                temp_jsonl_path,
                {
                    "file_name": source_name,
                    "image_name": p.image_name,
                    "text_chunk": text,
                    "order_index": p.index,
                    "method": "gpt",
                },
            )
            written += 1
            failed += int(is_fail)
        write_output_from_jsonl(
            temp_jsonl_path,
            output_txt_path,
            postprocessing_config,
            output_format=output_format,
        )
        if failed:
            raise PageTranscriptionError(source_name, failed, written)

    return fake


@pytest.mark.unit
class TestBudgetExhaustionWithPageFailure:
    @pytest.mark.asyncio
    async def test_page_failure_on_exhausting_pass_still_withholds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from modules.batch.jsonl import get_processed_image_names
        from modules.transcribe.pipeline import BudgetExhaustedError

        # 4-page doc; pass 1 admits 2 pages (page 1 FAILS), then exhausts and the
        # wait gives up (False). The pipeline raises PageTranscriptionError after
        # finalizing — the manager must catch it, withhold, and raise the budget
        # failure rather than let the page error bypass the withhold.
        _patch_streaming_env(
            monkeypatch,
            _make_exhaust_and_fail_pipeline(budget_per_pass=2, fail_index=1),
            wait_result=False,
        )
        jsonl = tmp_path / "doc.jsonl"
        jsonl.touch()
        out = tmp_path / "doc.txt"

        with pytest.raises(BudgetExhaustedError) as excinfo:
            await _run_stream(_make_bare_manager(), tmp_path, jsonl, out)

        assert excinfo.value.deferred_pages == 2
        # The truncated finalized output is withheld despite the page failure.
        assert not out.exists()
        # JSONL keeps the 2 attempted pages (1 ok + 1 error placeholder).
        assert len(get_processed_image_names(jsonl)) == 2

    @pytest.mark.asyncio
    async def test_completed_after_reset_keeps_output_but_still_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from modules.batch.jsonl import get_processed_image_names
        from modules.transcribe.pipeline import PageTranscriptionError

        # Pass 1 fails page 1 and exhausts; the wait succeeds (True); pass 2
        # transcribes the remaining pages. The output is COMPLETE with the error
        # placeholder — it must be KEPT (not withheld) and the item still fails.
        _patch_streaming_env(
            monkeypatch,
            _make_fail_first_pass_then_clean_pipeline(
                budget_first_pass=2, fail_index=1
            ),
            wait_result=True,
        )
        jsonl = tmp_path / "doc.jsonl"
        jsonl.touch()
        out = tmp_path / "doc.txt"

        with pytest.raises(PageTranscriptionError) as excinfo:
            await _run_stream(_make_bare_manager(), tmp_path, jsonl, out)

        assert excinfo.value.failed_pages == 1
        # Output retained (complete, with the error placeholder for page 1).
        assert out.exists()
        assert len(get_processed_image_names(jsonl)) == 4
        text = out.read_text(encoding="utf-8")
        assert "[transcription error" in text
        for i in (0, 2, 3):
            assert f"page {i} text" in text


# ---------------------------------------------------------------------------
# CT-8: the GPT image-folder path must propagate failures (ResumeFormatError,
#       render failure-rate guard, ...) like the PDF path, not swallow them as
#       "processed" with exit 0.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestImageFolderPropagatesRuntimeError:
    @pytest.mark.asyncio
    async def test_runtime_error_from_streaming_propagates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from modules.batch.jsonl import ResumeFormatError

        folder = tmp_path / "scans"
        folder.mkdir()
        (folder / "page1.png").write_bytes(b"\x89PNG")

        mgr = _make_bare_manager()
        mgr.use_input_as_output = False
        mgr.output_mode = "hash"
        mgr.input_root = tmp_path
        mgr.image_output_dir = tmp_path / "out"
        mgr.user_config = SimpleNamespace(
            transcription_method="gpt",
            additional_context_path=None,
            additional_context_image_path=None,
            use_hierarchical_context=True,
            page_range=None,
            output_format="txt",
            retry_errors=False,
        )

        async def _boom(**kwargs: Any) -> None:
            # A RuntimeError subclass (ResumeFormatError) that the removed
            # `except RuntimeError` used to swallow into a silent skip.
            raise ResumeFormatError("incompatible resume artifact")

        monkeypatch.setattr(mgr, "_process_gpt_streaming", _boom)

        with pytest.raises(ResumeFormatError):
            await mgr.process_single_image_folder(folder, transcriber=None)


# ---------------------------------------------------------------------------
# CT-10: EPUB/MOBI page ranges must resolve against the REAL section count, not
#        a 2**31 sentinel (which silently drops last:N and OOMs on open spans).
# ---------------------------------------------------------------------------


class _FakeExtraction:
    def __init__(self, sections: list[str]) -> None:
        self.sections = sections
        self.source_format: str | None = None

    def to_plain_text(self) -> str:
        return "\n".join(self.sections) + "\n"


def _make_fake_ebook_processor(n_sections: int) -> Any:
    class _FakeEbookProcessor:
        def __init__(self, path: Path) -> None:
            self.path = path

        def extract_text(
            self, section_indices: Any = None, page_range: Any = None
        ) -> Any:
            # The manager now extracts ALL sections and slices afterward.
            return _FakeExtraction([f"section {i}" for i in range(n_sections)])

        def prepare_output_folder(self, out_dir: Path) -> tuple[Path, Path]:
            parent = out_dir / "book"
            parent.mkdir(parents=True, exist_ok=True)
            return parent, parent / "book.txt"

    return _FakeEbookProcessor


def _make_ebook_manager(page_range: Any) -> Any:
    from modules.transcribe.manager import WorkflowManager

    mgr = WorkflowManager.__new__(WorkflowManager)
    mgr.use_input_as_output = False
    mgr.postprocessing_config = {}
    mgr.user_config = SimpleNamespace(page_range=page_range, output_format="txt")
    return mgr


@pytest.mark.unit
class TestEbookPageRangeResolvesAgainstRealCount:
    @pytest.mark.asyncio
    async def test_last_n_selects_final_sections_not_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import modules.transcribe.manager as mm
        from modules.documents.page_range import parse_page_range

        captured: dict[str, str] = {}

        def fake_writer(pages: Any, out: Path, **kw: Any) -> Path:
            captured["text"] = pages[0]["text"]
            Path(out).write_text(pages[0]["text"], encoding="utf-8")
            return Path(out)

        monkeypatch.setattr(mm, "write_transcription_output", fake_writer)

        mgr = _make_ebook_manager(parse_page_range("last:5"))
        await mgr._process_native_ebook(
            file_path=tmp_path / "book.epub",
            processor_cls=_make_fake_ebook_processor(10),
            format_label="EPUB",
            default_output_dir=tmp_path / "out",
        )
        # last:5 of 10 sections -> sections 5..9 (previously the 2**31 sentinel
        # pushed these indices out of range and produced EMPTY output).
        text = captured["text"]
        for i in range(5, 10):
            assert f"section {i}" in text
        assert "section 4" not in text

    @pytest.mark.asyncio
    async def test_open_span_clamps_to_real_count(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import modules.transcribe.manager as mm
        from modules.documents.page_range import parse_page_range

        captured: dict[str, str] = {}

        def fake_writer(pages: Any, out: Path, **kw: Any) -> Path:
            captured["text"] = pages[0]["text"]
            return Path(out)

        monkeypatch.setattr(mm, "write_transcription_output", fake_writer)

        # `3-` open span: previously materialized ~2.1B indices (OOM). With the
        # real count it clamps to sections 2..9.
        mgr = _make_ebook_manager(parse_page_range("3-"))
        await mgr._process_native_ebook(
            file_path=tmp_path / "book.epub",
            processor_cls=_make_fake_ebook_processor(10),
            format_label="EPUB",
            default_output_dir=tmp_path / "out",
        )
        text = captured["text"]
        assert "section 2" in text
        assert "section 9" in text
        assert "section 1" not in text

    @pytest.mark.asyncio
    async def test_out_of_range_warns_and_skips_without_empty_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import modules.transcribe.manager as mm
        from modules.documents.page_range import parse_page_range

        called = {"wrote": False}

        def fake_writer(pages: Any, out: Path, **kw: Any) -> Path:
            called["wrote"] = True
            return Path(out)

        monkeypatch.setattr(mm, "write_transcription_output", fake_writer)

        # A valid range beyond the section count selects nothing: the manager
        # must warn and skip, never write an empty output with a success message.
        mgr = _make_ebook_manager(parse_page_range("50-60"))
        await mgr._process_native_ebook(
            file_path=tmp_path / "book.epub",
            processor_cls=_make_fake_ebook_processor(3),
            format_label="EPUB",
            default_output_dir=tmp_path / "out",
        )
        assert called["wrote"] is False


# ---------------------------------------------------------------------------
# CT-9: CLI auto mode must build the per-method config as a FULL copy of the
#       user's config (only overriding the auto-decided fields), and re-point
#       the WorkflowManager's ResumeChecker at the redirected auto output dir.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAutoModeConfigPropagation:
    @pytest.mark.asyncio
    async def test_full_copy_and_resume_checker_repointed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import main.unified_transcriber as ut
        from modules.transcribe.manager import ProcessingSummary
        from modules.transcribe.user_config import UserConfiguration

        captured: dict[str, Any] = {}

        class _FakeResumeChecker:
            def __init__(self) -> None:
                self.pdf_output_dir: Any = None
                self.image_output_dir: Any = None
                self.epub_output_dir: Any = None
                self.mobi_output_dir: Any = None

        class _FakeManager:
            def __init__(self, cfg: Any, *a: Any, **k: Any) -> None:
                captured["cfg"] = cfg
                self.pdf_output_dir: Any = None
                self.image_output_dir: Any = None
                self.epub_output_dir: Any = None
                self.mobi_output_dir: Any = None
                self.resume_checker = _FakeResumeChecker()

            async def process_selected_items(self, transcriber: Any = None) -> Any:
                captured["rc"] = self.resume_checker
                return ProcessingSummary(processed=1, failed=0, total=1)

        class _FakePathConfig:
            @classmethod
            def from_paths_config(cls, pc: Any) -> Any:
                return SimpleNamespace(use_input_as_output=False)

        monkeypatch.setattr(ut, "WorkflowManager", _FakeManager)
        monkeypatch.setattr(ut, "PathConfig", _FakePathConfig)

        out_dir = tmp_path / "auto_out"
        input_root = tmp_path / "root"
        ctx = tmp_path / "ctx.txt"
        ctx_img = tmp_path / "ctx.png"
        decision = SimpleNamespace(method="native", file_path=tmp_path / "a.pdf")
        user_config = UserConfiguration(
            processing_type="auto",
            auto_decisions=[decision],
            selected_items=[out_dir],
            auto_selector=SimpleNamespace(print_decision_summary=lambda d: None),
            output_format="md",
            output_mode="mirror",
            input_root=input_root,
            additional_context_path=ctx,
            additional_context_image_path=ctx_img,
            sync_fallback=True,
            resume_mode="skip",
        )

        summary = await ut.process_auto_mode(user_config, {}, {}, {}, {})

        cfg = captured["cfg"]
        # Previously-dropped fields now carried through unchanged.
        assert cfg.output_format == "md"
        assert cfg.output_mode == "mirror"
        assert cfg.input_root == input_root
        assert cfg.additional_context_path == ctx
        assert cfg.additional_context_image_path == ctx_img
        assert cfg.sync_fallback is True
        # Auto-decided overrides applied.
        assert cfg.transcription_method == "native"
        assert cfg.processing_type == "auto"
        assert cfg.selected_items == [decision.file_path]
        assert cfg.use_batch_processing is False
        # ResumeChecker re-pointed at the redirected auto output dir.
        rc = captured["rc"]
        assert rc.pdf_output_dir == out_dir
        assert rc.image_output_dir == out_dir
        assert rc.epub_output_dir == out_dir
        assert rc.mobi_output_dir == out_dir
        assert summary.processed == 1


# ---------------------------------------------------------------------------
# CT-3: OutputParserException is retried like other validation failures
# ---------------------------------------------------------------------------


class _MiniProvider(BaseProvider):
    @property
    def provider_name(self) -> str:
        return "openai"

    def get_capabilities(self) -> Any:
        return None

    async def transcribe_image_from_base64(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def close(self) -> None:
        pass


class _QueueLLM:
    """LLM double returning a queue of prepared results (last one repeats)."""

    def __init__(self, results: list[Any]) -> None:
        self.results = list(results)
        self.calls = 0

    async def ainvoke(self, messages: Any, **kwargs: Any) -> Any:
        self.calls += 1
        if len(self.results) > 1:
            return self.results.pop(0)
        return self.results[0]


def _parse_failure_result() -> dict[str, Any]:
    return {
        "raw": SimpleNamespace(content="not json", response_metadata={}),
        "parsed": None,
        "parsing_error": OutputParserException("Invalid json output: not json"),
    }


def _good_result() -> dict[str, Any]:
    return {
        "raw": SimpleNamespace(
            content='{"transcription": "hi"}',
            response_metadata={},
            usage_metadata={
                "input_tokens": 1000,
                "output_tokens": 10,
                "total_tokens": 1010,
            },
        ),
        "parsed": {"transcription": "hi"},
        "parsing_error": None,
    }


@pytest.fixture
def fast_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero backoff and pin retry budgets so tests are config-independent."""
    import modules.llm.providers.base as base_mod

    monkeypatch.setattr(
        tenacity, "wait_exponential_jitter", lambda **kw: tenacity.wait_fixed(0)
    )
    monkeypatch.setattr(base_mod, "load_max_retries", lambda: 8)
    monkeypatch.setattr(base_mod, "load_max_validation_retries", lambda: 3)


@pytest.mark.unit
class TestOutputParserExceptionRetry:
    @pytest.mark.asyncio
    async def test_invalid_json_retried_then_succeeds(self, fast_retry: None) -> None:
        provider = _MiniProvider(api_key="k", model="gpt-4o")
        llm = _QueueLLM([_parse_failure_result(), _good_result()])

        result = await provider._ainvoke_with_retry(llm, [])

        assert llm.calls == 2
        assert result["parsed"] == {"transcription": "hi"}

    @pytest.mark.asyncio
    async def test_invalid_json_bounded_by_validation_budget(
        self, fast_retry: None
    ) -> None:
        provider = _MiniProvider(api_key="k", model="gpt-4o")
        llm = _QueueLLM([_parse_failure_result()])

        with pytest.raises(OutputParserException):
            await provider._ainvoke_with_retry(llm, [])

        # Bounded by the pinned validation retry budget (3).
        assert llm.calls == 3


# ---------------------------------------------------------------------------
# CT-4: --json summaries on the four remaining entry points
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCheckBatchesJsonSummary:
    def _make_args(self) -> Namespace:
        return Namespace(
            no_diagnostics=True,
            directory=None,
            output_format=None,
            json_summary=True,
        )

    def test_emits_summary_and_exits_0(
        self, monkeypatch: pytest.MonkeyPatch, capsys: Any
    ) -> None:
        import main.check_batches as cb
        from modules.batch.check import BatchCheckStats

        script = cb.CheckBatchesScript()
        script.paths_config = {"general": {}}
        monkeypatch.setattr(
            cb,
            "run_batch_finalization",
            lambda **kw: BatchCheckStats(finalized=2, pending=1, failed=0),
        )

        script.run_cli(self._make_args())
        payload = _extract_json_line(capsys.readouterr().out)
        assert payload == {
            "tool": "chronotranscriber",
            "command": "check_batches",
            "finalized": 2,
            "pending": 1,
            "failed": 0,
            "exit_code": 0,
        }

    def test_failure_exits_1_with_json(
        self, monkeypatch: pytest.MonkeyPatch, capsys: Any
    ) -> None:
        import main.check_batches as cb
        from modules.batch.check import BatchCheckStats

        script = cb.CheckBatchesScript()
        script.paths_config = {"general": {}}
        monkeypatch.setattr(
            cb,
            "run_batch_finalization",
            lambda **kw: BatchCheckStats(finalized=0, pending=0, failed=2),
        )

        with pytest.raises(SystemExit) as excinfo:
            script.run_cli(self._make_args())
        assert excinfo.value.code == 1
        payload = _extract_json_line(capsys.readouterr().out)
        assert payload["failed"] == 2
        assert payload["exit_code"] == 1


@pytest.mark.unit
class TestCancelBatchesJsonSummary:
    def test_emits_summary_with_counts(
        self, monkeypatch: pytest.MonkeyPatch, capsys: Any
    ) -> None:
        import main.cancel_batches as cbm

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setattr(
            cbm, "cancel_batch_by_id", lambda provider, bid: bid == "b1"
        )

        script = cbm.CancelBatchesScript()
        args = Namespace(batch_ids=["b1", "b2"], force=True, json_summary=True)

        with pytest.raises(SystemExit) as excinfo:
            script.run_cli(args)
        assert excinfo.value.code == 1
        payload = _extract_json_line(capsys.readouterr().out)
        assert payload == {
            "tool": "chronotranscriber",
            "command": "cancel_batches",
            "cancelled": 1,
            "failed": 1,
            "exit_code": 1,
        }

    def test_all_cancelled_exits_0(
        self, monkeypatch: pytest.MonkeyPatch, capsys: Any
    ) -> None:
        import main.cancel_batches as cbm

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setattr(cbm, "cancel_batch_by_id", lambda provider, bid: True)

        script = cbm.CancelBatchesScript()
        args = Namespace(batch_ids=["b1"], force=True, json_summary=True)

        script.run_cli(args)  # no SystemExit
        payload = _extract_json_line(capsys.readouterr().out)
        assert payload["cancelled"] == 1
        assert payload["failed"] == 0
        assert payload["exit_code"] == 0


@pytest.mark.unit
class TestRepairJsonSummary:
    @pytest.mark.asyncio
    async def test_emits_summary_all_repaired_exit_0(
        self, monkeypatch: pytest.MonkeyPatch, capsys: Any
    ) -> None:
        import main.repair_transcriptions as rt

        async def fake_main_cli(args: Any, paths_config: Any) -> dict[str, int]:
            return {"repaired": 3, "failed": 0}

        monkeypatch.setattr(rt, "main_cli", fake_main_cli)
        script = rt.RepairTranscriptionsScript()

        await script.run_cli(Namespace(json_summary=True))
        payload = _extract_json_line(capsys.readouterr().out)
        assert payload == {
            "tool": "chronotranscriber",
            "command": "repair_transcriptions",
            "repaired": 3,
            "failed": 0,
            "exit_code": 0,
        }

    @pytest.mark.asyncio
    async def test_unrepaired_line_reports_failure_and_exit_1(
        self, monkeypatch: pytest.MonkeyPatch, capsys: Any
    ) -> None:
        """A surviving placeholder must surface as a non-zero exit, not a
        silent success (the single-line unresolved-target incident)."""
        import main.repair_transcriptions as rt

        async def fake_main_cli(args: Any, paths_config: Any) -> dict[str, int]:
            return {"repaired": 3, "failed": 1}

        monkeypatch.setattr(rt, "main_cli", fake_main_cli)
        script = rt.RepairTranscriptionsScript()

        with pytest.raises(SystemExit) as exc_info:
            await script.run_cli(Namespace(json_summary=True))
        assert exc_info.value.code == 1
        payload = _extract_json_line(capsys.readouterr().out)
        assert payload == {
            "tool": "chronotranscriber",
            "command": "repair_transcriptions",
            "repaired": 3,
            "failed": 1,
            "exit_code": 1,
        }


@pytest.mark.unit
class TestPostprocessJsonSummary:
    def _make_args(self, input_path: Path, **overrides: Any) -> Namespace:
        base: dict[str, Any] = {
            "input": str(input_path),
            "output": None,
            "in_place": True,
            "recursive": False,
            "use_config": False,
            "merge_hyphenation": None,
            "no_collapse_spaces": False,
            "max_blank_lines": None,
            "tab_size": None,
            "wrap_width": None,
            "auto_wrap": None,
            "json_summary": True,
        }
        base.update(overrides)
        return Namespace(**base)

    def test_in_place_emits_summary(self, tmp_path: Path, capsys: Any) -> None:
        import main.postprocess_transcriptions as pp

        f = tmp_path / "doc_transcription.txt"
        f.write_text("hello   world\n\n\n\n\nend\n", encoding="utf-8")

        code = pp.postprocess_cli(self._make_args(f))
        assert code == 0
        payload = _extract_json_line(capsys.readouterr().out)
        assert payload == {
            "tool": "chronotranscriber",
            "command": "postprocess_transcriptions",
            "files_processed": 1,
            "files_failed": 0,
            "exit_code": 0,
        }

    def test_missing_input_emits_summary_with_exit_1(
        self, tmp_path: Path, capsys: Any
    ) -> None:
        import main.postprocess_transcriptions as pp

        code = pp.postprocess_cli(self._make_args(tmp_path / "missing.txt"))
        assert code == 1
        payload = _extract_json_line(capsys.readouterr().out)
        assert payload["files_processed"] == 0
        assert payload["exit_code"] == 1


# ---------------------------------------------------------------------------
# CT-5: pydantic serializer warning suppression is narrow
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPydanticSerializerWarningSuppression:
    def test_filter_is_targeted(self) -> None:
        from modules.llm.providers.openai_provider import (
            suppress_pydantic_serializer_warnings,
        )

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            suppress_pydantic_serializer_warnings()

            warnings.warn_explicit(
                "Pydantic serializer warnings:\n"
                "  PydanticSerializationUnexpectedValue(...)",
                UserWarning,
                "pydantic/main.py",
                477,
                module="pydantic.main",
                registry={},
            )
            warnings.warn_explicit(
                "some other pydantic warning",
                UserWarning,
                "pydantic/main.py",
                1,
                module="pydantic.main",
                registry={},
            )
            warnings.warn_explicit(
                "Pydantic serializer warnings: from elsewhere",
                UserWarning,
                "other/mod.py",
                1,
                module="somewhere.else",
                registry={},
            )

        messages = [str(w.message) for w in record]
        # The noisy pydantic serializer warning is gone ...
        assert not any(
            m.startswith("Pydantic serializer warnings")
            and "PydanticSerializationUnexpectedValue" in m
            for m in messages
        )
        # ... but everything else still surfaces.
        assert "some other pydantic warning" in messages
        assert "Pydantic serializer warnings: from elsewhere" in messages
