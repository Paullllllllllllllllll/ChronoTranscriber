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
    async def test_emits_summary(
        self, monkeypatch: pytest.MonkeyPatch, capsys: Any
    ) -> None:
        import main.repair_transcriptions as rt

        async def fake_main_cli(args: Any, paths_config: Any) -> dict[str, int]:
            return {"repaired": 3, "failed": 1}

        monkeypatch.setattr(rt, "main_cli", fake_main_cli)
        script = rt.RepairTranscriptionsScript()

        await script.run_cli(Namespace(json_summary=True))
        payload = _extract_json_line(capsys.readouterr().out)
        assert payload == {
            "tool": "chronotranscriber",
            "command": "repair_transcriptions",
            "repaired": 3,
            "failed": 1,
            "exit_code": 0,
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
