"""Unit tests for the call-label ContextVar used in retry logging.

Covers ``call_label``/``current_call_label`` semantics (bind, reset, nesting,
no-op for falsy labels, per-task isolation) and the call sites that bind a
page name: ``LangChainTranscriber.transcribe_image``,
``LangChainTranscriber.transcribe_image_from_base64``, and
``modules.transcribe.pipeline.transcribe_payload``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from modules.llm.providers.base import (
    TranscriptionResult,
    call_label,
    current_call_label,
)


class _StubProvider:
    """Provider stub recording the label visible during each call."""

    provider_name = "stub"
    model = "stub-model"

    def __init__(self) -> None:
        self.seen: list[str | None] = []

    async def transcribe_image(self, *args: Any, **kwargs: Any) -> TranscriptionResult:
        self.seen.append(current_call_label())
        return TranscriptionResult(content="ok")

    async def transcribe_image_from_base64(
        self, *args: Any, **kwargs: Any
    ) -> TranscriptionResult:
        self.seen.append(current_call_label())
        return TranscriptionResult(content="ok")


def _make_transcriber(tmp_path: Path) -> tuple[Any, _StubProvider]:
    """Build a LangChainTranscriber backed by a label-recording stub provider."""
    from modules.config.config_loader import PROJECT_ROOT

    schema_path = tmp_path / "schema.json"
    schema_path.write_text('{"type": "object", "properties": {}}', encoding="utf-8")
    prompt_path = PROJECT_ROOT / "system_prompt" / "transcription_prompt_schema.txt"

    provider = _StubProvider()
    mock_cs = MagicMock()
    mock_cs.get_model_config.return_value = {
        "transcription_model": {
            "provider": "openai",
            "name": "gpt-4o",
            "max_output_tokens": 4096,
            "temperature": 0.0,
        }
    }
    mock_cs.get_concurrency_config.return_value = {}
    mock_cs.get_paths_config.return_value = {"general": {}}
    mock_cs.get_image_processing_config.return_value = {}

    with (
        patch("modules.llm.transcriber.get_config_service", return_value=mock_cs),
        patch("modules.llm.transcriber.get_provider", return_value=provider),
    ):
        from modules.llm import transcriber as tr

        instance = tr.LangChainTranscriber(
            schema_path=schema_path,
            system_prompt_path=prompt_path,
            use_hierarchical_context=False,
        )

    return instance, provider


class TestCallLabelContextManager:
    """Tests for call_label/current_call_label semantics."""

    @pytest.mark.unit
    def test_sets_and_resets_label(self) -> None:
        """The label is visible inside the block and cleared afterwards."""
        assert current_call_label() is None
        with call_label("page_0001.jpg"):
            assert current_call_label() == "page_0001.jpg"
        assert current_call_label() is None

    @pytest.mark.unit
    def test_nesting_restores_outer_label(self) -> None:
        """An inner binding shadows the outer one, which is then restored."""
        with call_label("outer.jpg"):
            with call_label("inner.jpg"):
                assert current_call_label() == "inner.jpg"
            assert current_call_label() == "outer.jpg"
        assert current_call_label() is None

    @pytest.mark.unit
    @pytest.mark.parametrize("falsy", [None, ""])
    def test_falsy_label_is_noop(self, falsy: str | None) -> None:
        """A falsy label leaves whatever label is already bound untouched."""
        with call_label(falsy):
            assert current_call_label() is None
        with call_label("outer.jpg"):
            with call_label(falsy):
                assert current_call_label() == "outer.jpg"
            assert current_call_label() == "outer.jpg"

    @pytest.mark.unit
    async def test_concurrent_tasks_are_isolated(self) -> None:
        """Gathered coroutines each observe only their own label."""

        async def run(name: str) -> list[str | None]:
            with call_label(name):
                seen = [current_call_label()]
                await asyncio.sleep(0)
                seen.append(current_call_label())
            return seen

        first, second = await asyncio.gather(run("a.jpg"), run("b.jpg"))
        assert first == ["a.jpg", "a.jpg"]
        assert second == ["b.jpg", "b.jpg"]
        assert current_call_label() is None


class TestTranscriberBindsLabel:
    """Tests that LangChainTranscriber binds the page name for provider calls."""

    @pytest.mark.unit
    async def test_transcribe_image_binds_file_name(self, tmp_path: Path) -> None:
        """transcribe_image binds the image file name."""
        transcriber, provider = _make_transcriber(tmp_path)
        image_path = tmp_path / "page_0007.jpg"
        image_path.write_bytes(b"not-a-real-jpeg")

        await transcriber.transcribe_image(image_path)

        assert provider.seen == ["page_0007.jpg"]
        assert current_call_label() is None

    @pytest.mark.unit
    async def test_base64_binds_explicit_label(self, tmp_path: Path) -> None:
        """transcribe_image_from_base64 binds the label keyword argument."""
        transcriber, provider = _make_transcriber(tmp_path)

        await transcriber.transcribe_image_from_base64(
            "ZmFrZQ==", "image/jpeg", label="p1.png"
        )

        assert provider.seen == ["p1.png"]

    @pytest.mark.unit
    async def test_base64_without_label_leaves_none(self, tmp_path: Path) -> None:
        """Omitting the label keeps the label unbound."""
        transcriber, provider = _make_transcriber(tmp_path)

        await transcriber.transcribe_image_from_base64("ZmFrZQ==", "image/jpeg")

        assert provider.seen == [None]


class TestPipelineForwardsLabel:
    """Tests that transcribe_payload forwards the payload's image name."""

    @pytest.mark.unit
    async def test_transcribe_payload_forwards_label(self) -> None:
        """transcribe_payload passes label=payload.image_name to the transcriber."""
        from modules.images.page_stream import PagePayload
        from modules.transcribe.pipeline import transcribe_payload

        captured: dict[str, Any] = {}

        class _Transcriber:
            async def transcribe_image_from_base64(
                self, *args: Any, **kwargs: Any
            ) -> dict[str, Any]:
                captured["args"] = args
                captured["kwargs"] = kwargs
                return {"output_text": "hello"}

        payload = PagePayload(
            index=3,
            image_name="page_0004_pre_processed.jpg",
            base64="ZmFrZQ==",
            mime_type="image/jpeg",
        )

        await transcribe_payload(payload, _Transcriber())

        assert captured["kwargs"].get("label") == "page_0004_pre_processed.jpg"
