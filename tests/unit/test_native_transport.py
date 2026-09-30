"""PNG survives provider serialization, retries and batch submission."""

import base64
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from langchain_core.messages import AIMessage
from PIL import Image

from modules.batch.backends.anthropic_backend import AnthropicBatchBackend
from modules.batch.backends.base import BatchRequest
from modules.batch.backends.google_backend import GoogleBatchBackend
from modules.batch.backends.openai_backend import OpenAIBatchBackend
from modules.images.page_stream import load_image_payload
from modules.llm.providers.anthropic_provider import AnthropicProvider
from modules.llm.providers.google_provider import GoogleProvider
from modules.llm.providers.openai_provider import OpenAIProvider


def png_payload(tmp_path: Path) -> Any:
    path = tmp_path / "image.png"
    Image.new("L", (32, 48), 128).save(path)
    return load_image_payload(
        path,
        0,
        img_cfg={"payload_format": "png", "llm_detail": "original"},
        model_type="openai",
    )


@pytest.mark.parametrize(
    "provider_cls,client,model",
    [
        (
            OpenAIProvider,
            "modules.llm.providers.openai_provider.ChatOpenAI",
            "gpt-6-astra",
        ),
        (
            AnthropicProvider,
            "modules.llm.providers.anthropic_provider.ChatAnthropic",
            "claude-opus-5",
        ),
        (
            GoogleProvider,
            "modules.llm.providers.google_provider.ChatGoogleGenerativeAI",
            "gemini-3-flash-preview",
        ),
    ],
)
async def test_png_sync_and_retry(
    tmp_path: Path,
    provider_cls: Any,
    client: str,
    model: str,
) -> None:
    payload = png_payload(tmp_path)
    with patch(client):
        provider = provider_cls(api_key="test", model=model)
    captured = []

    async def invoke(messages: Any, **kwargs: Any) -> AIMessage:
        captured.append(json.dumps([message.model_dump() for message in messages]))
        if len(captured) == 1:
            raise httpx.ConnectError("Synthetic transient failure")
        return AIMessage(content="Synthetic transcription.")

    provider._llm.ainvoke = invoke
    with (
        patch("tenacity.wait_exponential_jitter", return_value=lambda _: 0),
        patch("modules.llm.providers.base.load_min_input_tokens", return_value=0),
        patch("modules.infra.token_budget.get_token_tracker"),
        patch.object(provider, "_process_llm_response", new=AsyncMock()),
    ):
        await provider.transcribe_image_from_base64(
            payload.base64,
            payload.mime_type,
            system_prompt="Transcribe.",
        )
    assert len(captured) == 2
    assert captured[0] == captured[1]
    assert "image/png" in captured[0]
    assert payload.base64 in captured[0]
    assert base64.b64decode(payload.base64).startswith(b"\x89PNG")


@pytest.mark.parametrize(
    "backend_cls,model",
    [
        (OpenAIBatchBackend, "gpt-6-astra"),
        (AnthropicBatchBackend, "claude-opus-5"),
        (GoogleBatchBackend, "gemini-3-flash-preview"),
    ],
)
def test_png_batch(tmp_path: Path, backend_cls: Any, model: str) -> None:
    payload = png_payload(tmp_path)
    backend = backend_cls()
    client = MagicMock()
    backend._client = client
    captured = []

    def upload(**kwargs: Any) -> Any:
        captured.append(kwargs["file"].read().decode("utf-8"))
        return MagicMock(id="file")

    client.files.create.side_effect = upload
    client.batches.create.return_value.name = "batch"
    backend.submit_batch(
        [
            BatchRequest(
                "page", image_base64=payload.base64, mime_type=payload.mime_type
            )
        ],
        {"name": model},
        system_prompt="Transcribe.",
    )
    if backend_cls is AnthropicBatchBackend:
        captured.append(json.dumps(client.messages.batches.create.call_args.kwargs))
    elif backend_cls is GoogleBatchBackend:
        captured.append(json.dumps(client.batches.create.call_args.kwargs))
    assert len(captured) == 1
    assert "image/png" in captured[0]
    assert payload.base64 in captured[0]


async def test_recorded_detail_override_is_per_request() -> None:
    from modules.llm.transcriber import LangChainTranscriber

    transcriber = object.__new__(LangChainTranscriber)
    transcriber._provider = MagicMock()
    transcriber._provider.transcribe_image_from_base64 = AsyncMock(return_value={})
    with (
        patch.object(
            transcriber,
            "_transcribe_kwargs",
            return_value={
                "image_detail": "high",
                "media_resolution": "low",
            },
        ),
        patch.object(transcriber, "_result_to_dict", return_value={}),
    ):
        await transcriber.transcribe_image_from_base64(
            "fixture",
            "image/png",
            image_detail="original",
            media_resolution="high",
        )
        await transcriber.transcribe_image_from_base64("fixture", "image/png")
    calls = transcriber._provider.transcribe_image_from_base64.call_args_list
    assert calls[0].kwargs["image_detail"] == "original"
    assert calls[0].kwargs["media_resolution"] == "high"
    assert calls[1].kwargs["image_detail"] == "high"
    assert calls[1].kwargs["media_resolution"] == "low"
