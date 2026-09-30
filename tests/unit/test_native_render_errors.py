"""Pages that cannot be rendered or encoded become transcription errors."""

import json
import random
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from modules.batch.backends import BatchHandle
from modules.batch.submission import BatchSubmissionError, submit_batch
from modules.images.page_stream import PagePayload, stream_folder_payloads
from modules.transcribe.pipeline import transcribe_payload


def noisy_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "scans"
    folder.mkdir()
    Image.new("L", (40, 60), 200).save(folder / "page_1.png")
    noise = random.Random(7).randbytes(300 * 300 * 3)
    Image.frombytes("RGB", (300, 300), noise).save(folder / "page_2.png")
    return folder


async def test_oversized_page_becomes_error_record(tmp_path: Path) -> None:
    folder = noisy_folder(tmp_path)
    cfg = {
        "payload_format": "jpeg",
        "jpeg_quality": 95,
        "grayscale_conversion": False,
        "max_image_bytes": 20000,
        "resize_profile": "none",
    }
    payloads = [
        payload
        async for payload in stream_folder_payloads(
            folder, img_cfg=cfg, model_type="custom"
        )
    ]
    assert [p.render_error is None for p in payloads] == [True, False]
    failed = payloads[1]
    assert failed.image_name == "page_2.png_pre_processed.jpg"
    assert "max_image_bytes" in failed.provenance()["render_error"]
    transcriber = MagicMock()
    transcriber.transcribe_image_from_base64 = AsyncMock()
    result = await transcribe_payload(failed, transcriber)
    assert result[2] == f"[transcription error: {failed.image_name}]"
    transcriber.transcribe_image_from_base64.assert_not_called()


def payload(index: int, render_error: str | None = None) -> PagePayload:
    return PagePayload(
        index=index,
        image_name=f"page_{index + 1:04d}_pre_processed.jpg",
        base64="" if render_error else "ZmFrZQ==",
        source_file="scan.pdf",
        page_index=index,
        render_error=render_error,
    )


async def test_batch_records_unrendered_pages_without_requests(tmp_path: Path) -> None:
    temp = tmp_path / "scan_temporary.jsonl"
    backend = MagicMock(max_batch_size=100, max_batch_bytes=10_000_000)
    submitted: list[Any] = []

    def submit(part: list[Any], *args: Any, **kwargs: Any) -> BatchHandle:
        submitted.extend(part)
        return BatchHandle(provider="openai", batch_id="batch-1")

    backend.submit_batch.side_effect = submit
    with (
        patch("modules.batch.submission.get_batch_backend", return_value=backend),
        patch("modules.batch.submission._load_system_prompt", return_value="S"),
        patch(
            "modules.batch.submission._resolve_additional_context", return_value=None
        ),
        patch("modules.batch.submission._resolve_context_image", return_value=None),
    ):
        await submit_batch(
            [payload(0), payload(1, "Payload exceeds max_image_bytes")],
            temp,
            tmp_path,
            "scan",
            {"transcription_model": {"provider": "openai", "name": "gpt-6-astra"}},
            MagicMock(selected_schema_path=None),
        )
        assert [req.custom_id for req in submitted] == ["req-1"]
        metadata = [
            json.loads(line)["image_metadata"]
            for line in temp.read_text(encoding="utf-8").splitlines()
            if "image_metadata" in line
        ]
        assert [m["custom_id"] for m in metadata] == ["req-1", "unrendered-2"]
        with pytest.raises(BatchSubmissionError, match="could be rendered"):
            await submit_batch(
                [payload(0, "broken")],
                temp,
                tmp_path,
                "scan",
                {"transcription_model": {"provider": "openai"}},
                MagicMock(selected_schema_path=None),
            )
