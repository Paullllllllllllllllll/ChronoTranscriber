"""Batch submission: submit_batch on the store."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from chronotranscriber import ext
from chronotranscriber.batch.submit import (
    REQUEST_TOO_LARGE,
    BatchSubmissionError,
    SubmitOutcome,
    submit_batch,
)
from chronotranscriber.common.image_presets import MAX_IMAGE_BYTES
from chronotranscriber.llm.prompts import DEFAULT_SCHEMA_PATH
from chronotranscriber.output.store import DATABASE_NAME, open_store
from chronotranscriber.settings import Settings
from chronotranscriber.sources.images import PageImage
from chronotranscriber.spec import resolve_spec
from tests.batch.conftest import World, leased_document, page_images, run_spec


async def submit(root: Path, payloads: list[Any]) -> SubmitOutcome:
    async with open_store(root) as store:
        await store.begin_run("transcribe")
        document = await leased_document(store)
        return await submit_batch(
            payloads,
            document=document,
            spec=run_spec(root),
            settings=Settings(),
            schema_path=DEFAULT_SCHEMA_PATH,
        )


async def test_the_submit_attempts_name_the_model(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identities: list[ext.Identity] = []
    real = ext.attempt

    def attempt(*args: Any, **kwargs: Any) -> ext.Attempt[Any]:
        opened: ext.Attempt[Any] = real(*args, **kwargs)
        identities.append(opened.identity)
        return opened

    monkeypatch.setattr(ext, "attempt", attempt)
    await submit(tmp_path, page_images(1))
    assert identities
    assert {(i.provider, i.role, i.model) for i in identities} == {
        ("openai", "transcription", "gpt-5.6-luna")
    }


async def test_submit_batch_reads_the_run_and_writes_no_working_file(
    world: World, tmp_path: Path
) -> None:
    spec = resolve_spec(
        {"input": tmp_path, "provider": "anthropic", "model": "claude-sonnet-4-6"}
    ).spec
    payloads = [
        PageImage(index=0, image_name="page_0001_pre_processed.jpg", base64="UEFHRQ=="),
        PageImage(
            index=1, image_name="page_0002_pre_processed.jpg", render_error="broken"
        ),
    ]
    root = tmp_path / "out"
    root.mkdir()
    async with open_store(root) as store:
        await store.begin_run("transcribe")
        document = await leased_document(store, provider="anthropic")
        outcome = await submit_batch(
            payloads,
            document=document,
            spec=spec,
            settings=Settings(),
            schema_path=DEFAULT_SCHEMA_PATH,
        )
    assert outcome.status == "submitted"
    assert len(outcome.submission_ids) == 1
    (batch,) = world.batches.values()
    assert batch.provider == "anthropic"
    (body,) = world.bodies()
    assert body["model"] == "claude-sonnet-4-6"
    assert sorted(path.name for path in root.iterdir()) == [DATABASE_NAME]


async def test_an_oversized_page_fails_with_its_error_and_the_others_are_sent(
    world: World, tmp_path: Path
) -> None:
    spec = resolve_spec(
        {"input": tmp_path, "provider": "anthropic", "model": "claude-sonnet-4-6"}
    ).spec
    oversized = "A" * (MAX_IMAGE_BYTES["anthropic"] + 4)
    payloads = [
        PageImage(index=index, image_name=f"page_{index + 1:04d}.jpg", base64=data)
        for index, data in enumerate(["UEFHRQ==", oversized, "UEFHRQ=="])
    ]
    async with open_store(tmp_path) as store:
        await store.begin_run("transcribe")
        document = await leased_document(store, provider="anthropic")
        outcome = await submit_batch(
            payloads,
            document=document,
            spec=spec,
            settings=Settings(),
            schema_path=DEFAULT_SCHEMA_PATH,
        )
        (submission_id,) = outcome.submission_ids
        members = store.submission_requests(submission_id)
        units = {row.idx: row for row in store.units(document.doc_id)}
    assert outcome.status == "submitted"
    assert sorted(key.idx for key in members.values()) == [0, 2]
    assert len(world.bodies()) == 2
    assert [units[idx].status for idx in range(3)] == ["pending", "failed", "pending"]
    failed = units[1]
    assert failed.error_type == REQUEST_TOO_LARGE
    assert failed.error is not None and "over the anthropic limit" in failed.error
    assert failed.content == f"[transcription error: {failed.error}]"


async def test_submit_batch_refuses_providers_without_a_batch_api(
    world: World, tmp_path: Path
) -> None:
    spec = resolve_spec(
        {"input": tmp_path, "provider": "openrouter", "model": "openai/gpt-5.4"}
    ).spec
    async with open_store(tmp_path) as store:
        await store.begin_run("transcribe")
        document = await leased_document(store)
        with pytest.raises(BatchSubmissionError, match="does not support batch"):
            await submit_batch(
                [],
                document=document,
                spec=spec,
                settings=Settings(),
                schema_path=None,
            )
        assert store.submissions() == []
    assert world.calls == []
