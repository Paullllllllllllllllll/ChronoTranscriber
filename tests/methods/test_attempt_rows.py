"""Every provider attempt of a unit is a response row with the body verbatim.

Run on Windows with:

    uv --directory <repo> run pytest tests/methods/test_attempt_rows.py -q

The unit pipeline runs the real ``PageTranscriber`` (through ``ModelCaller``)
and the real ``OpenAIAudioBackend``; an ``httpx.MockTransport`` answers
behind their httpx clients (``tests.llm.network``), so no request leaves the
machine. When request *n* arrives, the transport reads the store through a
``mode=ro`` connection and expects the rows of the *n - 1* attempts before
it.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
from pathlib import Path
from typing import Any

import pytest

from chronotranscriber.common.store import Store
from chronotranscriber.llm.caller import CallEnv
from chronotranscriber.llm.transcription import PageTranscriber
from chronotranscriber.methods.audio_api import OpenAIAudioBackend
from chronotranscriber.methods.pages import transcribe_audio_payload, transcribe_payload
from chronotranscriber.methods.units import UnitsFailedError, run_unit_stream
from chronotranscriber.output.attempts import AttemptRows
from chronotranscriber.settings import RetrySettings
from chronotranscriber.sources.audio.stream import AudioChunkPayload
from chronotranscriber.sources.images import PageImage
from tests.audio.test_audio_capture import AUDIO, TRANSCRIPT
from tests.audio.test_audio_capture import MALFORMED as AUDIO_MALFORMED
from tests.audio.test_audio_capture import REFUSAL as AUDIO_REFUSAL
from tests.conftest import make_settings
from tests.llm import helpers
from tests.llm.conftest import Sleeps
from tests.llm.network import MockNetwork
from tests.llm.test_attempt_capture import (
    OPENAI_MALFORMED,
    OPENAI_OK,
    OPENAI_REFUSAL,
    SERVER_ERROR,
)
from tests.methods.conftest import TOOL, open_handle, responses, stream, units

PAGE = PageImage(
    index=0,
    image_name="page_0001.png",
    source_file="doc.pdf",
    page_index=0,
    base64="QUJD" * 600,
    mime_type="image/jpeg",
)


class StoredRows:
    """The response rows of the store, read through a ``mode=ro`` connection."""

    def __init__(self, db: Path) -> None:
        self.db = db

    @property
    def records(self) -> list[dict[str, Any]]:
        return responses(self.db) if self.db.exists() else []


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return tmp_path / "chronotranscriber.sqlite"


@pytest.fixture
def network(monkeypatch: pytest.MonkeyPatch, db: Path) -> MockNetwork:
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.setenv(name, f"test-{name}")
    view: Any = StoredRows(db)
    return MockNetwork(view).install(monkeypatch)


def _bodies(rows: list[dict[str, Any]]) -> list[bytes]:
    return [str(row["raw"]).encode("utf-8") for row in rows]


def _run(
    db: Path,
    transcriber: Any,
    handler: Any,
    payload: Any,
    model: str,
    kind: str = "page",
) -> UnitsFailedError | None:
    async def main() -> UnitsFailedError | None:
        async with Store(db, TOOL) as store:
            handle = await open_handle(store, pages=1, model=model, kind=kind)
            try:
                await run_unit_stream(
                    handle, stream([payload]), transcriber, handler, concurrency=1
                )
            except UnitsFailedError as exc:
                return exc
            finally:
                await transcriber.close()
        return None

    return asyncio.run(main())


def _page(db: Path, sleeps: Sleeps, validation_attempts: int = 1) -> Any:
    retry = dataclasses.replace(
        RetrySettings(), validation_attempts=validation_attempts
    )
    env = helpers.env(retry, clock=sleeps.clock)
    transcriber = PageTranscriber(helpers.model_spec(response_format="text"), env)
    return _run(db, transcriber, transcribe_payload, PAGE, "gpt-5.6-luna")


def _check_rows(db: Path, network: MockNetwork, expected: list[bytes]) -> Any:
    """Check the bodies and stamps of the rows; return the unit row."""
    rows = responses(db)
    (unit,) = units(db)
    assert _bodies(rows) == expected
    assert [row["raw_kind"] for row in rows] == ["http"] * len(expected)
    assert [row["attempt"] for row in rows] == list(range(1, len(expected) + 1))
    for row in rows:
        assert (row["doc_id"], row["kind"], row["idx"]) == (
            unit["doc_id"],
            unit["kind"],
            0,
        )
        assert (row["tool"], row["run_id"]) == (TOOL, unit["run_id"])
    assert network.violations == []
    assert len(network.requests) == len(expected)
    assert network.scoped == [True] * len(expected)
    return unit, rows


class TestPageTranscriber:
    """The oracle cases through ``ModelCaller`` on the Responses API."""

    def test_a_success_is_one_row_and_the_units_response(
        self, db: Path, network: MockNetwork, sleeps: Sleeps
    ) -> None:
        network.answers = [(200, OPENAI_OK)]

        assert _page(db, sleeps) is None

        unit, rows = _check_rows(db, network, [OPENAI_OK])
        assert (rows[0]["provider"], rows[0]["model"]) == ("openai", "gpt-6-luna")
        assert (rows[0]["http_status"], rows[0]["error"]) == (200, None)
        assert "QUJD" not in rows[0]["request"]
        assert unit["status"] == "done"
        assert unit["content"] == "Grüße aus  Zürich"
        assert (unit["response_id"], unit["attempts"]) == (rows[0]["response_id"], 1)

    def test_a_malformed_output_is_two_rows_and_the_second_is_selected(
        self, db: Path, network: MockNetwork, sleeps: Sleeps
    ) -> None:
        network.answers = [(200, OPENAI_MALFORMED), (200, OPENAI_OK)]

        assert _page(db, sleeps, validation_attempts=2) is None

        unit, rows = _check_rows(db, network, [OPENAI_MALFORMED, OPENAI_OK])
        assert [row["error_type"] for row in rows] == ["TypeError", None]
        assert (unit["response_id"], unit["attempts"]) == (rows[1]["response_id"], 2)
        assert unit["status"] == "done"

    def test_a_refusal_is_one_row_and_a_failed_unit(
        self, db: Path, network: MockNetwork, sleeps: Sleeps
    ) -> None:
        network.answers = [(200, OPENAI_REFUSAL)]

        failure = _page(db, sleeps)

        assert failure is not None and failure.failed_pages == 1
        unit, rows = _check_rows(db, network, [OPENAI_REFUSAL])
        assert unit["status"] == "failed"
        assert unit["outcome"] == "failed"
        assert unit["error"]
        assert unit["response_id"] == rows[0]["response_id"]

    def test_a_500_then_a_200_is_two_rows_committed_before_the_retry(
        self, db: Path, network: MockNetwork, sleeps: Sleeps
    ) -> None:
        network.answers = [(500, SERVER_ERROR), (200, OPENAI_OK)]

        assert _page(db, sleeps) is None

        unit, rows = _check_rows(db, network, [SERVER_ERROR, OPENAI_OK])
        assert [row["http_status"] for row in rows] == [500, 200]
        assert rows[0]["error_type"] == "InternalServerError"
        assert unit["response_id"] == rows[1]["response_id"]
        assert unit["content"] == "Grüße aus  Zürich"


def _chunk(tmp_path: Path) -> AudioChunkPayload:
    chunk = tmp_path / "rec_chunk_0001.mp3"
    chunk.write_bytes(AUDIO)
    return AudioChunkPayload(
        index=0,
        image_name=chunk.name,
        path=chunk,
        mime_type="audio/mp3",
        source_file=str(tmp_path / "rec.mp3"),
        start_seconds=0.0,
        duration_seconds=30.0,
        byte_size=len(AUDIO),
        sha256=hashlib.sha256(AUDIO).hexdigest(),
    )


def _audio(db: Path, tmp_path: Path) -> UnitsFailedError | None:
    env = CallEnv.create(
        make_settings(retry=RetrySettings(max_attempts=3), rate_limits=((100_000, 1),))
    )
    backend = OpenAIAudioBackend("gpt-transcribe", env=env)
    return _run(
        db,
        backend,
        transcribe_audio_payload,
        _chunk(tmp_path),
        "gpt-transcribe",
        kind="chunk",
    )


class TestOpenAIAudio:
    """The same oracle cases on the transcription endpoint."""

    def test_a_success_is_one_row_and_the_units_response(
        self, db: Path, tmp_path: Path, network: MockNetwork, sleeps: Sleeps
    ) -> None:
        network.answers = [(200, TRANSCRIPT)]

        assert _audio(db, tmp_path) is None

        unit, rows = _check_rows(db, network, [TRANSCRIPT])
        assert (rows[0]["role"], rows[0]["model"]) == ("audio", "gpt-transcribe")
        assert unit["content"] == "Grüße aus  Zürich"
        assert (unit["kind"], unit["source_index"]) == ("chunk", "0.000-30.000")
        assert unit["response_id"] == rows[0]["response_id"]

    def test_a_malformed_output_is_one_row_and_a_failed_unit(
        self, db: Path, tmp_path: Path, network: MockNetwork, sleeps: Sleeps
    ) -> None:
        network.answers = [(200, AUDIO_MALFORMED)]

        failure = _audio(db, tmp_path)

        assert failure is not None
        unit, rows = _check_rows(db, network, [AUDIO_MALFORMED])
        assert rows[0]["error_type"] == "JSONDecodeError"
        assert (unit["status"], unit["response_id"]) == (
            "failed",
            rows[0]["response_id"],
        )
        assert unit["error_type"] == "JSONDecodeError"

    def test_a_refusal_is_one_row_and_a_failed_unit(
        self, db: Path, tmp_path: Path, network: MockNetwork, sleeps: Sleeps
    ) -> None:
        network.answers = [(400, AUDIO_REFUSAL)]

        failure = _audio(db, tmp_path)

        assert failure is not None
        unit, rows = _check_rows(db, network, [AUDIO_REFUSAL])
        assert rows[0]["http_status"] == 400
        assert "content policy" in unit["error"]
        assert unit["response_id"] == rows[0]["response_id"]

    def test_a_500_then_a_200_is_two_rows_committed_before_the_retry(
        self, db: Path, tmp_path: Path, network: MockNetwork, sleeps: Sleeps
    ) -> None:
        network.answers = [(500, SERVER_ERROR), (200, TRANSCRIPT)]

        assert _audio(db, tmp_path) is None

        unit, rows = _check_rows(db, network, [SERVER_ERROR, TRANSCRIPT])
        assert [row["http_status"] for row in rows] == [500, 200]
        assert unit["response_id"] == rows[1]["response_id"]
        assert unit["attempts"] == 2


def test_the_recorder_selects_the_last_attempt_without_an_error(
    tmp_path: Path,
) -> None:
    """Without a successful attempt the last attempt is selected."""
    from chronotranscriber.common.capture import AttemptRecord

    def record(attempt: int, error: str | None) -> AttemptRecord:
        return AttemptRecord("openai", "m", "transcription", attempt, error=error)

    async def main() -> list[int | None]:
        async with Store(tmp_path / "chronotranscriber.sqlite", TOOL) as store:
            await store.begin_run("transcribe")
            rows = AttemptRows(store)
            picked = [rows.selected()]
            await rows.record(record(1, "boom"))
            picked.append(rows.selected().response_id)  # type: ignore[union-attr]
            await rows.record(record(2, None))
            await rows.record(record(3, "late"))
            picked.append(rows.selected().response_id)  # type: ignore[union-attr]
            return picked

    assert asyncio.run(main()) == [None, 1, 2]
