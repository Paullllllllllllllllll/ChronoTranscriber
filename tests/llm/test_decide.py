"""The decision pre-pass: grouping, thresholds, skips and the spec checks."""

from __future__ import annotations

import asyncio
import io
import json
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any

import httpx
import pytest
from PIL import Image as PILImage

from chronotranscriber.common.settings_schema import Decisions
from chronotranscriber.llm.caller import CallEnv
from chronotranscriber.llm.decide import SKIPPED_OUTCOME, PageDecider, PageDecision
from chronotranscriber.methods.llm import LlmTranscriber
from chronotranscriber.settings import Settings
from chronotranscriber.spec import DecisionSpec, SpecError, resolve_repair, resolve_spec


def _jpeg(size: tuple[int, int] = (1200, 1800)) -> bytes:
    buffer = io.BytesIO()
    PILImage.new("RGB", size, "white").save(buffer, "JPEG")
    return buffer.getvalue()


def _env(**block: Any) -> CallEnv:
    return CallEnv.create(Settings(decisions=replace(Decisions(), **block)))


def _answers(body: dict[str, Any], page_type: str) -> dict[str, Any]:
    names = list(body["questions"])
    return {
        "model": "typesafe/jev-1.14",
        "answers": {
            name: {
                "type": "choice",
                "choice": page_type,
                "probabilities": {page_type: 0.97, "body_text": 0.03},
            }
            for name in names
        },
        "usage": {"input_tokens": 100, "output_tokens": 0},
    }


def _decider(env: CallEnv, seen: list[dict[str, Any]], page_type: str) -> PageDecider:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, json=_answers(body, page_type))

    decider = PageDecider(DecisionSpec("shadow", "typesafe/jev-1.14", None, None), env)
    decider._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return decider


async def _stream(pages: list[tuple[int, bytes]]) -> AsyncIterator[tuple[int, bytes]]:
    for page in pages:
        yield page


@pytest.fixture(autouse=True)
def _key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


def test_pages_go_out_in_groups_downscaled_with_the_document_as_context() -> None:
    seen: list[dict[str, Any]] = []
    decider = _decider(_env(pages_per_request=2, image_max_px=300), seen, "blank")
    pages = [(i, _jpeg()) for i in (4, 5, 6)]

    decisions = asyncio.run(decider.classify("doc-1/doc", _stream(pages)))

    assert [d.idx for d in decisions] == [4, 5, 6]
    assert [len(body["questions"]) for body in seen] == [2, 1]
    assert json.loads(seen[0]["state"][0]["text"]) == {"document": "doc-1/doc"}
    assert decisions[0].no_text == pytest.approx(0.97)
    assert decider.env.usage.get("decide").input_tokens == 200


def test_a_client_error_follows_the_attempt_decision_and_is_not_retried() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(400, json={"error": {"message": "bad request"}})

    env = _env()
    decider = PageDecider(DecisionSpec("shadow", "typesafe/jev-1.14", None, None), env)
    decider._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(decider.classify("doc", _stream([(0, _jpeg((60, 90)))])))
    assert calls == [1]


def test_a_threshold_applies_only_to_its_request_fingerprint() -> None:
    decider = _decider(_env(), [], "blank")
    fingerprint = decider.fingerprint({"dpi": 300})
    assert decider.threshold(fingerprint) is None
    tuned = _decider(_env(thresholds={fingerprint: 0.95}), [], "blank")
    assert tuned.threshold(fingerprint) == 0.95
    assert decider.fingerprint({"dpi": 200}) != fingerprint
    other = _decider(_env(pages_per_request=12), [], "blank")
    assert other.fingerprint({"dpi": 300}) != fingerprint


def test_a_missing_key_fails_before_any_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY")
    with pytest.raises(OSError, match="OPENROUTER_API_KEY"):
        PageDecider(DecisionSpec("shadow", "typesafe/jev-1.14", None, None), _env())


class _Store:
    def __init__(self) -> None:
        self.stored: list[Any] = []

    async def write(self, lease: object, func: Any) -> None:
        self.stored.append(func)


class _Handle:
    """The parts of a document handle the pre-pass touches."""

    name = "doc-1/doc"
    run_id = "run-1"
    lease = object()

    def __init__(self) -> None:
        self.writes: list[tuple[int, dict[str, Any]]] = []
        self.store = _Store()

    async def write(self, idx: int, **fields: Any) -> None:
        self.writes.append((idx, fields))


class _Decider:
    profile = "general"
    dialect = "openrouter"
    spec = DecisionSpec("apply", "typesafe/jev-1.14", None, None)

    def __init__(self, threshold: float | None) -> None:
        self._threshold = threshold
        self.upstream: Any = None

    async def classify(self, document: str, pages: Any) -> list[PageDecision]:
        assert [idx async for idx, _data in pages] == [0, 1]
        return [
            PageDecision(0, "blank", {"blank": 0.99, "body_text": 0.01}),
            PageDecision(1, "body_text", {"body_text": 0.9, "blank": 0.1}),
        ]

    def fingerprint(self, upstream: Any = None) -> str:
        self.upstream = upstream
        return "fp"

    def threshold(self, fingerprint: str) -> float | None:
        return self._threshold


class _Events:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, text: str) -> None:
        self.warnings.append(text)


def _run_decide(
    mode: str, threshold: float | None
) -> tuple[list[int], _Handle, _Events]:
    events = _Events()
    job = type("Job", (), {})()
    job.spec = type("Spec", (), {"decision": DecisionSpec(mode, "m", None, None)})()
    job.events = events
    transcriber = LlmTranscriber(job)
    transcriber._decider = _Decider(threshold)  # type: ignore[assignment]
    handle = _Handle()

    async def source(indices: list[int]) -> Any:
        for idx in indices:
            yield type(
                "Page", (), {"index": idx, "base64": "AA==", "render_error": None}
            )()

    async def main() -> list[int]:
        return await transcriber._decide(handle, [0, 1], source, {"dpi": 300})  # type: ignore[arg-type]

    needed = asyncio.run(main())
    return needed, handle, events


def test_apply_skips_pages_at_the_threshold_and_records_why() -> None:
    needed, handle, _ = _run_decide("apply", 0.95)
    assert needed == [1]
    assert len(handle.store.stored) == 1
    [(idx, fields)] = handle.writes
    assert idx == 0
    assert fields["outcome"] == SKIPPED_OUTCOME
    assert fields["content"] == "[No transcribable text: blank]"
    assert fields["provenance"]["decision"]["fingerprint"] == "fp"


def test_shadow_and_apply_without_a_threshold_skip_nothing() -> None:
    needed, handle, _ = _run_decide("shadow", 0.95)
    assert (needed, handle.writes) == ([0, 1], [])
    needed, handle, events = _run_decide("apply", None)
    assert (needed, handle.writes) == ([0, 1], [])
    assert "No validated decision threshold" in events.warnings[0]


def test_decisions_need_the_llm_method() -> None:
    with pytest.raises(SpecError, match="--method llm"):
        resolve_spec({"input": "in.pdf", "method": "tesseract", "decisions": "shadow"})


def test_repair_skipped_selects_only_skipped_pages_unless_told_more() -> None:
    assert resolve_repair({"input": ".", "skipped": True}).spec.selection.causes == (
        "skipped",
    )
    causes = resolve_repair({"input": ".", "all_failures": True, "skipped": True})
    assert causes.spec.selection.causes == (
        "api_error",
        "not_possible",
        "no_text",
        "skipped",
    )
    plain = resolve_repair({"input": ".", "all_failures": True})
    assert "skipped" not in plain.spec.selection.causes
