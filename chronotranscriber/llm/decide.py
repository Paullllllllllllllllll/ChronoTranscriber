"""The decision pre-pass: the page type of every page still to transcribe.

:class:`PageDecider` sends the pages of one document to a decision model in
consecutive groups (``decisions.pages_per_request``), each page downscaled to
``decisions.image_max_px`` on its longer side, with the document name as
context, and asks one page-type question per page
(:func:`~chronotranscriber.common.page_types.page_type_question`). Pages are
read from their stream one group at a time. Calls run under the role
``decide`` through the extension seam (client kind ``decision``), the
provider's rate limiter and the shared retry ladder, which follows each failed
attempt's decision; their usage, failed attempts included, counts toward the
run's totals. The attempts are not stored in ``page_attempts``.

A threshold is looked up by the request's fingerprint
(:meth:`PageDecider.fingerprint`), which covers the upstream image settings
of the pages as well; without one, ``apply`` skips nothing and the pass only
records.
"""

from __future__ import annotations

import base64
import contextlib
import io
import os
from collections.abc import AsyncIterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import httpx
from PIL import Image as PILImage

from chronotranscriber import ext
from chronotranscriber.common.capture import async_capture_hooks
from chronotranscriber.common.decision_client import DecisionClient
from chronotranscriber.common.decisions import (
    MAX_QUESTIONS,
    ChoiceAnswer,
    DecisionResult,
    Image,
    Json,
    MalformedAnswer,
    Part,
    Question,
    Text,
    dialect_for,
    request_fingerprint,
)
from chronotranscriber.common.page_types import (
    VERSION,
    Profile,
    no_text_probability,
    page_type_question,
)
from chronotranscriber.common.retry import (
    Decision,
    ErrorKind,
    RetryHooks,
    call_with_retry,
    classify_error,
)
from chronotranscriber.common.usage import Usage
from chronotranscriber.llm.caller import CallEnv
from chronotranscriber.llm.retrying import AttemptDecisions
from chronotranscriber.spec import DecisionSpec

__all__ = ["KIND", "ROLE", "SKIPPED_OUTCOME", "PageDecider", "PageDecision"]

ROLE: Final = "decide"
KIND: Final = "decision"
SKIPPED_OUTCOME: Final = "skipped_no_text"
"""The outcome of a page ``apply`` skipped as carrying no text of the book."""


@dataclass(frozen=True)
class PageDecision:
    """The page type of one page with every type's probability."""

    idx: int
    page_type: str
    probabilities: Mapping[str, float]

    @property
    def no_text(self) -> float:
        """The probability that the page carries no text of the book."""
        return no_text_probability(self.probabilities)


def _downscale(data: bytes, max_px: int) -> bytes:
    """Return *data* as a JPEG whose longer side is at most *max_px*."""
    with PILImage.open(io.BytesIO(data)) as opened:
        image = opened.convert("RGB")
    image.thumbnail((max_px, max_px))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=85)
    return buffer.getvalue()


class PageDecider:
    """Classify page images with the run's decision model.

    Raises:
        OSError: None of the key variables of the decision provider is set.
    """

    def __init__(self, spec: DecisionSpec, env: CallEnv) -> None:
        settings = env.settings
        block = settings.decisions
        self.spec = spec
        self.env = env
        self.dialect = dialect_for(spec.model, spec.provider)
        self.profile: Profile = block.profile  # type: ignore[assignment]
        self.image_max_px = block.image_max_px
        self.pages_per_request = min(block.pages_per_request, MAX_QUESTIONS)
        self._thresholds = block.thresholds
        self._policy = settings.retry.policy()
        self._timeout_attempts = settings.retry.timeout_attempts
        variables = ext.key_variables(self.dialect, kind=KIND, settings=settings)
        if variables and not any(os.environ.get(name) for name in variables):
            raise OSError(f"the decision model needs the key variable {variables[0]}")
        self._http = httpx.AsyncClient(
            timeout=block.timeout_s, event_hooks=async_capture_hooks()
        )
        self._clients: dict[str, DecisionClient] = {}

    def fingerprint(self, upstream: Mapping[str, Any] | None = None) -> str:
        """Return the fingerprint of this pass's requests (see the settings).

        *upstream* is the image-settings record the pages were rendered with.
        """
        return request_fingerprint(
            tool="chronotranscriber",
            dialect=self.dialect,
            model=self.spec.model,
            endpoint=self.spec.endpoint,
            taxonomy=VERSION,
            profile=self.profile,
            image_max_px=self.image_max_px,
            pages_per_request=self.pages_per_request,
            context="document",
            upstream=dict(upstream or {}),
        )

    def threshold(self, fingerprint: str) -> float | None:
        """Return the no-text probability at which ``apply`` skips, or None."""
        return self._thresholds.get(fingerprint)

    async def classify(
        self, document: str, pages: AsyncIterable[tuple[int, bytes]]
    ) -> list[PageDecision]:
        """Return the page type of each ``(idx, image bytes)`` in *pages*.

        Raises:
            MalformedAnswer: An answer does not fit its question.
            httpx.HTTPError: A request failed after the retries.
        """
        decisions: list[PageDecision] = []
        group: list[tuple[int, bytes]] = []
        async for page in pages:
            group.append((page[0], _downscale(page[1], self.image_max_px)))
            if len(group) == self.pages_per_request:
                decisions += await self._classify_group(document, group)
                group = []
        if group:
            decisions += await self._classify_group(document, group)
        return decisions

    async def _classify_group(
        self, document: str, group: Sequence[tuple[int, bytes]]
    ) -> list[PageDecision]:
        parts: list[Part] = [Json({"document": document})]
        questions: list[Question] = []
        for n, (_idx, data) in enumerate(group, start=1):
            parts += [Text(f"Image {n}:"), Image(data)]
            questions.append(
                page_type_question(f"page_type__{n:02d}", f"image {n}", self.profile)
            )
        label = f"{document} decisions {group[0][0]}-{group[-1][0]}"
        result = await self._call(parts, questions, label)
        decisions: list[PageDecision] = []
        for n, (idx, _data) in enumerate(group, start=1):
            answer = result.answers[f"page_type__{n:02d}"]
            if isinstance(answer, ChoiceAnswer):
                decisions.append(PageDecision(idx, answer.choice, answer.probabilities))
        return decisions

    def client_for(self, identity: ext.Identity) -> DecisionClient:
        """Return the client for an attempt identity, building it if new.

        The client takes the extension's client options of the identity.

        Raises:
            OSError: The identity's key variable is unset.
        """
        client = self._clients.get(identity.api_key_env)
        if client is None:
            options = ext.client_options(identity)
            key = options.api_key or os.environ.get(identity.api_key_env, "").strip()
            if not key:
                raise OSError(
                    f"the decision model needs the key variable {identity.api_key_env}"
                )
            client = DecisionClient(
                self._http,
                dialect=self.dialect,
                api_key=key,
                endpoint=options.base_url or self.spec.endpoint,
            )
            self._clients[identity.api_key_env] = client
        return client

    async def _call(
        self, parts: Sequence[Part], questions: Sequence[Question], label: str
    ) -> DecisionResult:
        decisions = AttemptDecisions()

        async def one_attempt() -> DecisionResult:
            attempt = ext.attempt(
                ROLE,
                self.dialect,
                None,
                settings=self.env.settings,
                build_client=self.client_for,
                endpoint=None,
                model=self.spec.model,
                kind=KIND,
            )
            async with contextlib.AsyncExitStack() as stack:
                call = await decisions.enter(
                    stack, attempt, on_failure=lambda: self._record(attempt.usage)
                )
                try:
                    result = await call.client.decide(self.spec.model, parts, questions)
                except Exception as exc:
                    decisions.failed(call, exc)
                    self._record(call.usage)
                    raise
                call.commit(
                    Usage(
                        input_tokens=result.usage.input_tokens,
                        output_tokens=result.usage.output_tokens,
                        total_tokens=result.usage.input_tokens
                        + result.usage.output_tokens,
                        cached_tokens=result.usage.cached_tokens,
                    )
                )
                self._record(call.usage)
                return result

        timeouts = 0

        def decide(exc: BaseException) -> Decision:
            nonlocal timeouts
            decision = decisions.decide(exc)
            retrying = decision in (Decision.RETRY, Decision.SWITCH_KEY)
            if not retrying or classify_error(exc) is not ErrorKind.TIMEOUT:
                return decision
            timeouts += 1
            return decision if timeouts < self._timeout_attempts else Decision.FAIL

        hooks = RetryHooks(on_error=decide, gate=self.env.limiters.get(self.dialect))
        return await call_with_retry(
            one_attempt,
            policy=self._policy,
            hooks=hooks,
            label=label,
            validation_errors=(MalformedAnswer,),
        )

    def _record(self, usage: Usage | None) -> None:
        if usage is not None:
            self.env.record(ROLE, usage)

    async def aclose(self) -> None:
        """Close the HTTP client. Never raises."""
        with contextlib.suppress(Exception):
            await self._http.aclose()


def page_bytes(base64_data: str) -> bytes:
    """Return the image bytes of a page payload's base64 text."""
    return base64.b64decode(base64_data)
