"""Decision-model calls: the page-type pre-pass and the classify commands.

:class:`DecisionModel` holds one run's decision model: its dialect, the
HTTP client with the capture hooks, one
:class:`~chronotranscriber.common.decision_client.DecisionClient` per key
variable and the shared
:class:`~chronotranscriber.common.decision_runner.DecisionRunner`. Each try
of a request opens an attempt of the extension seam (client kind
``decision``) under the model's role, behind the provider's rate limiter and
the shared retry ladder, which follows each failed attempt's decision; the
usage of every attempt, failed ones included, counts toward the run's totals
under that role. The attempts are not stored in ``page_attempts``.

:class:`PageDecider` is the pre-pass of ``--decisions``: it sends the pages
of one document in consecutive groups (``decisions.pages_per_request``),
each page downscaled to ``decisions.image_max_px`` on its longer side, with
the document name as context, and asks one page-type question per page
(:func:`~chronotranscriber.common.page_types.page_type_question`) under the
role ``decide``. Pages are read from their stream one group at a time. A
threshold is looked up by the request's fingerprint
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
from chronotranscriber.common.classifiers import Classifier, ClassifierQuestion
from chronotranscriber.common.decision_client import DecisionClient
from chronotranscriber.common.decision_runner import (
    DecisionRunner,
    Request,
    Seam,
    Unit,
)
from chronotranscriber.common.decisions import (
    MAX_QUESTIONS,
    Image,
    Json,
    Part,
    Question,
    Text,
    dialect_for,
    request_fingerprint,
)
from chronotranscriber.common.page_types import (
    DEFAULT_PROFILE,
    VERSION,
    Profile,
    no_text_probability,
    page_type_question,
)
from chronotranscriber.common.usage import Usage
from chronotranscriber.llm.caller import CallEnv
from chronotranscriber.spec import DecisionSpec

__all__ = [
    "KIND",
    "ROLE",
    "SKIPPED_OUTCOME",
    "DecisionModel",
    "PageDecider",
    "PageDecision",
]

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


class DecisionModel:
    """One run's decision model, asked through the shared runner.

    *model*, *provider* (None: from the model id) and *endpoint* (None: the
    dialect's) name the model; *role* is the usage role of its calls.

    Raises:
        OSError: None of the key variables of the decision provider is set.
        ValueError: *provider* is no decision dialect.
    """

    def __init__(
        self,
        model: str,
        provider: str | None,
        endpoint: str | None,
        env: CallEnv,
        *,
        role: str,
    ) -> None:
        settings = env.settings
        self.model = model
        self.endpoint = endpoint
        self.env = env
        self.role = role
        self.dialect = dialect_for(model, provider)
        variables = ext.key_variables(self.dialect, kind=KIND, settings=settings)
        if variables and not any(os.environ.get(name) for name in variables):
            raise OSError(f"the decision model needs the key variable {variables[0]}")
        self._http = httpx.AsyncClient(
            timeout=settings.decisions.timeout_s, event_hooks=async_capture_hooks()
        )
        self._clients: dict[str, DecisionClient] = {}
        seam = Seam(
            attempt=self._attempt,
            record=self._record,
            retry=settings.retry,
            gate=env.limiters.get(self.dialect),
        )
        self.runner = DecisionRunner(seam, model=model, dialect=self.dialect)

    def _attempt(self) -> ext.Attempt[DecisionClient]:
        return ext.attempt(
            self.role,
            self.dialect,
            None,
            settings=self.env.settings,
            build_client=self.client_for,
            endpoint=None,
            model=self.model,
            kind=KIND,
        )

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
                endpoint=options.base_url or self.endpoint,
            )
            self._clients[identity.api_key_env] = client
        return client

    def _record(self, usage: Usage) -> None:
        self.env.record(self.role, usage)

    async def aclose(self) -> None:
        """Close the HTTP client. Never raises."""
        with contextlib.suppress(Exception):
            await self._http.aclose()


class _PageTypeRequest(Request):
    """A pre-pass request, sent as the pass always sent it.

    Its pages are labelled ``Image n:`` and its questions name ``image n``,
    named ``page_type__nn`` also for a single page.
    """

    def slot(self, position: int) -> int | None:
        """Return the 1-based number of the page at *position*."""
        return position + 1

    def parts(self) -> list[Part]:
        """Return the context, then each page after its ``Image n:`` label."""
        parts: list[Part] = list(self.context)
        for number, unit in enumerate(self.units, start=1):
            parts += [Text(f"Image {number}:"), *unit.parts]
        return parts

    def questions(self) -> list[Question]:
        """Return one page-type question per page."""
        profile = self.classifier.questions[0].profile or DEFAULT_PROFILE
        return [
            page_type_question(f"page_type__{number:02d}", f"image {number}", profile)
            for number in range(1, len(self.units) + 1)
        ]


class PageDecider(DecisionModel):
    """Classify page images with the run's decision model.

    Raises:
        OSError: None of the key variables of the decision provider is set.
    """

    def __init__(self, spec: DecisionSpec, env: CallEnv) -> None:
        super().__init__(spec.model, spec.provider, spec.endpoint, env, role=ROLE)
        block = env.settings.decisions
        self.spec = spec
        self.profile: Profile = block.profile  # type: ignore[assignment]
        self.image_max_px = block.image_max_px
        self.pages_per_request = min(block.pages_per_request, MAX_QUESTIONS)
        self._thresholds = block.thresholds
        self._classifier = Classifier(
            name="page_types",
            version=1,
            unit="page",
            description="the page-type pre-pass",
            questions=(
                ClassifierQuestion("page_type", "page_types", profile=self.profile),
            ),
            per_request=self.pages_per_request,
        )

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
        request = _PageTypeRequest(
            self._classifier,
            document,
            tuple(Unit(str(idx), idx, (Image(data),)) for idx, data in group),
            (Json({"document": document}),),
        )
        label = f"{document} decisions {group[0][0]}-{group[-1][0]}"
        result = await self.runner.ask(request, label=label)
        return [
            PageDecision(
                unit.unit.ordinal, str(answer.value), dict(answer.probabilities)
            )
            for unit in result.units
            for answer in unit.answers
            if answer.refusal is None
            and answer.value is not None
            and answer.probabilities is not None
        ]


def page_bytes(base64_data: str) -> bytes:
    """Return the image bytes of a page payload's base64 text."""
    return base64.b64decode(base64_data)
