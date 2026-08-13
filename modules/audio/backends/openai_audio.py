"""OpenAI speech-to-text backend (``/v1/audio/transcriptions``).

Called through the raw ``AsyncOpenAI`` SDK rather than LangChain, because the
transcriptions endpoint is a multipart file upload with no chat-message shape.
The SDK client is therefore built with ``max_retries=0`` and every call is
wrapped in :func:`modules.audio.retry.acall_with_retry`, keeping this backend
under the same single retry authority as the chat providers.

Parameter routing follows the installed SDK's ``transcriptions.create``
signature: ``file``, ``model``, ``language``, ``prompt``, and ``temperature``
are native keyword arguments, while the ``languages[]`` / ``keywords[]`` list
parameters that only ``gpt-transcribe`` accepts are not in the signature and
travel through ``extra_body``.
"""

from __future__ import annotations

import os
from typing import Any

from openai import AsyncOpenAI

from modules.audio.audio_stream import AudioChunkPayload
from modules.audio.backends.base import build_legacy_response, error_response
from modules.audio.constants import (
    OPENAI_ALLOWED_EXTENSIONS,
    OPENAI_PLURAL_PARAM_MODELS,
    WHISPER1_PROMPT_TOKEN_CAP,
)
from modules.audio.retry import acall_with_retry
from modules.config.capabilities import ensure_audio_support
from modules.infra.logger import setup_logger
from modules.llm.providers.factory import (
    ProviderType,
    get_api_key_for_provider,
    resolve_api_key_env_var,
)
from modules.llm.providers.http_timeouts import build_httpx_timeout

logger = setup_logger(__name__)

DEFAULT_OPENAI_AUDIO_MODEL = "gpt-transcribe"

# Audio uploads are long-lived multipart requests; when the concurrency config
# carries no request_timeout, five minutes is a safer floor than the SDK's
# ten-minute default applied to a chat call.
DEFAULT_AUDIO_REQUEST_TIMEOUT_S = 300.0

# whisper-1's prompt cap is expressed in tokens; ~4 characters per token is the
# standard rough conversion and only needs to be right to within a wide margin.
_CHARS_PER_TOKEN = 4


class OpenAIAudioBackend:
    """Transcribes chunks through the OpenAI speech-to-text endpoint."""

    def __init__(
        self,
        audio_config: dict[str, Any],
        concurrency_config: dict[str, Any],
    ) -> None:
        """Build the SDK client and resolve the model, key, and request knobs.

        Args:
            audio_config: Parsed ``audio_config.yaml``.
            concurrency_config: Parsed ``concurrency_config.yaml``; supplies
                ``concurrency.transcription.request_timeout``.

        Raises:
            CapabilityError: When the configured model cannot accept audio.
            ValueError: When no OpenAI API key can be resolved.
        """
        settings = (audio_config.get("audio_transcription", {}) or {}).get(
            "openai", {}
        ) or {}
        self.provider_name = "openai"
        self.model = str(settings.get("model") or DEFAULT_OPENAI_AUDIO_MODEL).strip()
        ensure_audio_support(self.model)

        self._settings = settings
        self._timeout = self._resolve_timeout(concurrency_config)
        self._key_env = resolve_api_key_env_var(ProviderType.OPENAI)
        # Typed as Any: transcriptions.create is a five-way overload and the
        # request kwargs are assembled dynamically, which no single overload
        # can be matched against statically.
        self._client: Any = self._build_client(
            get_api_key_for_provider(ProviderType.OPENAI)
        )
        # Clients replaced by rekey(); closed together at the end of the run
        # (rekey is synchronous and cannot await a close).
        self._retired_clients: list[Any] = []

        logger.info(
            "OpenAI audio backend initialized: model=%s, key_env=%s, timeout=%ss",
            self.model,
            self._key_env,
            self._timeout,
        )

    # -- construction helpers ---------------------------------------------

    @staticmethod
    def _resolve_timeout(concurrency_config: dict[str, Any]) -> float:
        """Read ``concurrency.transcription.request_timeout``, else the default."""
        try:
            trans = (concurrency_config.get("concurrency", {}) or {}).get(
                "transcription", {}
            ) or {}
            raw = trans.get("request_timeout")
            if raw is not None:
                return float(raw)
        except (AttributeError, TypeError, ValueError) as exc:
            logger.debug("Could not read request_timeout, using default: %s", exc)
        return DEFAULT_AUDIO_REQUEST_TIMEOUT_S

    def _build_client(self, api_key: str) -> AsyncOpenAI:
        """Build an SDK client with retries disabled (see module docstring).

        The timeout is per-phase: a scalar float would apply the long upload
        budget to the connect phase as well, hiding a dead peer for minutes.
        """
        return AsyncOpenAI(
            api_key=api_key,
            max_retries=0,
            timeout=build_httpx_timeout(self._timeout),
        )

    # -- request construction ----------------------------------------------

    def _resolve_prompt(self) -> str:
        """Return the configured prompt, trimmed to whisper-1's token cap."""
        prompt = str(self._settings.get("prompt") or "").strip()
        if not prompt or self.model != "whisper-1":
            return prompt
        max_chars = WHISPER1_PROMPT_TOKEN_CAP * _CHARS_PER_TOKEN
        if len(prompt) <= max_chars:
            return prompt
        logger.warning(
            "whisper-1 caps the prompt at %d tokens; truncating from %d to %d "
            "characters.",
            WHISPER1_PROMPT_TOKEN_CAP,
            len(prompt),
            max_chars,
        )
        return prompt[:max_chars]

    def _build_kwargs(
        self, filename: str, data: bytes, mime_type: str
    ) -> dict[str, Any]:
        """Assemble the ``transcriptions.create`` keyword arguments."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            # The SDK accepts a (name, bytes, content_type) tuple for file
            # parameters, so the chunk never needs an open file handle.
            "file": (filename, data, mime_type),
        }

        prompt = self._resolve_prompt()
        if prompt:
            kwargs["prompt"] = prompt

        if self.model in OPENAI_PLURAL_PARAM_MODELS:
            extra_body: dict[str, Any] = {}
            languages = [
                str(x).strip() for x in (self._settings.get("languages") or []) if x
            ]
            keywords = [
                str(x).strip() for x in (self._settings.get("keywords") or []) if x
            ]
            if languages:
                extra_body["languages"] = languages
            if keywords:
                extra_body["keywords"] = keywords
            if extra_body:
                kwargs["extra_body"] = extra_body
        else:
            language = str(self._settings.get("language") or "").strip()
            if language:
                kwargs["language"] = language

        temperature = self._settings.get("temperature")
        if temperature is not None:
            kwargs["temperature"] = float(temperature)

        return kwargs

    # -- usage accounting ---------------------------------------------------

    def _commit_usage(self, usage: Any) -> tuple[int, int, int]:
        """Commit token usage to the daily budget; return the token triple.

        The transcriptions response carries either a token-billed usage object
        (``type == "tokens"``) or a duration-billed one (``type ==
        "duration"``), and older models may report none at all. Only the token
        variant is committed, stamped ``(provider, key_env, model)`` exactly as
        :meth:`BaseProvider._track_token_usage` stamps a chat call, so audio
        spend lands in the same per-key bucket as everything else. Duration
        billing has no token equivalent and is deliberately not converted.
        """
        if usage is None or getattr(usage, "type", None) != "tokens":
            logger.debug(
                "No token usage on the transcription response (type=%s); "
                "nothing committed to the daily budget.",
                getattr(usage, "type", None),
            )
            return 0, 0, 0

        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
        if total_tokens <= 0:
            total_tokens = input_tokens + output_tokens

        if total_tokens > 0:
            try:
                from modules.infra.token_budget import get_token_tracker

                get_token_tracker().add_tokens(
                    total_tokens,
                    provider=self.provider_name,
                    key_env=self._key_env,
                    model=self.model,
                )
            except Exception as exc:
                logger.warning("Error tracking audio tokens: %s", exc)

        return input_tokens, output_tokens, total_tokens

    # -- backend protocol ---------------------------------------------------

    async def transcribe_chunk(self, payload: AudioChunkPayload) -> dict[str, Any]:
        """Transcribe one chunk; see :mod:`modules.audio.backends.base`."""
        suffix = payload.path.suffix.lower()
        if suffix not in OPENAI_ALLOWED_EXTENSIONS:
            allowed = ", ".join(sorted(OPENAI_ALLOWED_EXTENSIONS))
            return error_response(
                f"OpenAI transcription does not accept '{suffix}' files "
                f"(allowed: {allowed}). Set chunking.chunk_format to 'mp3' or "
                f"'wav16' to convert on the fly.",
                provider=self.provider_name,
                model=self.model,
                chunk_index=payload.index,
            )

        try:
            data = await payload.read_bytes()
            kwargs = self._build_kwargs(payload.image_name, data, payload.mime_type)
            response = await acall_with_retry(
                lambda: self._client.audio.transcriptions.create(**kwargs),
                provider_name=self.provider_name,
            )
        except Exception as exc:
            logger.error(
                "OpenAI audio transcription failed for %s: %s",
                payload.image_name,
                exc,
            )
            return error_response(
                str(exc),
                provider=self.provider_name,
                model=self.model,
                chunk_index=payload.index,
            )

        input_tokens, output_tokens, total_tokens = self._commit_usage(
            getattr(response, "usage", None)
        )
        return build_legacy_response(
            output_text=str(getattr(response, "text", "") or ""),
            provider=self.provider_name,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            metadata={"chunk_index": payload.index},
        )

    def rekey(self) -> None:
        """Rebuild the SDK client against a freshly resolved key env var.

        Satisfies the :class:`~modules.audio.backends.base.AudioBackend`
        protocol. Nothing calls it mid-run: it re-resolves the provider's key
        env var through :func:`resolve_api_key_env_var` and rebuilds the client
        so a caller that has swapped the mapping (or the environment) can adopt
        it without a restart. A no-op when the name is unchanged or the new
        variable is unset. The replaced client is retired rather than closed
        here: ``rekey`` is synchronous, and disposal happens once in
        :meth:`close`.
        """
        fresh = resolve_api_key_env_var(ProviderType.OPENAI)
        if not fresh or fresh == self._key_env:
            return
        api_key = os.environ.get(fresh)
        if not api_key:
            logger.warning(
                "Audio backend re-key skipped: %s is not set in the environment.",
                fresh,
            )
            return

        self._retired_clients.append(self._client)
        self._key_env = fresh
        self._client = self._build_client(api_key)
        logger.info("Re-keyed OpenAI audio backend to %s", fresh)

    async def close(self) -> None:
        """Close the active client and every client retired by ``rekey``."""
        for client in [*self._retired_clients, self._client]:
            try:
                await client.close()
            except Exception as exc:
                logger.debug("Error closing OpenAI audio client: %s", exc)
        self._retired_clients.clear()


__all__ = ["DEFAULT_OPENAI_AUDIO_MODEL", "OpenAIAudioBackend"]
