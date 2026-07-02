"""Unit tests for modules/llm/providers/openai_provider.py.

Tests OpenAI provider initialization, capabilities, and parameter forwarding.
Includes CT-3 regression tests for text.verbosity forwarding.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest


class TestOpenAIProviderCapabilities:
    """Tests for _get_model_capabilities helper."""

    @pytest.mark.unit
    def test_gpt5_is_reasoning_model(self) -> None:
        """GPT-5 family is classified as a reasoning model."""
        from modules.config.capabilities import detect_capabilities

        caps = detect_capabilities("gpt-5-mini")
        assert caps.is_reasoning_model is True
        assert caps.supports_reasoning_effort is True

    @pytest.mark.unit
    def test_gpt5_1_is_reasoning_model(self) -> None:
        """GPT-5.1 family is classified as a reasoning model."""
        from modules.config.capabilities import detect_capabilities

        caps = detect_capabilities("gpt-5.1")
        assert caps.is_reasoning_model is True
        assert caps.supports_reasoning_effort is True

    @pytest.mark.unit
    def test_gpt4o_is_not_reasoning_model(self) -> None:
        """GPT-4o is NOT classified as a reasoning model."""
        from modules.config.capabilities import detect_capabilities

        caps = detect_capabilities("gpt-4o")
        assert caps.is_reasoning_model is False
        assert caps.supports_reasoning_effort is False

    @pytest.mark.unit
    def test_o3_is_reasoning_model(self) -> None:
        """o3 is classified as a reasoning model."""
        from modules.config.capabilities import detect_capabilities

        caps = detect_capabilities("o3")
        assert caps.is_reasoning_model is True
        assert caps.supports_reasoning_effort is True

    @pytest.mark.unit
    def test_gpt4o_supports_temperature(self) -> None:
        """GPT-4o supports sampler controls."""
        from modules.config.capabilities import detect_capabilities

        caps = detect_capabilities("gpt-4o")
        assert caps.supports_sampler_controls is True

    @pytest.mark.unit
    def test_gpt5_does_not_support_temperature(self) -> None:
        """GPT-5 reasoning models do not support temperature."""
        from modules.config.capabilities import detect_capabilities

        caps = detect_capabilities("gpt-5")
        assert caps.supports_sampler_controls is False

    @pytest.mark.unit
    def test_gpt4_1_family_capabilities(self) -> None:
        """GPT-4.1 family has correct capability flags."""
        from modules.config.capabilities import detect_capabilities

        for model in ("gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"):
            caps = detect_capabilities(model)
            assert caps.is_reasoning_model is False
            assert caps.supports_image_input is True
            assert caps.supports_structured_outputs is True


class TestOpenAIProviderInit:
    """Tests for OpenAIProvider.__init__ parameter handling."""

    @pytest.mark.unit
    def test_service_tier_stored(self) -> None:
        """service_tier is stored on the provider instance."""
        from modules.llm.providers.openai_provider import OpenAIProvider

        with (
            patch("modules.llm.providers.openai_provider.ChatOpenAI"),
            patch(
                "modules.llm.providers.openai_provider.load_max_retries", return_value=3
            ),
        ):
            provider = OpenAIProvider(
                api_key="sk-test",
                model="gpt-4o",
                service_tier="auto",
            )

        assert provider.service_tier == "auto"

    @pytest.mark.unit
    def test_reasoning_config_stored(self) -> None:
        """reasoning_config is stored on the provider instance."""
        from modules.llm.providers.openai_provider import OpenAIProvider

        with (
            patch("modules.llm.providers.openai_provider.ChatOpenAI"),
            patch(
                "modules.llm.providers.openai_provider.load_max_retries", return_value=3
            ),
        ):
            provider = OpenAIProvider(
                api_key="sk-test",
                model="gpt-5-mini",
                reasoning_config={"effort": "high"},
            )

        assert provider.reasoning_config == {"effort": "high"}

    @pytest.mark.unit
    def test_reasoning_dict_forwarded_to_chat_openai(self) -> None:
        """reasoning dict is passed to ChatOpenAI for reasoning models."""
        from modules.llm.providers.openai_provider import OpenAIProvider

        captured: dict[str, Any] = {}

        with (
            patch(
                "modules.llm.providers.openai_provider.ChatOpenAI",
                side_effect=lambda **kw: captured.update(kw) or MagicMock(),
            ),
            patch(
                "modules.llm.providers.openai_provider.load_max_retries", return_value=3
            ),
        ):
            OpenAIProvider(
                api_key="sk-test",
                model="gpt-5-mini",
                reasoning_config={"effort": "high"},
            )

        assert captured.get("reasoning") == {"effort": "high"}
        assert "reasoning_effort" not in captured

    @pytest.mark.unit
    def test_service_tier_forwarded_to_chat_openai(self) -> None:
        """service_tier is passed to ChatOpenAI when set."""
        from modules.llm.providers.openai_provider import OpenAIProvider

        captured: dict[str, Any] = {}

        with (
            patch(
                "modules.llm.providers.openai_provider.ChatOpenAI",
                side_effect=lambda **kw: captured.update(kw) or MagicMock(),
            ),
            patch(
                "modules.llm.providers.openai_provider.load_max_retries", return_value=3
            ),
        ):
            OpenAIProvider(
                api_key="sk-test",
                model="gpt-4o",
                service_tier="flex",
            )

        assert captured.get("service_tier") == "flex"

    @pytest.mark.unit
    def test_provider_name_is_openai(self) -> None:
        """provider_name property returns 'openai'."""
        from modules.llm.providers.openai_provider import OpenAIProvider

        with (
            patch("modules.llm.providers.openai_provider.ChatOpenAI"),
            patch(
                "modules.llm.providers.openai_provider.load_max_retries", return_value=3
            ),
        ):
            provider = OpenAIProvider(api_key="sk-test", model="gpt-4o")

        assert provider.provider_name == "openai"

    @pytest.mark.unit
    def test_max_completion_tokens_used_for_reasoning_model(self) -> None:
        """Reasoning models use max_completion_tokens instead of max_tokens."""
        from modules.llm.providers.openai_provider import OpenAIProvider

        captured: dict[str, Any] = {}

        with (
            patch(
                "modules.llm.providers.openai_provider.ChatOpenAI",
                side_effect=lambda **kw: captured.update(kw) or MagicMock(),
            ),
            patch(
                "modules.llm.providers.openai_provider.load_max_retries", return_value=3
            ),
        ):
            OpenAIProvider(
                api_key="sk-test",
                model="gpt-5-mini",
                max_tokens=8192,
            )

        assert captured.get("max_completion_tokens") == 8192
        assert "max_tokens" not in captured

    @pytest.mark.unit
    def test_max_tokens_used_for_non_reasoning_model(self) -> None:
        """Non-reasoning models use max_tokens."""
        from modules.llm.providers.openai_provider import OpenAIProvider

        captured: dict[str, Any] = {}

        with (
            patch(
                "modules.llm.providers.openai_provider.ChatOpenAI",
                side_effect=lambda **kw: captured.update(kw) or MagicMock(),
            ),
            patch(
                "modules.llm.providers.openai_provider.load_max_retries", return_value=3
            ),
        ):
            OpenAIProvider(
                api_key="sk-test",
                model="gpt-4o",
                max_tokens=4096,
            )

        assert captured.get("max_tokens") == 4096
        assert "max_completion_tokens" not in captured

    @pytest.mark.unit
    def test_reasoning_model_no_sampler_in_kwargs(self) -> None:
        """Sampler params must be None in ChatOpenAI kwargs for reasoning models.

        Regression test: presence_penalty / frequency_penalty leaked into the
        Responses API payload causing ``AsyncResponses.parse() got an
        unexpected keyword argument 'presence_penalty'``.
        """
        from modules.llm.providers.openai_provider import OpenAIProvider

        captured: dict[str, Any] = {}

        with (
            patch(
                "modules.llm.providers.openai_provider.ChatOpenAI",
                side_effect=lambda **kw: captured.update(kw) or MagicMock(),
            ),
            patch(
                "modules.llm.providers.openai_provider.load_max_retries", return_value=3
            ),
        ):
            OpenAIProvider(
                api_key="sk-test",
                model="gpt-5-mini",
                presence_penalty=0.01,
                frequency_penalty=0.01,
                top_p=0.9,
                temperature=0.5,
            )

        for param in ("presence_penalty", "frequency_penalty", "top_p", "temperature"):
            val = captured.get(param)
            assert val is None, (
                f"{param} should be None for reasoning models but was {val!r}"
            )

    @pytest.mark.unit
    def test_use_responses_api_always_set(self) -> None:
        """use_responses_api=True is always passed to ChatOpenAI."""
        from modules.llm.providers.openai_provider import OpenAIProvider

        captured: dict[str, Any] = {}

        with (
            patch(
                "modules.llm.providers.openai_provider.ChatOpenAI",
                side_effect=lambda **kw: captured.update(kw) or MagicMock(),
            ),
            patch(
                "modules.llm.providers.openai_provider.load_max_retries", return_value=3
            ),
        ):
            OpenAIProvider(api_key="sk-test", model="gpt-4o")

        assert captured.get("use_responses_api") is True


# =============================================================================
# CT-3: text.verbosity forwarding through OpenAIProvider (Responses API)
# =============================================================================


class TestOpenAIProviderTextVerbosity:
    """CT-3 regression tests: text_config stored and verbosity forwarded.

    Verbosity is forwarded via the LangChain `verbosity` parameter
    (Responses API native) rather than `model_kwargs["text"]`.
    """

    @pytest.mark.unit
    def test_text_config_stored_on_provider(self) -> None:
        """OpenAIProvider stores text_config attribute when provided."""
        from modules.llm.providers.openai_provider import OpenAIProvider

        with (
            patch("modules.llm.providers.openai_provider.ChatOpenAI"),
            patch(
                "modules.llm.providers.openai_provider.load_max_retries", return_value=3
            ),
        ):
            provider = OpenAIProvider(
                api_key="sk-test",
                model="gpt-5-mini",
                text_config={"verbosity": "concise"},
            )

        assert provider.text_config == {"verbosity": "concise"}

    @pytest.mark.unit
    def test_text_config_defaults_to_none(self) -> None:
        """text_config defaults to None when not provided."""
        from modules.llm.providers.openai_provider import OpenAIProvider

        with (
            patch("modules.llm.providers.openai_provider.ChatOpenAI"),
            patch(
                "modules.llm.providers.openai_provider.load_max_retries", return_value=3
            ),
        ):
            provider = OpenAIProvider(api_key="sk-test", model="gpt-4o")

        assert provider.text_config is None

    @pytest.mark.unit
    def test_verbosity_passed_as_parameter_for_reasoning_model(self) -> None:
        """For GPT-5 reasoning models, text verbosity is passed as direct parameter."""
        from modules.llm.providers.openai_provider import OpenAIProvider

        captured: dict[str, Any] = {}

        with (
            patch(
                "modules.llm.providers.openai_provider.ChatOpenAI",
                side_effect=lambda **kw: captured.update(kw) or MagicMock(),
            ),
            patch(
                "modules.llm.providers.openai_provider.load_max_retries", return_value=3
            ),
        ):
            OpenAIProvider(
                api_key="sk-test",
                model="gpt-5-mini",
                text_config={"verbosity": "verbose"},
            )

        assert captured.get("verbosity") == "verbose"

    @pytest.mark.unit
    def test_verbosity_not_set_for_non_reasoning_model(self) -> None:
        """For non-reasoning models (gpt-4o), verbosity is NOT passed."""
        from modules.llm.providers.openai_provider import OpenAIProvider

        captured: dict[str, Any] = {}

        with (
            patch(
                "modules.llm.providers.openai_provider.ChatOpenAI",
                side_effect=lambda **kw: captured.update(kw) or MagicMock(),
            ),
            patch(
                "modules.llm.providers.openai_provider.load_max_retries", return_value=3
            ),
        ):
            OpenAIProvider(
                api_key="sk-test",
                model="gpt-4o",
                text_config={"verbosity": "concise"},
            )

        assert "verbosity" not in captured

    @pytest.mark.unit
    def test_verbosity_skipped_when_text_config_empty(self) -> None:
        """Empty text_config dict does not add verbosity parameter."""
        from modules.llm.providers.openai_provider import OpenAIProvider

        captured: dict[str, Any] = {}

        with (
            patch(
                "modules.llm.providers.openai_provider.ChatOpenAI",
                side_effect=lambda **kw: captured.update(kw) or MagicMock(),
            ),
            patch(
                "modules.llm.providers.openai_provider.load_max_retries", return_value=3
            ),
        ):
            OpenAIProvider(
                api_key="sk-test",
                model="gpt-5-mini",
                text_config={},
            )

        assert "verbosity" not in captured

    @pytest.mark.unit
    def test_all_verbosity_levels_forwarded(self) -> None:
        """All verbosity levels (concise/medium/verbose) are forwarded correctly."""
        from modules.llm.providers.openai_provider import OpenAIProvider

        for level in ("concise", "medium", "verbose"):
            captured: dict[str, Any] = {}

            with (
                patch(
                    "modules.llm.providers.openai_provider.ChatOpenAI",
                    side_effect=lambda _c=captured, **kw: _c.update(kw) or MagicMock(),
                ),
                patch(
                    "modules.llm.providers.openai_provider.load_max_retries",
                    return_value=3,
                ),
            ):
                OpenAIProvider(
                    api_key="sk-test",
                    model="gpt-5-mini",
                    text_config={"verbosity": level},
                )

            assert captured.get("verbosity") == level


class TestOpenAIProviderInvokeLLM:
    """Tests for OpenAIProvider._invoke_llm() parameter forwarding."""

    @pytest.mark.unit
    def test_invoke_llm_forwards_expect_image_tokens(self) -> None:
        """_invoke_llm passes expect_image_tokens=True to _ainvoke_with_retry."""
        import asyncio

        from modules.llm.providers.openai_provider import OpenAIProvider

        with (
            patch("modules.llm.providers.openai_provider.ChatOpenAI"),
            patch(
                "modules.llm.providers.openai_provider.load_max_retries", return_value=3
            ),
        ):
            provider = OpenAIProvider(api_key="sk-test", model="gpt-4o")

        captured_kwargs = {}

        async def _capture_retry(llm, messages, **kwargs):
            captured_kwargs.update(kwargs)
            # Return a minimal AIMessage-like response
            msg = MagicMock()
            msg.content = "transcribed"
            msg.response_metadata = {}
            msg.usage_metadata = {
                "input_tokens": 1000,
                "output_tokens": 100,
                "total_tokens": 1100,
            }
            return msg

        provider._ainvoke_with_retry = _capture_retry

        with patch("modules.infra.token_budget.get_token_tracker"):
            asyncio.run(provider._invoke_llm(MagicMock(), ["msg"]))

        assert captured_kwargs.get("expect_image_tokens") is True


class TestOpenAIProviderContextImage:
    """Tests for context image handling in transcribe_image_from_base64."""

    def _make_provider(self) -> Any:
        """Create an OpenAIProvider with mocked dependencies."""
        from modules.llm.providers.openai_provider import OpenAIProvider

        with (
            patch("modules.llm.providers.openai_provider.ChatOpenAI"),
            patch(
                "modules.llm.providers.openai_provider.load_max_retries",
                return_value=3,
            ),
        ):
            return OpenAIProvider(api_key="sk-test", model="gpt-4o")

    @pytest.mark.unit
    async def test_context_image_blocks_present(self) -> None:
        """HumanMessage contains context image blocks when provided."""
        provider = self._make_provider()

        captured_messages: list[Any] = []

        async def _capture(llm: Any, messages: Any, **kw: Any) -> Any:
            captured_messages.extend(messages)
            msg = MagicMock()
            msg.content = "transcribed"
            msg.response_metadata = {}
            msg.usage_metadata = {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            }
            return msg

        provider._ainvoke_with_retry = _capture

        with patch("modules.infra.token_budget.get_token_tracker"):
            await provider.transcribe_image_from_base64(
                image_base64="AAAA",
                mime_type="image/png",
                system_prompt="Transcribe.",
                user_instruction="Please transcribe.",
                context_image_base64="BBBB",
                context_image_mime_type="image/jpeg",
                context_image_detail="high",
            )

        assert len(captured_messages) == 2
        human_msg = captured_messages[1]
        content = human_msg.content
        # Expect 4 blocks: "Context image:" text, ctx image, instruction, page image
        assert len(content) == 4
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Context image:"
        assert content[1]["type"] == "image_url"
        assert "data:image/jpeg;base64,BBBB" in content[1]["image_url"]["url"]

    @pytest.mark.unit
    async def test_no_context_image_two_blocks(self) -> None:
        """HumanMessage has only instruction + page image when no context."""
        provider = self._make_provider()

        captured_messages: list[Any] = []

        async def _capture(llm: Any, messages: Any, **kw: Any) -> Any:
            captured_messages.extend(messages)
            msg = MagicMock()
            msg.content = "transcribed"
            msg.response_metadata = {}
            msg.usage_metadata = {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            }
            return msg

        provider._ainvoke_with_retry = _capture

        with patch("modules.infra.token_budget.get_token_tracker"):
            await provider.transcribe_image_from_base64(
                image_base64="AAAA",
                mime_type="image/png",
                system_prompt="Transcribe.",
                user_instruction="Please transcribe.",
            )

        human_msg = captured_messages[1]
        content = human_msg.content
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"

    @pytest.mark.unit
    async def test_context_image_detail_forwarded(self) -> None:
        """Context image uses the detail level passed in."""
        provider = self._make_provider()

        captured_messages: list[Any] = []

        async def _capture(llm: Any, messages: Any, **kw: Any) -> Any:
            captured_messages.extend(messages)
            msg = MagicMock()
            msg.content = "transcribed"
            msg.response_metadata = {}
            msg.usage_metadata = {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            }
            return msg

        provider._ainvoke_with_retry = _capture

        with patch("modules.infra.token_budget.get_token_tracker"):
            await provider.transcribe_image_from_base64(
                image_base64="AAAA",
                mime_type="image/png",
                system_prompt="Transcribe.",
                context_image_base64="BBBB",
                context_image_mime_type="image/png",
                context_image_detail="high",
            )

        human_msg = captured_messages[1]
        ctx_block = human_msg.content[1]
        assert ctx_block["image_url"].get("detail") == "high"
