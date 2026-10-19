"""PageTranscriber: messages per provider, response formats, results, retries."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage

import chronotranscriber.llm.caller as caller_module
from chronotranscriber.common.testing import FakeRequest
from chronotranscriber.common.usage import Usage
from chronotranscriber.llm.caller import CallEnv
from chronotranscriber.llm.prompts import PLAIN_PROMPT_PATH, SCHEMAS_DIR
from chronotranscriber.llm.transcription import (
    ANTHROPIC_SCHEMA_DESCRIPTION,
    ANTHROPIC_SCHEMA_TITLE,
    PageTranscriber,
    transcribe_image_with_llm,
    usage_record,
)
from chronotranscriber.settings import PromptCaching, RetrySettings
from chronotranscriber.spec import ModelSpec
from tests.llm import helpers
from tests.llm.conftest import Sleeps
from tests.llm.helpers import RecordingExt, StatusError, message, model_spec

PAGE = "QUJD"
SCHEMA_PATH = SCHEMAS_DIR / "markdown_transcription_schema.json"
ANSWER: dict[str, Any] = {
    "transcription": "Bread prices rose.",
    "no_transcribable_text": False,
    "transcription_not_possible": False,
}
SPECS: dict[str, ModelSpec] = {
    "openai": model_spec(),
    "anthropic": model_spec("claude-sonnet-4-5", "anthropic"),
    "google": model_spec("gemini-3-flash", "google"),
    "openrouter": model_spec("anthropic/claude-sonnet-4.5", "openrouter"),
    "custom": helpers.custom_spec(),
}


def usage_dict(
    input_tokens: int, output_tokens: int, total_tokens: int, *, cached: int = 0
) -> dict[str, int]:
    """Return the stored usage shape of these counts."""
    return Usage(input_tokens, output_tokens, total_tokens, cached).as_dict()


def _transcriber(
    monkeypatch: pytest.MonkeyPatch,
    answers: list[Any],
    spec: ModelSpec | None = None,
    *,
    env: CallEnv | None = None,
    **options: Any,
) -> tuple[PageTranscriber, helpers.Script]:
    script = helpers.install(monkeypatch, answers)
    options.setdefault("schema_path", SCHEMA_PATH)
    transcriber = PageTranscriber(spec or model_spec(), env or helpers.env(), **options)
    return transcriber, script


def _transcribe(transcriber: PageTranscriber, **kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("label", "page_0001.jpg")
    return asyncio.run(
        transcriber.transcribe_image_from_base64(PAGE, "image/jpeg", **kwargs)
    )


def _human(request: FakeRequest) -> list[Any]:
    content = request.messages[-1].content
    assert isinstance(content, list)
    return content


def _system(request: FakeRequest) -> Any:
    return request.messages[0].content


def _context_image(tmp_path: Path) -> Path:
    path = tmp_path / "title.png"
    path.write_bytes(b"\x89PNG context")
    return path


def test_the_openai_request_carries_detail_and_the_enforced_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcriber, script = _transcriber(monkeypatch, [ANSWER])

    result = _transcribe(transcriber)

    (request,) = script.requests
    instruction, block = _human(request)
    assert instruction == {"type": "text", "text": "The image:"}
    assert block == {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{PAGE}", "detail": "original"},
    }
    assert request.kwargs["text"]["format"]["name"] == "markdown_transcription_schema"
    assert "The JSON schema:" in _system(request)
    assert result["parsed"] == ANSWER


@pytest.mark.parametrize(
    ("model", "requested", "sent"),
    [
        ("gpt-5.6-luna", None, "original"),
        ("gpt-5.6-luna", "low", "low"),
        ("o1-pro", "original", "high"),
        ("gpt-4o", "auto", "auto"),
        ("gpt-4o", "nonsense", "high"),
    ],
)
def test_openai_keeps_only_the_detail_tiers_the_model_takes(
    monkeypatch: pytest.MonkeyPatch, model: str, requested: str | None, sent: str
) -> None:
    transcriber, script = _transcriber(
        monkeypatch, [ANSWER], model_spec(model), image_detail=requested
    )

    _transcribe(transcriber)

    assert _human(script.requests[0])[-1]["image_url"]["detail"] == sent


def test_a_per_call_detail_replaces_the_page_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcriber, script = _transcriber(monkeypatch, [ANSWER])

    _transcribe(transcriber, image_detail="low", media_resolution="high")

    assert _human(script.requests[0])[-1]["image_url"]["detail"] == "low"


@pytest.mark.parametrize(
    ("provider", "page_block"),
    [
        (
            "anthropic",
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": PAGE},
            },
        ),
        (
            "google",
            {"type": "image", "base64": PAGE, "mime_type": "image/jpeg"},
        ),
        (
            "openrouter",
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{PAGE}"},
            },
        ),
        (
            "custom",
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{PAGE}"},
            },
        ),
    ],
)
def test_each_provider_gets_its_image_block(
    monkeypatch: pytest.MonkeyPatch, provider: str, page_block: dict[str, Any]
) -> None:
    transcriber, script = _transcriber(monkeypatch, [ANSWER], SPECS[provider])

    _transcribe(transcriber)

    assert _human(script.requests[0])[-1] == page_block


def test_a_gemini_page_carries_the_media_resolution_of_its_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcriber, script = _transcriber(monkeypatch, [ANSWER], SPECS["google"])
    transcriber.page_detail = "high"

    _transcribe(transcriber)

    assert _human(script.requests[0])[-1] == {
        "type": "image",
        "base64": PAGE,
        "mime_type": "image/jpeg",
        "media_resolution": "MEDIA_RESOLUTION_HIGH",
    }


@pytest.mark.parametrize("provider", ["openai", "anthropic", "google", "custom"])
def test_the_context_image_precedes_the_page(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, provider: str
) -> None:
    transcriber, script = _transcriber(monkeypatch, [ANSWER], SPECS[provider])
    transcriber.update_context_image(_context_image(tmp_path))

    _transcribe(transcriber)

    blocks = _human(script.requests[0])
    assert [block["type"] for block in blocks] == [
        "text",
        blocks[1]["type"],
        "text",
        blocks[3]["type"],
    ]
    assert blocks[0]["text"] == "Context image:"
    assert blocks[2]["text"] == "The image:"
    assert "image/png" in json.dumps(blocks[1])
    if provider == "openai":
        assert blocks[1]["image_url"]["detail"] == "original"


def test_the_context_image_detail_follows_the_page_tiers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    transcriber, script = _transcriber(
        monkeypatch, [ANSWER], model_spec("gpt-4o"), image_detail="original"
    )
    transcriber.update_context_image(_context_image(tmp_path))

    _transcribe(transcriber)

    assert _human(script.requests[0])[1]["image_url"]["detail"] == "high"


def test_a_missing_or_cleared_context_image_sends_two_blocks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    transcriber, script = _transcriber(monkeypatch, [ANSWER])
    transcriber.update_context_image(_context_image(tmp_path))
    transcriber.update_context_image(None)
    transcriber.update_context_image(tmp_path / "missing.png")

    _transcribe(transcriber)

    assert len(_human(script.requests[0])) == 2


def test_openrouter_sends_only_low_or_high_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = model_spec("openai/gpt-5.4", "openrouter")
    transcriber, script = _transcriber(
        monkeypatch, [ANSWER, ANSWER], spec, image_detail="original"
    )

    _transcribe(transcriber)
    _transcribe(transcriber, image_detail="low")

    first, second = script.requests
    assert _human(first)[-1]["image_url"]["detail"] == "high"
    assert _human(second)[-1]["image_url"]["detail"] == "low"


def test_the_context_text_fills_and_empties_its_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcriber, script = _transcriber(monkeypatch, [ANSWER])

    transcriber.update_context("Register of the city market of Lyon.")
    _transcribe(transcriber)
    transcriber.update_context(None)
    _transcribe(transcriber)

    with_context, without = (str(_system(r)) for r in script.requests)
    assert "Register of the city market of Lyon." in with_context
    assert "{{ADDITIONAL_CONTEXT}}" not in with_context + without
    assert "Additional context" not in without


@pytest.mark.parametrize(
    ("fmt", "schema_prompt", "enforced"),
    [("schema", True, True), ("json", True, False), ("text", False, False)],
)
def test_the_response_format_picks_prompt_and_enforcement(
    monkeypatch: pytest.MonkeyPatch, fmt: str, schema_prompt: bool, enforced: bool
) -> None:
    answer = "Plain page text." if fmt == "text" else ANSWER
    transcriber, script = _transcriber(
        monkeypatch, [answer], model_spec(response_format=fmt)
    )

    result = _transcribe(transcriber)

    (request,) = script.requests
    system = str(_system(request))
    assert ("The JSON schema:" in system) is schema_prompt
    if fmt == "text":
        first_line = PLAIN_PROMPT_PATH.read_text(encoding="utf-8").splitlines()[0]
        assert system.startswith(first_line)
    assert ("text" in request.kwargs) is enforced
    expected = "Plain page text." if fmt == "text" else json.dumps(ANSWER)
    assert result["output_text"] == expected
    assert "error" not in result


def test_a_prompt_file_replaces_the_default_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prompt = tmp_path / "prompt.txt"
    prompt.write_text(
        "Transcribe this register.\n{{ADDITIONAL_CONTEXT}}", encoding="utf-8"
    )
    transcriber, script = _transcriber(monkeypatch, [ANSWER], prompt_path=prompt)

    _transcribe(transcriber)

    system = str(_system(script.requests[0]))
    assert system.startswith("Transcribe this register.")
    assert "The JSON schema:" in system


def test_missing_prompt_or_schema_files_fail_at_construction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(FileNotFoundError, match="System prompt missing"):
        _transcriber(monkeypatch, [ANSWER], prompt_path=tmp_path / "none.txt")
    with pytest.raises(FileNotFoundError, match="Schema file missing"):
        _transcriber(monkeypatch, [ANSWER], schema_path=tmp_path / "none.json")


def test_the_anthropic_schema_is_titled_and_keeps_its_nullable_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcriber, script = _transcriber(monkeypatch, [ANSWER], SPECS["anthropic"])
    original = json.dumps(transcriber.schema, sort_keys=True)

    _transcribe(transcriber)

    structured = script.requests[0].structured
    assert structured is not None and structured.method == "json_schema"
    schema = structured.schema
    assert schema["title"] == ANTHROPIC_SCHEMA_TITLE
    assert schema["description"] == ANTHROPIC_SCHEMA_DESCRIPTION
    assert schema["properties"]["transcription"]["anyOf"] == [
        {"type": "string"},
        {"type": "null"},
    ]
    assert json.dumps(transcriber.schema, sort_keys=True) == original


def test_an_anthropic_schema_over_the_union_limit_is_prompted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loaded = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    properties = loaded["schema"]["properties"]
    for index in range(17):
        properties[f"extra_{index}"] = {"type": ["string", "null"]}
    schema_path = tmp_path / "wide.json"
    schema_path.write_text(json.dumps(loaded), encoding="utf-8")
    transcriber, script = _transcriber(
        monkeypatch, [ANSWER], SPECS["anthropic"], schema_path=schema_path
    )

    _transcribe(transcriber)

    assert transcriber.response_format == "json"
    assert script.requests[0].structured is None
    assert transcriber.request().required_keys == tuple(loaded["schema"]["required"])


def test_a_bare_schema_file_is_sent_under_the_transcription_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bare = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))["schema"]
    schema_path = tmp_path / "bare.json"
    schema_path.write_text(json.dumps(bare), encoding="utf-8")
    transcriber, script = _transcriber(monkeypatch, [ANSWER], schema_path=schema_path)

    _transcribe(transcriber)

    text_format = script.requests[0].kwargs["text"]["format"]
    assert (text_format["name"], text_format["strict"]) == ("transcription", True)
    assert text_format["schema"] == bare


@pytest.mark.parametrize("provider", ["google", "openrouter", "custom"])
def test_the_selected_schema_is_enforced_on_every_provider(
    monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    schema_path = SCHEMAS_DIR / "plain_text_transcription_schema.json"
    transcriber, script = _transcriber(
        monkeypatch, [ANSWER], SPECS[provider], schema_path=schema_path
    )

    result = _transcribe(transcriber)

    request = script.requests[0]
    selected = json.loads(schema_path.read_text(encoding="utf-8"))
    if provider == "google":
        assert request.kwargs["response_schema"] == selected["schema"]
    elif provider == "custom":
        sent = request.kwargs["response_format"]["json_schema"]
        assert sent["name"] == selected["name"]
        assert sent["schema"] == selected["schema"]
    else:
        assert request.structured is not None
        assert request.structured.schema["title"] == selected["name"]
    assert result["parsed"] == ANSWER


def _caching_env(**values: Any) -> CallEnv:
    return helpers.env(prompt_caching=PromptCaching(enabled=True, **values))


@pytest.mark.parametrize(
    ("ttl", "control"),
    [("5m", {"type": "ephemeral"}), ("1h", {"type": "ephemeral", "ttl": "1h"})],
)
def test_anthropic_caches_the_system_prompt_when_caching_is_on(
    monkeypatch: pytest.MonkeyPatch, ttl: str, control: dict[str, str]
) -> None:
    env = _caching_env(anthropic_ttl=ttl)
    transcriber, script = _transcriber(
        monkeypatch, [ANSWER], SPECS["anthropic"], env=env
    )

    _transcribe(transcriber)

    (block,) = _system(script.requests[0])
    assert block["type"] == "text"
    assert block["cache_control"] == control
    assert "The JSON schema:" in block["text"]


@pytest.mark.parametrize(
    ("spec", "env", "cached"),
    [
        (SPECS["openrouter"], _caching_env(), True),
        (model_spec("openai/gpt-5.4", "openrouter"), _caching_env(), False),
        (SPECS["openrouter"], _caching_env(openrouter_cache_control=False), False),
        (SPECS["openrouter"], helpers.env(), True),
        (SPECS["anthropic"], helpers.env(), True),
        (SPECS["anthropic"], helpers.env(prompt_caching=PromptCaching(False)), False),
    ],
)
def test_cache_control_marks_only_cacheable_system_prompts(
    monkeypatch: pytest.MonkeyPatch, spec: ModelSpec, env: CallEnv, cached: bool
) -> None:
    transcriber, script = _transcriber(monkeypatch, [ANSWER], spec, env=env)

    _transcribe(transcriber)

    system = _system(script.requests[0])
    assert isinstance(system, list) is cached
    if cached:
        assert system[0]["cache_control"] == {"type": "ephemeral"}


def test_the_response_dict_and_the_role_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    env = helpers.env()
    answer = message(json.dumps(ANSWER), response_metadata={"model_name": "m"})
    transcriber, _ = _transcriber(monkeypatch, [answer], env=env)

    result = _transcribe(transcriber)

    assert result == {
        "output_text": json.dumps(ANSWER),
        "usage": usage_dict(100, 20, 120),
        "metadata": {"model_name": "m"},
        "parsed": ANSWER,
    }
    assert env.usage.get("transcription") == Usage(100, 20, 120)


@pytest.mark.parametrize(
    ("provider", "fields", "expected"),
    [
        pytest.param(
            "anthropic",
            {
                "response_metadata": {
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 20,
                        "cache_read_input_tokens": 80,
                        "cache_creation_input_tokens": 5,
                    }
                }
            },
            usage_dict(185, 20, 205, cached=80),
            id="anthropic-cache",
        ),
        pytest.param(
            "anthropic",
            {
                "response_metadata": {
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 20,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 90,
                    }
                }
            },
            usage_dict(190, 20, 210),
            id="anthropic-zero-read",
        ),
        pytest.param(
            "openai",
            {
                "response_metadata": {
                    "token_usage": {
                        "prompt_tokens": 300,
                        "completion_tokens": 50,
                        "total_tokens": 350,
                        "prompt_tokens_details": {"cached_tokens": 200},
                    }
                },
                "usage_metadata": {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                },
            },
            usage_dict(1, 1, 2),
            id="usage-metadata-first",
        ),
        pytest.param(
            "openai",
            {
                "usage_metadata": {
                    "input_tokens": 300,
                    "output_tokens": 50,
                    "total_tokens": 0,
                    "input_token_details": {"cache_read": 200},
                }
            },
            usage_dict(300, 50, 350, cached=200),
            id="usage-metadata",
        ),
        pytest.param(
            "anthropic",
            {
                "usage_metadata": {
                    "input_tokens": 300,
                    "output_tokens": 50,
                    "total_tokens": 350,
                    "input_token_details": {"cache_read": 180, "cache_creation": 0},
                }
            },
            usage_dict(300, 50, 350, cached=180),
            id="usage-metadata-zero-creation",
        ),
        pytest.param(
            "google",
            {
                "response_metadata": {
                    "usage_metadata": {
                        "prompt_token_count": 7,
                        "candidates_token_count": 3,
                        "total_token_count": 10,
                    }
                }
            },
            usage_dict(7, 3, 10),
            id="google",
        ),
        pytest.param(
            "openai",
            {},
            usage_dict(0, 0, 0),
            id="none",
        ),
    ],
)
def test_usage_record(
    provider: str, fields: dict[str, Any], expected: dict[str, int]
) -> None:
    assert usage_record(AIMessage(content="x", **fields), provider) == expected


@pytest.mark.parametrize(
    ("text", "parsed"),
    [
        (json.dumps(ANSWER), ANSWER),
        (f"```json\n{json.dumps(ANSWER)}\n```", ANSWER),
        (f"Here is the page:\n\n{json.dumps(ANSWER)}", ANSWER),
    ],
    ids=["json", "fenced", "preamble"],
)
def test_the_parsed_answer(
    monkeypatch: pytest.MonkeyPatch, text: str, parsed: dict[str, Any] | None
) -> None:
    transcriber, _ = _transcriber(
        monkeypatch, [text], model_spec(response_format="json")
    )

    result = _transcribe(transcriber)

    assert result["output_text"] == text
    assert result.get("parsed") == parsed


@pytest.mark.parametrize(
    "text",
    ['{"transcription": "cut', "Plain page text.", "[1, 2]"],
    ids=["malformed", "text", "array"],
)
def test_a_json_answer_without_an_object_fails_the_page(
    monkeypatch: pytest.MonkeyPatch, text: str
) -> None:
    transcriber, _ = _transcriber(
        monkeypatch, [text], model_spec(response_format="json")
    )

    result = _transcribe(transcriber)

    assert result["output_text"] == ""
    assert result["error"].startswith("Answer holds no JSON object")


def test_an_unsupported_image_file_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    transcriber, script = _transcriber(monkeypatch, [ANSWER])
    path = tmp_path / "scan.xyz"
    path.write_bytes(b"data")

    with pytest.raises(ValueError, match="Unsupported image format"):
        asyncio.run(transcriber.transcribe_image(path))
    assert script.requests == []


def test_a_model_without_vision_fails_without_a_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcriber, script = _transcriber(monkeypatch, [ANSWER], model_spec("o3-mini"))

    result = _transcribe(transcriber)

    assert script.requests == []
    assert result["output_text"] == ""
    assert "does not support vision" in result["error"]


def test_an_api_error_after_the_ladder_is_an_error_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcriber, _ = _transcriber(monkeypatch, [StatusError(401)])

    result = _transcribe(transcriber)

    assert result == {
        "output_text": "",
        "usage": usage_dict(0, 0, 0),
        "error": "status 401",
    }


@pytest.mark.parametrize(
    "answer",
    [
        message(
            "{}",
            response_metadata={
                "status": "incomplete",
                "incomplete_details": {"reason": "content_filter"},
            },
        ),
        message("{}", additional_kwargs={"refusal": "I cannot help."}),
        message(
            '{"transcription": "cut', response_metadata={"stop_reason": "max_tokens"}
        ),
        message(
            '{"transcription": "cut', response_metadata={"finish_reason": "MAX_TOKENS"}
        ),
        message(
            '{"transcription": "cut',
            response_metadata={
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
            },
        ),
    ],
    ids=["content-filter", "refusal", "anthropic-max", "google-max", "openai-max"],
)
def test_a_stopped_schema_answer_fails_the_page_after_one_call(
    monkeypatch: pytest.MonkeyPatch, answer: AIMessage
) -> None:
    retry = RetrySettings(validation_attempts=3)
    transcriber, script = _transcriber(
        monkeypatch, [answer, ANSWER], env=helpers.env(retry)
    )

    result = _transcribe(transcriber)

    assert len(script.requests) == 1
    assert result["output_text"] == ""
    assert result["error"]
    assert result["usage"]["total_tokens"] == 120


@pytest.mark.parametrize("response_format", ["json", "text"])
def test_a_truncated_answer_fails_in_every_format_without_a_retry(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps, response_format: str
) -> None:
    truncated = message(
        '{"transcription": "cut', response_metadata={"stop_reason": "max_tokens"}
    )
    transcriber, script = _transcriber(
        monkeypatch,
        [truncated, ANSWER],
        model_spec(response_format=response_format),
        env=helpers.env(RetrySettings(validation_attempts=2)),
    )

    result = _transcribe(transcriber)

    assert len(script.requests) == 1
    assert result["output_text"] == ""
    assert result["error_type"] == "truncated"


def test_invalid_json_is_asked_again_within_the_validation_attempts(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    env = helpers.env(RetrySettings(validation_attempts=2))
    missing = {"transcription": ANSWER["transcription"]}
    transcriber, script = _transcriber(
        monkeypatch, [missing, ANSWER], model_spec(response_format="json"), env=env
    )

    result = _transcribe(transcriber)

    assert len(script.requests) == 2
    assert result["parsed"] == ANSWER
    assert env.usage.get("transcription") == Usage(200, 40, 240)


def test_an_exhausted_json_answer_without_an_object_fails_the_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcriber, script = _transcriber(
        monkeypatch, ["Here is the page: no JSON"], model_spec(response_format="json")
    )

    result = _transcribe(transcriber)

    assert len(script.requests) == 1
    assert result["output_text"] == ""
    assert "error" in result


def test_an_exhausted_json_object_missing_keys_is_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcriber, _ = _transcriber(
        monkeypatch, ['{"other": 1}'], model_spec(response_format="json")
    )

    result = _transcribe(transcriber)

    assert result["output_text"] == '{"other": 1}'
    assert "error" not in result


def test_an_unparsed_schema_answer_is_retried_then_fails_the_page(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    env = helpers.env(RetrySettings(validation_attempts=2))
    transcriber, script = _transcriber(
        monkeypatch, ["not json"], SPECS["anthropic"], env=env
    )

    result = _transcribe(transcriber)

    assert len(script.requests) == 2
    assert result["output_text"] == ""
    assert result["error"].startswith("Structured output validation failed")


def test_an_unparsed_schema_answer_is_retried_to_success(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    env = helpers.env(RetrySettings(validation_attempts=2))
    transcriber, script = _transcriber(
        monkeypatch, ["not json", ANSWER], SPECS["anthropic"], env=env
    )

    result = _transcribe(transcriber)

    assert len(script.requests) == 2
    assert result["parsed"] == ANSWER


def test_an_empty_answer_is_returned_for_the_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcriber, script = _transcriber(monkeypatch, [""])

    result = _transcribe(transcriber)

    assert len(script.requests) == 1
    assert result["output_text"] == ""
    assert "error" not in result and "parsed" not in result


def _low_input(request: FakeRequest) -> AIMessage:
    usage = {"input_tokens": 40, "output_tokens": 20, "total_tokens": 60}
    return message(json.dumps(ANSWER), usage_metadata=usage)


def test_an_answer_billed_below_the_input_minimum_is_retried(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    env = helpers.env(RetrySettings(validation_attempts=2, min_input_tokens=50))
    transcriber, script = _transcriber(monkeypatch, [_low_input, ANSWER], env=env)

    result = _transcribe(transcriber)

    assert len(script.requests) == 2
    assert result["parsed"] == ANSWER
    assert env.usage.get("transcription") == Usage(140, 40, 180)


def test_the_input_minimum_is_bounded_by_the_validation_attempts(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    env = helpers.env(RetrySettings(validation_attempts=2, min_input_tokens=50))
    transcriber, script = _transcriber(monkeypatch, [_low_input], env=env)

    result = _transcribe(transcriber)

    assert len(script.requests) == 2
    assert "below minimum threshold (50)" in result["error"]


def test_a_malformed_sdk_response_is_retried_as_invalid_output(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    env = helpers.env(RetrySettings(validation_attempts=2))
    shape_error = TypeError("'NoneType' object is not iterable")
    transcriber, script = _transcriber(monkeypatch, [shape_error, ANSWER], env=env)

    result = _transcribe(transcriber)

    assert len(script.requests) == 2
    assert result["parsed"] == ANSWER


def _timeout() -> httpx.ReadTimeout:
    return httpx.ReadTimeout("read timed out")


def test_timeouts_stop_at_the_timeout_attempts(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    env = helpers.env(
        RetrySettings(max_attempts=8, timeout_attempts=2), clock=sleeps.clock
    )
    transcriber, script = _transcriber(monkeypatch, [_timeout()], env=env)

    result = _transcribe(transcriber)

    assert len(script.requests) == 2
    assert "read timed out" in result["error"]


def test_connection_errors_keep_the_full_attempt_limit(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    env = helpers.env(
        RetrySettings(max_attempts=4, timeout_attempts=1), clock=sleeps.clock
    )
    wrapped = RuntimeError("APIConnectionError")
    wrapped.__cause__ = httpx.ConnectError("connection refused")
    transcriber, script = _transcriber(monkeypatch, [wrapped], env=env)

    _transcribe(transcriber)

    assert len(script.requests) == 4


def test_a_server_retry_after_raises_the_wait(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps
) -> None:
    error = StatusError(429)
    error.response = httpx.Response(429, headers={"retry-after": "30"})  # type: ignore[attr-defined]
    env = helpers.env(clock=sleeps.clock)
    transcriber, _ = _transcriber(monkeypatch, [error, ANSWER], env=env)

    _transcribe(transcriber)

    assert sleeps[0] >= 30.0


def _hang(monkeypatch: pytest.MonkeyPatch, error: BaseException | None = None) -> None:
    async def invoke(*_args: Any) -> Any:
        if error is not None:
            raise error
        await asyncio.Event().wait()

    monkeypatch.setattr(caller_module, "default_invoke", invoke)


def test_the_page_watchdog_ends_a_page_that_runs_too_long(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    recorder = RecordingExt().install(monkeypatch)
    env = helpers.env(RetrySettings(page_timeout=0.05))
    transcriber, _ = _transcriber(monkeypatch, [ANSWER], env=env)
    _hang(monkeypatch)

    with caplog.at_level(logging.ERROR):
        result = _transcribe(transcriber, label="page_0007.jpg")

    assert result["error"] == (
        "Page 'page_0007.jpg' exceeded its 0s wall-clock ceiling across all "
        "retry attempts"
    )
    assert "Page watchdog fired for page_0007.jpg" in caplog.text
    assert all(attempt.released for attempt in recorder.attempts)


def test_a_timeout_error_of_the_call_is_not_the_watchdog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = helpers.env(RetrySettings(page_timeout=60.0, timeout_attempts=1))
    transcriber, _ = _transcriber(monkeypatch, [ANSWER], env=env)
    _hang(monkeypatch, TimeoutError("socket timed out"))

    result = _transcribe(transcriber)

    assert result["error"] == "socket timed out"


def test_a_cancellation_during_a_timeout_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    async def invoke(*_args: Any) -> Any:
        calls.append(1)
        try:
            raise _timeout()
        except httpx.ReadTimeout:
            raise asyncio.CancelledError from None

    transcriber, _ = _transcriber(monkeypatch, [ANSWER])
    monkeypatch.setattr(caller_module, "default_invoke", invoke)

    with pytest.raises(asyncio.CancelledError):
        _transcribe(transcriber)
    assert calls == [1]


def test_page_timeout_zero_turns_the_watchdog_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcriber, _ = _transcriber(
        monkeypatch, [ANSWER], env=helpers.env(RetrySettings(page_timeout=0))
    )

    assert transcriber.caller.deadline is None
    assert _transcribe(transcriber)["parsed"] == ANSWER


def test_the_auto_page_timeout_covers_the_timeout_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcriber, _ = _transcriber(
        monkeypatch, [ANSWER], env=helpers.env(RetrySettings(timeout_attempts=3))
    )

    assert transcriber.caller.deadline == 900.0 * 3 + 300.0


def test_retry_logs_name_the_page(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps, caplog: pytest.LogCaptureFixture
) -> None:
    env = helpers.env(clock=sleeps.clock)
    transcriber, _ = _transcriber(monkeypatch, [StatusError(503), ANSWER], env=env)

    with caplog.at_level(logging.WARNING):
        _transcribe(transcriber, label="page_0042.jpg")

    assert "Retrying page_0042.jpg" in caplog.text


def test_an_image_file_is_sent_under_its_name(
    monkeypatch: pytest.MonkeyPatch,
    sleeps: Sleeps,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    image = tmp_path / "scan_07.png"
    image.write_bytes(b"\x89PNG page")
    env = helpers.env(clock=sleeps.clock)
    transcriber, script = _transcriber(monkeypatch, [StatusError(503), ANSWER], env=env)

    with caplog.at_level(logging.WARNING):
        result = asyncio.run(transcribe_image_with_llm(image, transcriber))

    assert "Retrying scan_07.png" in caplog.text
    assert "data:image/png;base64," in json.dumps(_human(script.requests[-1]))
    assert result["parsed"] == ANSWER


def test_every_llm_call_path_runs_in_an_attempt(
    monkeypatch: pytest.MonkeyPatch, sleeps: Sleeps, tmp_path: Path
) -> None:
    """Page calls with a context image and Google audio calls, retries included."""
    from chronotranscriber.methods.audio_api import GoogleAudioBackend

    recorder = RecordingExt().install(monkeypatch)
    env = helpers.env(clock=sleeps.clock)
    transcriber, script = _transcriber(
        monkeypatch,
        [StatusError(503), ANSWER, StatusError(503), "Spoken words."],
        env=env,
    )
    transcriber.update_context_image(_context_image(tmp_path))
    audio = GoogleAudioBackend("gemini-3.6-flash", env=env)

    page = _transcribe(transcriber)
    chunk = asyncio.run(audio.transcribe_audio_chunk(_audio_chunk(tmp_path)))

    assert page["parsed"] == ANSWER
    assert chunk["output_text"] == "Spoken words."
    assert recorder.roles == ["transcription", "transcription", "audio", "audio"]
    assert [type(error) for error in recorder.errors] == [StatusError, StatusError]
    assert len(recorder.commits) == 2
    assert all(attempt.released for attempt in recorder.attempts)
    assert len(script.requests) == 4
    assert env.usage.get("audio") == Usage(100, 20, 120)


def _audio_chunk(tmp_path: Path) -> Any:
    from chronotranscriber.sources.audio.stream import AudioChunkPayload

    path = tmp_path / "talk.wav"
    path.write_bytes(b"RIFF fake audio")
    return AudioChunkPayload(
        index=0,
        image_name="talk.wav",
        path=path,
        mime_type="audio/wav",
        source_file=str(path),
        byte_size=path.stat().st_size,
    )
