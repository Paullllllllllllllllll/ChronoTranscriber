"""Request bodies of batch transcription per provider and response format."""

from __future__ import annotations

import base64
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from langchain_anthropic import ChatAnthropic
from PIL import Image

from chronotranscriber.batch.bodies import BodyPlan, build_plan
from chronotranscriber.common.batch import BatchRequest
from chronotranscriber.common.structured import ANTHROPIC_UNION_LIMIT, prepare_request
from chronotranscriber.llm.options import (
    constructor_kwargs,
    model_capabilities,
    response_format_for,
)
from chronotranscriber.llm.prompts import DEFAULT_SCHEMA_PATH, PLAIN_PROMPT_PATH
from chronotranscriber.llm.transcription import (
    ANTHROPIC_SCHEMA_DESCRIPTION,
    ANTHROPIC_SCHEMA_TITLE,
    anthropic_schema,
)
from chronotranscriber.settings import PromptCaching, Settings
from chronotranscriber.spec import REASONING_EFFORTS
from tests.batch.conftest import model_spec

REQUEST = BatchRequest("req-1", image_base64="UEFHRQ==", mime_type="image/jpeg")


def plan(provider: str, settings: Settings | None = None, **kwargs: Any) -> BodyPlan:
    flags = kwargs.pop("flags", {})
    return build_plan(
        model_spec(provider, kwargs.pop("model", None), **flags),
        settings or Settings(),
        schema_path=kwargs.pop("schema_path", DEFAULT_SCHEMA_PATH),
        **kwargs,
    )


def bare_schema() -> dict[str, Any]:
    loaded = json.loads(DEFAULT_SCHEMA_PATH.read_text(encoding="utf-8"))
    return dict(loaded["schema"])


def test_openai_body_enforces_the_schema_by_default() -> None:
    body = plan("openai", detail="original").body(REQUEST)
    assert set(body) == {
        "model",
        "input",
        "max_output_tokens",
        "service_tier",
        "prompt_cache_options",
        "text",
        "reasoning",
    }
    assert body["model"] == "gpt-5.6-luna"
    assert body["max_output_tokens"] == 128000
    assert body["service_tier"] == "default"
    assert body["prompt_cache_options"] == {"ttl": "30m"}
    assert body["reasoning"] == {"effort": "xhigh"}
    assert body["text"] == {
        "format": {
            "type": "json_schema",
            "name": "markdown_transcription_schema",
            "schema": bare_schema(),
            "strict": True,
        },
        "verbosity": "medium",
    }
    system, user = body["input"]
    assert system["role"] == "system"
    assert "The JSON schema:" in system["content"][0]["text"]
    assert user["content"] == [
        {"type": "input_text", "text": "The image:"},
        {
            "type": "input_image",
            "image_url": "data:image/jpeg;base64,UEFHRQ==",
            "detail": "original",
        },
    ]


def test_openai_json_format_keeps_the_schema_in_the_prompt_only() -> None:
    body = plan("openai", flags={"response_format": "json"}).body(REQUEST)
    assert body["text"] == {"verbosity": "medium"}
    assert "The JSON schema:" in body["input"][0]["content"][0]["text"]


def test_text_format_uses_the_plain_prompt_without_a_schema() -> None:
    body = plan("openai", flags={"response_format": "text"}).body(REQUEST)
    prompt = body["input"][0]["content"][0]["text"]
    plain = PLAIN_PROMPT_PATH.read_text(encoding="utf-8")
    assert "return exactly [No transcribable text]" in plain
    assert "return exactly [No transcribable text]" in prompt
    assert "The JSON schema:" not in prompt
    assert "format" not in body["text"]


def test_anthropic_body_enforces_the_schema_by_default() -> None:
    body = plan("anthropic").body(REQUEST)
    assert set(body) == {
        "model",
        "max_tokens",
        "system",
        "messages",
        "thinking",
        "output_config",
    }
    assert body["max_tokens"] == 128000
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["messages"][0]["content"] == [
        {"type": "text", "text": "The image:"},
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": "UEFHRQ==",
            },
        },
    ]


def test_anthropic_json_format_sends_no_output_format() -> None:
    body = plan("anthropic", flags={"response_format": "json"}).body(REQUEST)
    assert "format" not in body.get("output_config", {})
    assert "The JSON schema:" in body["system"][0]["text"]


def sync_anthropic_schema(spec: dict[str, Any]) -> Any:
    """Return the ``output_config.format`` schema a synchronous call sends."""
    model = ChatAnthropic(
        api_key="unused",  # type: ignore[arg-type]
        model_name="claude-sonnet-4-6",
        timeout=None,
        stop=None,
    )
    prepared = prepare_request(model, "anthropic", "schema", anthropic_schema(spec))
    return prepared.runnable.first.steps__["raw"].kwargs["output_config"]["format"][
        "schema"
    ]


def nullable_schema(count: int) -> dict[str, Any]:
    """Return a schema with *count* nullable fields and unsupported keywords."""
    properties: dict[str, Any] = {
        f"field_{index}": {"type": ["integer", "null"], "minimum": 1, "maximum": 7}
        for index in range(count)
    }
    properties["lines"] = {"type": "array", "items": {"type": "string"}, "maxItems": 3}
    return {
        "name": "Nullables",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": properties,
            "required": sorted(properties),
            "additionalProperties": False,
        },
    }


def write_schema(tmp_path: Path, spec: dict[str, Any]) -> Path:
    path = tmp_path / "schema.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path


def test_anthropic_batch_schema_is_the_synchronous_schema() -> None:
    body = plan("anthropic", flags={"response_format": "schema"}).body(REQUEST)
    schema = body["output_config"]["format"]["schema"]
    spec = json.loads(DEFAULT_SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema == sync_anthropic_schema(spec)
    assert schema["title"] == ANTHROPIC_SCHEMA_TITLE
    assert schema["description"] == ANTHROPIC_SCHEMA_DESCRIPTION
    assert {"type": "null"} in schema["properties"]["transcription"]["anyOf"]


def test_anthropic_batch_schema_drops_unsupported_keywords_like_sync(
    tmp_path: Path,
) -> None:
    spec = nullable_schema(2)
    path = write_schema(tmp_path, spec)
    body = plan("anthropic", schema_path=path).body(REQUEST)
    schema = body["output_config"]["format"]["schema"]
    assert schema == sync_anthropic_schema(spec)
    sent = json.dumps(schema)
    assert '"minimum"' not in sent and '"maxItems"' not in sent
    assert schema["properties"]["field_0"]["anyOf"][1] == {"type": "null"}


@pytest.mark.parametrize(
    ("count", "fmt"), [(ANTHROPIC_UNION_LIMIT, "schema"), (17, "json")]
)
def test_an_anthropic_schema_over_the_union_limit_uses_prompted_json(
    tmp_path: Path, count: int, fmt: str
) -> None:
    built = plan(
        "anthropic", schema_path=write_schema(tmp_path, nullable_schema(count))
    )
    body = built.body(REQUEST)
    assert built.response_format == fmt
    assert ("format" in body["output_config"]) is (fmt == "schema")
    assert "The JSON schema:" in body["system"][0]["text"]


def test_anthropic_caching_marks_the_system_prompt() -> None:
    (default,) = plan("anthropic").body(REQUEST)["system"]
    assert default["cache_control"] == {"type": "ephemeral"}
    off = Settings(prompt_caching=PromptCaching(enabled=False))
    assert isinstance(plan("anthropic", off).body(REQUEST)["system"], str)
    settings = Settings(prompt_caching=PromptCaching(enabled=True, anthropic_ttl="1h"))
    body = plan("anthropic", settings).body(REQUEST)
    (block,) = body["system"]
    assert block["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


@pytest.mark.parametrize(
    ("model", "enabled", "expected"),
    [
        ("gpt-5.6-luna", True, {"prompt_cache_options": {"ttl": "30m"}}),
        ("gpt-5.4", True, {"prompt_cache_retention": "24h"}),
        ("gpt-5.6-luna", False, {}),
    ],
)
def test_openai_batch_bodies_carry_the_cache_parameter_of_sync_requests(
    model: str, enabled: bool, expected: dict[str, Any]
) -> None:
    caching = PromptCaching(enabled=enabled, openai_retention="24h")
    body = plan("openai", Settings(prompt_caching=caching), model=model).body(REQUEST)
    keys = ("prompt_cache_options", "prompt_cache_retention")
    assert {key: body[key] for key in keys if key in body} == expected


def test_google_body_takes_the_family_output_limit() -> None:
    body = plan("google").body(REQUEST)
    assert body["model"] == "gemini-3-flash-preview"
    assert body["config"]["max_output_tokens"] == 65536
    assert body["config"]["temperature"] == 1.0
    assert body["contents"][0]["parts"][0] == {"text": "The image:"}


@pytest.mark.parametrize("response_format", [None, "schema"])
def test_google_schema_format_sends_the_json_schema(
    response_format: str | None,
) -> None:
    flags = {"response_format": response_format} if response_format else {}
    body = plan("google", flags=flags).body(REQUEST)
    assert body["config"]["response_mime_type"] == "application/json"
    assert body["config"]["response_json_schema"] == bare_schema()


def test_google_json_format_sends_no_json_schema() -> None:
    body = plan("google", flags={"response_format": "json"}).body(REQUEST)
    assert "response_json_schema" not in body["config"]
    assert "response_mime_type" not in body["config"]


@pytest.mark.parametrize(
    ("provider", "model", "expected"),
    [
        ("openai", "gpt-5.6-luna", "schema"),
        ("anthropic", "claude-sonnet-4-6", "schema"),
        ("google", "gemini-2.5-flash", "schema"),
        ("openai", "o3-deep-research", "json"),
        ("anthropic", "claude-3-5-haiku", "json"),
    ],
)
def test_batch_and_sync_pick_the_same_default_response_format(
    provider: str, model: str, expected: str
) -> None:
    spec = model_spec(provider, model)
    caps = model_capabilities(spec, Settings())
    batch = build_plan(spec, Settings(), schema_path=DEFAULT_SCHEMA_PATH)
    assert spec.response_format is None
    assert batch.response_format == response_format_for(spec, caps) == expected


_SAMPLERS = ("temperature", "top_p")


@pytest.mark.parametrize("effort", REASONING_EFFORTS)
@pytest.mark.parametrize(
    ("model", "max_output_tokens"),
    [
        ("claude-opus-5-5", None),
        ("claude-sonnet-4-6", None),
        ("claude-opus-4-5", None),
        ("claude-sonnet-4-5", None),
        ("claude-opus-4-1", 2000),
        ("claude-sonnet-4-5", 1200),
        ("claude-3-5-haiku", None),
    ],
)
def test_anthropic_batch_thinking_matches_the_sync_client(
    effort: str, model: str, max_output_tokens: int | None
) -> None:
    flags: dict[str, Any] = {"reasoning_effort": effort}
    if max_output_tokens is not None:
        flags["max_output_tokens"] = max_output_tokens
    spec = model_spec("anthropic", model, **flags)
    kwargs = constructor_kwargs(spec, Settings())

    body = build_plan(spec, Settings(), schema_path=DEFAULT_SCHEMA_PATH).body(REQUEST)

    assert body.get("thinking") == kwargs.get("thinking")
    assert body.get("output_config", {}).get("effort") == kwargs.get("effort")
    assert body["max_tokens"] == kwargs["max_tokens"]
    if "thinking" in body:
        assert not set(_SAMPLERS) & set(body)
    else:
        assert body.get("temperature") == kwargs.get("temperature")


def test_anthropic_batch_thinking_follows_the_effort() -> None:
    budgets = {
        effort: plan(
            "anthropic", flags={"reasoning_effort": effort}, model="claude-sonnet-4-5"
        )
        for effort in ("none", "low", "high")
    }
    assert "thinking" not in budgets["none"].body(REQUEST)
    assert budgets["low"].body(REQUEST)["thinking"] == {
        "type": "enabled",
        "budget_tokens": 1024,
    }
    assert budgets["high"].body(REQUEST)["thinking"]["budget_tokens"] == 16384


@pytest.mark.parametrize("response_format", ["schema", "json"])
def test_anthropic_batch_body_carries_the_effort(response_format: str) -> None:
    bodies = {
        effort: plan(
            "anthropic",
            flags={"reasoning_effort": effort, "response_format": response_format},
            model="claude-opus-5-5",
        ).body(REQUEST)
        for effort in ("low", "high")
    }
    assert bodies["low"] != bodies["high"]
    assert bodies["low"]["output_config"]["effort"] == "low"
    assert bodies["high"]["output_config"]["effort"] == "high"
    assert bodies["low"]["thinking"] == {"type": "adaptive"}
    if response_format == "schema":
        assert bodies["low"]["output_config"]["format"]["type"] == "json_schema"


def test_anthropic_batch_body_without_effort_ignores_the_level() -> None:
    bodies = [
        plan(
            "anthropic", flags={"reasoning_effort": effort}, model="claude-3-7-sonnet"
        ).body(REQUEST)
        for effort in ("low", "high")
    ]
    assert bodies[0] == bodies[1]
    assert "effort" not in bodies[0].get("output_config", {})


@pytest.mark.parametrize("effort", REASONING_EFFORTS)
@pytest.mark.parametrize(
    "model", ["gemini-3-flash-preview", "gemini-3.1-pro", "gemini-2.5-flash"]
)
def test_google_batch_thinking_matches_the_sync_client(effort: str, model: str) -> None:
    spec = model_spec("google", model, reasoning_effort=effort)
    kwargs = constructor_kwargs(spec, Settings())

    body = build_plan(spec, Settings(), schema_path=DEFAULT_SCHEMA_PATH).body(REQUEST)

    expected = {
        key: kwargs[key]
        for key in ("thinking_level", "thinking_budget")
        if kwargs.get(key) is not None
    }
    assert body["config"].get("thinking_config") == (expected or None)
    if model == "gemini-3-flash-preview" and effort == "low":
        assert expected == {"thinking_level": "low"}


@pytest.mark.parametrize("effort", REASONING_EFFORTS)
def test_openai_batch_body_carries_the_effort_and_no_thinking(effort: str) -> None:
    body = plan("openai", flags={"reasoning_effort": effort}).body(REQUEST)
    # GPT-5.6 documents no minimal effort; it takes the next level, low.
    assert body["reasoning"] == {"effort": "low" if effort == "minimal" else effort}
    assert "thinking" not in body
    assert "thinking_config" not in json.dumps(body)


def test_a_set_output_limit_is_clamped_to_the_family_on_google() -> None:
    body = plan("google", flags={"max_output_tokens": 500000}).body(REQUEST)
    assert body["config"]["max_output_tokens"] == 65536


@pytest.mark.parametrize("provider", ["openai", "anthropic", "google"])
def test_the_context_image_precedes_the_page(provider: str, tmp_path: Path) -> None:
    image = tmp_path / "context.png"
    Image.new("L", (4, 4), "white").save(image)
    body = plan(provider, context_image=image, context="Register of 1791.").body(
        BatchRequest("req-1", image_base64="UEFHRQ==", mime_type="image/png")
    )
    text = json.dumps(body)
    assert text.index("Context image:") < text.index("The image:")
    assert "Register of 1791." in text
    assert "image/png" in text


@pytest.mark.parametrize("provider", ["openai", "anthropic", "google"])
def test_a_png_page_is_sent_as_png(provider: str, tmp_path: Path) -> None:
    path = tmp_path / "page.png"
    Image.new("L", (32, 48), 128).save(path)
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    body = plan(provider).body(
        BatchRequest("req-1", image_base64=data, mime_type="image/png")
    )
    text = json.dumps(body)
    assert "image/png" in text and data in text


def test_a_request_detail_replaces_the_plan_detail() -> None:
    request = BatchRequest(
        "req-1", image_base64="UEFHRQ==", mime_type="image/png", image_detail="low"
    )
    body = plan("openai", detail="original").body(request)
    image = body["input"][1]["content"][-1]
    assert image["detail"] == "low"
    assert image["image_url"].startswith("data:image/png;base64,")


def test_providers_without_a_batch_api_are_refused() -> None:
    routed = replace(model_spec("openai"), provider="openrouter")
    with pytest.raises(ValueError, match="no batch API"):
        build_plan(routed, Settings(), schema_path=None)


def test_a_request_without_image_data_is_refused() -> None:
    with pytest.raises(ValueError, match="no image data"):
        plan("openai").body(BatchRequest("req-1"))
