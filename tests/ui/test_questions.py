"""The shared model, response and image questions and their review lines."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from chronotranscriber.common.testing import KEEP, ScriptedPrompter, ScriptError
from chronotranscriber.common.wizard import (
    BACK,
    DONE,
    SETTINGS_LEGEND,
    Nav,
    Review,
    State,
    StepContext,
    render_review,
)
from chronotranscriber.spec import OPTIONS, REPAIR_OPTIONS
from chronotranscriber.ui.questions import (
    OTHER,
    ImageQuestions,
    ModelQuestions,
    Question,
    ResponseQuestions,
    current_model,
    images_line,
    model_line,
    resolution_of,
    response_line,
    support_of,
)
from chronotranscriber.ui.session import Session, new_state
from tests.ui.conftest import ENDPOINT, ENDPOINT_URL, SessionMaker

Group = Callable[[StepContext, Session], Any]


@dataclasses.dataclass
class Asked:
    result: Nav
    state: State
    prompter: ScriptedPrompter

    @property
    def lines(self) -> list[str]:
        return self.prompter.lines


def _ask(
    group: Group,
    session: Session,
    answers: Sequence[Any],
    *,
    table: Any = OPTIONS,
    given: Mapping[str, Any] | None = None,
) -> Asked:
    state = new_state(session, table)
    for name, value in (given or {}).items():
        state.set(name, value)
    prompter = ScriptedPrompter(answers)
    with prompter:
        questions: list[Question] = group(StepContext(state, prompter), session)
        result = StepContext.sequence(questions)
    return Asked(result, state, prompter)


def _model(ctx: StepContext, session: Session) -> list[Question]:
    return ModelQuestions(ctx, session).questions()


def _response(ctx: StepContext, session: Session) -> list[Question]:
    return ResponseQuestions(ctx, session).questions()


def _images(ctx: StepContext, session: Session) -> list[Question]:
    return ImageQuestions(ctx, session).questions()


def _effective(asked: Asked, session: Session) -> Any:
    model = current_model(asked.state, session)
    assert model is not None
    return model


def test_provider_and_model_with_recommended_tuning(session: Session) -> None:
    asked = _ask(
        _model, session, ["anthropic", OTHER, "claude-haiku-4-5", "medium", KEEP]
    )
    assert asked.result is DONE
    model = _effective(asked, session)
    assert (model.provider, model.model) == ("anthropic", "claude-haiku-4-5")
    assert model.reasoning_effort == "medium"
    assert asked.state.answers == {
        "model": "claude-haiku-4-5",
        "reasoning_effort": "medium",
    }
    assert "? Verbosity:" not in asked.lines


def test_back_returns_to_the_provider(session: Session) -> None:
    asked = _ask(
        _model,
        session,
        ["anthropic", BACK, "openai", "gpt-5.6-terra", KEEP, "recommended"],
    )
    assert asked.lines.count("? Provider:") == 2
    model = _effective(asked, session)
    assert (model.provider, model.model) == ("openai", "gpt-5.6-terra")


def test_back_on_the_first_question_leaves_the_group(session: Session) -> None:
    asked = _ask(_model, session, [BACK])
    assert asked.result is BACK
    assert asked.state.answers == {}


def test_a_typed_model_switches_to_the_provider_it_implies(session: Session) -> None:
    asked = _ask(
        _model, session, ["openai", OTHER, "claude-sonnet-5", KEEP, "recommended"]
    )
    assert "  provider anthropic, as the model name implies" in asked.lines
    model = _effective(asked, session)
    assert (model.provider, model.model) == ("anthropic", "claude-sonnet-5")
    assert "? Reasoning effort:" in asked.lines


@pytest.mark.parametrize(
    ("model", "asked"),
    [
        ("claude-opus-5-5", True),
        ("claude-sonnet-5-5", True),
        ("claude-haiku-5-5", True),
        ("claude-3-7-sonnet", False),
    ],
)
def test_the_effort_is_asked_for_claude_models_that_take_it(
    session: Session, model: str, asked: bool
) -> None:
    effort = [KEEP] if asked else []
    answers = ["anthropic", OTHER, model, *effort, "recommended"]
    lines = _ask(_model, session, answers).lines
    assert ("? Reasoning effort:" in lines) is asked


def test_a_named_endpoint_sets_provider_and_model_together(session: Session) -> None:
    asked = _ask(_model, session, [f"endpoint:{ENDPOINT}", "llama-3", "recommended"])
    assert f"   Endpoint {ENDPOINT}  {ENDPOINT_URL}" in asked.lines
    model = _effective(asked, session)
    assert (model.provider, model.endpoint, model.model) == (
        "custom",
        ENDPOINT,
        "llama-3",
    )
    assert asked.state.answers == {
        "model": "llama-3",
        "provider": "custom",
        "endpoint": ENDPOINT,
    }


def test_customized_tuning_asks_only_what_the_core_sends(session: Session) -> None:
    reasoning = _ask(
        _model,
        session,
        ["openai", "gpt-5.6-luna", KEEP, "customize", "low", "", "priority"],
    )
    assert reasoning.state.value("verbosity") == "low"
    assert reasoning.state.value("service_tier") == "priority"
    assert reasoning.state.value("max_output_tokens") is None
    assert "? Temperature (0.0 to 2.0):" not in reasoning.lines
    sampling = _ask(
        _model,
        session,
        ["openai", OTHER, "gpt-4o", "customize", "2000", "0.2", "0.9", KEEP],
    )
    assert "? Reasoning effort:" not in sampling.lines
    assert "? Verbosity:" not in sampling.lines
    model = _effective(sampling, session)
    assert (model.max_output_tokens, model.temperature, model.top_p) == (
        2000,
        0.2,
        0.9,
    )


def test_recommended_tuning_clears_customized_values(session: Session) -> None:
    asked = _ask(
        _model,
        session,
        ["openai", "gpt-5.6-luna", KEEP, "recommended"],
        given={"verbosity": "low", "service_tier": "priority"},
    )
    assert " > Customize" in asked.lines
    assert asked.state.value("verbosity") == "medium"
    assert asked.state.value("service_tier") == "default"


def test_no_unset_reset_where_a_settings_default_applies(
    make_session: SessionMaker,
) -> None:
    answers = ["openai", "gpt-5.6-luna", KEEP, "customize", KEEP, KEEP, KEEP]
    free = _ask(_model, make_session(), answers)
    assert "? Max output tokens (empty = the model's limit):" in free.lines
    session = make_session({"max_output_tokens": 4000})
    asked = _ask(_model, session, answers)
    assert "? Max output tokens: [4000 (s)] 4000" in asked.lines
    assert not any("empty =" in line for line in asked.lines)
    assert asked.state.from_settings("max_output_tokens")
    with pytest.raises(ScriptError, match="rejected"):
        _ask(_model, session, ["openai", "gpt-5.6-luna", KEEP, "customize", KEEP, ""])


def test_settings_defaults_are_preselected_and_marked(
    make_session: SessionMaker,
) -> None:
    session = make_session({"model": "gpt-5.6-terra", "reasoning_effort": "low"})
    asked = _ask(_model, session, [KEEP, KEEP, KEEP, "recommended"])
    assert " > OpenAI (s)" in asked.lines
    assert " > gpt-5.6-terra (s)" in asked.lines
    assert " > low (s)" in asked.lines
    assert asked.state.answers == {}
    assert asked.state.from_settings("model")


def test_audio_mode_offers_audio_providers_and_models_without_tuning(
    session: Session,
) -> None:
    given = {"type": "audio", "method": "audio-api"}
    kept = _ask(_model, session, [KEEP, KEEP], given=given)
    assert [line for line in kept.lines if line.startswith("   ") or " > " in line][
        :3
    ] == [" > OpenAI", "   Google", "   Back"]
    assert " > gpt-transcribe" in kept.lines
    assert _effective(kept, session).model == "gpt-transcribe"
    google = _ask(_model, session, ["google", "gemini-3.6-flash"], given=given)
    model = _effective(google, session)
    assert (model.provider, model.model) == ("google", "gemini-3.6-flash")
    assert "? Model parameters:" not in google.lines
    with pytest.raises(ScriptError, match="audio-api runs on"):
        _ask(_model, session, ["openai", OTHER, "claude-sonnet-5"], given=given)


def test_model_questions_work_on_the_repair_table(session: Session) -> None:
    asked = _ask(
        _model,
        session,
        ["google", "gemini-3.7-flash", KEEP, "recommended"],
        table=REPAIR_OPTIONS,
    )
    model = _effective(asked, session)
    assert (model.provider, model.model) == ("google", "gemini-3.7-flash")


def test_response_questions_default_schema_and_context_choices(
    session: Session, image_folder: Path
) -> None:
    image = next(image_folder.iterdir())
    asked = _ask(
        _response,
        session,
        [KEEP, KEEP, KEEP, "none", "file", f'"{image}"'],
    )
    assert " > Automatic  schema for this model" in asked.lines
    assert " > markdown_transcription_schema" in asked.lines
    assert "   plain_text_transcription_schema" in asked.lines
    assert asked.state.value("response_format") is None
    assert asked.state.value("prompt") is None
    assert asked.state.value("context") == "none"
    assert asked.state.value("context_image") == image


def test_plain_text_skips_the_schema(session: Session) -> None:
    asked = _ask(_response, session, ["text", "default", "auto", "none", "auto"])
    assert "? Schema:" not in asked.lines
    assert asked.state.answers == {"response_format": "text"}


def test_schema_and_prompt_files(session: Session, tmp_path: Path) -> None:
    schema = tmp_path / "custom_schema.json"
    schema.write_text(json.dumps({"type": "object"}), encoding="utf-8")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Transcribe.", encoding="utf-8")
    asked = _ask(
        _response,
        session,
        ["schema", OTHER, str(schema), "file", f"'{prompt}'", "auto", "none", "auto"],
    )
    assert asked.state.value("schema") == str(schema)
    assert asked.state.value("prompt") == prompt
    with pytest.raises(ScriptError, match="no such file"):
        _ask(_response, session, ["json", OTHER, str(tmp_path / "missing.json")])


def test_response_format_offers_no_automatic_over_a_settings_default(
    make_session: SessionMaker,
) -> None:
    session = make_session({"response_format": "json"})
    asked = _ask(_response, session, [KEEP, KEEP, KEEP, KEEP, KEEP, KEEP])
    assert not any("Automatic" in line for line in asked.lines[:6])
    assert " > JSON (prompted, validated) (s)" in asked.lines
    assert asked.state.from_settings("response_format")


def test_recommended_images_follow_the_model_family(session: Session) -> None:
    asked = _ask(_images, session, [KEEP])
    assert (
        " > Recommended for openai models  native DPI, detail original, JPEG 95, color"
    ) in asked.lines
    google = _ask(
        _images,
        session,
        ["customize", "200", "high", "png", False],
        given={"model": "gemini-3.7-flash"},
    )
    assert " > Recommended (high)" in google.lines
    assert "   original" not in google.lines
    assert "? JPEG quality (1 to 100, empty = 95):" not in google.lines
    values = [google.state.value(name) for name in ("dpi", "image_format")]
    assert values == [200, "png"]
    assert google.state.value("grayscale") is False


def test_image_resets_are_withheld_over_settings_defaults(
    make_session: SessionMaker,
) -> None:
    session = make_session({"dpi": 300, "image_detail": "high"})
    asked = _ask(_images, session, ["customize", KEEP, KEEP, KEEP, KEEP, KEEP])
    assert "? DPI (native or a number): [300 (s)] 300" in asked.lines
    assert not any(line.startswith("   Recommended (") for line in asked.lines[:12])
    assert " > high (s)" in asked.lines
    note = _ask(_images, session, [KEEP]).lines
    assert any("300 DPI (s), detail high (s)" in line for line in note)


def test_model_line_marks_a_settings_default_model(make_session: SessionMaker) -> None:
    session = make_session({"model": "gpt-5.6-terra"})
    resolved = resolution_of(new_state(session, OPTIONS), session)
    spec = resolved.spec
    line = model_line(spec.model, resolved.sources, session)
    screen = render_review(Review([line]), width=80)
    assert SETTINGS_LEGEND in screen.splitlines()[0]
    assert (
        "  model   openai / gpt-5.6-terra (s) | reasoning xhigh | tier default"
        in screen
    )
    audio = model_line(spec.model, resolved.sources, session, audio=True)
    assert len(audio.parts) == 1


def test_response_and_images_lines(session: Session) -> None:
    resolved = resolution_of(new_state(session, OPTIONS), session)
    spec = resolved.spec
    screen = render_review(
        Review(
            [
                response_line(spec.model, spec.prompt, resolved.sources, session),
                images_line(spec.model, spec.images, resolved.sources),
            ]
        )
    ).splitlines()
    assert screen[1] == (
        "  response   automatic (schema) | markdown_transcription_schema | "
        "default prompt | context auto | no fallback context | context image auto"
    )
    assert screen[2] == (
        "  images     native DPI | detail original | JPEG 95 | color | "
        "recommended for openai"
    )


def test_support_of_a_custom_endpoint_takes_its_declaration(
    session: Session,
) -> None:
    state = new_state(session, OPTIONS)
    state.set("model", "llama-3")
    state.set("provider", "custom")
    state.set("endpoint", ENDPOINT)
    model = current_model(state, session)
    assert model is not None
    support = support_of(model, session.settings)
    assert support.schema and support.temperature
    assert not (support.reasoning or support.service_tier)
    unknown = dataclasses.replace(model, endpoint="nowhere")
    assert not support_of(unknown, session.settings).schema
