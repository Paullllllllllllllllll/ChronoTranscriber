"""Tests for spec.py: code defaults, resolution rules, conditions, suggestions.

Round trips of the option tables live in ``test_spec_roundtrip.py``, the
hand-written command lines in ``test_argv_to_spec.py``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from chronotranscriber.common.options import Source
from chronotranscriber.providers import infer_provider
from chronotranscriber.spec import (
    AUDIO_API_PROVIDERS,
    DEFAULT_AUDIO_MODEL,
    DEFAULT_MODEL,
    MODEL_SUGGESTIONS,
    OPTIONS,
    BatchCancelSpec,
    ImageSpec,
    RunSpec,
    SpecError,
    effective_spec,
    inapplicable,
    model_suggestions,
    parse,
    parse_batch_cancel,
    parse_batch_check,
    parse_repair,
    recommended_images,
    resolve_batch_cancel,
    resolve_repair,
    resolve_spec,
    to_argv,
)


def _code_defaults(path: str = "in.pdf") -> RunSpec:
    return resolve_spec({"input": Path(path)}).spec


def test_an_audio_api_model_names_its_provider() -> None:
    audio = ["--input", "t.mp3", "--type", "audio"]
    google = parse([*audio, "--method", "audio-api", "--model", "gemini-3.6-flash"])
    assert google.model.provider == "google"
    whisper = parse([*audio, "--method", "whisper"])
    assert whisper.model.model == DEFAULT_MODEL


def test_settings_model_default_applies_to_audio_api() -> None:
    defaults = OPTIONS.check_defaults({"model": "gemini-3.6-flash"})
    flags = {"input": Path("t.mp3"), "type": "audio", "method": "audio-api"}
    resolution = resolve_spec(flags, defaults)
    assert resolution.spec.model.model == "gemini-3.6-flash"
    assert resolution.sources["model.model"] is Source.SETTINGS


def test_a_type_taking_llm_keeps_an_invalid_settings_method() -> None:
    defaults = OPTIONS.check_defaults({"method": "whisper"})
    with pytest.raises(SpecError, match="--type pdf takes"):
        resolve_spec({"input": Path("x.pdf"), "type": "pdf"}, defaults)


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        ({"type": "auto", "method": "native"}, "llm or tesseract"),
        ({"type": "images", "method": "whisper"}, "--type images"),
        (
            {"type": "audio", "method": "audio-api", "provider": "anthropic"},
            "audio-api runs on provider openai or google",
        ),
    ],
)
def test_type_and_method_must_fit(flags: dict[str, Any], message: str) -> None:
    with pytest.raises(SpecError, match=message):
        resolve_spec({"input": Path("x"), **flags})


def test_the_effective_spec_names_the_source_of_every_option() -> None:
    defaults = OPTIONS.check_defaults(
        {"model": "gpt-5.6-terra", "concurrency": 40, "formats": ["md", "txt"]}
    )
    flags = {"input": Path("x"), "concurrency": 8, "layout": "mirror"}

    resolution = resolve_spec(flags, defaults)

    payload = effective_spec(resolution.spec, resolution.sources)
    assert payload["run.concurrency"] == {"value": 8, "source": "flag"}
    assert payload["model.model"] == {"value": "gpt-5.6-terra", "source": "settings"}
    assert payload["input.path"] == {"value": "x", "source": "flag"}
    assert payload["output.formats"]["value"] == ["txt", "md"]
    assert len(payload) == len([o for o in OPTIONS if o.target is not None])


def test_to_argv_spells_out_values_the_settings_defaults_change() -> None:
    spec = _code_defaults()

    def argv(**defaults: Any) -> list[str]:
        return to_argv(spec, OPTIONS.check_defaults(defaults))[2:]

    assert argv(output="out") == ["--beside-input"]
    assert argv(context="ctx.txt") == ["--context", "auto"]
    assert argv(prompt="p.txt") == ["--prompt", "default"]
    assert argv(language="de") == ["--language", "auto"]
    assert argv(model="gpt-5.6-terra") == ["--model", DEFAULT_MODEL]
    assert argv(endpoint="local") == ["--provider", "openai"]
    assert argv(batch=True) == ["--no-batch"]
    assert argv(provider="openai", concurrency=16, context="auto") == []


@pytest.mark.parametrize(
    ("defaults", "option"),
    [
        ({"max_output_tokens": 100}, "max_output_tokens"),
        ({"response_format": "json"}, "response_format"),
        ({"dpi": 300}, "dpi"),
        ({"grayscale": False}, "grayscale"),
    ],
)
def test_to_argv_rejects_a_default_without_an_unset_spelling(
    defaults: dict[str, Any], option: str
) -> None:
    with pytest.raises(SpecError, match=option):
        to_argv(_code_defaults(), OPTIONS.check_defaults(defaults))


def test_custom_provider_without_endpoint_is_an_error() -> None:
    with pytest.raises(SpecError, match="endpoint"):
        resolve_spec({"input": Path("x"), "provider": "custom"})
    with pytest.raises(SpecError, match="endpoint"):
        resolve_repair({"input": Path("x"), "provider": "custom"})


@pytest.mark.parametrize(
    ("flags", "defaults"),
    [
        ({"provider": "openai", "model": "anthropic:claude-x"}, {}),
        ({"model": "google:gemini-y"}, {"provider": "custom", "endpoint": "e"}),
        ({"provider": "openai", "model": "anthropic:claude-x"}, {"model": "gpt-5.5"}),
    ],
)
def test_a_provider_prefix_contradicting_the_provider_is_an_error(
    flags: dict[str, Any], defaults: dict[str, Any]
) -> None:
    with pytest.raises(SpecError, match="names provider"):
        resolve_spec({"input": Path("x"), **flags}, OPTIONS.check_defaults(defaults))


def test_a_provider_prefix_selects_the_provider() -> None:
    spec = parse(["--input", "x", "--model", "anthropic:claude-x"])
    assert (spec.model.provider, spec.model.model) == ("anthropic", "claude-x")
    defaults = OPTIONS.check_defaults({"provider": "openai"})
    spec = parse(["--input", "x", "--model", "claude-x"], defaults)
    assert spec.model.provider == "anthropic"


def test_a_dotted_anthropic_alias_resolves_to_its_api_id() -> None:
    spec = parse(["--input", "x", "--model", "claude-haiku-5.5"])
    assert (spec.model.provider, spec.model.model) == ("anthropic", "claude-haiku-5-5")
    assert parse(to_argv(spec)) == spec
    defaults = OPTIONS.check_defaults({"model": "claude-sonnet-5.5"})
    assert resolve_spec({"input": Path("x")}, defaults).spec.model.model == (
        "claude-sonnet-5-5"
    )
    repair = parse_repair(["--input", "d.txt", "--model", "anthropic:claude-opus-4.8"])
    assert repair.model.model == "claude-opus-4-8"
    routed = parse(["--input", "x", "--model", "anthropic/claude-sonnet-4.5"])
    assert routed.model.model == "anthropic/claude-sonnet-4.5"


def test_missing_input_is_an_error_and_repair_takes_the_current_directory() -> None:
    with pytest.raises(SpecError, match="input"):
        resolve_spec({})
    assert resolve_repair({}).spec.input == Path(".")


def test_empty_formats_are_an_error() -> None:
    with pytest.raises(SpecError, match="formats"):
        resolve_spec({"input": Path("x"), "formats": ()})


@pytest.mark.parametrize(
    "argv",
    [
        ["--input", "x", "--formats", "docx"],
        ["--input", "x", "--formats", "json"],
        ["--input", "x", "--formats", "sqlite"],
        ["--input", "x", "--formats", "txt,sqlite"],
        ["--input", "x", "--temperature", "2.5"],
        ["--input", "x", "--top-p", "nan"],
        ["--input", "x", "--jpeg-quality", "0"],
        ["--input", "x", "--dpi", "-3"],
        ["--input", "x", "--pages", "0"],
        ["--input", "x", "--pages", "7-3"],
        ["--input", "x", "--output", "o", "--beside-input"],
        ["--input", "x", "--response-format", "xml"],
        ["--input", "x", "--type", "pdfs"],
        ["--input", "x", "--method", "gpt"],
        ["--input", "x", "--verbosity", "concise"],
        ["--input", "x", "--layout", "flat"],
    ],
)
def test_invalid_arguments_exit_with_status_2(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as info:
        parse(argv)
    assert info.value.code == 2


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (
            [
                "--provider",
                "anthropic",
                "--model",
                "claude-x",
                "--service-tier",
                "flex",
            ],
            ["service_tier"],
        ),
        (["--provider", "openai", "--service-tier", "flex"], []),
        (
            ["--ocr-config", "--psm 4", "--ocr-preprocess", "none"],
            ["ocr_config", "ocr_preprocess"],
        ),
        (["--method", "tesseract", "--ocr-preprocess", "none"], []),
        (
            ["--whisper-model", "small", "--language", "de"],
            ["whisper_model", "language"],
        ),
        (
            ["--type", "audio", "--method", "audio-api", "--whisper-model", "small"],
            ["whisper_model"],
        ),
        (["--type", "audio", "--method", "whisper", "--language", "de"], []),
        (
            ["--method", "tesseract", "--batch", "--sync-fallback"],
            ["batch", "sync_fallback"],
        ),
        (["--batch"], []),
        (["--type", "pdf", "--searchable-pdfs", "text"], ["searchable_pdfs"]),
        (["--files", "a.pdf", "--recursive"], ["files", "recursive"]),
        (["--type", "pdf", "--files", "a.pdf", "--recursive"], []),
    ],
)
def test_condition_rules(argv: list[str], expected: list[str]) -> None:
    flags = OPTIONS.parse(["--input", "x", *argv])
    spec = resolve_spec(flags).spec
    assert inapplicable(spec, flags) == expected


@pytest.mark.parametrize("value", ["-1", "a", "5-2", ","])
def test_invalid_repair_indices_exit_with_status_2(value: str) -> None:
    with pytest.raises(SystemExit) as info:
        parse_repair(["--input", "d.txt", "--indices", value])
    assert info.value.code == 2


def test_batch_check_keeps_the_run_formats_and_refuses_database_formats() -> None:
    assert parse_batch_check([]).formats is None
    defaults = OPTIONS.check_defaults({"formats": ["md"]})
    assert parse_batch_check([], defaults).formats is None
    assert parse_batch_check(["--formats", "md"]).formats == ("md",)
    for name in ("json", "sqlite"):
        with pytest.raises(SystemExit) as info:
            parse_batch_check(["--formats", name])
        assert info.value.code == 2


def test_batch_cancel_needs_ids_or_all() -> None:
    cancel = parse_batch_cancel(["--batch-ids", "b1,b2"])
    assert cancel == BatchCancelSpec(("b1", "b2"), False, False)
    assert parse_batch_cancel(["--all"]).all
    with pytest.raises(SpecError, match="--batch-ids IDS or --all"):
        resolve_batch_cancel({})
    with pytest.raises(SystemExit):
        parse_batch_cancel(["--all", "--batch-ids", "b1"])


def test_model_suggestions_imply_their_provider() -> None:
    for provider, names in MODEL_SUGGESTIONS.items():
        assert all(infer_provider(name) == provider for name in names)
    for provider in AUDIO_API_PROVIDERS:
        names = model_suggestions(provider, audio=True)
        assert names and all(infer_provider(name) == provider for name in names)
    assert model_suggestions("openai")[0] == DEFAULT_MODEL
    assert model_suggestions("openai", audio=True)[0] == DEFAULT_AUDIO_MODEL
    assert model_suggestions("custom") == ()
    suggest = OPTIONS["model"].choice_provider
    assert OPTIONS["model"].choices is None
    assert suggest is not None
    assert tuple(suggest()) == model_suggestions()


def test_recommended_images_fill_only_the_unset_fields() -> None:
    recommended = recommended_images("openai")
    filled = ImageSpec(jpeg_quality=70).over(recommended)
    assert filled == replace(recommended, jpeg_quality=70)
    assert recommended_images("unknown") == recommended_images("custom")


def test_to_argv_spells_only_changed_values() -> None:
    spec = _code_defaults()
    tiered = replace(spec, model=replace(spec.model, service_tier="priority"))
    assert to_argv(tiered) == ["--input", "in.pdf", "--service-tier", "priority"]
