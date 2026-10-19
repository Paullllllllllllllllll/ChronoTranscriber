"""A provider set above a model of another provider exits 2.

The rule applies when the run calls a model: the llm method (auto mode
included), audio-api and every repair. Equal precedence levels keep the
provider as given; Tesseract, native and whisper runs ignore the provider.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pytest

from chronotranscriber import __main__ as entry
from chronotranscriber.common.options import Source
from chronotranscriber.spec import (
    DEFAULT_AUDIO_MODEL,
    DEFAULT_GOOGLE_AUDIO_MODEL,
    OPTIONS,
    SpecError,
    parse,
    resolve_repair,
    resolve_spec,
    to_argv,
)
from tests.cli.conftest import json_lines


def _settings(tmp_path: Path, defaults: str = "{}") -> Path:
    path = tmp_path / "settings.yaml"
    path.write_text(f"defaults: {defaults}\n", encoding="utf-8")
    return path


def _main(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    defaults: str,
    *argv: str,
    input_name: str = "missing.pdf",
) -> tuple[int, dict[str, Any], str]:
    settings = _settings(tmp_path, defaults)
    code = entry.main(
        [
            "transcribe",
            "--input",
            str(tmp_path / input_name),
            *argv,
            "--settings",
            str(settings),
            "--dry-run",
            "--json",
        ]
    )
    captured = capsys.readouterr()
    assert len(captured.out.splitlines()) == 1, captured.out
    (payload,) = json_lines(captured.out)
    return code, payload, captured.err


@pytest.mark.parametrize(
    ("defaults", "model"),
    [("{}", "gpt-6-luna"), ("{model: gpt-5.6-terra}", "gpt-5.6-terra")],
    ids=["code default", "settings default"],
)
def test_a_flag_provider_above_the_model_exits_2(
    defaults: str,
    model: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, payload, err = _main(capsys, tmp_path, defaults, "--provider", "anthropic")

    assert code == 2
    assert payload["dry_run"] is True
    assert "--model" in err
    assert "anthropic" in err and model in err


def test_a_settings_provider_above_the_code_default_model_is_an_error() -> None:
    defaults = OPTIONS.check_defaults({"provider": "google"})
    with pytest.raises(SpecError, match="--model"):
        resolve_spec({"input": Path("x")}, defaults)


def test_every_repair_applies_the_rule() -> None:
    with pytest.raises(SpecError, match="--model"):
        resolve_repair({"input": Path("d.txt"), "provider": "anthropic"})
    spec = resolve_repair(
        {"input": Path("d.txt"), "provider": "anthropic", "model": "claude-x"}
    ).spec
    assert (spec.model.provider, spec.model.model) == ("anthropic", "claude-x")


@pytest.mark.parametrize(
    ("flags", "defaults"),
    [
        ({"provider": "openrouter", "model": "gpt-5.6-luna"}, {}),
        ({}, {"provider": "openrouter", "model": "gpt-5.6-luna"}),
        ({"provider": "anthropic", "model": "claude-x"}, {}),
        ({"provider": "anthropic"}, {"model": "my-local-model"}),
    ],
    ids=["flags", "settings", "matching model", "model of no provider"],
)
def test_equal_levels_and_matching_models_pass(
    flags: dict[str, Any], defaults: dict[str, Any]
) -> None:
    checked = OPTIONS.check_defaults(defaults)
    resolution = resolve_spec({"input": Path("x"), **flags}, checked)
    expected = flags.get("provider") or defaults["provider"]
    assert resolution.spec.model.provider == expected


def test_a_provider_below_the_model_still_yields() -> None:
    defaults = OPTIONS.check_defaults({"provider": "anthropic"})
    spec = parse(["--input", "x", "--model", "gpt-5.6-terra"], defaults)
    assert spec.model.provider == "openai"


def test_a_flag_provider_with_a_flag_model_runs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    document = fitz.open()
    document.new_page(width=200, height=300).insert_text((50, 100), "Page 1")
    document.save(tmp_path / "doc.pdf")
    document.close()

    code, payload, err = _main(
        capsys,
        tmp_path,
        "{}",
        "--provider",
        "anthropic",
        "--model",
        "claude-x",
        input_name="doc.pdf",
    )

    assert code == 0, err
    assert payload["spec"]["model.provider"]["value"] == "anthropic"


@pytest.mark.parametrize(
    "argv",
    [
        ["--method", "tesseract"],
        ["--type", "pdf", "--method", "native"],
        ["--type", "audio", "--method", "whisper"],
        ["--type", "epub"],
    ],
    ids=["tesseract", "native", "whisper", "epub"],
)
def test_local_methods_ignore_the_provider(argv: list[str]) -> None:
    spec = parse(["--input", "x", "--provider", "anthropic", *argv])
    assert spec.model.provider == "anthropic"
    assert parse(to_argv(spec)) == spec


def test_a_custom_endpoint_needs_a_model_at_its_level() -> None:
    with pytest.raises(SpecError, match="--model"):
        resolve_spec({"input": Path("x"), "endpoint": "local"})
    defaults = OPTIONS.check_defaults({"model": "llama-x"})
    with pytest.raises(SpecError, match="--model"):
        resolve_spec({"input": Path("x"), "endpoint": "local"}, defaults)
    defaults = OPTIONS.check_defaults({"endpoint": "local", "model": "llama-x"})
    spec = resolve_spec({"input": Path("x")}, defaults).spec
    assert (spec.model.provider, spec.model.model) == ("custom", "llama-x")


_AUDIO = ["--input", "talk.mp3", "--type", "audio", "--method", "audio-api"]


def test_audio_api_on_google_takes_the_google_audio_model() -> None:
    resolution = resolve_spec(
        {"input": Path("t.mp3"), "type": "audio", "method": "audio-api"},
        OPTIONS.check_defaults({"provider": "google"}),
    )
    assert resolution.spec.model.model == DEFAULT_GOOGLE_AUDIO_MODEL
    assert resolution.sources["model.model"] is Source.DEFAULT

    spec = parse([*_AUDIO, "--provider", "google"])
    assert (spec.model.provider, spec.model.model) == (
        "google",
        DEFAULT_GOOGLE_AUDIO_MODEL,
    )
    assert parse(to_argv(spec)) == spec
    defaults = OPTIONS.check_defaults({"model": DEFAULT_AUDIO_MODEL})
    assert parse([*_AUDIO, "--provider", "google"], defaults) == spec


def test_audio_api_keeps_a_model_named_at_the_provider_level() -> None:
    spec = parse([*_AUDIO, "--provider", "google", "--model", "gemini-3-pro"])
    assert spec.model.model == "gemini-3-pro"
    openai = parse([*_AUDIO, "--provider", "openai"])
    assert openai.model.model == DEFAULT_AUDIO_MODEL
