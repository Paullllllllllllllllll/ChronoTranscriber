"""Tests for settings.py: the packaged example, typed values and resolution."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from chronotranscriber import settings as settings_module
from chronotranscriber.common.options import Source
from chronotranscriber.common.retry import RetryPolicy
from chronotranscriber.settings import (
    EXAMPLE_SETTINGS_PATH,
    Endpoint,
    Images,
    PromptCaching,
    RetrySettings,
    Settings,
    SettingsError,
    SpecError,
    Timeouts,
    load,
    resolve,
    resolve_batch_check,
    resolve_repair,
)
from chronotranscriber.spec import OPTIONS


@pytest.fixture
def settings_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the repository settings file and the example into ``tmp_path``."""
    monkeypatch.setattr(
        settings_module, "DEFAULT_SETTINGS_PATH", tmp_path / "settings.yaml"
    )
    monkeypatch.setattr(
        settings_module, "EXAMPLE_SETTINGS_PATH", tmp_path / "settings.example.yaml"
    )
    return tmp_path


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_packaged_example_loads_and_matches_code_values() -> None:
    loaded = load(EXAMPLE_SETTINGS_PATH)
    assert loaded.source == EXAMPLE_SETTINGS_PATH
    assert dict(loaded.defaults) == {"fallback_context": "context/transcr_context.txt"}
    expected = Settings(source=EXAMPLE_SETTINGS_PATH)
    assert replace(loaded, defaults=expected.defaults) == expected


def test_settings_flag_beats_the_repository_file_and_the_example(
    settings_paths: Path,
) -> None:
    _write(settings_paths / "settings.example.yaml", "concurrency:\n  audio: 5\n")
    assert load().concurrency.audio == 5
    _write(settings_paths / "settings.yaml", "concurrency:\n  audio: 7\n")
    assert load().concurrency.audio == 7
    other = _write(settings_paths / "other.yaml", "concurrency:\n  audio: 9\n")
    assert load(other).concurrency.audio == 9
    assert load(other).source == other


def test_without_any_file_the_code_values_apply(settings_paths: Path) -> None:
    assert load() == Settings()


def test_a_missing_settings_flag_file_is_an_error(settings_paths: Path) -> None:
    with pytest.raises(SettingsError, match="missing.yaml"):
        load(settings_paths / "missing.yaml")


def test_partial_file_keeps_code_values(settings_paths: Path) -> None:
    _write(
        settings_paths / "settings.yaml",
        "timeouts:\n  request: 300\nretry:\n  validation_attempts: 3\n"
        "prompt_caching:\n  openai:\n    prompt_cache_retention: null\n",
    )
    value = load()
    assert value.timeouts == Timeouts(300.0, 10.0, 30.0, 30.0)
    assert value.retry == RetrySettings(8, 3, 3, 0, "auto")
    assert value.prompt_caching == PromptCaching(True, "5m", None, True)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("images:\n  original_max_side_px: 2500\n", 2500),
        ("images:\n  original_max_side_px: null\n", None),
        ("images: {}\n", None),
    ],
)
def test_original_max_side_setting(
    settings_paths: Path, text: str, expected: int | None
) -> None:
    _write(settings_paths / "settings.yaml", text)
    assert load().images == Images(expected)


def test_paths_resolve_against_the_current_directory(
    settings_paths: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work = settings_paths / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    absolute = settings_paths / "models"
    _write(
        settings_paths / "settings.yaml",
        f"logs_dir: logs\nwhisper:\n  model_path: '{absolute.as_posix()}'\n"
        "  download_root: cache/whisper\n"
        "tools:\n  tesseract_cmd: tesseract\n  ffmpeg_cmd: bin/ffmpeg\n",
    )
    value = load()
    assert value.logs_dir == work / "logs"
    assert value.whisper.model_path == absolute
    assert value.whisper.download_root == work / "cache" / "whisper"
    assert value.tools.tesseract_cmd == "tesseract"
    assert value.tools.ffmpeg_cmd == str(work / "bin" / "ffmpeg")
    assert value.tools.ffprobe_cmd is None


def test_endpoints_and_key_variables(settings_paths: Path) -> None:
    _write(
        settings_paths / "settings.yaml",
        "api_keys:\n  openai: MY_OPENAI_KEY\n"
        "endpoints:\n  local:\n    base_url: http://127.0.0.1:8000/v1/\n"
        "    api_key_env: LOCAL_KEY\n    supports_schema: true\n",
    )
    value = load()
    assert value.endpoint("local") == Endpoint(
        "local", "http://127.0.0.1:8000/v1/", "LOCAL_KEY", True, True
    )
    assert value.api_key_env("openai") == "MY_OPENAI_KEY"
    assert value.api_key_env("anthropic") == "ANTHROPIC_API_KEY"
    assert value.api_key_env("custom", "local") == "LOCAL_KEY"
    with pytest.raises(SettingsError, match="unknown endpoint"):
        value.endpoint("remote")
    with pytest.raises(SettingsError):
        value.api_key_env("custom")


@pytest.mark.parametrize(
    "text",
    [
        "unknown: 1\n",
        "api_keys:\n  mistral: MISTRAL_KEY\n",
        "endpoints:\n  local:\n    base_url: http://x/\n    api_key_env: K\n"
        "    model: m\n",
        "timeouts:\n  read: 5\n",
        "retry:\n  attempts: 3\n",
        "concurrency:\n  transcription: 3\n",
        "tools:\n  tesseract: x\n",
        "whisper:\n  model_size: small\n",
        "audio_chunking:\n  target_seconds: 60\n",
        "prompt_caching:\n  google: {}\n",
        "prompt_caching:\n  anthropic:\n    ttl_seconds: 300\n",
        "prompt_caching:\n  openai:\n    retention: 24h\n",
        "prompt_caching:\n  openrouter:\n    cache: true\n",
        "images:\n  max_side_px: 3000\n",
        "defaults:\n  modle: gpt-5.6-terra\n",
        "keep_preprocessed_images: true\n",
    ],
)
def test_unknown_keys_at_every_level_are_errors(
    settings_paths: Path, text: str
) -> None:
    _write(settings_paths / "settings.yaml", text)
    with pytest.raises(SettingsError, match="settings.yaml"):
        load()


@pytest.mark.parametrize(
    "text",
    [
        "timeouts:\n  request: fast\n",
        "rate_limits:\n  - [10, 0]\n",
        "rate_limits: []\n",
        "endpoints:\n  local:\n    base_url: http://x/\n",
        "retry:\n  max_attempts: 0\n",
        "retry:\n  page_timeout: soon\n",
        "concurrency:\n  audio: 0\n",
        "whisper:\n  device: tpu\n",
        "audio_chunking:\n  chunk_format: ogg\n",
        "audio_chunking:\n  mono: maybe\n",
        "prompt_caching:\n  anthropic:\n    ttl: 2h\n",
        "logs_dir: 5\n",
        "images:\n  original_max_side_px: 0\n",
        "images:\n  original_max_side_px: wide\n",
        "defaults:\n  dry_run: true\n",
        "defaults:\n  force: true\n",
        "defaults:\n  input: x.pdf\n",
        "defaults:\n  beside_input: true\n",
        "defaults:\n  concurrency: 0\n",
        "defaults:\n  method: gpt\n",
        "defaults:\n  formats: [json]\n",
        "defaults:\n  formats: [txt, sqlite]\n",
        "defaults:\n  output: out\n  beside_input: true\n",
        "defaults: [model]\n",
    ],
)
def test_invalid_values_are_errors(settings_paths: Path, text: str) -> None:
    _write(settings_paths / "settings.yaml", text)
    with pytest.raises(SettingsError, match="settings.yaml"):
        load()


@pytest.mark.parametrize(
    ("value", "expected", "deadline"),
    [("auto", "auto", 900.0 * 3 + 300.0), ("off", 0.0, None), (False, 0.0, None)],
)
def test_page_timeout_values(
    settings_paths: Path, value: object, expected: object, deadline: float | None
) -> None:
    text = "retry:\n  page_timeout: " + ("false" if value is False else str(value))
    _write(settings_paths / "settings.yaml", text + "\n")
    retry = load().retry
    assert retry.page_timeout == expected
    assert retry.page_deadline(900.0) == deadline
    assert RetrySettings(page_timeout=120.0).page_deadline(900.0) == 120.0


def test_retry_policy() -> None:
    policy = Settings().retry.policy()
    assert isinstance(policy, RetryPolicy)
    assert (policy.max_attempts, policy.validation_attempts) == (8, 1)
    assert (policy.backoff_base, policy.backoff_cap) == (2.0, 120.0)
    assert policy.backoff(0, "timeout", jitter=0.0) == 2.0
    assert policy.backoff(2, "rate_limit", jitter=0.5) == 8.5
    assert policy.backoff(10, "other", jitter=0.0) == 120.0


def test_precedence_flag_then_settings_default_then_code_default(
    settings_paths: Path,
) -> None:
    _write(
        settings_paths / "settings.yaml",
        "defaults:\n  model: gpt-5.6-terra\n  concurrency: 40\n"
        "  verbosity: high\n  formats: [md]\n",
    )
    value = load()
    flags = OPTIONS.parse(["--input", "x.pdf", "--concurrency", "8"])
    resolution = resolve(value, flags)
    spec, sources = resolution.spec, resolution.sources
    assert (spec.run.concurrency, sources["run.concurrency"]) == (8, Source.FLAG)
    assert spec.model.model == "gpt-5.6-terra"
    assert sources["model.model"] is Source.SETTINGS
    assert (spec.model.verbosity, sources["model.verbosity"]) == (
        "high",
        Source.SETTINGS,
    )
    assert spec.output.formats == ("md",)
    assert (spec.model.reasoning_effort, sources["model.reasoning_effort"]) == (
        "xhigh",
        Source.DEFAULT,
    )
    check = resolve_batch_check(value, {}).spec
    assert check.formats is None
    repair = resolve_repair(value, {"input": Path("x.txt")}).spec
    assert repair.model.model == "gpt-5.6-terra"


def test_resolve_checks_endpoint_names() -> None:
    value = Settings(
        endpoints={"local": Endpoint("local", "http://127.0.0.1/v1/", "LOCAL_KEY")}
    )
    named = {"input": Path("x"), "model": "llama-x"}
    spec = resolve(value, {**named, "endpoint": "local"}).spec
    assert spec.model.provider == "custom"
    with pytest.raises(SettingsError, match="remote"):
        resolve(value, {**named, "endpoint": "remote"})
    with pytest.raises(SettingsError, match="remote"):
        resolve_repair(value, {**named, "endpoint": "remote"})
    with pytest.raises(SpecError):
        resolve(value, {"input": Path("x"), "provider": "custom"})


def test_example_names_key_variables_by_provider_only() -> None:
    text = EXAMPLE_SETTINGS_PATH.read_text(encoding="utf-8")
    variables = {
        line.split(":", 1)[1].strip()
        for line in text.splitlines()
        if line.strip().endswith("_API_KEY")
    }
    assert variables == {
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "OPENROUTER_API_KEY",
        "YOUR_ENDPOINT_API_KEY",
    }
