"""Conditional options: warnings when a given option does not apply."""

from __future__ import annotations

from pathlib import Path

import pytest

from chronotranscriber import __main__ as entry
from chronotranscriber.cli.command import condition_warnings
from chronotranscriber.spec import OPTIONS
from tests.cli.conftest import MakePdf
from tests.conftest import make_spec


@pytest.fixture(autouse=True)
def _anthropic_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dry runs check the key variable of the provider they would call."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


MODEL = "claude-sonnet-5"
"""An Anthropic model, named because a provider flag outranks the default."""


def _dry_run(pdf: Path, *extra: str) -> list[str]:
    return [
        "transcribe",
        "--input",
        str(pdf),
        "--model",
        MODEL,
        *extra,
        "--dry-run",
        "--json",
    ]


def _warnings(err: str) -> list[str]:
    return [line for line in err.splitlines() if "[WARNING]" in line]


def test_service_tier_with_another_provider_warns_and_runs(
    make_pdf: MakePdf, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = _dry_run(make_pdf(), "--provider", "anthropic", "--service-tier", "priority")

    assert entry.main(argv) == 0

    captured = capsys.readouterr()
    assert len(captured.out.splitlines()) == 1
    (warning,) = _warnings(captured.err)
    assert "--service-tier" in warning
    assert "only for provider openai" in warning
    assert "anthropic" in warning


def test_without_the_conditional_flag_nothing_warns(
    make_pdf: MakePdf, capsys: pytest.CaptureFixture[str]
) -> None:
    assert entry.main(_dry_run(make_pdf(), "--provider", "anthropic")) == 0
    assert _warnings(capsys.readouterr().err) == []


def test_method_conditions_name_the_method() -> None:
    flags = {"method": "native", "type": "pdf", "ocr_config": "--psm 3"}

    (warning,) = condition_warnings(OPTIONS, flags, {}, make_spec(**flags))

    assert warning.startswith("--ocr-config applies only for method tesseract")
    assert "the method is native" in warning


def test_code_defaults_never_warn() -> None:
    flags = {"provider": "anthropic", "model": MODEL}
    assert condition_warnings(OPTIONS, flags, {}, make_spec(**flags)) == []


def test_settings_default_warns_with_its_name() -> None:
    flags = {"provider": "anthropic", "model": MODEL}
    defaults = {"service_tier": "priority"}

    (warning,) = condition_warnings(OPTIONS, flags, defaults, make_spec(**flags))

    assert warning.startswith("the settings default service_tier (--service-tier)")


def test_settings_default_warns_through_the_cli(
    tmp_path: Path, make_pdf: MakePdf, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = tmp_path / "settings.yaml"
    settings.write_text("defaults:\n  service_tier: priority\n", encoding="utf-8")
    argv = _dry_run(make_pdf(), "--provider", "anthropic", "--settings", str(settings))

    assert entry.main(argv) == 0

    captured = capsys.readouterr()
    assert len(captured.out.splitlines()) == 1
    (warning,) = _warnings(captured.err)
    assert "settings default service_tier" in warning


def test_flag_wins_over_a_settings_default_in_the_warning() -> None:
    flags = {"provider": "anthropic", "model": MODEL, "service_tier": "auto"}
    defaults = {"service_tier": "priority"}

    (warning,) = condition_warnings(OPTIONS, flags, defaults, make_spec(**flags))

    assert warning.startswith("--service-tier ")
