"""Plan-time checks: batch API, image input and the selection in auto mode."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pymupdf
import pytest

import chronotranscriber.run as core
from chronotranscriber import __main__ as entry
from chronotranscriber.common.contract import ExitCode
from chronotranscriber.settings import Endpoint
from tests.cli.conftest import json_lines
from tests.conftest import make_settings, make_spec

KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY")


@pytest.fixture(autouse=True)
def _keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dummy keys, so only the check under test can fail the plan."""
    for name in (*KEYS, "LOCAL_KEY"):
        monkeypatch.setenv(name, "test-key")


@pytest.fixture
def docs(tmp_path: Path) -> Path:
    """An input folder with one born-digital PDF."""
    folder = tmp_path / "in"
    folder.mkdir()
    document = pymupdf.open()
    document.new_page(width=200, height=300).insert_text((50, 100), "Page 1")
    document.save(folder / "x.pdf")
    document.close()
    return folder


def _settings_file(tmp_path: Path, *, vision: bool) -> Path:
    path = tmp_path / "settings.yaml"
    path.write_text(
        "endpoints:\n"
        "  local:\n"
        "    base_url: http://127.0.0.1:8000/v1\n"
        "    api_key_env: LOCAL_KEY\n"
        f"    supports_vision: {'true' if vision else 'false'}\n",
        encoding="utf-8",
    )
    return path


def _transcribe(docs: Path, out: Path, *extra: str) -> list[str]:
    return [
        "transcribe",
        "--input",
        str(docs),
        "--output",
        str(out),
        *extra,
        "--dry-run",
        "--json",
    ]


def _result(capsys: pytest.CaptureFixture[str]) -> tuple[dict[str, Any], str]:
    captured = capsys.readouterr()
    (payload,) = json_lines(captured.out)
    return payload, captured.err


def test_batch_on_a_provider_without_a_batch_api_exits_2(
    tmp_path: Path, docs: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out"
    argv = _transcribe(
        docs,
        out,
        "--type",
        "pdf",
        "--provider",
        "openrouter",
        "--model",
        "openai/gpt-5.6-luna",
        "--batch",
    )

    assert entry.main(argv) == 2

    payload, err = _result(capsys)
    assert payload["items_total"] == 1
    assert "batch API (openai, anthropic, google), not openrouter" in err
    assert not out.exists()


@pytest.mark.parametrize(
    ("model", "code"), [("gpt-7-foo", ExitCode.USAGE), ("gpt-5.6-luna", ExitCode.OK)]
)
def test_the_model_must_accept_images(
    tmp_path: Path,
    docs: Path,
    capsys: pytest.CaptureFixture[str],
    model: str,
    code: ExitCode,
) -> None:
    out = tmp_path / "out"

    assert entry.main(_transcribe(docs, out, "--type", "pdf", "--model", model)) == code

    payload, err = _result(capsys)
    assert payload["items_total"] == 1
    assert ("does not support image inputs" in err) is (code == ExitCode.USAGE)
    assert not out.exists()


def test_local_methods_need_no_image_model(docs: Path) -> None:
    spec = make_spec(
        docs, type="pdf", method="tesseract", model="gpt-7-foo", dry_run=True
    )

    assert len(core.plan(spec, make_settings()).to_process) == 1


@pytest.mark.parametrize(("vision", "code"), [(False, 2), (True, 0)])
def test_a_custom_endpoint_declares_its_image_input(
    tmp_path: Path,
    docs: Path,
    capsys: pytest.CaptureFixture[str],
    vision: bool,
    code: int,
) -> None:
    settings = _settings_file(tmp_path, vision=vision)
    out = tmp_path / "out"
    argv = _transcribe(
        docs,
        out,
        "--type",
        "pdf",
        "--endpoint",
        "local",
        "--model",
        "my-model",
        "--settings",
        str(settings),
    )

    assert entry.main(argv) == code

    _payload, err = _result(capsys)
    assert ("'my-model' does not support image inputs" in err) is (not vision)
    assert not out.exists()


def test_a_missing_endpoint_keeps_its_settings_error(docs: Path) -> None:
    spec = make_spec(docs, type="pdf", endpoint="local", model="m", dry_run=True)
    endpoint = Endpoint("other", "http://127.0.0.1:8000/v1", "LOCAL_KEY")

    with pytest.raises(core.ConfigurationError, match="unknown endpoint 'local'"):
        core.plan(spec, make_settings(endpoints={"other": endpoint}))


def test_files_in_auto_mode_take_paths_relative_to_the_input(
    tmp_path: Path, docs: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out"
    (docs / "vol" / "scans").mkdir(parents=True)
    (docs / "vol" / "scans" / "p1.png").write_bytes(b"png")
    (docs / "vol" / "talk.mp3").write_bytes(b"mp3")
    files = "vol\\scans,x.pdf"

    assert entry.main(_transcribe(docs, out, "--files", files)) == 0

    payload, _err = _result(capsys)
    assert [item["name"] for item in payload["to_process"]] == ["scans", "x.pdf"]
    assert entry.main(_transcribe(docs, out, "--files", "vol/talk.mp3,vol")) == 2
    _payload, err = _result(capsys)
    assert "'vol/talk.mp3', 'vol'" in err
    assert not out.exists()


def test_recursive_in_auto_mode_takes_every_level(
    tmp_path: Path, docs: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out"
    (docs / "vol" / "deep").mkdir(parents=True)
    (docs / "x.pdf").replace(docs / "vol" / "deep" / "x.pdf")
    (docs / "vol" / "scans").mkdir()
    (docs / "vol" / "scans" / "p1.png").write_bytes(b"png")
    flat = _transcribe(docs, out, "--type", "auto")

    assert entry.main(flat) == 2
    capsys.readouterr()
    assert entry.main([*flat, "--recursive", "--all"]) == 0

    payload, err = _result(capsys)
    assert payload["items_total"] == 2
    assert "[WARNING]" not in err
    assert not out.exists()
