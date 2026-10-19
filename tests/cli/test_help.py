"""Help of the console script and of every subcommand."""

from __future__ import annotations

import re

import pytest

from chronotranscriber import __main__ as entry

WIDTHS = (80, 100)


def _help(capsys: pytest.CaptureFixture[str], *argv: str) -> str:
    with pytest.raises(SystemExit) as exc:
        entry.main([*argv, "--help"])
    assert exc.value.code == 0
    return capsys.readouterr().out


@pytest.fixture(params=WIDTHS, ids=[f"columns{w}" for w in WIDTHS])
def columns(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> int:
    width: int = request.param
    monkeypatch.setenv("COLUMNS", str(width))
    return width


def test_the_bare_command_prints_help_and_exits_0(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert entry.main([]) == 0
    bare = capsys.readouterr().out
    assert "usage: chronotranscriber" in bare
    for name in ("transcribe", "repair", "batch", "render", "drop", "ui"):
        assert name in bare
    assert "migrate" not in bare
    assert "guided run in an interactive terminal" in bare
    assert _help(capsys) == bare


def test_ui_help_lists_settings_and_fits_the_width(
    columns: int, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _help(capsys, "ui")
    assert "--settings" in set(re.findall(r"--[\w-]+", out))
    assert "--input" not in out
    for line in out.splitlines():
        assert len(line) <= columns, line


@pytest.mark.parametrize("command", ["render", "drop"])
def test_store_command_help_names_the_exit_codes_and_fits_the_width(
    command: str, columns: int, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _help(capsys, command)
    assert "Exit codes:" in out
    for line in out.splitlines():
        assert len(line) <= columns, line


def test_transcribe_help_never_splits_a_flag(
    columns: int, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _help(capsys, "transcribe")
    for line in out.splitlines():
        assert len(line) <= columns, line
        assert not re.search(r"[\w-]-$", line), line


def test_batch_without_a_subcommand_prints_its_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert entry.main(["batch"]) == 0
    out = capsys.readouterr().out
    assert "check" in out
    assert "cancel" in out


def test_removed_flags_are_usage_errors(capsys: pytest.CaptureFixture[str]) -> None:
    assert entry.main(["transcribe", "--auto"]) == 2
    assert "unrecognized arguments" in capsys.readouterr().err
