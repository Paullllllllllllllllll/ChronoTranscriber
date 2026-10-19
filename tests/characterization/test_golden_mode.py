"""The rendered-only comparison of ``CT_GOLDEN_RENDERED_ONLY=1``.

Each test writes a golden case into ``tmp_path``, changes one file of the
run's snapshot and checks it with and without the mode.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from tests.characterization import golden

CASE = "mode/case"
RENDERED_TXT = "files/out/doc-1/doc.txt"
RENDERED_MD = "files/out/doc-1/doc.md"
SNAPSHOT = {
    RENDERED_TXT: "Page one.\nPage two.\n",
    RENDERED_MD: "Page one.\n\nPage two.\n",
    "files/out/doc-1/state.json": '{\n  "order_index": 0\n}\n',
    "files/out/chronotranscriber.sqlite.dump.txt": "## documents\n",
    "files/out/doc-1/page.png.bytes.txt": "<binary file>\n",
    "stdout.txt": "[DRY RUN] 1 discovered\n",
    "summary.json": json.dumps({"exit_code": 0, "json_line": {"outputs": []}}),
    "requests.json": "{}\n",
}

Change = Callable[[dict[str, str]], None]


@pytest.fixture
def golden_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write ``SNAPSHOT`` as the golden case below ``tmp_path``."""
    monkeypatch.setattr(golden, "GOLDEN_DIR", tmp_path)
    monkeypatch.delenv(golden.UPDATE_ENV, raising=False)
    monkeypatch.delenv(golden.RENDERED_ONLY_ENV, raising=False)
    for name, content in SNAPSHOT.items():
        target = tmp_path / CASE / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="\n")
    return tmp_path


def _set(name: str, content: str) -> Change:
    def change(files: dict[str, str]) -> None:
        files[name] = content

    return change


def _drop(name: str) -> Change:
    def change(files: dict[str, str]) -> None:
        del files[name]

    return change


def _check(change: Change) -> None:
    actual = dict(SNAPSHOT)
    change(actual)
    golden.check_files(CASE, actual)


IGNORED = [
    _set("files/out/doc-1/state.json", '{\n  "order_index": 1\n}\n'),
    _drop("files/out/doc-1/state.json"),
    _set("files/out/chronotranscriber.sqlite.dump.txt", "## units\n"),
    _set("files/out/doc-1/page.png.bytes.txt", "<other>\n"),
    _set("stdout.txt", "[DRY RUN] 2 discovered\n"),
    _set("requests.json", '{"calls": []}\n'),
    _set("files/out/doc-1/doc.json", "[]\n"),
    _set("summary.json", json.dumps({"exit_code": 0, "json_line": None})),
]
CHECKED = [
    _set(RENDERED_TXT, "Page one.\nPage 2.\n"),
    _set(RENDERED_MD, "Page one.\n\nPage two."),
    _drop(RENDERED_TXT),
    _set("files/out/doc-1/extra.md", "New.\n"),
    _set("summary.json", json.dumps({"exit_code": 1, "json_line": {}})),
]


@pytest.mark.parametrize("change", IGNORED)
def test_the_mode_ignores_files_other_than_the_views(
    change: Change, golden_case: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(AssertionError, match="Golden mismatch"):
        _check(change)

    monkeypatch.setenv(golden.RENDERED_ONLY_ENV, "1")
    _check(change)


@pytest.mark.parametrize("change", CHECKED)
def test_the_mode_fails_on_a_changed_view_or_exit_code(
    change: Change, golden_case: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(golden.RENDERED_ONLY_ENV, "1")
    with pytest.raises(AssertionError, match="Golden mismatch"):
        _check(change)


def test_the_mode_passes_an_unchanged_snapshot(
    golden_case: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(golden.RENDERED_ONLY_ENV, "1")
    golden.check_files(CASE, dict(SNAPSHOT))


def test_the_view_keeps_rendered_files_and_exit_codes() -> None:
    view = golden.rendered_view(SNAPSHOT)
    assert sorted(view) == [RENDERED_MD, RENDERED_TXT, "summary.json"]
    assert json.loads(view["summary.json"]) == {"exit_code": 0}
