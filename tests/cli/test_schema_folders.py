"""Schema and classifier folders: ``--schemas-dir``, ``--classifiers-dir``, settings.

The lookup takes an existing path, then the user folders (flag folders
before settings folders), then the repository's folder. The commands run
in-process through the console entry point with a temporary settings file;
every schema and classifier is an invented file in a temporary folder.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from chronotranscriber import __main__ as entry
from chronotranscriber import classify
from chronotranscriber.common.classifiers import AmbiguousNameError
from chronotranscriber.llm.prompts import (
    SCHEMAS_DIR,
    AmbiguousSchemaError,
    list_schema_options,
    resolve_schema,
)
from chronotranscriber.settings import Settings
from tests.characterization.inputs import make_scanned_pdf
from tests.cli.conftest import json_lines

BUNDLED = SCHEMAS_DIR / "markdown_transcription_schema.json"


def _schema(folder: Path, file_name: str, name: str, *, valid: bool = True) -> Path:
    """Write a transcription schema named *name* to *folder*."""
    body = json.loads(BUNDLED.read_text(encoding="utf-8"))
    body["name"] = name
    if not valid:
        del body["schema"]["properties"]["transcription_not_possible"]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / file_name
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return path


def _classifier(folder: Path, file_name: str, name: str) -> Path:
    body = {
        "name": name,
        "version": 1,
        "unit": "page",
        "questions": [
            {"name": "title", "type": "predicate", "instructions": "A title page."}
        ],
    }
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / file_name
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def _settings_file(tmp_path: Path, **data: Any) -> Path:
    path = tmp_path / "settings.yaml"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _main(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, Any, str]:
    code = entry.main([*argv, "--json"])
    captured = capsys.readouterr()
    lines = json_lines(captured.out)
    return code, lines[-1] if lines else None, captured.err


# ============================================================================
# Lookup
# ============================================================================
def test_a_user_folder_schema_hides_the_bundled_file_of_its_name(
    tmp_path: Path,
) -> None:
    own = _schema(tmp_path / "project", BUNDLED.name, "ledger_pages")

    assert resolve_schema("markdown_transcription_schema", [own.parent]) == (
        "ledger_pages",
        own.resolve(),
    )
    names = [name for name, _path in list_schema_options([own.parent])]
    assert "ledger_pages" in names
    assert "markdown_transcription_schema" not in names
    assert "plain_text_transcription_schema" in names


def test_a_user_schema_is_found_by_its_name_field(tmp_path: Path) -> None:
    own = _schema(tmp_path / "project", "pages.json", "ledger_pages")

    assert resolve_schema("ledger_pages", [own.parent])[1] == own.resolve()
    assert resolve_schema("pages", [own.parent])[1] == own.resolve()
    assert resolve_schema(str(own))[1] == own.resolve()


def test_a_schema_name_in_two_user_folders_is_ambiguous(tmp_path: Path) -> None:
    first = _schema(tmp_path / "one", "pages.json", "ledger_pages")
    second = _schema(tmp_path / "two", "other.json", "ledger_pages")

    with pytest.raises(AmbiguousSchemaError) as caught:
        resolve_schema("ledger_pages", [first.parent, second.parent])

    assert isinstance(caught.value, AmbiguousNameError)
    assert caught.value.paths == (first, second)
    assert str(first) in str(caught.value) and str(second) in str(caught.value)


def test_a_folder_given_twice_is_searched_once(tmp_path: Path) -> None:
    own = _schema(tmp_path / "project", "pages.json", "ledger_pages")
    again = tmp_path / "project" / ".." / "project"

    assert resolve_schema("ledger_pages", [own.parent, again])[1] == own.resolve()


def test_a_missing_settings_folder_is_skipped(tmp_path: Path) -> None:
    own = _schema(tmp_path / "project", "pages.json", "ledger_pages")

    assert resolve_schema("ledger_pages", [tmp_path / "absent", own.parent])[1] == (
        own.resolve()
    )


def test_a_user_schema_found_by_name_needs_the_transcription_fields(
    tmp_path: Path,
) -> None:
    own = _schema(tmp_path / "project", "pages.json", "ledger_pages", valid=False)

    with pytest.raises(ValueError, match="transcription_not_possible") as caught:
        resolve_schema("ledger_pages", [own.parent])

    assert not isinstance(caught.value, AmbiguousNameError)
    assert "ledger_pages" not in [name for name, _ in list_schema_options([own.parent])]


def test_an_unknown_schema_names_the_folders_searched(tmp_path: Path) -> None:
    folder = tmp_path / "project"
    folder.mkdir()

    with pytest.raises(ValueError, match="not found") as caught:
        resolve_schema("missing_schema", [folder])

    assert str(folder) in str(caught.value)
    assert str(SCHEMAS_DIR.resolve()) in str(caught.value)


def test_flag_folders_come_before_the_settings_folders(tmp_path: Path) -> None:
    data = {"schemas_dirs": ["listed"], "classifiers_dirs": ["sorted"]}
    settings = Settings.from_mapping(data, base=tmp_path)
    assert settings.schemas_dirs == (tmp_path / "listed",)

    joined = settings.with_folders((tmp_path / "flag",), (tmp_path / "rules",))

    assert joined.schemas_dirs == (tmp_path / "flag", tmp_path / "listed")
    assert joined.classifiers_dirs == (tmp_path / "rules", tmp_path / "sorted")
    assert settings.with_folders() is settings


def test_classifier_names_take_the_user_folders(tmp_path: Path) -> None:
    folder = tmp_path / "rules"
    _classifier(folder, "front.json", "front_pages")

    assert "front_pages" in classify.classifier_names([folder])
    assert "front_pages" not in classify.classifier_names()


# ============================================================================
# Commands
# ============================================================================
def test_transcribe_finds_a_schema_of_a_flag_folder(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    own = _schema(tmp_path / "project", "pages.json", "ledger_pages")
    pdf = make_scanned_pdf(tmp_path / "in" / "book.pdf", pages=1)
    argv = ["transcribe", "--input", str(pdf), "--output", str(tmp_path / "out")]
    argv += ["--schema", "ledger_pages", "--dry-run"]
    settings = str(_settings_file(tmp_path))

    code, payload, err = _main(
        capsys, *argv, "--schemas-dir", str(own.parent), "--settings", settings
    )

    assert code == 0, err
    assert payload["spec"]["prompt.schema"]["value"] == "ledger_pages"
    assert not any("schemas_dir" in key for key in payload["spec"])
    code, _payload, err = _main(capsys, *argv, "--settings", settings)
    assert code == 2
    assert "ledger_pages" in err


def test_a_name_in_a_flag_and_a_settings_folder_is_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first = _schema(tmp_path / "flag", "pages.json", "ledger_pages")
    second = _schema(tmp_path / "listed", "pages.json", "ledger_pages")
    settings = _settings_file(tmp_path, schemas_dirs=[str(second.parent)])
    pdf = make_scanned_pdf(tmp_path / "in" / "book.pdf", pages=1)

    code, _payload, err = _main(
        capsys,
        "transcribe",
        "--input",
        str(pdf),
        "--schema",
        "ledger_pages",
        "--schemas-dir",
        str(first.parent),
        "--dry-run",
        "--settings",
        str(settings),
    )

    assert code == 2
    assert str(first) in err and str(second) in err


def test_a_missing_flag_folder_is_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pdf = make_scanned_pdf(tmp_path / "in" / "book.pdf", pages=1)

    code, _payload, err = _main(
        capsys,
        "transcribe",
        "--input",
        str(pdf),
        "--schemas-dir",
        str(tmp_path / "absent"),
        "--dry-run",
    )

    assert code == 2
    assert "not a folder" in err


def test_classify_finds_a_classifier_of_the_folders(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    flag = _classifier(tmp_path / "flag", "front.json", "front_pages")
    listed = _classifier(tmp_path / "listed", "other.json", "front_pages")
    pdf = make_scanned_pdf(tmp_path / "in" / "book.pdf", pages=1)
    argv = ["classify", "pages", "--input", str(pdf), "--classifier", "front_pages"]
    argv += ["--dry-run", "--classifiers-dir", str(flag.parent)]

    alone = _settings_file(tmp_path)
    code, payload, err = _main(capsys, *argv, "--settings", str(alone))
    assert code == 0, err
    assert payload["classifier"] == "front_pages"

    both = _settings_file(tmp_path, classifiers_dirs=[str(listed.parent)])
    code, _payload, err = _main(capsys, *argv, "--settings", str(both))
    assert code == 2
    assert str(flag) in err and str(listed) in err
