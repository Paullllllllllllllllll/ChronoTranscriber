"""Core entry points: discover, plan with typed errors, run with events."""

from __future__ import annotations

import asyncio
import json
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pytest

import chronotranscriber.run as core
from chronotranscriber import __main__ as entry
from chronotranscriber.common.contract import ExitCode
from chronotranscriber.events import ItemFinished, ItemStarted
from chronotranscriber.output.store import DATABASE_NAME, ItemState
from tests.conftest import make_settings, make_spec
from tests.recording_events import RecordingEvents

MakePdf = Callable[..., Path]
KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY")


@pytest.fixture
def make_pdf(tmp_path: Path) -> MakePdf:
    """Return a factory for PDFs under ``tmp_path``; *text* False scans them."""

    def make(name: str = "doc.pdf", pages: int = 1, text: bool = True) -> Path:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        document = fitz.open()
        for index in range(pages):
            page = document.new_page(width=200, height=300)
            if text:
                page.insert_text((50, 100), f"Page {index + 1}")
        document.save(path)
        document.close()
        return path

    return make


@pytest.fixture
def keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the key variables of the providers."""
    for name in KEYS:
        monkeypatch.setenv(name, "test-key")


def _plan(input_path: Path, **flags: Any) -> core.RunPlan:
    return core.plan(make_spec(input_path, **flags), make_settings())


def _wav(path: Path, seconds: float = 0.5) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * int(16000 * seconds))
    return path


def test_discover_finds_the_files_of_the_type(
    tmp_path: Path, make_pdf: MakePdf
) -> None:
    make_pdf("in/a.pdf")
    make_pdf("in/sub/b.pdf")
    (tmp_path / "in" / "notes.txt").write_text("x", encoding="utf-8")

    flat = core.discover(make_spec(tmp_path / "in", type="pdf", method="native"))
    deep = core.discover(
        make_spec(tmp_path / "in", type="pdf", method="native", recursive=True)
    )

    assert [item.name for item in flat] == ["a.pdf"]
    assert sorted(item.name for item in deep) == ["a.pdf", "b.pdf"]
    assert all(item.type == "pdf" and item.path.is_absolute() for item in deep)


def test_discover_in_auto_mode_types_every_item(
    tmp_path: Path, make_pdf: MakePdf
) -> None:
    make_pdf("in/a.pdf")
    (tmp_path / "in" / "scans").mkdir()
    (tmp_path / "in" / "scans" / "p1.png").write_bytes(b"png")

    items = core.discover(make_spec(tmp_path / "in", method="tesseract"))

    assert [(item.type, item.name) for item in items] == [
        ("pdf", "a.pdf"),
        ("images", "scans"),
    ]


def _missing_input(tmp_path: Path, make_pdf: MakePdf) -> tuple[Path, dict[str, Any]]:
    return tmp_path / "missing.pdf", {"type": "pdf", "method": "native"}


def _empty_folder(tmp_path: Path, make_pdf: MakePdf) -> tuple[Path, dict[str, Any]]:
    (tmp_path / "in").mkdir()
    return tmp_path / "in", {"type": "pdf", "method": "native"}


def _empty_auto_folder(
    tmp_path: Path, make_pdf: MakePdf
) -> tuple[Path, dict[str, Any]]:
    (tmp_path / "in").mkdir()
    return tmp_path / "in", {"method": "tesseract"}


def _images_from_a_file(
    tmp_path: Path, make_pdf: MakePdf
) -> tuple[Path, dict[str, Any]]:
    return make_pdf(), {"type": "images", "method": "tesseract"}


def _output_is_a_file(tmp_path: Path, make_pdf: MakePdf) -> tuple[Path, dict[str, Any]]:
    (tmp_path / "out").write_text("x", encoding="utf-8")
    return make_pdf(), {"type": "pdf", "method": "native", "output": tmp_path / "out"}


def _unknown_schema(tmp_path: Path, make_pdf: MakePdf) -> tuple[Path, dict[str, Any]]:
    return make_pdf(), {"type": "pdf", "schema": "no_such_schema"}


def _missing_context(tmp_path: Path, make_pdf: MakePdf) -> tuple[Path, dict[str, Any]]:
    return make_pdf(), {"type": "pdf", "context": tmp_path / "nope.txt"}


def _missing_prompt(tmp_path: Path, make_pdf: MakePdf) -> tuple[Path, dict[str, Any]]:
    return make_pdf(), {"type": "pdf", "prompt": tmp_path / "nope.txt"}


@pytest.mark.usefixtures("keys")
@pytest.mark.parametrize(
    ("setup", "error", "message"),
    [
        (_missing_input, core.InputError, "Input path does not exist"),
        (_empty_folder, core.NoItemsError, "No items found to process"),
        (
            _empty_auto_folder,
            core.NoItemsError,
            "No PDFs, image folders, EPUB or MOBI files found",
        ),
        (_images_from_a_file, core.InputError, "input must be a directory"),
        (_output_is_a_file, core.InputError, "not a directory"),
        (_unknown_schema, core.ConfigurationError, "'no_such_schema' not found"),
        (_missing_context, core.ConfigurationError, "Context file not found"),
        (_missing_prompt, core.ConfigurationError, "System prompt missing"),
    ],
)
def test_plan_errors_carry_exit_code_2(
    tmp_path: Path,
    make_pdf: MakePdf,
    setup: Callable[[Path, MakePdf], tuple[Path, dict[str, Any]]],
    error: type[core.PlanError],
    message: str,
) -> None:
    input_path, flags = setup(tmp_path, make_pdf)

    with pytest.raises(error) as raised:
        _plan(input_path, **flags)

    assert raised.value.exit_code == ExitCode.USAGE
    assert message in str(raised.value)


def test_two_items_sharing_an_output_beside_the_input_exit_2(
    tmp_path: Path, make_pdf: MakePdf
) -> None:
    pdf = make_pdf("in/book.pdf")
    scans = tmp_path / "in" / "book"
    scans.mkdir()
    (scans / "p1.png").write_bytes(b"png")

    with pytest.raises(core.ConfigurationError) as raised:
        _plan(tmp_path / "in", method="tesseract")

    message = str(raised.value)
    assert raised.value.exit_code == ExitCode.USAGE
    assert str(pdf) in message and str(scans) in message
    assert "--output DIR" in message
    apart = _plan(tmp_path / "in", method="tesseract", output=tmp_path / "out")
    assert len({item.document for item in apart.to_process}) == 2


def test_document_names_differing_only_in_case_share_an_output(
    tmp_path: Path, make_pdf: MakePdf
) -> None:
    pdf = make_pdf("in/Book.pdf")
    scans = tmp_path / "in" / "book"
    scans.mkdir()
    (scans / "p1.png").write_bytes(b"png")

    with pytest.raises(core.ConfigurationError) as raised:
        _plan(tmp_path / "in", method="tesseract")

    assert str(pdf) in str(raised.value) and str(scans) in str(raised.value)


def test_auto_mode_on_a_recording_points_to_the_audio_type(tmp_path: Path) -> None:
    talk = _wav(tmp_path / "talk.wav")

    with pytest.raises(core.NoItemsError) as raised:
        _plan(talk, method="tesseract")

    message = str(raised.value)
    assert str(talk) in message
    assert "--type audio" in message
    assert "directory" not in message


def test_auto_mode_on_an_audio_folder_points_to_the_audio_type(
    tmp_path: Path,
) -> None:
    _wav(tmp_path / "in" / "talk.wav")

    with pytest.raises(core.NoItemsError, match="--type audio"):
        _plan(tmp_path / "in", method="tesseract")


def test_a_dry_run_of_a_recording_routes_it_to_whisper(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    talk = _wav(tmp_path / "talk.wav")
    argv = ["transcribe", "--input", str(talk), "--type", "audio"]

    assert entry.main([*argv, "--dry-run", "--json"]) == 0

    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert [item["method"] for item in summary["to_process"]] == ["whisper"]


KEYED_RUNS: dict[str, tuple[str, dict[str, Any]]] = {
    "llm": ("OPENAI_API_KEY", {"type": "pdf", "method": "llm"}),
    "batch": (
        "ANTHROPIC_API_KEY",
        {"type": "pdf", "batch": True, "model": "claude-sonnet-5"},
    ),
    "audio-api": (
        "GOOGLE_API_KEY",
        {"type": "audio", "method": "audio-api", "provider": "google"},
    ),
}


def _keyed_input(case: str, tmp_path: Path, make_pdf: MakePdf) -> Path:
    if case == "audio-api":
        return _wav(tmp_path / "talk.wav")
    return make_pdf()


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("case", list(KEYED_RUNS))
def test_the_cli_exits_2_on_an_unset_key_variable(
    tmp_path: Path,
    make_pdf: MakePdf,
    capsys: pytest.CaptureFixture[str],
    case: str,
    dry_run: bool,
) -> None:
    variable, flags = KEYED_RUNS[case]
    out = tmp_path / "out"
    spec = make_spec(_keyed_input(case, tmp_path, make_pdf), output=out, **flags)
    from chronotranscriber.spec import to_argv

    argv = ["transcribe", *to_argv(spec), "--json"]
    if dry_run:
        argv.append("--dry-run")

    # A dry run plans without the key and names it in a warning.
    assert entry.main(argv) == (0 if dry_run else 2)

    captured = capsys.readouterr()
    (line,) = captured.out.splitlines()
    assert json.loads(line)["items_total"] == 1
    assert variable in captured.err
    assert not out.exists()


def test_with_the_key_set_the_plan_passes(
    tmp_path: Path, make_pdf: MakePdf, monkeypatch: pytest.MonkeyPatch
) -> None:
    for case, (variable, flags) in KEYED_RUNS.items():
        monkeypatch.setenv(variable, "test-key")
        run_plan = _plan(_keyed_input(case, tmp_path, make_pdf), **flags)
        assert len(run_plan.to_process) == 1


def test_a_named_endpoint_takes_its_own_key_variable(
    make_pdf: MakePdf, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chronotranscriber.settings import Endpoint

    settings = make_settings(
        endpoints={"local": Endpoint("local", "http://127.0.0.1:1/v1", "LOCAL_KEY")}
    )
    spec = make_spec(make_pdf(), type="pdf", endpoint="local", model="m")
    with pytest.raises(core.ConfigurationError, match="LOCAL_KEY"):
        core.plan(spec, settings)
    monkeypatch.setenv("LOCAL_KEY", "k")
    assert core.plan(spec, settings).to_process


def test_auto_mode_needs_a_key_only_for_llm_items(
    tmp_path: Path, make_pdf: MakePdf
) -> None:
    make_pdf("in/born.pdf")

    text_only = _plan(tmp_path / "in", searchable_pdfs="text")

    assert [item.method for item in text_only.to_process] == ["native"]
    with pytest.raises(core.ConfigurationError, match="OPENAI_API_KEY"):
        _plan(tmp_path / "in")


@pytest.mark.usefixtures("keys")
def test_auto_mode_routes_each_item_and_groups_them_by_method(
    tmp_path: Path, make_pdf: MakePdf
) -> None:
    make_pdf("in/born.pdf")
    make_pdf("in/scan.pdf", text=False)
    (tmp_path / "in" / "scans").mkdir()
    (tmp_path / "in" / "scans" / "p1.png").write_bytes(b"png")
    (tmp_path / "in" / "book.epub").write_bytes(b"PK")

    run_plan = _plan(tmp_path / "in", searchable_pdfs="text")

    assert run_plan.auto and not run_plan.batch
    assert [(item.name, item.method) for item in run_plan.to_process] == [
        ("born.pdf", "native"),
        ("book.epub", "native"),
        ("scan.pdf", "llm"),
        ("scans", "llm"),
    ]
    assert run_plan.methods == ("native", "llm")


@pytest.mark.usefixtures("keys")
def test_a_dry_run_plan_classifies_resume_and_writes_nothing(
    tmp_path: Path, make_pdf: MakePdf
) -> None:
    make_pdf("in/a.pdf")
    make_pdf("in/b.pdf")
    out = tmp_path / "out"
    flags = {"type": "pdf", "method": "native", "output": out}
    asyncio.run(core.run(_plan(tmp_path / "in", files=("a.pdf",), **flags)))
    before = sorted(path.name for path in out.rglob("*"))

    run_plan = _plan(tmp_path / "in", dry_run=True, **flags)

    assert [item.name for item in run_plan.skipped] == ["a.pdf"]
    assert [(item.name, item.state) for item in run_plan.to_process] == [
        ("b.pdf", ItemState.NONE.value)
    ]
    assert sorted(path.name for path in out.rglob("*")) == before
    with pytest.raises(ValueError, match="dry-run"):
        asyncio.run(core.run(run_plan))


@pytest.mark.usefixtures("keys")
def test_a_dry_run_counts_the_chunks_of_a_recording(tmp_path: Path) -> None:
    talk = _wav(tmp_path / "in" / "talk.wav")

    run_plan = _plan(talk, type="audio", method="audio-api", dry_run=True)
    real_run = _plan(talk, type="audio", method="audio-api")

    assert [item.chunks for item in run_plan.to_process] == ["1 chunk"]
    assert [item.chunks for item in real_run.to_process] == [None]


def test_a_run_reports_its_items_and_writes_every_text_format(
    tmp_path: Path, make_pdf: MakePdf
) -> None:
    make_pdf("in/a.pdf")
    make_pdf("in/b.pdf", pages=2)
    events = RecordingEvents()
    run_plan = _plan(
        tmp_path / "in",
        type="pdf",
        method="native",
        output=tmp_path / "out",
        formats=("txt", "md"),
    )

    result = asyncio.run(core.run(run_plan, events))

    assert (result.complete, result.failed) == (2, 0)
    started = events.of("item_started")
    finished = events.of("item_finished")
    assert [(e.name, e.index, e.total) for e in started] == [
        ("a.pdf", 1, 2),
        ("b.pdf", 2, 2),
    ]
    assert all(isinstance(e, ItemStarted) for e in started)
    assert all(isinstance(e, ItemFinished) and e.success for e in finished)
    assert sorted(Path(path).suffix for path in result.outputs) == [
        ".md",
        ".md",
        ".sqlite",
        ".txt",
        ".txt",
    ]
    database = str((tmp_path / "out" / DATABASE_NAME).resolve())
    assert result.outputs[-1] == database
    assert result.databases == (database,)
    assert all(Path(path).exists() for path in result.outputs)


def test_a_failing_item_is_reported_and_the_run_continues(
    tmp_path: Path, make_pdf: MakePdf, monkeypatch: pytest.MonkeyPatch
) -> None:
    import chronotranscriber.methods.native as native_module

    make_pdf("in/a.pdf")
    make_pdf("in/b.pdf")
    real = native_module.native_extract_pdf_pages

    def extract(path: Path, page_indices: list[int]) -> dict[int, str]:
        if path.name == "a.pdf":
            raise RuntimeError("broken text layer")
        return real(path, page_indices)

    monkeypatch.setattr(native_module, "native_extract_pdf_pages", extract)
    events = RecordingEvents()
    run_plan = _plan(
        tmp_path / "in", type="pdf", method="native", output=tmp_path / "out"
    )

    result = asyncio.run(core.run(run_plan, events))

    assert [(item.name, item.success) for item in result.items] == [
        ("a.pdf", False),
        ("b.pdf", True),
    ]
    assert result.items[0].error == "broken text layer"
    failed = events.of("item_finished")[0]
    assert (failed.success, failed.error) == (False, "broken text layer")


@pytest.mark.usefixtures("keys")
def test_a_client_that_cannot_be_built_stops_the_run_before_any_write(
    tmp_path: Path, make_pdf: MakePdf, monkeypatch: pytest.MonkeyPatch
) -> None:
    import chronotranscriber.methods.llm as llm_module

    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("model cannot read images")

    monkeypatch.setattr(llm_module, "PageTranscriber", refuse)
    out = tmp_path / "out"
    run_plan = _plan(make_pdf(), type="pdf", output=out)

    with pytest.raises(core.ConfigurationError, match="cannot read images"):
        asyncio.run(core.run(run_plan))

    assert not out.exists()


def test_a_run_with_nothing_to_process_opens_no_client(
    tmp_path: Path, make_pdf: MakePdf, monkeypatch: pytest.MonkeyPatch
) -> None:
    import chronotranscriber.methods.native as native_module

    pdf = make_pdf()
    out = tmp_path / "out"
    flags = {"type": "pdf", "method": "native", "output": out}
    asyncio.run(core.run(_plan(pdf, **flags)))
    (txt,) = out.rglob("*.txt")
    txt.unlink()

    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("no client without items")

    monkeypatch.setattr(native_module, "NativeTranscriber", refuse)
    run_plan = _plan(pdf, **flags)
    assert not run_plan.to_process and run_plan.render_only

    result = asyncio.run(core.run(run_plan))

    assert result.failed == 0
    assert result.rendered and txt.exists()


def test_preprocessed_images_are_removed_after_a_failed_item(
    tmp_path: Path, make_pdf: MakePdf, monkeypatch: pytest.MonkeyPatch
) -> None:
    import chronotranscriber.methods.tesseract as tesseract_module

    monkeypatch.setattr(tesseract_module, "ensure_tesseract_available", lambda: True)

    async def preprocess(self: Any, folder: Path, dpi: int, page_indices: Any) -> Any:
        (folder / "page_0001_tess_preprocessed.png").write_bytes(b"png")
        raise RuntimeError("renderer crashed")

    monkeypatch.setattr(
        tesseract_module.PDFProcessor, "process_images_for_tesseract", preprocess
    )
    out = tmp_path / "out"
    run_plan = _plan(make_pdf(text=False), type="pdf", method="tesseract", output=out)

    result = asyncio.run(core.run(run_plan))

    assert result.failed == 1
    assert not list(out.rglob("preprocessed_images_tesseract"))
