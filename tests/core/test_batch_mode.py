"""Batch mode through the core: auto routing, dry runs, submission outcomes.

``chronotranscriber.batch.submit.submit_batch`` is replaced by a stub with
the signature of the real function; each outcome (submitted, refused,
unresolved) and an unbuildable submission drive the LLM method's reaction.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pytest

import chronotranscriber.batch.submit as submit_module
import chronotranscriber.methods.llm as llm_module
import chronotranscriber.run as core
from chronotranscriber import __main__ as entry
from chronotranscriber import ext
from chronotranscriber.common.contract import ExitCode
from chronotranscriber.common.sqlite import read_rows
from chronotranscriber.common.store import SubmissionRejectedError
from chronotranscriber.methods.units import DocumentHandle
from chronotranscriber.output.store import DATABASE_NAME
from chronotranscriber.spec import to_argv
from tests.conftest import make_settings, make_spec
from tests.llm import helpers

MakePdf = Callable[..., Path]
PAGE: dict[str, Any] = {
    "image_analysis": "One line of print.",
    "transcription": "Synchronous page text",
    "no_transcribable_text": False,
    "transcription_not_possible": False,
}


@pytest.fixture
def make_pdf(tmp_path: Path) -> MakePdf:
    """Return a factory for PDFs under ``tmp_path``; *text* False scans them."""

    def make(name: str, pages: int = 1, text: bool = True) -> Path:
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
def mixed_folder(tmp_path: Path, make_pdf: MakePdf) -> Path:
    """Return a folder with a born-digital PDF and a two-page scanned PDF."""
    make_pdf("in/born.pdf")
    make_pdf("in/scan.pdf", pages=2, text=False)
    return tmp_path / "in"


@pytest.fixture(autouse=True)
def openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the key variable of the default provider."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


@dataclass(frozen=True)
class Outcome:
    """A submission outcome as ``submit_batch`` reports it."""

    status: str
    submission_ids: tuple[str, ...] = ()
    detail: str | None = None


class FakeSubmit:
    """Stand in for ``submit_batch``; record each document and page count.

    *outcome* is returned, or raised when it is an exception.
    """

    def __init__(self, outcome: Outcome | Exception | None = None) -> None:
        self.outcome = outcome or Outcome("submitted", ("sub-1",))
        self.items: list[tuple[str, int]] = []

    async def __call__(
        self,
        payloads: Sequence[Any],
        *,
        document: DocumentHandle,
        spec: Any,
        settings: Any,
        schema_path: Path | None,
        context: str | None = None,
        context_image: Path | None = None,
    ) -> Outcome:
        self.items.append((Path(document.name).name, len(payloads)))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome

    def install(self, monkeypatch: pytest.MonkeyPatch) -> FakeSubmit:
        monkeypatch.setattr(submit_module, "submit_batch", self, raising=False)
        return self


def _no_client(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("batch mode without --sync-fallback builds no client")


def _statuses(out: Path) -> list[str]:
    rows = read_rows(out / DATABASE_NAME, "units", order_by=["doc_id", "idx"])
    return [row["status"] for row in rows]


def _scan_spec(make_pdf: MakePdf, out: Path, **flags: Any) -> Any:
    return make_spec(
        make_pdf("in/scan.pdf", pages=2, text=False),
        type="pdf",
        batch=True,
        output=out,
        formats=("txt",),
        **flags,
    )


def test_auto_mode_submits_the_llm_items_and_runs_the_others(
    tmp_path: Path, mixed_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    submitted = FakeSubmit().install(monkeypatch)
    monkeypatch.setattr(llm_module, "PageTranscriber", _no_client)
    out = tmp_path / "out"
    run_plan = core.plan(
        make_spec(mixed_folder, batch=True, searchable_pdfs="text", output=out),
        make_settings(),
    )

    result = asyncio.run(core.run(run_plan))

    assert run_plan.auto and run_plan.batch
    assert [(item.name, item.method) for item in run_plan.to_process] == [
        ("born.pdf", "native"),
        ("scan.pdf", "llm"),
    ]
    assert submitted.items == [("scan", 2)]
    assert [(item.name, item.success) for item in result.items] == [
        ("born.pdf", True),
        ("scan.pdf", True),
    ]
    assert any(Path(path).stem == "born" for path in result.items[0].outputs)
    assert result.items[1].outputs == ()
    assert sorted(_statuses(out)) == ["done", "pending", "pending"]


def test_auto_mode_without_llm_items_does_not_batch(
    mixed_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ext, "batch_allowed", lambda spec, **_: False)
    (mixed_folder / "scan.pdf").unlink()

    run_plan = core.plan(
        make_spec(mixed_folder, batch=True, searchable_pdfs="text"), make_settings()
    )

    assert run_plan.methods == ("native",)
    assert not run_plan.batch


def test_a_refused_batch_in_auto_mode_fails_the_plan(
    mixed_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ext, "batch_allowed", lambda spec, **_: False)

    with pytest.raises(core.ConfigurationError, match="Batch submission") as raised:
        core.plan(
            make_spec(mixed_folder, batch=True, searchable_pdfs="text"),
            make_settings(),
        )

    assert raised.value.exit_code == ExitCode.USAGE
    assert raised.value.items_total == 2


@pytest.mark.parametrize("batch", [True, False])
def test_a_dry_run_reports_batch_mode(
    tmp_path: Path,
    mixed_folder: Path,
    capsys: pytest.CaptureFixture[str],
    batch: bool,
) -> None:
    out = tmp_path / "out"
    spec = make_spec(mixed_folder, batch=batch, searchable_pdfs="text", output=out)

    assert entry.main(["transcribe", *to_argv(spec), "--dry-run"]) == 0

    err = capsys.readouterr().err
    note = "Batch mode: 1 LLM item(s) would be submitted to the openai batch API."
    assert (note in err) is batch
    assert not out.exists()


@pytest.mark.parametrize(
    "outcome",
    [
        Outcome("refused", detail="400 invalid request"),
        submit_module.BatchSubmissionError("unbuildable request"),
    ],
    ids=["refused", "unbuildable"],
)
def test_sync_fallback_transcribes_the_pages_of_a_refused_submission(
    tmp_path: Path,
    make_pdf: MakePdf,
    monkeypatch: pytest.MonkeyPatch,
    outcome: Outcome | Exception,
) -> None:
    submitted = FakeSubmit(outcome).install(monkeypatch)
    script = helpers.install(monkeypatch, [PAGE])
    out = tmp_path / "out"
    spec = _scan_spec(make_pdf, out, sync_fallback=True)
    run_plan = core.plan(spec, make_settings(rate_limits=((100_000, 1),)))

    result = asyncio.run(core.run(run_plan))

    assert submitted.items == [("scan", 2)]
    assert (result.complete, result.failed) == (1, 0)
    assert len(script.requests) == 2
    (item,) = result.items
    (output,) = item.outputs
    assert Path(output).read_text(encoding="utf-8").count("Synchronous page text") == 2
    assert _statuses(out) == ["done", "done"]


class PartlyRefusedSubmit(FakeSubmit):
    """Open a batch for page 1, then refuse the part of page 2 in the store."""

    async def __call__(
        self, payloads: Sequence[Any], *, document: DocumentHandle, **_: Any
    ) -> Outcome:
        self.items.append((Path(document.name).name, len(payloads)))

        async def upload(_submission: Any) -> None:
            return None

        async def create(_submission: Any) -> str:
            return "batch-1"

        async def refuse(_submission: Any) -> str:
            raise SubmissionRejectedError("400 invalid request")

        store = document.store
        accepted = await store.submit(
            provider="openai",
            model=document.model,
            units=[document.key(0)],
            upload=upload,
            create=create,
            part=1,
        )
        with pytest.raises(SubmissionRejectedError):
            await store.submit(
                provider="openai",
                model=document.model,
                units=[document.key(1)],
                upload=upload,
                create=refuse,
                part=2,
            )
        return Outcome("refused", (accepted.submission_id,), "400 invalid request")


def test_sync_fallback_skips_the_pages_of_an_accepted_part(
    tmp_path: Path, make_pdf: MakePdf, monkeypatch: pytest.MonkeyPatch
) -> None:
    submitted = PartlyRefusedSubmit().install(monkeypatch)
    script = helpers.install(monkeypatch, [PAGE])
    out = tmp_path / "out"
    spec = _scan_spec(make_pdf, out, sync_fallback=True)

    asyncio.run(core.run(core.plan(spec, make_settings(rate_limits=((100_000, 1),)))))

    assert submitted.items == [("scan", 2)]
    assert len(script.requests) == 1
    assert _statuses(out) == ["pending", "done"]


def test_without_sync_fallback_a_refused_submission_fails_the_item(
    tmp_path: Path, make_pdf: MakePdf, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeSubmit(Outcome("refused", detail="400 invalid request")).install(monkeypatch)
    monkeypatch.setattr(llm_module, "PageTranscriber", _no_client)
    out = tmp_path / "out"

    result = asyncio.run(
        core.run(core.plan(_scan_spec(make_pdf, out), make_settings()))
    )

    assert [(item.success, item.error) for item in result.items] == [
        (False, "400 invalid request")
    ]
    assert _statuses(out) == ["pending", "pending"]
    assert not list(out.rglob("*.txt"))


def test_an_unresolved_submission_fails_the_item_without_a_fallback(
    tmp_path: Path, make_pdf: MakePdf, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeSubmit(Outcome("unresolved", ("sub-1",), "timeout")).install(monkeypatch)
    script = helpers.install(monkeypatch, [PAGE])
    out = tmp_path / "out"
    spec = _scan_spec(make_pdf, out, sync_fallback=True)

    result = asyncio.run(core.run(core.plan(spec, make_settings())))

    ((success, error),) = [(item.success, item.error) for item in result.items]
    assert not success
    assert error is not None
    assert "unresolved" in error and "chronotranscriber batch check" in error
    assert script.requests == []
    assert _statuses(out) == ["pending", "pending"]
