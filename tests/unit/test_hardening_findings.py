"""Focused tests for a batch of transcription-hardening findings.

Covers:

* Item 1 -- ProgressTracker is wired into the image pipeline (page-level
  progress classification and tracker driving).
* Item 2 -- folder-Tesseract preprocessing returns an absolute-order map, and
  ``_absolute_order_index`` honours it as the positional-fallback override.
* Item 3 -- silently dropped Tesseract pages are surfaced via the
  failure-rate guard in the manager's folder path.
* Item 5 -- a cancelled token-budget wait re-raises ``CancelledError`` instead
  of swallowing it into a ``False`` return.
* Item 6 -- EPUB nav/toc/cover filtering is anchored on the basename stem, so
  real chapters that merely contain those tokens survive.
"""

from __future__ import annotations

import asyncio
import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Item 1: ProgressTracker wiring + page classification
# ---------------------------------------------------------------------------


class TestPageResultClassification:
    @pytest.mark.unit
    def test_none_text_is_failure(self) -> None:
        from modules.transcribe.pipeline import _page_result_failed

        assert _page_result_failed(("p", "img.png", None, None, 0)) is True

    @pytest.mark.unit
    def test_error_placeholder_is_failure(self) -> None:
        from modules.transcribe.pipeline import _page_result_failed

        result = ("p", "img.png", "[transcription error: img.png]", None, 0)
        assert _page_result_failed(result) is True

    @pytest.mark.unit
    def test_good_text_is_not_failure(self) -> None:
        from modules.transcribe.pipeline import _page_result_failed

        assert _page_result_failed(("p", "img.png", "real text", None, 0)) is False

    @pytest.mark.unit
    def test_short_or_non_tuple_is_failure(self) -> None:
        from modules.transcribe.pipeline import _page_result_failed

        assert _page_result_failed(("a", "b")) is True
        assert _page_result_failed(None) is True


class TestProgressTrackerWiring:
    @pytest.mark.asyncio
    async def test_tesseract_pass_drives_tracker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A completed page increments the tracker and finalize() is awaited."""
        from modules.transcribe.pipeline import run_transcription_pipeline

        img = tmp_path / "page1.png"
        img.write_bytes(b"")
        jsonl = tmp_path / "temp.jsonl"
        jsonl.write_text("", encoding="utf-8")
        out = tmp_path / "out.txt"

        fake_tracker = MagicMock()
        fake_tracker.increment_completed = AsyncMock()
        fake_tracker.increment_failed = AsyncMock()
        fake_tracker.finalize = AsyncMock()

        async def fake_runner(
            func: Any, args_list: Any, limit: Any, delay: Any, on_result: Any = None
        ) -> list[Any]:
            results = [(str(img), "page1.png", "ok text", None, 0)]
            if on_result is not None:
                for r in results:
                    await on_result(r)
            return results

        monkeypatch.setattr(
            "modules.transcribe.pipeline.perform_ocr", lambda p, c: "ok text"
        )
        with (
            patch(
                "modules.transcribe.pipeline.ProgressTracker",
                return_value=fake_tracker,
            ),
            patch(
                "modules.transcribe.pipeline.run_concurrent_transcription_tasks",
                side_effect=fake_runner,
            ),
            patch(
                "modules.postprocess.writer.postprocess_transcription",
                return_value="final",
            ),
        ):
            await run_transcription_pipeline(
                image_files=[img],
                method="tesseract",
                transcriber=None,
                temp_jsonl_path=jsonl,
                output_txt_path=out,
                source_name="src",
                concurrency_config={"concurrency": {"transcription": {}}},
                image_processing_config={},
                postprocessing_config={},
            )

        fake_tracker.increment_completed.assert_awaited_once()
        fake_tracker.increment_failed.assert_not_awaited()
        fake_tracker.finalize.assert_awaited_once()


# ---------------------------------------------------------------------------
# Item 2: folder-Tesseract absolute-order map + override
# ---------------------------------------------------------------------------


def _run_tess_preprocess(
    tmp_path: Path, filenames: list[str], page_indices: list[int] | None
) -> tuple[list[Path], dict[str, int]]:
    from modules.images.pipeline import ImageProcessor

    source = tmp_path / "src"
    source.mkdir()
    for name in filenames:
        (source / name).write_bytes(b"fake image data")
    out_dir = tmp_path / "pre"

    def fake_pool(func: Any, args_list: Any, processes: Any = None) -> list[str]:
        results = []
        for args in args_list:
            args[1].write_bytes(b"out")
            results.append(str(args[1]))
        return results

    with (
        patch(
            "modules.images.pipeline.run_multiprocessing_tasks",
            side_effect=fake_pool,
        ),
        patch("modules.images.pipeline.get_config_service") as mock_cs,
    ):
        mock_cs.return_value.get_image_processing_config.return_value = {}
        mock_cs.return_value.get_concurrency_config.return_value = {}
        return ImageProcessor.process_and_save_images_for_tesseract(
            source, out_dir, page_indices=page_indices
        )


class TestTesseractOrderMap:
    @pytest.mark.unit
    def test_order_map_uses_absolute_indices_under_page_range(
        self, tmp_path: Path
    ) -> None:
        """With a page subset, each output maps to its FULL-listing index."""
        files, order_map = _run_tess_preprocess(
            tmp_path,
            ["page_1.png", "page_2.png", "page_3.png", "page_4.png"],
            page_indices=[2, 3],
        )
        # Only two files preprocessed, but keyed to absolute indices 2 and 3.
        assert order_map == {
            "page_3_tess_preprocessed.png": 2,
            "page_4_tess_preprocessed.png": 3,
        }
        assert len(files) == 2

    @pytest.mark.unit
    def test_order_map_covers_full_listing_without_range(self, tmp_path: Path) -> None:
        _files, order_map = _run_tess_preprocess(
            tmp_path, ["a.png", "b.png"], page_indices=None
        )
        assert order_map == {
            "a_tess_preprocessed.png": 0,
            "b_tess_preprocessed.png": 1,
        }


class TestAbsoluteOrderIndexOverride:
    @pytest.mark.unit
    def test_override_beats_positional_for_folder_names(self) -> None:
        from modules.transcribe.pipeline import _absolute_order_index

        # Folder-derived names (no page number) in a resumed, reordered subset.
        files = [
            Path("b_tess_preprocessed.png"),
            Path("a_tess_preprocessed.png"),
        ]
        override = {
            "b_tess_preprocessed.png": 5,
            "a_tess_preprocessed.png": 3,
        }
        mapping = _absolute_order_index(files, override)
        # Absolute indices from the override, not positional 0/1.
        assert mapping == {
            "b_tess_preprocessed.png": 5,
            "a_tess_preprocessed.png": 3,
        }

    @pytest.mark.unit
    def test_missing_override_entry_falls_back_to_positional(self) -> None:
        from modules.transcribe.pipeline import _absolute_order_index

        files = [Path("x_tess_preprocessed.png"), Path("y_tess_preprocessed.png")]
        mapping = _absolute_order_index(files, {"x_tess_preprocessed.png": 9})
        assert mapping["x_tess_preprocessed.png"] == 9
        assert mapping["y_tess_preprocessed.png"] == 1  # positional fallback


# ---------------------------------------------------------------------------
# Item 3: dropped Tesseract pages surface via the failure-rate guard
# ---------------------------------------------------------------------------


def _make_folder_tesseract_manager(tmp_path: Path) -> Any:
    from modules.transcribe.manager import TransientFileTracker, WorkflowManager

    mgr = WorkflowManager.__new__(WorkflowManager)
    mgr.resume_mode = "skip"
    mgr.output_mode = "hash"
    mgr.use_input_as_output = False
    mgr.input_root = tmp_path
    mgr.image_output_dir = tmp_path / "out"
    mgr.image_output_dir.mkdir(parents=True, exist_ok=True)
    mgr.concurrency_config = {"concurrency": {"transcription": {}}}
    mgr.image_processing_config = {}
    mgr.postprocessing_config = {}
    mgr.processing_settings = {
        "retain_temporary_jsonl": True,
        "keep_preprocessed_images": True,
    }
    mgr.user_config = SimpleNamespace(
        transcription_method="tesseract",
        additional_context_path=None,
        additional_context_image_path=None,
        use_hierarchical_context=True,
        page_range=None,
        output_format="txt",
        retry_errors=False,
    )
    mgr._transient_tracker = TransientFileTracker()
    mgr._transient_tracker.configure({"retain_temporary_jsonl": True})
    return mgr


class TestDroppedTesseractPagesGuard:
    @pytest.mark.asyncio
    async def test_excessive_drop_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """4 attempted, 1 written -> above the 50% threshold -> raises."""
        import modules.transcribe.manager as mm

        folder = tmp_path / "scans"
        folder.mkdir()
        (folder / "page1.png").write_bytes(b"")

        mgr = _make_folder_tesseract_manager(tmp_path)
        monkeypatch.setattr(mgr, "_ensure_tesseract_available", lambda: True)

        written = [tmp_path / "pre" / "page_1_tess_preprocessed.png"]
        order_map = {f"page_{i}_tess_preprocessed.png": i - 1 for i in range(1, 5)}

        def fake_pre(src: Any, dst: Any, page_indices: Any = None) -> Any:
            return written, order_map

        monkeypatch.setattr(
            mm.ImageProcessor,
            "process_and_save_images_for_tesseract",
            staticmethod(fake_pre),
        )
        called = AsyncMock()
        monkeypatch.setattr(mgr, "_process_images_with_method", called)

        with pytest.raises(RuntimeError):
            await mgr.process_single_image_folder(folder, transcriber=None)
        called.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_minor_drop_warns_and_proceeds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """4 attempted, 3 written -> below threshold -> proceeds with override."""
        import modules.transcribe.manager as mm

        folder = tmp_path / "scans"
        folder.mkdir()
        (folder / "page1.png").write_bytes(b"")

        mgr = _make_folder_tesseract_manager(tmp_path)
        monkeypatch.setattr(mgr, "_ensure_tesseract_available", lambda: True)

        pre = tmp_path / "pre"
        pre.mkdir(exist_ok=True)
        written = [pre / f"page_{i}_tess_preprocessed.png" for i in (1, 2, 3)]
        order_map = {f"page_{i}_tess_preprocessed.png": i - 1 for i in range(1, 5)}

        def fake_pre(src: Any, dst: Any, page_indices: Any = None) -> Any:
            return written, order_map

        monkeypatch.setattr(
            mm.ImageProcessor,
            "process_and_save_images_for_tesseract",
            staticmethod(fake_pre),
        )
        called = AsyncMock()
        monkeypatch.setattr(mgr, "_process_images_with_method", called)

        await mgr.process_single_image_folder(folder, transcriber=None)

        called.assert_awaited_once()
        # The absolute-order override is threaded through to the pipeline.
        assert called.await_args.kwargs["order_override"] == order_map


# ---------------------------------------------------------------------------
# Item 5: cancelled token-budget wait re-raises CancelledError
# ---------------------------------------------------------------------------


def _blocked_tracker() -> MagicMock:
    tracker = MagicMock()
    tracker.is_limit_reached.return_value = True
    tracker.would_block_next_page.return_value = True
    tracker.estimate_exceeds_daily_limit.return_value = False
    tracker.get_stats.return_value = {"tokens_used_today": 10, "daily_limit": 100}
    tracker.get_reset_time.return_value = datetime.datetime.now()
    tracker.get_seconds_until_reset.return_value = 100
    tracker._shared_enabled = False
    return tracker


class TestTokenWaitCancellation:
    @pytest.mark.asyncio
    async def test_cancelled_error_propagates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import modules.infra.token_budget as tb

        monkeypatch.setattr(tb, "get_token_tracker", _blocked_tracker)
        monkeypatch.setattr(tb, "_log_per_key_exhaustion", lambda *a, **k: None)
        monkeypatch.setattr(
            tb.asyncio, "sleep", AsyncMock(side_effect=asyncio.CancelledError)
        )

        with pytest.raises(asyncio.CancelledError):
            await tb.check_and_wait_for_token_limit(
                {"daily_token_limit": {"enabled": True}}
            )

    @pytest.mark.asyncio
    async def test_keyboard_interrupt_returns_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import modules.infra.token_budget as tb

        monkeypatch.setattr(tb, "get_token_tracker", _blocked_tracker)
        monkeypatch.setattr(tb, "_log_per_key_exhaustion", lambda *a, **k: None)
        monkeypatch.setattr(
            tb.asyncio, "sleep", AsyncMock(side_effect=KeyboardInterrupt)
        )

        result = await tb.check_and_wait_for_token_limit(
            {"daily_token_limit": {"enabled": True}}
        )
        assert result is False


# ---------------------------------------------------------------------------
# Item 6: EPUB nav/toc/cover filtering anchored on the basename stem
# ---------------------------------------------------------------------------


class _FakeEpubItem:
    def __init__(self, name: str) -> None:
        self._name = name

    def get_type(self) -> int:
        import ebooklib

        return ebooklib.ITEM_DOCUMENT

    def get_name(self) -> str:
        return self._name


class _FakeEpubBook:
    def __init__(self, names: list[str]) -> None:
        self._items = {f"id{i}": _FakeEpubItem(n) for i, n in enumerate(names)}
        self.spine = [(f"id{i}", "yes") for i in range(len(names))]

    def get_item_with_id(self, idref: str) -> Any:
        return self._items.get(idref)

    def get_items_of_type(self, item_type: int) -> list[Any]:
        return list(self._items.values())


class TestEpubNavFiltering:
    @pytest.mark.unit
    def test_anchored_match_keeps_real_chapters(self) -> None:
        from modules.documents.epub import EPUBProcessor

        book = _FakeEpubBook(
            [
                "nav.xhtml",  # dropped
                "toc.ncx",  # dropped
                "cover.xhtml",  # dropped
                "discovery.xhtml",  # kept (contains 'cover' substring? no; 'toc'? no)
                "navarra.xhtml",  # kept (starts with 'nav' but not anchored token)
                "chapter1.xhtml",  # kept
            ]
        )
        ordered = EPUBProcessor._ordered_documents(book)
        names = [i.get_name() for i in ordered]
        assert names == ["discovery.xhtml", "navarra.xhtml", "chapter1.xhtml"]

    @pytest.mark.unit
    def test_hyphen_and_underscore_variants_dropped(self) -> None:
        from modules.documents.epub import EPUBProcessor

        book = _FakeEpubBook(["nav-1.xhtml", "toc_2.xhtml", "body.xhtml"])
        ordered = EPUBProcessor._ordered_documents(book)
        assert [i.get_name() for i in ordered] == ["body.xhtml"]
