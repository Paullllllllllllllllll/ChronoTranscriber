"""Streaming page-payload producer for the in-memory GPT pipeline.

Renders PDF pages (or loads folder images) one at a time, preprocesses
them fully in memory via :meth:`ImageProcessor.process_pil`, and yields
compact base64 JPEG payloads with reproducibility provenance. No
preprocessed image files are written to disk; peak memory is one raw
page plus the compact payloads currently in flight.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz
from PIL import Image, ImageOps

from modules.config.constants import SUPPORTED_IMAGE_EXTENSIONS
from modules.images.encoding import encode_bytes_to_base64
from modules.images.pipeline import IMAGE_FAILURE_RATE_THRESHOLD, ImageProcessor
from modules.infra.logger import setup_logger
from modules.infra.paths import natural_sort_key

logger = setup_logger(__name__)

_PDF_PAGE_NAME_RE = re.compile(
    r"^page_(\d+)_pre_processed\.(jpg|jpeg|png)$", re.IGNORECASE
)


@dataclass
class PagePayload:
    """A single preprocessed page ready for an LLM vision request.

    ``index`` is the absolute 0-based page/source index, so page numbering
    stays correct under page ranges and resume. ``image_name`` keeps the
    historical ``*_pre_processed.jpg`` naming so old partial JSONLs resume
    cleanly even though no file of that name exists anymore.
    """

    index: int
    image_name: str
    base64: str
    mime_type: str = "image/jpeg"
    sha256: str = ""
    width: int = 0
    height: int = 0
    byte_size: int = 0
    effective_dpi: int | None = None
    source_file: str = ""
    page_index: int | None = None

    def provenance(self) -> dict[str, Any]:
        """Per-page reproducibility record for JSONL persistence."""
        return {
            "sha256": self.sha256,
            "width": self.width,
            "height": self.height,
            "byte_size": self.byte_size,
            "effective_dpi": self.effective_dpi,
        }


def pdf_page_image_name(page_index: int) -> str:
    """Virtual image name for a 0-based PDF page index."""
    return f"page_{page_index + 1:04d}_pre_processed.jpg"


def parse_pdf_page_index(image_name: str) -> int | None:
    """Parse the 0-based PDF page index from a ``page_NNNN_pre_processed``
    image name, or None if the name does not follow that pattern."""
    m = _PDF_PAGE_NAME_RE.match(image_name.strip())
    if m:
        return int(m.group(1)) - 1
    return None


def folder_image_name(source: Path) -> str:
    """Virtual image name for a source folder image.

    Includes the source extension so two files that share a stem but differ by
    extension (e.g. ``scan_001.png`` and ``scan_001.tif``) map to distinct
    virtual names and do not collide in the JSONL dedup (CT-9).
    """
    return f"{source.name}_pre_processed.jpg"


def legacy_folder_image_name(source: Path) -> str:
    """Pre-CT-9 stem-based virtual name, kept for backward resume matching.

    Old partial JSONLs recorded ``{stem}_pre_processed.jpg``; the skip-set logic
    still matches this form so existing runs resume without re-transcribing.
    """
    return f"{source.stem}_pre_processed.jpg"


def list_folder_images(folder: Path) -> list[Path]:
    """List supported images in a folder, sorted as the legacy pipeline did."""
    files = [
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    files.sort(key=lambda p: natural_sort_key(p.name))
    return files


def resolve_image_settings(
    provider: str, model_name: str
) -> tuple[dict[str, Any], str, int, int]:
    """Resolve provider-specific image settings from configuration.

    Returns:
        Tuple of (img_cfg, model_type, target_dpi, max_pixels_per_page).
    """
    from modules.config.capabilities.detection import (
        detect_model_type,
        get_image_config_section_name,
    )
    from modules.config.service import get_config_service

    model_type = detect_model_type(provider, (model_name or "").lower())
    full_cfg = get_config_service().get_image_processing_config()
    img_cfg = full_cfg.get(get_image_config_section_name(model_type), {})
    target_dpi = int(img_cfg.get("target_dpi", 300))
    max_pixels = int(full_cfg.get("max_pixels_per_page", 0))
    return img_cfg, model_type, target_dpi, max_pixels


def compute_pdf_skip_indices(
    jsonl_path: Path, *, exclude_errors: bool = False
) -> set[int]:
    """0-based page indices already transcribed according to the temp JSONL."""
    from modules.batch.jsonl import get_processed_image_names

    skip: set[int] = set()
    for name in get_processed_image_names(jsonl_path, exclude_errors=exclude_errors):
        m = _PDF_PAGE_NAME_RE.match(name)
        if m:
            skip.add(int(m.group(1)) - 1)
    return skip


def compute_folder_skip_names(
    jsonl_path: Path, *, exclude_errors: bool = False
) -> set[str]:
    """Image names already transcribed according to the temp JSONL."""
    from modules.batch.jsonl import get_processed_image_names

    return get_processed_image_names(jsonl_path, exclude_errors=exclude_errors)


def _payload_from_pil(
    img: Image.Image,
    *,
    index: int,
    image_name: str,
    img_cfg: dict[str, Any],
    model_type: str,
    source_file: str,
    page_index: int | None,
    effective_dpi: int | None,
) -> PagePayload:
    """Run the in-memory transform chain and wrap the result in a payload."""
    jpeg_bytes, width, height = ImageProcessor.process_pil(img, img_cfg, model_type)
    return PagePayload(
        index=index,
        image_name=image_name,
        base64=encode_bytes_to_base64(jpeg_bytes),
        sha256=hashlib.sha256(jpeg_bytes).hexdigest(),
        width=width,
        height=height,
        byte_size=len(jpeg_bytes),
        effective_dpi=effective_dpi,
        source_file=source_file,
        page_index=page_index,
    )


def _render_pdf_page_payload(
    doc: fitz.Document,
    pdf_path: Path,
    page_index: int,
    target_dpi: int,
    max_pixels: int,
    img_cfg: dict[str, Any],
    model_type: str,
) -> PagePayload:
    """Render one PDF page and preprocess it in memory (thread worker)."""
    # Lazy import: modules.documents.pdf imports modules.images at module
    # level, so a top-level import here would be circular.
    from modules.documents.pdf import _get_effective_dpi

    page = doc[page_index]
    effective_dpi = _get_effective_dpi(page, target_dpi, max_pixels)
    mat = fitz.Matrix(effective_dpi / 72, effective_dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    pix = None  # free pixmap before preprocessing
    try:
        return _payload_from_pil(
            img,
            index=page_index,
            image_name=pdf_page_image_name(page_index),
            img_cfg=img_cfg,
            model_type=model_type,
            source_file=str(pdf_path),
            page_index=page_index,
            effective_dpi=effective_dpi,
        )
    finally:
        img.close()


def _apply_jpeg_draft(
    img: Image.Image, img_cfg: dict[str, Any], model_type: str
) -> None:
    """Best-effort decode-time downscaling for large JPEG sources."""
    if getattr(img, "format", None) != "JPEG":
        return
    try:
        desired_mode = "L" if img_cfg.get("grayscale_conversion", True) else "RGB"
        detail = ImageProcessor.resolve_detail(img_cfg, model_type).lower()
        w, h = img.size
        if detail == "low":
            max_side = int(img_cfg.get("low_max_side_px", 512))
            if max(w, h) > max_side:
                img.draft(desired_mode, (max_side, max_side))
        elif detail == "original":
            max_side = int(img_cfg.get("original_max_side_px", 6000))
            if max(w, h) > max_side:
                img.draft(desired_mode, (max_side, max_side))
        elif model_type == "anthropic":
            max_side = int(img_cfg.get("high_max_side_px", 1568))
            if max(w, h) > max_side:
                img.draft(desired_mode, (max_side, max_side))
        else:
            box = img_cfg.get("high_target_box", [768, 1536])
            try:
                target_w, target_h = int(box[0]), int(box[1])
            except (TypeError, ValueError, IndexError):
                target_w, target_h = 768, 1536
            if min(target_w / float(w), target_h / float(h)) < 1.0:
                img.draft(desired_mode, (target_w, target_h))
    except (OSError, ValueError) as exc:
        logger.debug("JPEG draft fast-path skipped: %s", exc)


def _load_image_payload(
    image_path: Path,
    index: int,
    img_cfg: dict[str, Any],
    model_type: str,
) -> PagePayload:
    """Load and preprocess one source image in memory (thread worker)."""
    try:
        with Image.open(image_path) as img:
            _apply_jpeg_draft(img, img_cfg, model_type)
            # Honor EXIF orientation so camera JPEGs are not processed sideways
            # (B14). exif_transpose returns a new upright image (or the original).
            oriented = ImageOps.exif_transpose(img) or img
            return _payload_from_pil(
                oriented,
                index=index,
                image_name=folder_image_name(image_path),
                img_cfg=img_cfg,
                model_type=model_type,
                source_file=str(image_path),
                page_index=None,
                effective_dpi=None,
            )
    except OSError:
        if image_path.suffix.lower() not in {".jp2", ".j2k"}:
            raise
        logger.warning(
            "Pillow/openjpeg failed on %s; attempting FFmpeg fallback.",
            image_path.name,
        )
        ffmpeg_img = ImageProcessor._decode_with_ffmpeg(image_path)
        return _payload_from_pil(
            ffmpeg_img,
            index=index,
            image_name=folder_image_name(image_path),
            img_cfg=img_cfg,
            model_type=model_type,
            source_file=str(image_path),
            page_index=None,
            effective_dpi=None,
        )


def _raise_if_failure_rate_excessive(source_name: str, total: int, failed: int) -> None:
    """Raise when too many pages failed, mirroring the legacy disk pipeline.

    Below the excessive-failure threshold, failed pages are skipped rather than
    raised; surface a WARNING so a partial output is never mistaken for a
    complete one (B8).
    """
    if total > 1 and failed >= 2 and (failed / total) >= IMAGE_FAILURE_RATE_THRESHOLD:
        raise RuntimeError(
            f"Page preprocessing failed for {failed}/{total} pages in "
            f"'{source_name}'. Check the source for corruption or unsupported "
            f"formats."
        )
    if failed > 0:
        from modules.ui import print_warning

        logger.warning(
            "Skipped %d of %d page(s) in '%s' that failed to render/preprocess.",
            failed,
            total,
            source_name,
        )
        print_warning(
            f"Skipped {failed} of {total} page(s) in '{source_name}' that failed "
            f"to render/preprocess; the output will be missing those pages."
        )


async def stream_pdf_payloads(
    pdf_path: Path,
    *,
    target_dpi: int,
    img_cfg: dict[str, Any],
    model_type: str,
    max_pixels: int = 0,
    page_indices: list[int] | None = None,
    skip_indices: set[int] | None = None,
) -> AsyncIterator[PagePayload]:
    """Yield preprocessed payloads for the needed PDF pages, one at a time.

    The document is opened once; pages are rendered strictly sequentially
    via ``asyncio.to_thread`` (the fitz document is only ever touched by
    one thread at a time). Resume skip-set and page slice are applied
    BEFORE any rendering happens.
    """
    skip = skip_indices or set()
    failed = 0
    needed: list[int] = []
    with fitz.open(pdf_path) as doc:
        total_pages = doc.page_count
        indices = page_indices if page_indices is not None else list(range(total_pages))
        needed = [i for i in indices if 0 <= i < total_pages and i not in skip]
        for page_index in needed:
            try:
                payload = await asyncio.to_thread(
                    _render_pdf_page_payload,
                    doc,
                    pdf_path,
                    page_index,
                    target_dpi,
                    max_pixels,
                    img_cfg,
                    model_type,
                )
            except Exception as e:
                failed += 1
                logger.error(
                    "Error rendering page %d of %s: %s",
                    page_index + 1,
                    pdf_path.name,
                    e,
                )
                continue
            yield payload
    _raise_if_failure_rate_excessive(pdf_path.name, len(needed), failed)


async def stream_folder_payloads(
    folder: Path,
    *,
    img_cfg: dict[str, Any],
    model_type: str,
    page_indices: list[int] | None = None,
    skip_names: set[str] | None = None,
) -> AsyncIterator[PagePayload]:
    """Yield preprocessed payloads for the needed folder images.

    ``skip_names`` is matched against both the virtual preprocessed name
    and the raw source filename, so JSONLs from old and new runs
    interoperate.
    """
    skip = skip_names or set()
    files = list_folder_images(folder)
    indices = page_indices if page_indices is not None else list(range(len(files)))
    needed: list[tuple[int, Path]] = []
    for i in indices:
        if not (0 <= i < len(files)):
            continue
        src = files[i]
        if (
            folder_image_name(src) in skip
            or legacy_folder_image_name(src) in skip
            or src.name in skip
        ):
            continue
        needed.append((i, src))

    failed = 0
    for index, src in needed:
        try:
            payload = await asyncio.to_thread(
                _load_image_payload, src, index, img_cfg, model_type
            )
        except Exception as e:
            failed += 1
            logger.error("Error preprocessing image %s: %s", src.name, e)
            continue
        yield payload
    _raise_if_failure_rate_excessive(folder.name, len(needed), failed)


def render_single_pdf_page_payload(
    pdf_path: Path,
    page_index: int,
    *,
    target_dpi: int,
    img_cfg: dict[str, Any],
    model_type: str,
    max_pixels: int = 0,
) -> PagePayload:
    """Re-render a single PDF page in memory (used by the repair workflow)."""
    with fitz.open(pdf_path) as doc:
        if not (0 <= page_index < doc.page_count):
            raise IndexError(
                f"Page index {page_index} out of range for {pdf_path.name} "
                f"({doc.page_count} pages)"
            )
        return _render_pdf_page_payload(
            doc, pdf_path, page_index, target_dpi, max_pixels, img_cfg, model_type
        )


def load_image_payload(
    image_path: Path,
    index: int,
    *,
    img_cfg: dict[str, Any],
    model_type: str,
) -> PagePayload:
    """Load and preprocess a single source image (used by the repair workflow)."""
    return _load_image_payload(image_path, index, img_cfg, model_type)
