"""Page images: folder listing, image settings and the payloads of LLM requests.

:func:`resolve_image_settings` builds the image settings of a model from the
recommended values of its provider family (:data:`IMAGE_PROCESSING`, built
from ``spec.IMAGE_PRESETS``), the run's ``ImageSpec`` and the model's
capabilities, with the detail rules of the presets. The page sources of
``common.images`` render and encode the pages; :class:`FolderPages` names
folder images ``<file name>_pre_processed.jpg``, and :func:`stream_pages`
turns their payloads into :class:`PageImage` units, a page that failed to
render included.
"""

from __future__ import annotations

import base64
import logging
import subprocess
import tempfile
from collections.abc import AsyncIterator, Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from chronotranscriber.common import image_presets
from chronotranscriber.common.images import (
    FolderPageSource,
    PageError,
    PagePayload,
    PageSource,
    sequence_number,
    stream_payloads,
)
from chronotranscriber.common.native import model_image_cap, validate_image_settings
from chronotranscriber.llm.capabilities import ORIGINAL_MAX_SIDE, detect_capabilities
from chronotranscriber.llm.media import (
    SUPPORTED_IMAGE_EXTENSIONS,
    SUPPORTED_IMAGE_FORMATS,
)
from chronotranscriber.output.paths import natural_sort_key
from chronotranscriber.sources.audio.ffmpeg import get_ffmpeg_command
from chronotranscriber.spec import IMAGE_PRESETS, ImageSpec

__all__ = [
    "IMAGE_PROCESSING",
    "PAYLOAD_SHA256_KEY",
    "RESAMPLING_ALGORITHM",
    "FolderPages",
    "PageImage",
    "count_images",
    "detect_model_type",
    "encode_image_to_base64",
    "folder_image_name",
    "image_section",
    "list_folder_images",
    "load_image",
    "page_image",
    "resolve_image_settings",
    "stream_pages",
    "with_resampling",
]

logger = logging.getLogger(__name__)

PAYLOAD_SHA256_KEY = "sha256"
"""The provenance key of the SHA-256 of an encoded page."""

RESAMPLING_ALGORITHM = IMAGE_PRESETS.resampling
"""The downscaling filter of page images.

The page sources add it to their own copy of the settings, so it stays out of
the image-settings fingerprint.
"""

_JPEG2000_SUFFIXES = frozenset({".jp2", ".j2k"})

IMAGE_PROCESSING: Mapping[str, Any] = MappingProxyType(
    {
        "render_strategy": IMAGE_PRESETS.render_strategy,
        "max_pixels_per_page": IMAGE_PRESETS.max_pixels_per_page,
        **IMAGE_PRESETS.sections(),
    }
)
"""Recommended image settings: the render strategy, the memory guard of a
numeric render and one section per model family."""


def list_folder_images(folder: Path) -> list[Path]:
    """List the supported images in *folder* in natural filename order."""
    files = [
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    files.sort(key=lambda path: natural_sort_key(path.name))
    return files


def count_images(folder: Path) -> int:
    """Return the number of supported images directly in *folder*."""
    return sum(
        1
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def folder_image_name(source: Path) -> str:
    """Return the page name of a folder image.

    The name keeps the extension, so ``scan_001.png`` and ``scan_001.tif``
    stay distinct pages.
    """
    return f"{source.name}_pre_processed.jpg"


def detect_model_type(provider: str, model_name: str | None = None) -> str:
    """Return the model family whose image settings apply.

    Direct providers decide; OpenRouter and other providers are detected from
    the model name, defaulting to OpenAI.
    """
    return image_presets.detect_model_type(provider, model_name)


def image_section(model_type: str) -> str:
    """Return the name of the settings section of *model_type*."""
    return image_presets.section_name(model_type)


def with_resampling(img_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of *img_cfg* that downscales with the page filter."""
    return image_presets.with_resampling(img_cfg, RESAMPLING_ALGORITHM)


def resolve_image_settings(
    provider: str | None,
    model_name: str | None,
    images: ImageSpec | None = None,
    original_max_side_px: int | None = None,
) -> tuple[dict[str, Any], str, str]:
    """Return the image settings, model family and section of a model.

    The settings are the section's recommended values with the run's values
    applied, the detail the model accepts (``resolved_detail``), the model's
    image caps and the rendering options. OpenRouter takes the OpenAI
    section's detail; a route that sends no detail sizes by the section.
    OpenAI pages at ``original`` detail are limited on the long side to
    *original_max_side_px*, else to the model's default limit, if any.

    Raises:
        ValueError: A setting is invalid, or a native render has no bound.
    """
    provider = (provider or "openai").lower()
    model = model_name or ""
    model_type = detect_model_type(provider, model)
    section = image_section(model_type)
    table = IMAGE_PROCESSING
    rules = IMAGE_PRESETS.detail
    result = image_presets.section_settings(table, section, images, rules, provider)
    validate_image_settings(result, section)
    caps = detect_capabilities(model)
    detail = image_presets.page_detail(provider, model_type, result, caps, rules)
    image_caps = (
        detect_capabilities(model.split("/")[-1])
        if provider == "openrouter" and model_type == "anthropic"
        else caps
    )
    result.update(
        model_name=model,
        provider=provider,
        resolved_detail=detail,
        image_original_patch_cap_30k=(
            caps.image_original_patch_cap_30k and provider == "openai"
        ),
        image_high_res_tier=image_caps.image_high_res_tier,
        max_pixels_per_page=int(table["max_pixels_per_page"]),
        render_strategy=str(table["render_strategy"]),
    )
    if provider == "openai" and detail == "original":
        side = original_max_side_px
        if side is None:
            side = caps.extra(ORIGINAL_MAX_SIDE)
        if side is not None:
            result["original_max_side_px"] = int(side)
    cap = model_image_cap(model_type, model, detail, result)
    result["cap_policy"] = cap.policy if cap else "profile-v1"
    if (
        result["target_dpi"] == "native"
        and not cap
        and result["resize_profile"] == "none"
    ):
        raise ValueError(f"Invalid resize_profile in {section}: native needs a bound")
    return result, model_type, section


@dataclass(frozen=True)
class PageImage:
    """One page as the transcription pipeline records it.

    A page that could not be rendered or encoded carries ``render_error`` and
    no image; it is recorded as a transcription error.
    """

    index: int
    image_name: str
    source_file: str = ""
    page_index: int | None = None
    base64: str = ""
    mime_type: str = "image/jpeg"
    render_error: str | None = None
    image_provenance: Mapping[str, Any] = field(default_factory=dict)

    def provenance(self) -> dict[str, Any]:
        """Per-page reproducibility record for the page's unit."""
        return dict(self.image_provenance)


def page_image(result: PagePayload | PageError, source: PageSource) -> PageImage:
    """Return the page unit of a payload or render error of *source*."""
    if isinstance(result, PagePayload):
        return PageImage(
            index=result.index,
            image_name=result.image_name,
            source_file=result.source_file,
            page_index=result.page_index,
            base64=result.base64,
            mime_type=result.mime_type,
            image_provenance=result.provenance,
        )
    if isinstance(source, FolderPageSource):
        source_file, page_index = str(source.image_paths[result.index]), None
    else:
        source_file, page_index = str(source.source_path), result.index
    return PageImage(
        index=result.index,
        image_name=result.image_name,
        source_file=source_file,
        page_index=page_index,
        render_error=result.error,
        image_provenance={"render_error": result.error},
    )


async def stream_pages(
    source: PageSource, indices: Iterable[int] | None = None
) -> AsyncIterator[PageImage]:
    """Yield the pages of *indices* (all by default) in order, one at a time.

    The source is closed after the last page.

    Raises:
        RuntimeError: Too many pages failed to render (``check_failure_rate``).
    """
    with source:
        async for result in stream_payloads(source, indices):
            yield page_image(result, source)


def _decode_with_ffmpeg(image_path: Path, target: Path) -> None:
    """Decode *image_path* into the image file *target* with FFmpeg.

    Raises:
        OSError: FFmpeg failed or is not installed.
    """
    result = subprocess.run(
        [
            get_ffmpeg_command(),
            "-y",
            "-i",
            str(image_path),
            "-frames:v",
            "1",
            str(target),
        ],
        capture_output=True,
        timeout=120,
    )
    if result.returncode != 0:
        tail = result.stderr.decode(errors="replace")[-300:]
        raise OSError(f"FFmpeg exited with code {result.returncode}: {tail}")


class FolderPages(FolderPageSource):
    """The images of a folder, named ``<file name>_pre_processed.jpg``.

    Images are downscaled with :data:`RESAMPLING_ALGORITHM`. A JPEG 2000 file
    that Pillow cannot decode is decoded with FFmpeg.
    """

    def __init__(
        self,
        folder_path: Path,
        img_cfg: dict[str, Any],
        model_type: str,
        *,
        image_paths: list[Path] | None = None,
        section: str = "",
    ) -> None:
        super().__init__(
            folder_path,
            with_resampling(img_cfg),
            model_type,
            image_paths=(
                list_folder_images(folder_path) if image_paths is None else image_paths
            ),
            section=section,
            payload_sha256_key=PAYLOAD_SHA256_KEY,
        )

    def image_name(self, index: int) -> str:
        """Page name of the image at *index* (0-based)."""
        return folder_image_name(self.image_paths[index])

    def build_payload(self, index: int) -> PagePayload:
        """Load, preprocess and encode the image at *index* (0-based)."""
        try:
            payload = super().build_payload(index)
        except OSError:
            path = self.image_paths[index]
            if path.suffix.lower() not in _JPEG2000_SUFFIXES:
                raise
            logger.warning("Pillow cannot decode %s; decoding it with FFmpeg.", path)
            payload = self._ffmpeg_payload(index)
        return replace(payload, image_name=self.image_name(index))

    def _ffmpeg_payload(self, index: int) -> PagePayload:
        path = self.image_paths[index]
        with tempfile.TemporaryDirectory() as folder:
            decoded = Path(folder) / f"{path.stem}.png"
            _decode_with_ffmpeg(path, decoded)
            payload = FolderPageSource(
                path.parent,
                self.img_cfg,
                self.model_type,
                image_paths=[decoded],
                section=self.section,
                payload_sha256_key=self.payload_sha256_key,
            ).build_payload(0)
        return replace(
            payload, index=index, number=sequence_number(path), source_file=str(path)
        )


def load_image(
    image_path: Path, index: int, img_cfg: dict[str, Any], model_type: str
) -> PageImage:
    """Load and encode one image file as the page at *index*."""
    pages = FolderPages(
        image_path.parent, img_cfg, model_type, image_paths=[image_path]
    )
    return replace(page_image(pages.build_payload(0), pages), index=index)


def encode_image_to_base64(image_path: Path) -> tuple[str, str]:
    """Return the base64 text and MIME type of an image file, unchanged.

    Raises:
        ValueError: The file type is not a supported image format.
    """
    ext = image_path.suffix.lower()
    mime_type = SUPPORTED_IMAGE_FORMATS.get(ext)
    if not mime_type:
        raise ValueError(f"Unsupported image format: {ext}")
    return base64.b64encode(image_path.read_bytes()).decode("utf-8"), mime_type
