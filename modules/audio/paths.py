"""Output-path resolution for audio items.

Replicates, for a flat audio file, the three-branch layout
``WorkflowManager.process_single_pdf`` derives for a PDF (co-located,
mirrored, hash-suffixed). The naming must stay byte-identical to the PDF
path, because :meth:`modules.transcribe.resume.ResumeChecker._check_output_exists`
re-derives the same names independently: any divergence makes a completed item
look unprocessed and forces a full re-transcription.
"""

from __future__ import annotations

from pathlib import Path

from modules.infra.paths import (
    OutputMode,
    create_safe_directory_name,
    create_safe_filename,
    mirror_output_path,
)


def _relative_key(item: Path, input_root: Path | None) -> str | None:
    """Input-relative hash key for *item*, or None when it does not apply.

    Byte-identical to ``modules.transcribe.manager._relative_key`` (duplicated
    rather than imported: importing the manager from here would be circular
    once the manager wires in the audio workflow). A single-file ``--input``
    makes ``item == input_root`` and degenerates the key to ``"."``, which
    would hash into a hidden ``.-<hash>`` directory (CT-1); fall back to None
    so the directory name comes from the item's own stem.
    """
    if input_root is None:
        return None
    try:
        rel = str(item.relative_to(input_root))
    except ValueError:
        return None
    if rel in (".", ""):
        return None
    return rel


def prepare_audio_output(
    audio_path: Path,
    *,
    output_dir: Path,
    input_paths_is_output_path: bool,
    output_mode: str,
    input_root: Path | None,
    output_format: str = "txt",
) -> tuple[Path, Path, Path]:
    """Resolve the working folder, final transcript, and temp JSONL for a file.

    The three branches mirror ``WorkflowManager.process_single_pdf`` exactly:

    1. ``input_paths_is_output_path``: working files land in a hash-suffixed
       directory NEXT TO the recording, while the final transcript is written
       directly beside it.
    2. ``output_mode == "mirror"`` with an ``input_root``: both the transcript
       and the JSONL land in the mirrored directory, and the transcript is
       named with the real output extension (the mirror branch is the only one
       resume matches on the real extension).
    3. Otherwise: a hash-suffixed directory under *output_dir*, keyed by the
       input-relative path when one applies, holding both files.

    In branches 1 and 3 the transcript name is built with ``.txt``; the output
    writer swaps the suffix to the configured format afterwards, and resume
    replicates that same build-then-swap (CT-8). Passing *output_format*
    therefore only affects the mirror branch.

    Args:
        audio_path: The recording being transcribed.
        output_dir: Configured audio output directory (branches 2 and 3).
        input_paths_is_output_path: Write outputs beside the input.
        output_mode: ``"hash"`` or ``"mirror"`` (see
            :class:`modules.infra.paths.OutputMode`).
        input_root: Root the ``--input`` selection was made from; required for
            mirroring and for input-relative hash keys.
        output_format: Final transcript extension without the dot
            (``txt``/``md``/``json``); used by the mirror branch only.

    Returns:
        ``(parent_folder, output_txt_path, temp_jsonl_path)``. The parent
        folder exists and the JSONL has been touched into existence on return.
    """
    stem = audio_path.stem

    if input_paths_is_output_path:
        parent_folder = audio_path.parent / create_safe_directory_name(stem)
        parent_folder.mkdir(parents=True, exist_ok=True)
        temp_jsonl_path = parent_folder / create_safe_filename(
            stem, ".jsonl", parent_folder
        )
        output_txt_path = audio_path.parent / create_safe_filename(
            stem, ".txt", audio_path.parent
        )
    elif output_mode == OutputMode.MIRROR and input_root is not None:
        parent_folder = mirror_output_path(audio_path.parent, input_root, output_dir)
        parent_folder.mkdir(parents=True, exist_ok=True)
        ext = f".{output_format}"
        output_txt_path = parent_folder / create_safe_filename(stem, ext, parent_folder)
        temp_jsonl_path = parent_folder / create_safe_filename(
            stem, ".jsonl", parent_folder
        )
    else:
        rel_key = _relative_key(audio_path, input_root)
        parent_folder = output_dir / create_safe_directory_name(rel_key or stem)
        parent_folder.mkdir(parents=True, exist_ok=True)
        output_txt_path = parent_folder / create_safe_filename(
            stem, ".txt", parent_folder
        )
        temp_jsonl_path = parent_folder / create_safe_filename(
            stem, ".jsonl", parent_folder
        )

    if not temp_jsonl_path.exists():
        temp_jsonl_path.touch()

    return parent_folder, output_txt_path, temp_jsonl_path


__all__ = ["prepare_audio_output"]
