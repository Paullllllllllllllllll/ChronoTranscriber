"""Round trips of every option table: ``parse(to_argv(spec, defaults)) == spec``.

Seeded random specs of the transcribe, repair, batch check, batch cancel and
render tables, each with a random settings defaults block and
without one. The value pools include values that begin with ``-``, which
``to_argv`` writes as ``--flag=value``.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from chronotranscriber.spec import (
    AUDIO_API_PROVIDERS,
    CONTEXT_NONE,
    IMAGE_DETAILS,
    IMAGE_FORMATS,
    LAYOUTS,
    METHODS_BY_TYPE,
    OCR_PRESETS,
    OPTIONS,
    OUTPUT_FORMATS,
    PROVIDERS,
    REASONING_EFFORTS,
    REPAIR_CAUSES,
    RESPONSE_FORMATS,
    SEARCHABLE_PDF_MODES,
    SERVICE_TIERS,
    VERBOSITIES,
    AudioSpec,
    BatchCancelSpec,
    BatchCheckSpec,
    CommandSpec,
    ImageSpec,
    InputSpec,
    MethodSpec,
    ModelSpec,
    OcrSpec,
    OutputSpec,
    PromptSpec,
    RenderSpec,
    RepairControl,
    RepairSelection,
    RepairSpec,
    RunControl,
    RunSpec,
    parse,
    parse_batch_cancel,
    parse_batch_check,
    parse_render,
    parse_repair,
    to_argv,
)

SEEDS = range(200)
_PATHS = [Path("in.pdf"), Path("a b/c.pdf"), Path("-in.pdf"), Path(".")]
_CONTEXTS: list[Any] = [None, CONTEXT_NONE, "ctx.txt", "-ctx.txt", "fish, game"]
_IMAGES: list[Any] = [None, CONTEXT_NONE, Path("ctx.png"), Path("-ctx.png")]
_FALLBACKS: list[Any] = [None, "ctx.txt", "-ctx.txt", "fish, game"]
_MODELS = ["gpt-5.6-luna", "gpt-transcribe", "claude-x", "gemini-y", "v/m", "-m"]


def _formats(rng: random.Random) -> tuple[str, ...]:
    return tuple(f for f in OUTPUT_FORMATS if rng.random() < 0.6) or ("md",)


def _names(rng: random.Random) -> tuple[str, ...]:
    return rng.choice([(), ("a.pdf",), ("x y.pdf", "-b.pdf"), ("-a",)])


def _model(rng: random.Random, providers: tuple[str, ...]) -> ModelSpec:
    provider = rng.choice(providers)
    return ModelSpec(
        provider=provider,
        model=rng.choice(_MODELS),
        endpoint=rng.choice(["local", "-remote"]) if provider == "custom" else None,
        reasoning_effort=rng.choice(REASONING_EFFORTS),
        verbosity=rng.choice(VERBOSITIES),
        max_output_tokens=rng.choice([None, 1, 4096]),
        temperature=rng.choice([0.0, 0.7, 1.0]),
        top_p=rng.choice([0.0, 1.0]),
        service_tier=rng.choice(SERVICE_TIERS),
        response_format=rng.choice([None, *RESPONSE_FORMATS]),
    )


def _prompt(rng: random.Random) -> PromptSpec:
    return PromptSpec(
        schema=rng.choice(["markdown_transcription_schema", "-s.json"]),
        prompt=rng.choice([None, Path("p.txt"), Path("-p.txt")]),
        context=rng.choice(_CONTEXTS),
        context_image=rng.choice(_IMAGES),
        fallback_context=rng.choice(_FALLBACKS),
    )


def _images(rng: random.Random) -> ImageSpec:
    return ImageSpec(
        dpi=rng.choice([None, "native", 300]),
        detail=rng.choice([None, *IMAGE_DETAILS]),
        format=rng.choice([None, *IMAGE_FORMATS]),
        jpeg_quality=rng.choice([None, 1, 100]),
        grayscale=rng.choice([None, True, False]),
    )


def _flag(rng: random.Random) -> bool:
    return rng.random() < 0.5


def _transcribe(rng: random.Random) -> RunSpec:
    kind = rng.choice(sorted(METHODS_BY_TYPE))
    method = rng.choice(METHODS_BY_TYPE[kind])
    providers = AUDIO_API_PROVIDERS if method == "audio-api" else PROVIDERS
    selection = rng.choice(["none", "files", "select", "all"])
    return RunSpec(
        input=InputSpec(
            path=rng.choice(_PATHS),
            type=kind,
            files=_names(rng) if selection == "files" else (),
            recursive=_flag(rng),
            pages=rng.choice([None, "5", "-7", "3-", "1,3,5-8"]),
            select=rng.choice(["1-3", "-2", "letter"])
            if selection == "select"
            else None,
            all=selection == "all",
        ),
        method=MethodSpec(method, rng.choice(SEARCHABLE_PDF_MODES)),
        output=OutputSpec(
            directory=rng.choice([None, Path("out"), Path("-o")]),
            layout=rng.choice(LAYOUTS),
            formats=_formats(rng),
        ),
        prompt=_prompt(rng),
        model=_model(rng, providers),
        images=_images(rng),
        ocr=OcrSpec(
            config=rng.choice(["--oem 3 --psm 6", "--psm=6", "-l deu", "psm 4"]),
            preprocess=rng.choice(OCR_PRESETS),
        ),
        audio=AudioSpec(
            whisper_model=rng.choice(["large-v3", "-w"]),
            language=rng.choice([None, "de", "-x"]),
            chunk_seconds=rng.choice([30, 600]),
        ),
        run=RunControl(
            batch=_flag(rng),
            sync_fallback=_flag(rng),
            retry_errors=_flag(rng),
            force=_flag(rng),
            concurrency=rng.choice([1, 16]),
            keep_working_files=_flag(rng),
            dry_run=_flag(rng),
            json=_flag(rng),
        ),
    )


def _repair(rng: random.Random) -> RepairSpec:
    causes = tuple(c for c in REPAIR_CAUSES if rng.random() < 0.5)
    return RepairSpec(
        input=rng.choice(_PATHS),
        selection=RepairSelection(
            causes=causes or ("api_error",),
            indices=rng.choice([None, (0,), (3, 4, 5, 9)]),
        ),
        prompt=_prompt(rng),
        model=_model(rng, PROVIDERS),
        images=_images(rng),
        run=RepairControl(_flag(rng), _flag(rng), _flag(rng)),
    )


def _check(rng: random.Random) -> BatchCheckSpec:
    action = rng.choice([None, "adopt", "abandon"])
    return BatchCheckSpec(
        directory=rng.choice(_PATHS),
        batch_ids=_names(rng),
        formats=rng.choice([None, _formats(rng)]),
        json=_flag(rng),
        adopt=rng.choice(["b1", "-b2"]) if action == "adopt" else None,
        abandon=rng.choice(["s1", "-s2"]) if action == "abandon" else None,
    )


def _cancel(rng: random.Random) -> BatchCancelSpec:
    target = rng.choice(["ids", "all", "dir"])
    return BatchCancelSpec(
        batch_ids=rng.choice([("b1",), ("-b1", "b2")]) if target == "ids" else (),
        all=target == "all",
        json=_flag(rng),
        directory=rng.choice(_PATHS) if target == "dir" else None,
    )


def _render(rng: random.Random) -> RenderSpec:
    return RenderSpec(
        input=rng.choice(_PATHS),
        formats=rng.choice([None, _formats(rng)]),
        force=_flag(rng),
        dry_run=_flag(rng),
        json=_flag(rng),
    )


_DEFAULT_VALUES: dict[str, list[Any]] = {
    "type": ["auto", "pdf", "audio"],
    "method": ["llm", "tesseract", "whisper"],
    "recursive": [True, False],
    "searchable_pdfs": list(SEARCHABLE_PDF_MODES),
    "output": ["out", "-o"],
    "layout": list(LAYOUTS),
    "formats": [["md"], list(OUTPUT_FORMATS)],
    "schema": ["plain_text_transcription_schema", "-s.json"],
    "prompt": ["default", "-p.txt"],
    "context": ["auto", "none", "-ctx.txt"],
    "context_image": ["auto", "none"],
    "fallback_context": ["fb.txt", "-fb.txt", "none"],
    "select": ["2-4", "-ab"],
    "all": [True, False],
    "provider": list(PROVIDERS),
    "model": ["gpt-5.6-terra", "claude-x", "gemini-y", "-m"],
    "endpoint": ["local", "-remote"],
    "reasoning_effort": list(REASONING_EFFORTS),
    "verbosity": list(VERBOSITIES),
    "max_output_tokens": [1000],
    "temperature": [0.3],
    "top_p": [0.5],
    "service_tier": list(SERVICE_TIERS),
    "response_format": list(RESPONSE_FORMATS),
    "dpi": ["native", 200],
    "image_detail": list(IMAGE_DETAILS),
    "image_format": list(IMAGE_FORMATS),
    "jpeg_quality": [80],
    "grayscale": [True, False],
    "ocr_config": ["--psm=4", "-l deu"],
    "ocr_preprocess": list(OCR_PRESETS),
    "whisper_model": ["small", "-w"],
    "language": ["auto", "-x"],
    "audio_chunk_seconds": [60],
    "batch": [True, False],
    "sync_fallback": [True, False],
    "retry_errors": [True, False],
    "concurrency": [4],
    "keep_working_files": [True, False],
}

# Settings defaults whose unset value has no spelling, by the field they set.
_UNSPELLABLE = {
    "dpi": ("images", "dpi"),
    "image_detail": ("images", "detail"),
    "image_format": ("images", "format"),
    "jpeg_quality": ("images", "jpeg_quality"),
    "grayscale": ("images", "grayscale"),
    "max_output_tokens": ("model", "max_output_tokens"),
    "response_format": ("model", "response_format"),
}


def _defaults(rng: random.Random, spec: CommandSpec) -> dict[str, Any]:
    """Return a random checked defaults block that *spec* can undo as flags."""
    keys = rng.sample(sorted(_DEFAULT_VALUES), rng.randint(1, 12))
    raw: dict[str, Any] = {}
    for key in keys:
        part, field = _UNSPELLABLE.get(key, ("", ""))
        holder = getattr(spec, part, None)
        if holder is not None and getattr(holder, field) is None:
            continue
        raw[key] = rng.choice(_DEFAULT_VALUES[key])
    if "select" in raw and "all" in raw:
        del raw[rng.choice(["select", "all"])]
    selection = getattr(spec, "input", None)
    if isinstance(selection, InputSpec) and not (
        selection.files or selection.select or selection.all
    ):
        # No flag spells "no selection" over a settings selection.
        raw.pop("select", None)
        raw.pop("all", None)
    return OPTIONS.check_defaults(raw)


type Parser = Callable[[list[str], Mapping[str, Any] | None], CommandSpec]

TABLES: dict[str, tuple[Callable[[random.Random], CommandSpec], Parser]] = {
    "transcribe": (_transcribe, lambda argv, defaults: parse(argv, defaults)),
    "repair": (_repair, lambda argv, defaults: parse_repair(argv, defaults)),
    "batch check": (_check, lambda argv, defaults: parse_batch_check(argv, defaults)),
    "batch cancel": (_cancel, lambda argv, _: parse_batch_cancel(argv)),
    "render": (_render, lambda argv, _: parse_render(argv)),
}


@pytest.mark.parametrize("table", list(TABLES))
def test_every_table_round_trips_with_and_without_settings_defaults(
    table: str,
) -> None:
    generate, parse_table = TABLES[table]
    for seed in SEEDS:
        rng = random.Random(f"{table}-{seed}")
        spec = generate(rng)
        defaults = _defaults(rng, spec)

        assert parse_table(to_argv(spec), None) == spec, seed
        assert parse_table(to_argv(spec, defaults), defaults) == spec, seed


def test_a_value_beginning_with_a_dash_is_joined_to_its_flag() -> None:
    spec = parse(["--input", "x.pdf", "--method", "tesseract", "--ocr-config=--psm=6"])
    assert spec.ocr.config == "--psm=6"
    argv = to_argv(spec)
    assert "--ocr-config=--psm=6" in argv
    assert parse(argv) == spec
    cancel = BatchCancelSpec(("-b1", "b2"), False, False)
    assert to_argv(cancel) == ["--batch-ids=-b1,b2"]
    assert parse_batch_cancel(to_argv(cancel)) == cancel
