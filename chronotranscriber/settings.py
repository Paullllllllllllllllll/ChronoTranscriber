"""Typed settings: machine values and optional run defaults.

``load`` reads ``--settings FILE`` when given, else ``config/settings.yaml``
(gitignored), else the packaged ``resources/settings.example.yaml``; without
any file the code values apply. Every key is optional and falls back to the
code value; unknown keys are errors. Relative paths in the file resolve
against the current directory. The ``defaults`` block sets run options by
option name (see :mod:`chronotranscriber.spec`); a flag overrides it, and it
overrides the code default. Further top-level blocks are accepted only when
the caller passes a parser for them; :attr:`Settings.extensions` holds their
parsed values.

The shared blocks (``api_keys``, ``endpoints``, ``timeouts``,
``rate_limits``, ``retry``, ``concurrency``, ``images``, ``prompt_caching``,
``state_dir``, ``logs_dir``) are read by
:mod:`chronotranscriber.common.settings_schema` with this tool's defaults;
``tools``, ``whisper``, ``audio_chunking`` and ``audio_api`` are this tool's
own.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from chronotranscriber import spec as specs
from chronotranscriber.common import settings_schema as schema
from chronotranscriber.common.rate_limit import DEFAULT_RATE_LIMITS
from chronotranscriber.common.settings_loader import SettingsError, load_settings
from chronotranscriber.common.settings_schema import (
    DEFAULT_API_KEYS,
    BlockParser,
    Endpoint,
    Images,
    Reader,
    RetrySettings,
    Timeouts,
    ValueKind,
    check_blocks,
    log_file,
    read_api_keys,
    read_concurrency,
    read_endpoints,
    read_extensions,
    read_images,
    read_logs_dir,
    read_prompt_caching,
    read_rate_limits,
    read_retry,
    read_state_dir,
    read_timeouts,
    state_directory,
)
from chronotranscriber.spec import (
    OPTIONS,
    BatchCancelSpec,
    BatchCheckSpec,
    RepairSpec,
    Resolution,
    RunSpec,
    SpecError,
)

__all__ = [
    "DEFAULT_API_KEYS",
    "DEFAULT_LOGS_DIR",
    "DEFAULT_SETTINGS_PATH",
    "EXAMPLE_SETTINGS_PATH",
    "AudioApi",
    "AudioChunking",
    "BlockParser",
    "Concurrency",
    "Endpoint",
    "Images",
    "PromptCaching",
    "RetrySettings",
    "Settings",
    "SettingsError",
    "SpecError",
    "Timeouts",
    "Tools",
    "Whisper",
    "load",
    "resolve",
    "resolve_batch_cancel",
    "resolve_batch_check",
    "resolve_repair",
]

_PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_SETTINGS_PATH = _PACKAGE_DIR.parent / "config" / "settings.yaml"
EXAMPLE_SETTINGS_PATH = _PACKAGE_DIR / "resources" / "settings.example.yaml"
DEFAULT_LOGS_DIR = _PACKAGE_DIR.parent / "logs"
STATE_DIR_NAME = ".chronotranscriber"
"""The state folder in the home folder when ``state_dir`` is unset."""

RATE_LIMITS: tuple[tuple[int, int], ...] = DEFAULT_RATE_LIMITS
"""The code default of ``rate_limits``: ``[requests, seconds]`` windows."""

CHUNK_FORMATS = ("mp3", "wav16", "copy")
WHISPER_DEVICES = ("auto", "cpu", "cuda")
WHISPER_COMPUTE_TYPES = ("auto", "int8", "int8_float16", "float16", "float32")


@dataclass(frozen=True)
class Concurrency(schema.Concurrency):
    """Parallelism outside the LLM page requests and the delay before requests.

    ``image_processing`` is the number of workers that render and preprocess
    pages for Tesseract; it does not bound the rendering of the page images
    sent to a model. ``audio`` bounds the remote audio chunk requests.
    ``delay_between_tasks`` is the pause in seconds each request worker takes
    before its next request; none by default.
    """

    delay_between_tasks: float = 0.0
    audio: int = 4


_CONCURRENCY_KEYS: tuple[tuple[str, ValueKind], ...] = (
    ("image_processing", "positive"),
    ("audio", "positive"),
    ("delay_between_tasks", "number"),
)


@dataclass(frozen=True)
class PromptCaching(schema.PromptCaching):
    """Provider prompt caching of the static prompt part; on by default."""

    openai_retention: str | None = "24h"


@dataclass(frozen=True)
class Tools:
    """Executables; None finds them on PATH."""

    tesseract_cmd: str | None = None
    ffmpeg_cmd: str | None = None
    ffprobe_cmd: str | None = None


@dataclass(frozen=True)
class Whisper:
    """Local faster-whisper runtime; ``model_path`` overrides the model size.

    ``vad_filter`` drops non-speech segments with the built-in voice-activity
    detector.
    """

    device: str = "auto"
    compute_type: str = "auto"
    model_path: Path | None = None
    download_root: Path | None = None
    beam_size: int = 5
    vad_filter: bool = True


@dataclass(frozen=True)
class AudioApi:
    """Extra fields of OpenAI audio-api requests.

    ``prompt`` is a vocabulary and style hint; ``languages`` and
    ``keywords`` reach only the models that accept them (gpt-transcribe).
    """

    prompt: str = ""
    languages: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class AudioChunking:
    """How long recordings are cut before they are sent.

    ``max_request_bytes`` 0 takes the provider's request limit;
    ``apply_to_local`` chunks local whisper runs too.
    """

    max_request_bytes: int = 0
    overlap_seconds: float = 0.0
    chunk_format: str = "mp3"
    sample_rate: int = 16000
    mono: bool = True
    apply_to_local: bool = False


@dataclass(frozen=True)
class Settings:
    """Machine values, the optional run defaults and any extra blocks.

    *extensions* maps the name of each extra top-level block in the file to
    the value its parser returned.
    """

    api_keys: Mapping[str, str] = field(default_factory=lambda: DEFAULT_API_KEYS)
    endpoints: Mapping[str, Endpoint] = field(
        default_factory=lambda: MappingProxyType({})
    )
    timeouts: Timeouts = field(default_factory=Timeouts)
    rate_limits: tuple[tuple[int, int], ...] = RATE_LIMITS
    retry: RetrySettings = field(default_factory=RetrySettings)
    concurrency: Concurrency = field(default_factory=Concurrency)
    tools: Tools = field(default_factory=Tools)
    whisper: Whisper = field(default_factory=Whisper)
    audio_chunking: AudioChunking = field(default_factory=AudioChunking)
    audio_api: AudioApi = field(default_factory=AudioApi)
    images: Images = field(default_factory=Images)
    prompt_caching: PromptCaching = field(default_factory=PromptCaching)
    logs_dir: Path | None = None
    state_dir: Path | None = None
    defaults: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    source: Path | None = None
    extensions: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def state_directory(self) -> Path:
        """Return the folder of the tool's state files: ``state_dir``, else
        ``~/.chronotranscriber``."""
        return state_directory(self.state_dir, STATE_DIR_NAME)

    def log_file(self) -> Path:
        """Return the log file: ``logs_dir`` names a ``.log`` file or its folder.

        Without ``logs_dir`` the file is ``application.log`` in the
        repository's ``logs`` folder.
        """
        return log_file(self.logs_dir, DEFAULT_LOGS_DIR)

    def api_key_env(self, provider: str, endpoint: str | None = None) -> str:
        """Return the environment variable that holds the key of a provider.

        Raises:
            SettingsError: An unknown endpoint, or a provider without one.
        """
        if endpoint is not None:
            return self.endpoint(endpoint).api_key_env
        try:
            return self.api_keys[provider]
        except KeyError:
            raise SettingsError(
                f"no API key variable for provider {provider!r}"
            ) from None

    def endpoint(self, name: str) -> Endpoint:
        """Return a named endpoint.

        Raises:
            SettingsError: No endpoint of that name.
        """
        try:
            return self.endpoints[name]
        except KeyError:
            known = ", ".join(sorted(self.endpoints)) or "none"
            raise SettingsError(
                f"unknown endpoint {name!r}; the settings define: {known}"
            ) from None

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        defaults: Mapping[str, Any] | None = None,
        source: Path | None = None,
        blocks: Mapping[str, BlockParser] | None = None,
        base: Path | None = None,
    ) -> Settings:
        """Build settings from parsed YAML; missing keys keep code values.

        *blocks* names the extra top-level blocks to accept, each with its
        parser; a block present in *data* is parsed into :attr:`extensions`.
        Relative paths resolve against *base*, by default the current
        directory.

        Raises:
            SettingsError: An unknown key, a value of the wrong type, or a
                block its parser rejects.
            ValueError: A block name is a built-in key.
        """
        blocks = dict(blocks or {})
        check_blocks(_TOP_KEYS, blocks)
        where = str(source) if source is not None else "settings"
        reader = Reader(where, base)
        reader.keys(data, _TOP_KEYS | frozenset(blocks), "")
        return cls(
            api_keys=read_api_keys(reader, data.get("api_keys")),
            endpoints=read_endpoints(reader, data.get("endpoints")),
            timeouts=read_timeouts(reader, data.get("timeouts")),
            rate_limits=read_rate_limits(reader, data.get("rate_limits"), RATE_LIMITS),
            retry=read_retry(reader, data.get("retry"), RetrySettings()),
            concurrency=read_concurrency(
                reader, data.get("concurrency"), Concurrency(), _CONCURRENCY_KEYS
            ),
            tools=_tools(reader, data.get("tools")),
            whisper=_whisper(reader, data.get("whisper")),
            audio_chunking=_audio_chunking(reader, data.get("audio_chunking")),
            audio_api=_audio_api(reader, data.get("audio_api")),
            images=read_images(reader, data.get("images")),
            prompt_caching=read_prompt_caching(
                reader, data.get("prompt_caching"), PromptCaching()
            ),
            logs_dir=read_logs_dir(reader, data),
            state_dir=read_state_dir(reader, data),
            defaults=MappingProxyType(dict(defaults or {})),
            source=source,
            extensions=read_extensions(reader, data, blocks),
        )


_TOP_KEYS = frozenset(
    {
        "api_keys",
        "endpoints",
        "timeouts",
        "rate_limits",
        "retry",
        "concurrency",
        "tools",
        "whisper",
        "audio_chunking",
        "audio_api",
        "images",
        "prompt_caching",
        "logs_dir",
        "state_dir",
    }
)


def _tools(reader: Reader, raw: Any) -> Tools:
    keys = ("tesseract_cmd", "ffmpeg_cmd", "ffprobe_cmd")
    section = reader.section(raw, "tools", frozenset(keys))
    return Tools(*(reader.command(section, key, "tools.") for key in keys))


def _whisper(reader: Reader, raw: Any) -> Whisper:
    keys = frozenset(
        {
            "device",
            "compute_type",
            "model_path",
            "download_root",
            "beam_size",
            "vad_filter",
        }
    )
    section = reader.section(raw, "whisper", keys)
    base = Whisper()
    path = "whisper."
    return Whisper(
        device=reader.choice(section, "device", base.device, path, WHISPER_DEVICES),
        compute_type=reader.choice(
            section, "compute_type", base.compute_type, path, WHISPER_COMPUTE_TYPES
        ),
        model_path=reader.path(section, "model_path", path),
        download_root=reader.path(section, "download_root", path),
        beam_size=reader.positive(section, "beam_size", base.beam_size, path),
        vad_filter=reader.flag(section, "vad_filter", base.vad_filter, path),
    )


def _audio_api(reader: Reader, raw: Any) -> AudioApi:
    keys = frozenset({"prompt", "languages", "keywords"})
    section = reader.section(raw, "audio_api", keys)
    path = "audio_api."
    return AudioApi(
        prompt=reader.text(section, "prompt", "", path),
        languages=_texts(reader, section, "languages", path),
        keywords=_texts(reader, section, "keywords", path),
    )


def _texts(
    reader: Reader, raw: Mapping[str, Any], key: str, path: str
) -> tuple[str, ...]:
    value = raw.get(key)
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise reader.fail(path + key, "must be a list of strings")
    return tuple(item.strip() for item in value if item.strip())


def _audio_chunking(reader: Reader, raw: Any) -> AudioChunking:
    keys = frozenset(
        {
            "max_request_bytes",
            "overlap_seconds",
            "chunk_format",
            "sample_rate",
            "mono",
            "apply_to_local",
        }
    )
    section = reader.section(raw, "audio_chunking", keys)
    base = AudioChunking()
    path = "audio_chunking."
    return AudioChunking(
        max_request_bytes=reader.integer(
            section, "max_request_bytes", base.max_request_bytes, path
        ),
        overlap_seconds=reader.number(
            section, "overlap_seconds", base.overlap_seconds, path
        ),
        chunk_format=reader.choice(
            section, "chunk_format", base.chunk_format, path, CHUNK_FORMATS
        ),
        sample_rate=reader.positive(section, "sample_rate", base.sample_rate, path),
        mono=reader.flag(section, "mono", base.mono, path),
        apply_to_local=reader.flag(
            section, "apply_to_local", base.apply_to_local, path
        ),
    )


def load(
    path: Path | None = None, *, blocks: Mapping[str, BlockParser] | None = None
) -> Settings:
    """Load the settings; an explicit ``path`` (``--settings``) wins.

    Without ``path`` the repository's ``config/settings.yaml`` is read, else
    the packaged example; when neither exists the code values apply.
    *blocks* maps extra top-level block names to their parsers (see
    :meth:`Settings.from_mapping`).

    Raises:
        SettingsError: The file is missing or invalid.
    """
    if path is None and not (
        DEFAULT_SETTINGS_PATH.is_file() or EXAMPLE_SETTINGS_PATH.is_file()
    ):
        return Settings()
    loaded = load_settings(
        DEFAULT_SETTINGS_PATH,
        example=EXAMPLE_SETTINGS_PATH,
        explicit=path,
        options=OPTIONS,
    )
    return Settings.from_mapping(
        loaded.data, defaults=loaded.defaults, source=loaded.path, blocks=blocks
    )


def resolve(settings: Settings, flags: Mapping[str, Any]) -> Resolution[RunSpec]:
    """Return the effective transcribe spec and the source of each value.

    Precedence: flag, then settings default, then code default. A named
    endpoint must exist in the settings.

    Raises:
        SpecError: The options do not form a valid spec.
        SettingsError: A named endpoint is not defined.
    """
    resolution = specs.resolve_spec(flags, settings.defaults)
    _check_endpoint(settings, resolution.spec.model.endpoint)
    return resolution


def resolve_repair(
    settings: Settings, flags: Mapping[str, Any]
) -> Resolution[RepairSpec]:
    """Return the effective repair spec; see :func:`resolve`."""
    resolution = specs.resolve_repair(flags, settings.defaults)
    _check_endpoint(settings, resolution.spec.model.endpoint)
    return resolution


def resolve_batch_check(
    settings: Settings, flags: Mapping[str, Any]
) -> Resolution[BatchCheckSpec]:
    """Return the effective batch check spec; see :func:`resolve`."""
    return specs.resolve_batch_check(flags, settings.defaults)


def resolve_batch_cancel(
    settings: Settings, flags: Mapping[str, Any]
) -> Resolution[BatchCancelSpec]:
    """Return the effective batch cancel spec; no option takes a settings default."""
    return specs.resolve_batch_cancel(flags)


def _check_endpoint(settings: Settings, endpoint: str | None) -> None:
    if endpoint is not None:
        settings.endpoint(endpoint)
