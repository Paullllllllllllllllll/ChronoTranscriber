"""Question groups and review lines shared by the wizard tasks.

The transcribe and repair tables hold the same model, response and image
options; :class:`ModelQuestions`, :class:`ResponseQuestions` and
:class:`ImageQuestions` ask them for either table. Each returns its
questions for :meth:`StepContext.sequence`. Answers act like command-line
flags over the settings defaults; a reset to an unset code default is
offered only where no settings default applies, since the equivalent
command could not express it. :func:`model_line`, :func:`response_line` and
:func:`images_line` build the review lines and mark settings defaults.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Final

from chronotranscriber.common.options import Source
from chronotranscriber.common.wizard import (
    BACK,
    DONE,
    NO_DEFAULT,
    Choice,
    Line,
    Nav,
    Part,
    State,
    StepContext,
    Validator,
)
from chronotranscriber.llm.capabilities import custom_capabilities
from chronotranscriber.llm.options import constructor_kwargs, model_capabilities
from chronotranscriber.llm.prompts import list_schema_options
from chronotranscriber.providers import implied_provider, split_provider_prefix
from chronotranscriber.settings import Settings, resolve, resolve_repair
from chronotranscriber.sources.images import detect_model_type
from chronotranscriber.spec import (
    AUDIO_API_PROVIDERS,
    CONTEXT_NONE,
    IMAGE_DETAILS,
    IMAGE_FORMATS,
    REASONING_EFFORTS,
    REPAIR_OPTIONS,
    SERVICE_TIERS,
    VERBOSITIES,
    ContextChoice,
    ImageChoice,
    ImageSpec,
    ModelSpec,
    PromptSpec,
    Resolution,
    model_suggestions,
    recommended_images,
)
from chronotranscriber.ui.session import Session, absolute, can_reset

__all__ = [
    "ENDPOINT_PREFIX",
    "IMAGE_OPTIONS",
    "IMAGE_TEXT_INDEX",
    "MODEL_OPTIONS",
    "OTHER",
    "RESPONSE_OPTIONS",
    "TUNING_OPTIONS",
    "ImageQuestions",
    "ModelQuestions",
    "Question",
    "ResponseQuestions",
    "Support",
    "current_model",
    "describe_images",
    "image_family",
    "images_line",
    "is_audio",
    "model_line",
    "provider_key",
    "resolution_of",
    "response_line",
    "source_of",
    "support_of",
]

MODEL_OPTIONS: Final = (
    "provider",
    "model",
    "endpoint",
    "reasoning_effort",
    "verbosity",
    "max_output_tokens",
    "temperature",
    "top_p",
    "service_tier",
)
TUNING_OPTIONS: Final = (
    "verbosity",
    "max_output_tokens",
    "temperature",
    "top_p",
    "service_tier",
)
RESPONSE_OPTIONS: Final = (
    "response_format",
    "schema",
    "prompt",
    "context",
    "fallback_context",
    "context_image",
)
IMAGE_OPTIONS: Final = (
    "dpi",
    "image_detail",
    "image_format",
    "jpeg_quality",
    "grayscale",
)
IMAGE_TEXT_INDEX: Final = {
    "dpi": 0,
    "image_detail": 1,
    "image_format": 2,
    "jpeg_quality": 2,
    "grayscale": 3,
}

OTHER: Final = "<other>"
ENDPOINT_PREFIX: Final = "endpoint:"

_PROVIDER_TITLES = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "google": "Google",
    "openrouter": "OpenRouter",
}
_FORMAT_TEXTS = {
    "schema": "schema (API-enforced)",
    "json": "json (prompted, validated)",
    "text": "text (plain)",
}
_REASONING_KEYS = ("reasoning", "thinking", "effort", "thinking_level", "extra_body")
_NO_ORIGINAL_DETAIL = frozenset({"anthropic", "google"})

Question = Callable[[], Nav | None]
"""One question of a step: DONE, BACK, or None when skipped."""


def resolution_of(state: State, session: Session) -> Resolution[Any]:
    """Resolve the answers of a transcribe or repair state into its spec.

    A missing input path is replaced by the current directory, so the model,
    response and image options resolve before the input step is answered.

    Raises:
        ValueError: The answers do not form a valid spec, or a named
            endpoint is not in the settings.
    """
    flags: dict[str, Any] = state.answers
    if state.value("input") is None:
        flags["input"] = Path()
    if state.table is REPAIR_OPTIONS:
        return resolve_repair(session.settings, flags)
    return resolve(session.settings, flags)


def current_model(state: State, session: Session) -> ModelSpec | None:
    """Return the effective model, or None when the answers do not resolve."""
    try:
        model: ModelSpec = resolution_of(state, session).spec.model
    except ValueError:
        return None
    return model


def source_of(state: State, session: Session, target: str) -> Source | None:
    """Return where the value of spec field *target* comes from."""
    try:
        return resolution_of(state, session).sources.get(target)
    except ValueError:
        return None


def is_audio(state: State) -> bool:
    """Whether the state's method is audio-api."""
    return "method" in state.table and state.value("method") == "audio-api"


@dataclass(frozen=True)
class Support:
    """Which model options the core sends for one model.

    ``schema`` is whether API-enforced schemas are available: the
    endpoint's declaration for a custom endpoint, else the capability table.
    """

    reasoning: bool
    verbosity: bool
    temperature: bool
    top_p: bool
    service_tier: bool
    schema: bool

    @property
    def automatic_format(self) -> str:
        """The response format an unset ``--response-format`` takes."""
        return "schema" if self.schema else "json"


def _sent(model: ModelSpec, settings: Settings, caps: Any) -> dict[str, Any]:
    try:
        return constructor_kwargs(model, settings, caps)
    except ValueError:
        return {"temperature": model.temperature}


def support_of(model: ModelSpec, settings: Settings) -> Support:
    """Return what the core would send for *model*, from its client options.

    Reasoning counts as supported when the effort changes what is sent; an
    unknown custom endpoint is treated as one that declares no schema.
    """
    try:
        caps = model_capabilities(model, settings)
    except ValueError:
        caps = custom_capabilities(
            model.model, supports_vision=True, supports_schema=False
        )
    sent = _sent(model, settings, caps)
    low = _sent(replace(model, reasoning_effort="low"), settings, caps)
    high = _sent(replace(model, reasoning_effort="high"), settings, caps)
    return Support(
        reasoning=any(low.get(key) != high.get(key) for key in _REASONING_KEYS),
        verbosity=sent.get("verbosity") is not None,
        temperature=sent.get("temperature") is not None,
        top_p=sent.get("top_p") is not None,
        service_tier=model.provider == "openai",
        schema=caps.supports_structured_output,
    )


def provider_key(provider: str, endpoint: str | None) -> str:
    """Return the provider menu value: a provider name or ``endpoint:NAME``."""
    if provider == "custom" and endpoint is not None:
        return ENDPOINT_PREFIX + endpoint
    return provider


def _split_key(key: str) -> tuple[str, str | None]:
    if key.startswith(ENDPOINT_PREFIX):
        return "custom", key.removeprefix(ENDPOINT_PREFIX)
    return key, None


def _marker(state: State, source: Source | None) -> str | None:
    return state.settings_marker if source is Source.SETTINGS else None


def _provider_choices(session: Session, audio: bool) -> list[Choice[str]]:
    if audio:
        return [Choice(name, _PROVIDER_TITLES[name]) for name in AUDIO_API_PROVIDERS]
    choices = [Choice(name, title) for name, title in _PROVIDER_TITLES.items()]
    for name, endpoint in session.settings.endpoints.items():
        choices.append(
            Choice(provider_key("custom", name), f"Endpoint {name}", endpoint.base_url)
        )
    return choices


def _model_name(provider: str, audio: bool) -> Validator:
    def validate(text: str) -> str | None:
        if not text.strip():
            return "enter a model name"
        prefixed, name = split_provider_prefix(text.strip())
        if provider == "custom" and prefixed is not None:
            return "an endpoint model name takes no provider prefix"
        implied = prefixed or implied_provider(name)
        if audio and implied is not None and implied not in AUDIO_API_PROVIDERS:
            return f"audio-api runs on {' or '.join(AUDIO_API_PROVIDERS)}"
        return None

    return validate


def _tuning_note(state: State, session: Session) -> str:
    """Describe the tuning values the model takes without tuning answers."""
    kept = {
        name: value
        for name, value in state.answers.items()
        if name not in TUNING_OPTIONS
    }
    trial = State(state.table, state.defaults, kept)
    model = current_model(trial, session)
    if model is None:
        return ""
    support = support_of(model, session.settings)
    marker = state.settings_marker

    def shown(name: str, text: str) -> str:
        return f"{text} {marker}" if trial.from_settings(name) else text

    parts = []
    if support.verbosity:
        parts.append(shown("verbosity", f"verbosity {model.verbosity}"))
    limit = model.max_output_tokens
    parts.append(
        shown(
            "max_output_tokens",
            "model's output limit" if limit is None else f"max {limit} tokens",
        )
    )
    if support.temperature:
        parts.append(shown("temperature", f"temperature {model.temperature}"))
    if support.top_p:
        parts.append(shown("top_p", f"top p {model.top_p}"))
    if support.service_tier:
        parts.append(shown("service_tier", f"tier {model.service_tier}"))
    return ", ".join(parts)


class ModelQuestions:
    """The questions that choose and tune the model.

    Provider (with the settings' named endpoints) and model name are
    committed together; a name that implies another provider switches to
    it. Reasoning effort and the tuning values are asked only where the
    core sends them. Under the audio-api method the providers and models
    are the audio ones and nothing is tuned. *skip* passes over every
    question while it returns True.
    """

    def __init__(
        self,
        ctx: StepContext,
        session: Session,
        skip: Callable[[], bool] | None = None,
    ) -> None:
        self.ctx = ctx
        self.state = ctx.state
        self.session = session
        self.skip = skip or (lambda: False)
        self.key = ""
        self.tuning: str | None = None

    def questions(self) -> list[Question]:
        """Return the questions in order."""
        return [
            self.provider,
            self.name,
            self.reasoning,
            self.tune,
            self.verbosity,
            self.max_tokens,
            self.temperature,
            self.top_p,
            self.service_tier,
        ]

    def _audio(self) -> bool:
        return is_audio(self.state)

    def _model(self) -> ModelSpec | None:
        return current_model(self.state, self.session)

    def _support(self, model: ModelSpec) -> Support:
        return support_of(model, self.session.settings)

    def provider(self) -> Nav | None:
        """Ask the provider or named endpoint."""
        if self.skip():
            return None
        current = self._model()
        key: Any = NO_DEFAULT
        if current is not None:
            key = provider_key(current.provider, current.endpoint)
        source = source_of(self.state, self.session, "model.provider")
        answer = self.ctx.prompter.select(
            "Provider:",
            _provider_choices(self.session, self._audio()),
            default=key,
            marker=_marker(self.state, source),
        )
        if answer is BACK:
            return BACK
        self.key = answer
        return DONE

    def name(self) -> Nav | None:
        """Ask the model: from the suggestions, typed, or at an endpoint."""
        if self.skip():
            return None
        provider, endpoint = _split_key(self.key)
        current = self._model()
        source = source_of(self.state, self.session, "model.model")
        if provider == "custom":
            same = current is not None and (current.provider, current.endpoint) == (
                provider,
                endpoint,
            )
            return self._endpoint_model(endpoint, current if same else None, source)
        return self._listed_model(provider, current, source)

    def _endpoint_model(
        self, endpoint: str | None, same: ModelSpec | None, source: Source | None
    ) -> Nav:
        answer = self.ctx.prompter.text(
            f"Model at endpoint {endpoint}:",
            default=same.model if same is not None else "",
            validate=_model_name("custom", False),
            marker=_marker(self.state, source) if same is not None else None,
        )
        if answer is BACK:
            return BACK
        self._commit(answer)
        return DONE

    def _listed_model(
        self, provider: str, current: ModelSpec | None, source: Source | None
    ) -> Nav:
        audio = self._audio()
        names = list(model_suggestions(provider, audio=audio))
        if (
            current is not None
            and current.provider == provider
            and current.model not in names
        ):
            names.insert(0, current.model)
        choices = [Choice(value, value) for value in names]
        choices.append(Choice(OTHER, "Other (type a name)"))
        while True:
            default = current.model if current is not None else NO_DEFAULT
            answer = self.ctx.prompter.select(
                "Model:",
                choices,
                default=default,
                marker=_marker(self.state, source),
            )
            if answer is BACK:
                return BACK
            if answer != OTHER:
                self._commit(answer)
                return DONE
            typed = self.ctx.prompter.text(
                "Model name:", validate=_model_name(provider, audio)
            )
            if typed is not BACK:
                self._commit(typed)
                return DONE

    def _commit(self, text: str) -> None:
        """Answer the model and its provider; the implied provider wins."""
        state = self.state
        provider, endpoint = _split_key(self.key)
        prefixed, name = split_provider_prefix(text.strip())
        if provider != "custom":
            implied = prefixed or implied_provider(name)
            if implied is not None and implied != provider:
                self.ctx.prompter.show(
                    f"  provider {implied}, as the model name implies"
                )
                provider = implied
        state.set("model", name)
        state.clear("provider")
        state.clear("endpoint")
        model = self._model()
        if model is None or (model.provider, model.endpoint) != (provider, endpoint):
            state.set("provider", provider)
            state.set("endpoint", endpoint)

    def reasoning(self) -> Nav | None:
        """Ask the reasoning effort where it changes the request."""
        current = self._model()
        if self.skip() or self._audio() or current is None:
            return None
        if not self._support(current).reasoning:
            return None
        return self.ctx.select_option(
            "reasoning_effort",
            "Reasoning effort:",
            [Choice(value, value) for value in REASONING_EFFORTS],
        )

    def tune(self) -> Nav | None:
        """Ask whether to keep the recommended tuning values."""
        if self.skip() or self._audio():
            return None
        state = self.state
        customized = any(name in state.answers for name in TUNING_OPTIONS)
        answer = self.ctx.prompter.select(
            "Model parameters:",
            [
                Choice("recommended", "Recommended", _tuning_note(state, self.session)),
                Choice("customize", "Customize"),
            ],
            default="customize" if customized else "recommended",
        )
        if answer is BACK:
            return BACK
        self.tuning = answer
        if answer == "recommended":
            for name in TUNING_OPTIONS:
                state.clear(name)
        return DONE

    def _customizing(self) -> Support | None:
        if self.skip() or self._audio() or self.tuning != "customize":
            return None
        current = self._model()
        return None if current is None else self._support(current)

    def verbosity(self) -> Nav | None:
        """Ask the verbosity where the core sends it."""
        support = self._customizing()
        if support is None or not support.verbosity:
            return None
        return self.ctx.select_option(
            "verbosity", "Verbosity:", [Choice(value, value) for value in VERBOSITIES]
        )

    def max_tokens(self) -> Nav | None:
        """Ask the output limit; empty takes the model's limit when allowed."""
        if self._customizing() is None:
            return None
        if can_reset(self.state, "max_output_tokens"):
            return self.ctx.text_option(
                "max_output_tokens",
                "Max output tokens (empty = the model's limit):",
                empty=None,
            )
        return self.ctx.text_option("max_output_tokens", "Max output tokens:")

    def temperature(self) -> Nav | None:
        """Ask the temperature where the core sends it."""
        support = self._customizing()
        if support is None or not support.temperature:
            return None
        return self.ctx.text_option("temperature", "Temperature (0.0 to 2.0):")

    def top_p(self) -> Nav | None:
        """Ask top p where the core sends it."""
        support = self._customizing()
        if support is None or not support.top_p:
            return None
        return self.ctx.text_option("top_p", "Top p (0.0 to 1.0):")

    def service_tier(self) -> Nav | None:
        """Ask the OpenAI service tier."""
        support = self._customizing()
        if support is None or not support.service_tier:
            return None
        return self.ctx.select_option(
            "service_tier",
            "Service tier:",
            [Choice(value, value) for value in SERVICE_TIERS],
        )


def _existing_file(text: str) -> str | None:
    if not text.strip():
        return "enter a path"
    if not absolute(text).is_file():
        return "no such file"
    return None


def _image_mode(value: ImageChoice) -> str:
    if value is None:
        return "auto"
    if value == CONTEXT_NONE:
        return "none"
    return "file"


def _names_text_file(value: str) -> bool:
    """Whether context *value* is a text file, existing or by its name."""
    text = value.strip()
    if "\n" in text:
        return False
    try:
        if Path(text).expanduser().is_file():
            return True
    except (OSError, ValueError):
        return False
    return Path(text).suffix.lower() == ".txt"


def _context_mode(value: ContextChoice) -> str:
    """Return auto, none, file or text for a ``--context`` value."""
    if value is None:
        return "auto"
    if value == CONTEXT_NONE:
        return "none"
    return "file" if _names_text_file(str(value)) else "text"


def _fallback_mode(value: str | None) -> str:
    """Return none, file or text for a ``--fallback-context`` value."""
    if value is None:
        return "none"
    return "file" if _names_text_file(value) else "text"


def _topics(text: str) -> str | None:
    return None if text.strip() else "enter the topics"


class ResponseQuestions:
    """The questions of the response format, schema, prompt and context.

    The response format offers automatic only without a settings default
    and the schema only where the model supports it; the schema is asked
    unless the effective format is text.
    """

    def __init__(self, ctx: StepContext, session: Session) -> None:
        self.ctx = ctx
        self.state = ctx.state
        self.session = session
        self.chosen: dict[str, str] = {}

    def questions(self) -> list[Question]:
        """Return the questions in order."""
        return [
            self.response_format,
            self.schema,
            self.schema_file,
            self.prompt,
            self.prompt_file,
            *self._text_context(),
            *self._image_context(),
        ]

    def _support(self) -> Support | None:
        model = current_model(self.state, self.session)
        return None if model is None else support_of(model, self.session.settings)

    def response_format(self) -> Nav | None:
        """Ask the response format."""
        support = self._support()
        if support is None:
            return None
        choices: list[Choice[str | None]] = []
        if can_reset(self.state, "response_format"):
            choices.append(
                Choice(None, "Automatic", f"{support.automatic_format} for this model")
            )
        if support.schema:
            choices.append(Choice("schema", "Schema (API-enforced)"))
        choices.append(Choice("json", "JSON (prompted, validated)"))
        choices.append(Choice("text", "Plain text"))
        return self.ctx.select_option("response_format", "Response format:", choices)

    def _plain(self) -> bool:
        chosen: str | None = self.state.value("response_format")
        if chosen is None:
            support = self._support()
            chosen = support.automatic_format if support is not None else None
        return chosen == "text"

    def schema(self) -> Nav | None:
        """Ask the schema: a name from schemas/ or another file."""
        if self._plain():
            return None
        state = self.state
        names = [name for name, _path in list_schema_options()]
        current = state.value("schema")
        if current not in names:
            names.insert(0, current)
        choices = [Choice(name, name) for name in names]
        choices.append(Choice(OTHER, "Other (a schema file)"))
        answer = self.ctx.prompter.select(
            "Schema:", choices, default=current, marker=state.marker("schema")
        )
        if answer is BACK:
            return BACK
        self.chosen["schema"] = answer
        if answer != OTHER:
            state.set("schema", answer)
        return DONE

    def schema_file(self) -> Nav | None:
        """Ask the schema file after Other."""
        if self._plain() or self.chosen.get("schema") != OTHER:
            return None
        answer = self.ctx.prompter.path("Schema file:", validate=_existing_file)
        if answer is BACK:
            return BACK
        self.state.set("schema", str(absolute(answer)))
        return DONE

    def prompt(self) -> Nav:
        """Ask for the default prompt or a prompt file."""
        state = self.state
        current = "default" if state.value("prompt") is None else "file"
        answer = self.ctx.prompter.select(
            "System prompt:",
            [
                Choice("default", "Default", "the prompt of the response format"),
                Choice("file", "A prompt file"),
            ],
            default=current,
            marker=state.marker("prompt"),
        )
        if answer is BACK:
            return BACK
        self.chosen["prompt"] = answer
        if answer == "default":
            state.set("prompt", None)
        return DONE

    def prompt_file(self) -> Nav | None:
        """Ask the prompt file."""
        if self.chosen.get("prompt") != "file":
            return None
        current = self.state.value("prompt")
        answer = self.ctx.prompter.path(
            "Prompt file:",
            default="" if current is None else str(current),
            validate=_existing_file,
            marker=self.state.marker("prompt") if current is not None else None,
        )
        if answer is BACK:
            return BACK
        self.state.set("prompt", absolute(answer))
        return DONE

    def _value_questions(
        self, name: str, mode_of: Callable[[Any], str], key: str, label: str
    ) -> list[Question]:
        """Return the text and file questions of *name* after its mode
        (``self.chosen[key]``) is text or file; *label* opens the prompts."""
        state = self.state

        def text() -> Nav | None:
            if self.chosen.get(key) != "text":
                return None
            current = state.value(name)
            given = current is not None and mode_of(current) == "text"
            answer = self.ctx.prompter.text(
                f"{label} topics:",
                default=str(current) if given else "",
                validate=_topics,
                marker=state.marker(name) if given else None,
            )
            if answer is BACK:
                return BACK
            state.set(name, answer.strip())
            return DONE

        def file() -> Nav | None:
            if self.chosen.get(key) != "file":
                return None
            current = state.value(name)
            given = current is not None and mode_of(current) == "file"
            answer = self.ctx.prompter.path(
                f"{label} file:",
                default=str(current) if given else "",
                validate=_existing_file,
                marker=state.marker(name) if given else None,
            )
            if answer is BACK:
                return BACK
            state.set(name, str(absolute(answer)))
            return DONE

        return [text, file]

    def _text_context(self) -> list[Question]:
        """Return the context questions: automatic (with the fallback for
        items without a sidecar), topics as text, a text file, or none."""
        state = self.state

        def mode() -> Nav:
            answer = self.ctx.prompter.select(
                "Transcription context:",
                [
                    Choice("auto", "Automatic (sidecar)", "the item's sidecar file"),
                    Choice("text", "Topics as text"),
                    Choice("file", "A text file"),
                    Choice("none", "None"),
                ],
                default=_context_mode(state.value("context")),
                marker=state.marker("context"),
            )
            if answer is BACK:
                return BACK
            self.chosen["context"] = answer
            if answer == "auto":
                state.set("context", None)
            elif answer == "none":
                state.set("context", CONTEXT_NONE)
            return DONE

        def fallback() -> Nav | None:
            if self.chosen.get("context") != "auto":
                return None
            answer = self.ctx.prompter.select(
                "Items without a sidecar:",
                [
                    Choice("none", "No context"),
                    Choice("text", "Topics as text"),
                    Choice("file", "A text file"),
                ],
                default=_fallback_mode(state.value("fallback_context")),
                marker=state.marker("fallback_context"),
            )
            if answer is BACK:
                return BACK
            self.chosen["fallback"] = answer
            if answer == "none":
                state.set("fallback_context", None)
            return DONE

        return [
            mode,
            *self._value_questions("context", _context_mode, "context", "Context"),
            fallback,
            *self._value_questions(
                "fallback_context", _fallback_mode, "fallback", "Fallback context"
            ),
        ]

    def _image_context(self) -> list[Question]:
        state = self.state
        name = "context_image"

        def mode() -> Nav:
            answer = self.ctx.prompter.select(
                "Context image:",
                [
                    Choice("auto", "Automatic", "the item's sidecar file"),
                    Choice("none", "None"),
                    Choice("file", "A file"),
                ],
                default=_image_mode(state.value(name)),
                marker=state.marker(name),
            )
            if answer is BACK:
                return BACK
            self.chosen[name] = answer
            if answer == "auto":
                state.set(name, None)
            elif answer == "none":
                state.set(name, CONTEXT_NONE)
            return DONE

        def file() -> Nav | None:
            if self.chosen.get(name) != "file":
                return None
            current = state.value(name)
            given = _image_mode(current) == "file"
            answer = self.ctx.prompter.path(
                "Context image file:",
                default=str(current) if given else "",
                validate=_existing_file,
                marker=state.marker(name) if given else None,
            )
            if answer is BACK:
                return BACK
            state.set(name, absolute(answer))
            return DONE

        return [mode, file]


def image_family(model: ModelSpec | None) -> str:
    """Return the image-settings family of a model, openai when unknown."""
    if model is None:
        return "openai"
    return detect_model_type(model.provider, model.model)


def describe_images(images: ImageSpec) -> list[str]:
    """Return the texts of resolved image settings: DPI, detail, format, color."""
    dpi = "native DPI" if images.dpi == "native" else f"{images.dpi} DPI"
    encoding = "PNG" if images.format == "png" else f"JPEG {images.jpeg_quality}"
    return [
        dpi,
        f"detail {images.detail}",
        encoding,
        "grayscale" if images.grayscale else "color",
    ]


def _image_values(state: State) -> ImageSpec:
    return ImageSpec(*(state.value(name) for name in IMAGE_OPTIONS))


def _images_note(state: State, recommended: ImageSpec) -> str:
    kept = {k: v for k, v in state.answers.items() if k not in IMAGE_OPTIONS}
    trial = State(state.table, state.defaults, kept)
    texts = describe_images(_image_values(trial).over(recommended))
    flagged = {
        IMAGE_TEXT_INDEX[name] for name in IMAGE_OPTIONS if trial.from_settings(name)
    }
    return ", ".join(
        f"{text} {state.settings_marker}" if index in flagged else text
        for index, text in enumerate(texts)
    )


class ImageQuestions:
    """The page image questions: the recommended values or customized ones.

    The recommended values are those of the model's image family, which
    the run applies to every option left unset.
    """

    def __init__(self, ctx: StepContext, session: Session) -> None:
        self.ctx = ctx
        self.state = ctx.state
        self.session = session
        self.mode_answer: str | None = None

    def questions(self) -> list[Question]:
        """Return the questions in order."""
        return [
            self.mode,
            self.dpi,
            self.detail,
            self.image_format,
            self.quality,
            self.grayscale,
        ]

    def _family(self) -> str:
        return image_family(current_model(self.state, self.session))

    def _recommended(self) -> ImageSpec:
        return recommended_images(self._family())

    def _customizing(self) -> bool:
        return self.mode_answer == "customize"

    def _reset_choice(self, name: str, text: str) -> list[Choice[Any]]:
        if can_reset(self.state, name):
            return [Choice(None, f"Recommended ({text})")]
        return []

    def mode(self) -> Nav:
        """Ask whether to keep the recommended values."""
        state = self.state
        customized = any(name in state.answers for name in IMAGE_OPTIONS)
        answer = self.ctx.prompter.select(
            "Page images:",
            [
                Choice(
                    "recommended",
                    f"Recommended for {self._family()} models",
                    _images_note(state, self._recommended()),
                ),
                Choice("customize", "Customize"),
            ],
            default="customize" if customized else "recommended",
        )
        if answer is BACK:
            return BACK
        self.mode_answer = answer
        if answer == "recommended":
            for name in IMAGE_OPTIONS:
                state.clear(name)
        return DONE

    def dpi(self) -> Nav | None:
        """Ask the render resolution."""
        if not self._customizing():
            return None
        if can_reset(self.state, "dpi"):
            return self.ctx.text_option(
                "dpi",
                f"DPI (native or a number, empty = {self._recommended().dpi}):",
                empty=None,
            )
        return self.ctx.text_option("dpi", "DPI (native or a number):")

    def detail(self) -> Nav | None:
        """Ask the image detail the model family accepts."""
        if not self._customizing():
            return None
        family = self._family()
        details = [
            value
            for value in IMAGE_DETAILS
            if value != "original" or family not in _NO_ORIGINAL_DETAIL
        ]
        choices = [
            *self._reset_choice("image_detail", str(self._recommended().detail)),
            *(Choice(value, value) for value in details),
        ]
        return self.ctx.select_option("image_detail", "Image detail:", choices)

    def image_format(self) -> Nav | None:
        """Ask the page image format."""
        if not self._customizing():
            return None
        choices = [
            *self._reset_choice("image_format", str(self._recommended().format)),
            *(Choice(value, value.upper()) for value in IMAGE_FORMATS),
        ]
        return self.ctx.select_option("image_format", "Image format:", choices)

    def quality(self) -> Nav | None:
        """Ask the JPEG quality of JPEG pages."""
        if not self._customizing():
            return None
        recommended = self._recommended()
        if (self.state.value("image_format") or recommended.format) != "jpeg":
            return None
        if can_reset(self.state, "jpeg_quality"):
            return self.ctx.text_option(
                "jpeg_quality",
                f"JPEG quality (1 to 100, empty = {recommended.jpeg_quality}):",
                empty=None,
            )
        return self.ctx.text_option("jpeg_quality", "JPEG quality (1 to 100):")

    def grayscale(self) -> Nav | None:
        """Ask whether pages are converted to grayscale."""
        if not self._customizing():
            return None
        default = "yes" if self._recommended().grayscale else "no"
        choices = [
            *self._reset_choice("grayscale", default),
            Choice(True, "Yes"),
            Choice(False, "No"),
        ]
        return self.ctx.select_option("grayscale", "Grayscale:", choices)


def _marks(sources: Mapping[str, Source], prefix: str) -> Callable[[str, str], Part]:
    def mark(field: str, text: str) -> Part:
        return Part(text, sources.get(prefix + field) is Source.SETTINGS)

    return mark


def model_line(
    model: ModelSpec,
    sources: Mapping[str, Source],
    session: Session,
    *,
    audio: bool = False,
) -> Line:
    """Return the review line of a resolved model; *audio* omits tuning."""
    mark = _marks(sources, "model.")

    def given(field: str) -> bool:
        return sources.get("model." + field, Source.FLAG) is not Source.DEFAULT

    if model.provider == "custom":
        name = f"endpoint {model.endpoint} / {model.model}"
    else:
        name = f"{model.provider} / {model.model}"
    parts = [mark("model", name)]
    if audio:
        return Line("model", parts)
    support = support_of(model, session.settings)
    if support.reasoning:
        parts.append(mark("reasoning_effort", f"reasoning {model.reasoning_effort}"))
    if support.service_tier:
        parts.append(mark("service_tier", f"tier {model.service_tier}"))
    if support.verbosity and given("verbosity"):
        parts.append(mark("verbosity", f"verbosity {model.verbosity}"))
    if model.max_output_tokens is not None:
        parts.append(mark("max_output_tokens", f"max {model.max_output_tokens} tokens"))
    if support.temperature and given("temperature"):
        parts.append(mark("temperature", f"temperature {model.temperature}"))
    if support.top_p and given("top_p"):
        parts.append(mark("top_p", f"top p {model.top_p}"))
    return Line("model", parts)


def _context_text(value: ContextChoice | ImageChoice, label: str) -> str:
    if value is None:
        return f"{label} auto"
    if value == CONTEXT_NONE:
        return f"no {label}"
    if isinstance(value, str) and _context_mode(value) == "text":
        return f"{label} topics: {value}"
    return f"{label} {absolute(value)}"


def _fallback_text(value: str | None) -> str:
    if value is None:
        return "no fallback context"
    if _fallback_mode(value) == "text":
        return f"fallback topics: {value}"
    return f"fallback context {absolute(value)}"


def response_line(
    model: ModelSpec,
    prompt: PromptSpec,
    sources: Mapping[str, Source],
    session: Session,
) -> Line:
    """Return the review line of the response format, schema, prompt, context."""
    model_mark = _marks(sources, "model.")
    mark = _marks(sources, "prompt.")
    support = support_of(model, session.settings)
    chosen = model.response_format
    if chosen is None:
        effective = support.automatic_format
        text = f"automatic ({effective})"
    else:
        effective = chosen
        text = _FORMAT_TEXTS.get(chosen, chosen)
    parts = [model_mark("response_format", text)]
    if effective != "text":
        parts.append(mark("schema", prompt.schema))
    prompt_text = (
        "default prompt" if prompt.prompt is None else f"prompt {prompt.prompt}"
    )
    parts += [
        mark("prompt", prompt_text),
        mark("context", _context_text(prompt.context, "context")),
    ]
    if prompt.context is None:
        parts.append(mark("fallback_context", _fallback_text(prompt.fallback_context)))
    parts.append(
        mark("context_image", _context_text(prompt.context_image, "context image"))
    )
    return Line("response", parts)


def images_line(
    model: ModelSpec | None, images: ImageSpec, sources: Mapping[str, Source]
) -> Line:
    """Return the review line of the page image settings of *model*."""
    family = image_family(model)
    texts = describe_images(images.over(recommended_images(family)))
    flagged = {
        IMAGE_TEXT_INDEX[name]
        for name, field in zip(IMAGE_OPTIONS, fields(ImageSpec), strict=True)
        if sources.get(f"images.{field.name}") is Source.SETTINGS
    }
    parts = [Part(text, index in flagged) for index, text in enumerate(texts)]
    if all(getattr(images, field.name) is None for field in fields(ImageSpec)):
        parts.append(Part(f"recommended for {family}"))
    return Line("images", parts)
