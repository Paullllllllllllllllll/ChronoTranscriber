"""Unified LLM transcriber using LangChain providers.

This module provides a high-level transcription interface that:
- Supports multiple LLM providers (OpenAI, Anthropic, Google, OpenRouter)
- Uses the existing JSON schema system
- Maintains backward compatibility with the existing workflow
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional

from modules.config.config_loader import PROJECT_ROOT
from modules.config.service import get_config_service
from modules.llm.providers import BaseProvider, get_provider
from modules.llm.providers.base import TranscriptionResult
from modules.llm.prompt_utils import render_prompt_with_schema, inject_additional_context

logger = logging.getLogger(__name__)


class LangChainTranscriber:
    """High-level transcriber using LangChain providers.
    
    Drop-in replacement for OpenAITranscriber that supports multiple providers.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        *,
        schema_path: Optional[Path] = None,
        system_prompt_path: Optional[Path] = None,
        additional_context_path: Optional[Path] = None,
        use_hierarchical_context: bool = True,
    ):
        """Initialize the transcriber.
        
        Args:
            api_key: Optional API key (uses environment variable if not provided)
            model: Model name (uses config if not provided)
            provider: Provider name: openai, anthropic, google, openrouter
                     (auto-detected from model if not provided)
            schema_path: Path to JSON schema file
            system_prompt_path: Path to system prompt file
            additional_context_path: Path to additional context file
            use_hierarchical_context: Whether to use file/folder-specific context resolution
        """
        self.use_hierarchical_context = use_hierarchical_context
        config_service = get_config_service()
        
        # Load model config
        mc = config_service.get_model_config()
        tm = mc.get("transcription_model", {})
        
        self.model = model or tm.get("name", "gpt-4o")
        self.provider_name = provider or tm.get("provider")
        
        # Resolve prompt/schema paths
        pcfg = config_service.get_paths_config()
        general = pcfg.get("general", {})
        
        override_prompt = general.get("transcription_prompt_path")
        override_schema = general.get("transcription_schema_path")
        
        self.system_prompt_path = (
            Path(system_prompt_path)
            if system_prompt_path is not None
            else (
                Path(override_prompt)
                if override_prompt
                else (PROJECT_ROOT / "system_prompt" / "system_prompt.txt")
            )
        )
        self.schema_path = (
            Path(schema_path)
            if schema_path is not None
            else (
                Path(override_schema)
                if override_schema
                else (PROJECT_ROOT / "schemas" / "markdown_transcription_schema.json")
            )
        )
        
        # Validate paths
        if not self.system_prompt_path.exists():
            raise FileNotFoundError(f"System prompt missing: {self.system_prompt_path}")
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema file missing: {self.schema_path}")
        
        # Load prompt and schema
        raw_prompt = self.system_prompt_path.read_text(encoding="utf-8").strip()
        
        with self.schema_path.open("r", encoding="utf-8") as sf:
            loaded_schema = json.load(sf)
            self.full_schema_obj = loaded_schema
            
            # Accept wrapper form {name, strict, schema: {...}} or bare schema {...}
            if (
                isinstance(loaded_schema, dict)
                and "schema" in loaded_schema
                and isinstance(loaded_schema["schema"], dict)
            ):
                self.transcription_schema = loaded_schema["schema"]
            else:
                self.transcription_schema = loaded_schema
        
        # Render system prompt with schema (but NOT yet with context)
        self._base_prompt = render_prompt_with_schema(raw_prompt, self.full_schema_obj)
        
        # Inject additional context - use explicit path or hierarchical resolution
        additional_context = None
        if additional_context_path is not None and Path(additional_context_path).exists():
            try:
                additional_context = Path(additional_context_path).read_text(encoding="utf-8").strip()
            except Exception as e:
                logger.warning(f"Failed to load additional context: {e}")
        elif self.use_hierarchical_context:
            # Use hierarchical context resolution (general fallback only at init)
            from modules.llm.context_utils import _resolve_context, _SUFFIX
            context_content, context_path = _resolve_context(_SUFFIX)
            if context_content:
                additional_context = context_content
                logger.debug(f"Using general context from: {context_path}")
        
        # Inject context into prompt (or remove section if empty)
        self.system_prompt_text = inject_additional_context(
            self._base_prompt, additional_context or ""
        )
        
        # Load image processing config for detail level
        ipc = config_service.get_image_processing_config()
        
        # Load OpenAI-specific image_detail parameter
        openai_cfg = ipc.get("api_image_processing", {}) if isinstance(ipc, dict) else {}
        raw_detail = str(openai_cfg.get("llm_detail", "high")).lower().strip()
        
        if raw_detail in ("low", "high"):
            self.image_detail = raw_detail
        elif raw_detail == "auto":
            self.image_detail = "auto"
        else:
            self.image_detail = "auto"
        
        # Load Google-specific media_resolution parameter
        google_cfg = ipc.get("google_image_processing", {}) if isinstance(ipc, dict) else {}
        raw_resolution = str(google_cfg.get("media_resolution", "high")).lower().strip()
        
        if raw_resolution in ("low", "medium", "high", "ultra_high"):
            self.media_resolution = raw_resolution
        elif raw_resolution == "auto":
            self.media_resolution = "auto"
        else:
            self.media_resolution = "high"
        
        # Load max_output_tokens from model config (critical for reasoning models)
        max_tokens = int(
            tm.get("max_output_tokens")
            or tm.get("max_tokens", 20480)
        )
        
        # Load optional sampler parameters
        temperature = float(tm.get("temperature", 0.0))
        top_p = tm.get("top_p")
        frequency_penalty = tm.get("frequency_penalty")
        presence_penalty = tm.get("presence_penalty")
        reasoning_cfg = tm.get("reasoning")
        text_cfg = tm.get("text")
        
        # Load service_tier from concurrency config (synchronous mode)
        try:
            cc = config_service.get_concurrency_config()
            service_tier = (
                (cc.get("concurrency", {}) or {})
                .get("transcription", {})
                .get("service_tier")
            )
        except Exception:
            service_tier = None
        
        # Build kwargs for optional parameters
        provider_kwargs = {}
        if service_tier:
            provider_kwargs["service_tier"] = service_tier
        if top_p is not None:
            provider_kwargs["top_p"] = float(top_p)
        if frequency_penalty is not None:
            provider_kwargs["frequency_penalty"] = float(frequency_penalty)
        if presence_penalty is not None:
            provider_kwargs["presence_penalty"] = float(presence_penalty)
        if reasoning_cfg is not None:
            provider_kwargs["reasoning_config"] = reasoning_cfg
        if text_cfg:
            provider_kwargs["text_config"] = text_cfg
        
        # Create provider instance with full config
        self._provider = get_provider(
            provider=self.provider_name,
            model=self.model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            **provider_kwargs,
        )
        
        logger.info(
            f"LangChainTranscriber initialized: provider={self._provider.provider_name}, "
            f"model={self.model}, max_tokens={max_tokens}"
        )
    
    def update_context(self, context_content: Optional[str]) -> None:
        """Re-inject context into the system prompt.

        Call this before processing each item when per-file context
        resolution is needed.  The base prompt (schema-rendered, pre-context)
        is preserved from __init__ so only the context section changes.

        Parameters
        ----------
        context_content : Optional[str]
            New context text, or None/empty to remove the context section.
        """
        self.system_prompt_text = inject_additional_context(
            self._base_prompt, context_content or ""
        )

    @property
    def provider(self) -> BaseProvider:
        """Get the underlying provider instance."""
        return self._provider
    
    async def transcribe_image(self, image_path: Path) -> Dict[str, Any]:
        """Transcribe an image file.
        
        Args:
            image_path: Path to the image file
        
        Returns:
            Dictionary containing transcription response data
            (compatible with existing workflow expectations)
        """
        result = await self._provider.transcribe_image(
            image_path,
            system_prompt=self.system_prompt_text,
            user_instruction="The image:",
            json_schema=self.full_schema_obj,
            image_detail=self.image_detail,
            media_resolution=self.media_resolution,
        )
        
        # Convert TranscriptionResult to dict format expected by existing code
        return self._result_to_dict(result)
    
    async def transcribe_image_from_base64(
        self,
        image_base64: str,
        mime_type: str,
    ) -> Dict[str, Any]:
        """Transcribe from base64-encoded image data.
        
        Args:
            image_base64: Base64-encoded image data
            mime_type: MIME type of the image
        
        Returns:
            Dictionary containing transcription response data
        """
        result = await self._provider.transcribe_image_from_base64(
            image_base64=image_base64,
            mime_type=mime_type,
            system_prompt=self.system_prompt_text,
            user_instruction="The image:",
            json_schema=self.full_schema_obj,
            image_detail=self.image_detail,
            media_resolution=self.media_resolution,
        )
        
        return self._result_to_dict(result)
    
    def _result_to_dict(self, result: TranscriptionResult) -> Dict[str, Any]:
        """Convert TranscriptionResult to dict format for backward compatibility."""
        # Build response in format expected by existing workflow code
        response = {
            "output_text": result.content,
            "usage": {
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "total_tokens": result.total_tokens,
            },
        }
        
        # Include raw response metadata if available
        if result.raw_response:
            response["metadata"] = result.raw_response
        
        # Include parsed output if available
        if result.parsed_output:
            response["parsed"] = result.parsed_output
        
        # Include error if present
        if result.error:
            response["error"] = result.error
        
        return response
    
    async def close(self) -> None:
        """Clean up resources."""
        await self._provider.close()


@asynccontextmanager
async def open_transcriber(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    *,
    schema_path: Optional[Path] = None,
    system_prompt_path: Optional[Path] = None,
    additional_context_path: Optional[Path] = None,
    use_hierarchical_context: bool = True,
) -> AsyncGenerator[LangChainTranscriber, None]:
    """Context manager for LangChainTranscriber with automatic cleanup.
    
    Drop-in replacement for the old open_transcriber context manager.
    
    Args:
        api_key: Optional API key
        model: Model name
        provider: Provider name (openai, anthropic, google, openrouter)
        schema_path: Path to JSON schema file
        system_prompt_path: Path to system prompt file
        additional_context_path: Path to additional context file
        use_hierarchical_context: Whether to use file/folder-specific context resolution
    
    Yields:
        LangChainTranscriber instance with managed lifecycle
    """
    transcriber = LangChainTranscriber(
        api_key=api_key,
        model=model,
        provider=provider,
        schema_path=schema_path,
        system_prompt_path=system_prompt_path,
        additional_context_path=additional_context_path,
        use_hierarchical_context=use_hierarchical_context,
    )
    try:
        yield transcriber
    finally:
        await transcriber.close()


async def transcribe_image_with_llm(
    image_path: Path,
    transcriber: LangChainTranscriber,
) -> Dict[str, Any]:
    """Convenience helper for transcribing a single image.
    
    Drop-in replacement for transcribe_image_with_openai.
    
    Args:
        image_path: Path to the image file
        transcriber: Initialized LangChainTranscriber instance
    
    Returns:
        Dictionary containing transcription response data
    """
    return await transcriber.transcribe_image(image_path)
