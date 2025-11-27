# ChronoTranscriber Version 3.0 Release Notes

Release Date: November 2025

ChronoTranscriber 3.0 introduces a multi-provider LLM architecture powered by LangChain, automatic transcription method selection, EPUB document support, cost analysis utilities, and comprehensive token budget management. This major release transforms the application from an OpenAI-focused tool to a unified platform supporting four AI providers.

## Highlights

**Multi-Provider LLM Support**
Select from OpenAI, Anthropic, Google, or OpenRouter with automatic capability detection and parameter filtering. LangChain handles retry logic, token tracking, and structured output parsing across all providers.

**Auto Mode**
Automatically determine the optimal transcription method (native text extraction, Tesseract OCR, or GPT) based on file type, text searchability, and content characteristics. Process mixed directories without manual configuration.

**EPUB Support**
Natively extract structured text from EPUB ebooks without OCR. Preserves chapter structure and metadata for EPUB 2.0 and 3.0 formats.

**Cost Analysis and Token Budget**
Track API usage costs across all providers with daily token budget enforcement. View detailed cost breakdowns and historical usage through the new cost analysis utility.

**Additional Context Injection**
Provide domain-specific context to improve transcription quality for specialized documents. Context files are automatically injected into prompts for both synchronous and batch processing.

## What's New

### Multi-Provider Architecture

LangChain-based provider system supporting four AI providers through a unified interface:

**Supported Providers**
- **OpenAI**: GPT-5.1, GPT-5, o-series, GPT-4.1, GPT-4o models
- **Anthropic**: Claude 4.5, Claude 4.1, Claude 4, Claude 3.5 models
- **Google**: Gemini 3, Gemini 2.5, Gemini 2.0, Gemini 1.5 models
- **OpenRouter**: 200+ models from multiple providers

**Automatic Capability Detection**
- Reasoning models (GPT-5, o-series, Claude 4.5, Gemini 3) have temperature and sampling parameters automatically disabled
- Vision capability verification before image processing
- Model-specific parameter filtering via LangChain's `disabled_params`

**Configuration**
Set provider in `config/model_config.yaml`:
```yaml
model: gpt-4o  # or claude-sonnet-4-5, gemini-2.5-pro, etc.
provider: openai  # openai, anthropic, google, openrouter (auto-detected if omitted)
```

Environment variables per provider:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`
- `OPENROUTER_API_KEY`

### Auto Mode

Intelligent automatic method selection based on document characteristics:

**Features**
- Scans directories for supported file types (PDF, images, EPUB)
- Detects searchable PDFs and extracts text natively when possible
- Falls back to Tesseract OCR for scanned documents without API costs
- Uses GPT for high-quality transcription when configured
- Configurable OCR forcing for searchable PDFs with poor native extraction

**Configuration**
```yaml
# config/paths_config.yaml
auto_mode:
  force_ocr_for_searchable_pdfs: false  # Set true to OCR searchable PDFs
```

**Usage**
```bash
# Interactive mode
python main/unified_transcriber.py
# Select "auto" when prompted for method

# CLI mode
python main/unified_transcriber.py --input ./documents --output ./results --method auto
```

### EPUB Support

Native text extraction from EPUB ebooks:

**Features**
- Supports EPUB 2.0 and EPUB 3.0 formats
- Extracts text from all chapters preserving structure
- No OCR or API calls required
- Outputs structured JSON with chapter organization

**Usage**
EPUBs are automatically detected in auto mode or can be processed directly:
```bash
python main/unified_transcriber.py --input ./ebooks --output ./results --type epubs --method native
```

### Cost Analysis Utility

New script for tracking API usage and costs:

**Features**
- View cost breakdown by provider, model, and time period
- Track token usage (input and output tokens)
- Daily and cumulative cost summaries
- Export cost reports

**Usage**
```bash
python main/cost_analysis.py
```

### Token Budget Management

Configurable daily token limits with automatic enforcement:

**Features**
- Set daily token budgets per provider or globally
- Automatic rejection of requests exceeding budget
- Token state persistence across sessions
- Clear budget notifications in UI

**Configuration**
```yaml
# config/concurrency_config.yaml
token_budget:
  enabled: true
  daily_limit: 1000000  # tokens per day
```

### Additional Context Injection

Provide domain-specific context to improve transcription accuracy:

**Features**
- Context files in `additional_context/` directory
- Automatically injected into prompts for GPT transcription
- Supports both synchronous and batch processing
- Selectable via interactive prompts or CLI

**Usage**
1. Create context file: `additional_context/my_context.txt`
2. Select context during transcription or use CLI:
```bash
python main/unified_transcriber.py --input ./docs --output ./results --method gpt --context my_context.txt
```

### Custom Transcription Schemas

New schema for Swiss address book transcription:

**Available Schemas**
- `markdown_transcription_schema.json`: Rich markdown formatting
- `plain_text_transcription_schema.json`: Simple text output
- `swiss_address_book_schema.json`: Structured address book extraction

Select schemas interactively or via `--schema` CLI argument.

## Dependency Updates

### LangChain Stack (New)

| Package | Version | Purpose |
|---------|---------|---------|
| langchain | 1.1.0 | Multi-provider LLM integration |
| langchain-core | 1.1.0 | Core LangChain functionality |
| langchain-openai | 1.1.0 | OpenAI provider |
| langchain-anthropic | 1.2.0 | Anthropic provider |
| langchain-google-genai | 3.2.0 | Google Gemini provider |
| anthropic | 0.75.0 | Anthropic SDK |
| tiktoken | 0.12.0 | Token counting |
| langgraph | 1.0.4 | LangChain graph support |

### Key Package Updates

| Package | Previous | Updated |
|---------|----------|---------|
| aiofiles | 24.1.0 | 25.1.0 |
| aiohttp | 3.12.15 | 3.13.2 |
| openai | 2.1.0 | 2.8.1 |
| pillow | 11.3.0 | 12.0.0 |
| pydantic | 2.11.9 | 2.12.5 |
| PyMuPDF | 1.26.4 | 1.26.6 |
| EbookLib | 0.18 | 0.20 |

### Version Constraints

- **numpy**: Held at 2.2.6 (opencv-python requires <2.3.0)

## Bug Fixes

### Circular Import Resolution

**Issue:** Circular import occurred when `config_loader.py` imported from `modules.llm.model_capabilities`, triggering cascading imports.

**Resolution:** Implemented lazy imports using `__getattr__` in `modules/llm/__init__.py` to defer module loading until attributes are actually accessed.

**Impact:** Clean startup with no import errors across all entry points.

### Windows Long Path Handling

**Issue:** File operations failed on Windows when paths exceeded 260 characters.

**Resolution:** Centralized Windows long path handling in `modules/core/path_utils.py` with automatic `\\?\` prefix for extended paths.

**Impact:** Reliable processing of deeply nested document hierarchies on Windows.

## Technical Improvements

### New Modules

**modules/llm/providers/**
Provider-specific implementations for OpenAI, Anthropic, Google, and OpenRouter with automatic capability detection and parameter filtering.

**modules/core/auto_selector.py**
Automatic method selection logic analyzing file types, text searchability, and content characteristics.

**modules/processing/epub_utils.py**
EPUB parsing and text extraction with chapter structure preservation.

**modules/token_tracker.py**
Token usage tracking and daily budget enforcement with persistence.

**modules/operations/cost_analysis.py**
Cost calculation and reporting across providers.

**modules/ui/cost_display.py**
Cost visualization utilities for interactive display.

### Architecture Changes

**LangChain Integration**
- Retry logic handled by LangChain's `max_retries` parameter
- Token tracking via `response_metadata.token_usage`
- Structured output parsing via `with_structured_output()` with Pydantic models
- Capability-based parameter filtering via `disabled_params`

**Deprecated Custom Implementations**
- Custom retry logic → LangChain's built-in retry
- Manual capability filtering → LangChain's `disabled_params`
- Custom token extraction → LangChain's response metadata

**Retained for Batch API**
- `modules/llm/openai_utils.py`: OpenAI Batch API operations
- `modules/llm/model_capabilities.py`: Batch API capability detection and `ensure_image_support()`

### Repository Organization

Comprehensive cleanup with proper `__init__.py` files across all packages:
- `modules/__init__.py`
- `modules/config/__init__.py`
- `modules/core/__init__.py`
- `modules/infra/__init__.py`
- `modules/io/__init__.py`
- `modules/llm/__init__.py`
- `modules/llm/batch/__init__.py`
- `modules/operations/__init__.py`
- `modules/operations/batch/__init__.py`
- `modules/operations/repair/__init__.py`
- `modules/processing/__init__.py`

## Configuration Changes

### New Options

**config/model_config.yaml**
- `provider`: Explicit provider selection (openai, anthropic, google, openrouter)

**config/paths_config.yaml**
- `auto_mode.force_ocr_for_searchable_pdfs`: Force OCR on searchable PDFs
- `epubs` section: EPUB input/output directory configuration

**config/concurrency_config.yaml**
- `token_budget.enabled`: Enable daily token budget
- `token_budget.daily_limit`: Maximum tokens per day

### Environment Variables

| Variable | Provider |
|----------|----------|
| `OPENAI_API_KEY` | OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic |
| `GOOGLE_API_KEY` | Google |
| `OPENROUTER_API_KEY` | OpenRouter |

## Upgrading to 3.0

### For Existing Users

Existing workflows continue to work. OpenAI remains the default provider if no provider is specified.

**Recommended Steps**
1. Pull the latest version
2. Install updated dependencies: `pip install -r requirements.txt`
3. Set environment variables for desired providers
4. Review `config/model_config.yaml` for provider configuration
5. Test existing workflows (should work unchanged)

### New Provider Setup

To use providers other than OpenAI:

1. Set the appropriate environment variable
2. Update `config/model_config.yaml`:
```yaml
model: claude-sonnet-4-5
provider: anthropic
```
3. Run transcription as usual

### Auto Mode Setup

No configuration required. Auto mode is available as a method selection option in both interactive and CLI modes.

## Known Limitations

**Batch Processing**
- Only available for OpenAI provider (Batch API is OpenAI-specific)
- Other providers use synchronous processing

**Provider-Specific**
- OpenRouter models may have varying capability support
- Some reasoning models do not support structured outputs

**Auto Mode**
- Requires valid API key for GPT fallback
- May not detect all text in complex scanned documents

## Documentation

**Updated**
- README.md: Comprehensive multi-provider documentation
- Configuration files: Provider-specific examples and comments

**New**
- RELEASE_NOTES_v3.0.md: This document

## Support and Resources

**Getting Help**
- README.md: Comprehensive usage instructions
- `--help` flag: Available on all CLI scripts
- Configuration files: Inline comments for all options

**Provider Documentation**
- OpenAI: https://platform.openai.com/docs
- Anthropic: https://docs.anthropic.com
- Google: https://ai.google.dev/docs
- OpenRouter: https://openrouter.ai/docs
