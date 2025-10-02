# ChronoTranscriber Version 2.0 Release Notes

Release Date: October 2025

ChronoTranscriber 2.0 introduces a hybrid dual-mode operation system, enhanced user interface, comprehensive command-line support, and critical stability improvements. This major release accommodates both guided interactive workflows and automated CLI operation.

## Highlights

**Dual-Mode Operation**
Choose between Interactive Mode for guided workflows with visual prompts and navigation, or CLI Mode for automation and scripting. Switch modes via a single configuration flag.

**Enhanced User Experience**
Redesigned interface with navigation support, color-coded status messages, and Windows terminal compatibility. Users can now go back to previous steps or quit at any time during interactive workflows.

**Full Command-Line Support**
All functionality accessible via CLI arguments with comprehensive help documentation, proper exit codes, and path resolution for integration into automated pipelines.

**Critical Stability Fixes**
Resolved Windows Unicode encoding failures and batch API service tier incompatibility. Application now runs reliably across all Windows terminal environments with correct batch processing.

## What's New

### Interactive Mode

A guided, step-by-step experience designed for researchers and users who prefer visual workflows:

**Features**
- Visual progress indicators with color-coded status messages (green for success, yellow for warnings, red for errors)
- Navigate backward through workflow steps or quit at any time by entering 'b' or 'q'
- Multi-selection with intuitive input: specific items ("1,3,5"), ranges ("1-5"), or "all"
- Automatic validation with helpful error messages
- Confirmation screens before long-running operations
- Plain-language descriptions suitable for non-technical users

**Getting Started**
Set `interactive_mode: true` in `config/paths_config.yaml` and run scripts without arguments. The application guides you through each decision point.

### CLI Mode

Command-line operation for advanced users, automation, and integration:

**Features**
- Full argument support for all operations
- Comprehensive `--help` documentation for each script
- Absolute and relative path resolution
- Exit codes for shell scripting (0 for success, 1 for errors)
- Unattended batch processing
- Integration-friendly logging and error reporting

**Getting Started**
Set `interactive_mode: false` in `config/paths_config.yaml` and run scripts with arguments. See `--help` for each script or consult the README for examples.

**Available Commands**

unified_transcriber.py: Process documents with specified method
```bash
python main/unified_transcriber.py --input INPUT --output OUTPUT --type {images,pdfs} --method {native,tesseract,gpt}
```

repair_transcriptions.py: Fix failed or incomplete transcriptions
```bash
python main/repair_transcriptions.py --transcription FILE [--all-failures] [--batch]
```

check_batches.py: Monitor and retrieve batch job results
```bash
python main/check_batches.py [--directory DIR] [--no-diagnostics]
```

cancel_batches.py: Cancel non-terminal batch jobs
```bash
python main/cancel_batches.py [--batch-ids ID ...] [--force]
```

### User Interface Enhancements

**Visual Improvements**
- ASCII-safe characters replace Unicode box-drawing for universal Windows compatibility
- Color-coded output for clear status differentiation
- Graceful Unicode fallback with encoding error handling
- Consistent styling across all interactive prompts

**Navigation Support**
- State machine-based workflow management
- Ability to revisit previous configuration steps
- Clear visual hierarchy and progress indicators
- Confirmation screens with detailed summaries

**Separation of Concerns**
- User-facing prompts distinct from technical logging
- Interactive output via styled UI functions
- Technical diagnostics via logger for troubleshooting

## Bug Fixes

### Critical: Windows Unicode Encoding Error

**Issue:** Application crashed on Windows with `UnicodeEncodeError` when displaying UI prompts due to Unicode box-drawing characters incompatible with Windows cp1252 encoding.

**Resolution:** Replaced with ASCII-safe characters (=, -, .) and added error handling with fallback encoding. Application now runs correctly on all Windows terminal environments including PowerShell, Command Prompt, and Windows Terminal.

**Impact:** All Windows users can now use the application without crashes.

### Critical: Batch API Service Tier Incompatibility

**Issue:** All batch processing requests failed with "Flex is not available for this model" error because `service_tier` parameter was being sent to Batch API where it is not supported.

**Resolution:** Service tier parameter is now completely omitted from batch requests while preserved for synchronous calls. The application logs the omission for transparency.

**Impact:** Batch processing now works correctly. Synchronous processing unaffected. Users can configure service tier (flex, priority, etc.) and it will be used appropriately for synchronous calls only.

## Configuration Changes

### New Options

**paths_config.yaml**
- `interactive_mode`: Boolean flag to toggle between interactive prompts (true) or CLI mode (false). Default is true for backward compatibility.

**Behavior**
- When true: Scripts run with user prompts and guided workflows
- When false: Scripts accept command-line arguments for automation

**Path Resolution**
- In CLI mode, relative paths resolve relative to configured input/output directories
- Absolute paths work as-is in both modes
- Parent directories created automatically when needed

### Deprecated Options

The following configuration keys have been removed:
- `allow_relative_paths`: Path resolution now automatic based on mode
- `base_directory`: Replaced by input and output directory configuration

These keys are no longer needed and will be ignored if present in configuration files.

## Technical Improvements

### New Modules

**modules/core/cli_args.py**
Centralized argument parsers for all CLI entry points with path resolution, validation helpers, and consistent error handling.

**modules/ui/prompts.py**
Core prompt utilities with navigation support, visual styling, input validation, and Windows-compatible output handling.

**modules/ui/workflows.py**
Workflow orchestration with state machine navigation, styled summaries, and back/quit support throughout interactive flows.

### Updated Components

**Main Scripts**
- unified_transcriber.py: Routes between interactive and CLI execution paths
- repair_transcriptions.py: Dispatches to appropriate repair flow based on mode
- check_batches.py: Supports CLI diagnostics flags alongside interactive prompts
- cancel_batches.py: Adds CLI controls for batch targeting and force cancellation

**Core Modules**
- modules/llm/batch/batching.py: Omits service_tier for batch compliance
- modules/operations/repair/run.py: Exposes main_cli for automated workflows

### Testing and Validation

Comprehensive testing completed on Windows environment:
- 13/13 core functionality tests passed
- 2 critical bugs identified and fixed
- Performance metrics documented for all transcription methods
- Path resolution verified with absolute and relative paths
- Error handling and retry logic validated
- Batch submission and status checking confirmed working

**Performance Metrics** (from validation testing)
- Tesseract OCR (8 images): approximately 15 seconds
- Tesseract OCR (113-page PDF): approximately 2 minutes
- Native PDF extraction (20 pages): under 5 seconds
- GPT Synchronous (8 images): approximately 45 seconds
- GPT Synchronous (113-page PDF): approximately 8 minutes
- Batch submission: under 30 seconds (processing completes within 24 hours)

## Upgrading to 2.0

### For Existing Users

No breaking changes for users who continue using interactive mode. The application maintains full backward compatibility with existing workflows.

**Recommended Steps**

1. Pull the latest version
2. Review the updated README for new features
3. Test your existing workflows (they should work unchanged)
4. Optionally explore CLI mode for automation needs

### Configuration Updates

**Optional: Remove deprecated keys**

Edit `config/paths_config.yaml` and remove these lines if present:
```yaml
allow_relative_paths: false  # No longer needed
base_directory: "."          # No longer needed
```

These keys are ignored but can be removed for clarity.

**Optional: Enable CLI mode**

To use command-line automation:
1. Set `interactive_mode: false` in `config/paths_config.yaml`
2. Run scripts with arguments (see `--help` for each script)
3. Review README usage section for examples

### Migration Examples

**Before (Interactive only):**
```bash
python main/unified_transcriber.py
# User prompted for all options
```

**After (Interactive - unchanged):**
```bash
python main/unified_transcriber.py
# Same guided workflow
```

**After (CLI - new capability):**
```bash
python main/unified_transcriber.py \
  --input ./documents \
  --output ./results \
  --type pdfs \
  --method gpt
```

## Known Limitations

**Batch Processing**
- Service tier parameters not supported in batch mode (synchronous mode unaffected)
- Batch jobs can take up to 24 hours depending on queue length
- Cost savings: 50% discount on batch processing versus synchronous

**Interactive Mode**
- Navigation (back/quit) not available during long-running operations
- Some prompts require specific input formats (documented in help text)

**CLI Mode**
- File selection for images expects folder paths, not individual files
- Recursive processing requires explicit `--recursive` flag
- Relative paths resolve relative to configured base directories

## Documentation

**Updated**
- README.md: Comprehensive dual-mode documentation with examples
- Configuration files: Inline comments clarified for new options

**New**
- RELEASE_NOTES_v2.0.md: This document

## Support and Resources

**Getting Help**
- README.md: Comprehensive usage instructions and examples
- `--help` flag: Available on all CLI scripts
- Configuration files: Inline comments for all options

**Reporting Issues**
- Check README troubleshooting section first
- Review release notes for known limitations
- Consult logs directory for detailed error information