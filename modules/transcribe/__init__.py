"""Transcription workflow orchestration.

Deep module covering the high-level transcription pipeline: workflow
manager, async pipeline, resume checker, dual-mode CLI harness,
``UserConfiguration`` state carrier, and CLI-to-config builder.
"""

from modules.transcribe.config_builder import (
    create_config_from_cli_args,
)
from modules.transcribe.dual_mode import (
    AsyncDualModeScript,
    DualModeScript,
)
from modules.transcribe.manager import (
    TransientFileTracker,
    WorkflowManager,
)
from modules.transcribe.pipeline import (
    run_transcription_pipeline,
    transcribe_single_image,
    write_output_from_jsonl,
)
from modules.transcribe.resume import (
    ProcessingState,
    ResumeChecker,
    ResumeResult,
)
from modules.transcribe.user_config import UserConfiguration

__all__ = [
    "WorkflowManager",
    "TransientFileTracker",
    "run_transcription_pipeline",
    "transcribe_single_image",
    "write_output_from_jsonl",
    "ResumeChecker",
    "ResumeResult",
    "ProcessingState",
    "DualModeScript",
    "AsyncDualModeScript",
    "UserConfiguration",
    "create_config_from_cli_args",
]
