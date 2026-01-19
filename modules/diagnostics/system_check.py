"""System diagnostics and health checks.

Provides functions to check system requirements, dependencies, and configuration.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from modules.infra.logger import setup_logger
from modules.processing.tesseract_utils import is_tesseract_available

logger = setup_logger(__name__)


def check_python_version() -> Tuple[bool, str]:
    """Check if Python version meets requirements.
    
    Returns:
        Tuple of (is_valid, message).
    """
    version = sys.version_info
    if version >= (3, 8):
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    return False, f"Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)"


def check_tesseract() -> Tuple[bool, str]:
    """Check if Tesseract OCR is available.
    
    Returns:
        Tuple of (is_available, message).
    """
    if is_tesseract_available():
        return True, "Tesseract OCR available"
    return False, "Tesseract OCR not found"


def check_api_key() -> Tuple[bool, str]:
    """Check if OpenAI API key is configured.
    
    Returns:
        Tuple of (is_configured, message).
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        masked = api_key[:8] + "..." if len(api_key) > 8 else "***"
        return True, f"OpenAI API key configured ({masked})"
    return False, "OpenAI API key not set (OPENAI_API_KEY environment variable)"


def check_config_files() -> Tuple[bool, str]:
    """Check if required configuration files exist.
    
    Returns:
        Tuple of (all_exist, message).
    """
    from modules.config.config_loader import CONFIG_DIR
    
    required_files = [
        "model_config.yaml",
        "paths_config.yaml",
        "concurrency_config.yaml",
        "image_processing_config.yaml",
    ]
    
    missing = []
    for filename in required_files:
        if not (CONFIG_DIR / filename).exists():
            missing.append(filename)
    
    if not missing:
        return True, f"All configuration files present in {CONFIG_DIR}"
    return False, f"Missing config files: {', '.join(missing)}"


def check_system_requirements() -> Dict[str, Tuple[bool, str]]:
    """Run all system requirement checks.
    
    Returns:
        Dictionary mapping check names to (status, message) tuples.
    """
    checks = {
        "Python Version": check_python_version(),
        "Tesseract OCR": check_tesseract(),
        "OpenAI API Key": check_api_key(),
        "Configuration Files": check_config_files(),
    }
    
    return checks


def diagnose_api_connectivity() -> Dict[str, Any]:
    """Diagnose OpenAI API connectivity.
    
    Returns:
        Dictionary with diagnostic information.
    """
    import openai
    
    diagnostics = {
        "api_key_set": bool(os.environ.get("OPENAI_API_KEY")),
        "openai_version": openai.__version__,
        "connectivity_test": None,
        "error": None,
    }
    
    if not diagnostics["api_key_set"]:
        diagnostics["error"] = "API key not configured"
        return diagnostics
    
    try:
        from openai import OpenAI
        client = OpenAI()
        # Simple API test - list models
        models = client.models.list()
        diagnostics["connectivity_test"] = "SUCCESS"
        diagnostics["models_accessible"] = len(list(models.data))
    except Exception as e:
        diagnostics["connectivity_test"] = "FAILED"
        diagnostics["error"] = str(e)
    
    return diagnostics


def generate_diagnostic_report() -> str:
    """Generate comprehensive diagnostic report.
    
    Returns:
        Formatted diagnostic report string.
    """
    lines = ["=" * 80, "ChronoTranscriber System Diagnostics", "=" * 80, ""]
    
    # System requirements
    lines.append("System Requirements:")
    lines.append("-" * 40)
    checks = check_system_requirements()
    for name, (status, message) in checks.items():
        status_str = "✓" if status else "✗"
        lines.append(f"  {status_str} {name}: {message}")
    
    lines.append("")
    
    # API connectivity
    lines.append("API Connectivity:")
    lines.append("-" * 40)
    api_diag = diagnose_api_connectivity()
    for key, value in api_diag.items():
        lines.append(f"  {key}: {value}")
    
    lines.append("")
    lines.append("=" * 80)
    
    return "\n".join(lines)
