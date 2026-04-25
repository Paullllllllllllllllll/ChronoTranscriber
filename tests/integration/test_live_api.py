"""Live API smoke tests for ChronoTranscriber.

Exercises real API calls against a small PDF and pre-processed images
in staging/, organized as a mode x provider matrix.

Run via pytest:
    pytest tests/integration/test_live_api.py -m api -v

Run standalone for a summary table:
    python tests/integration/test_live_api.py

Skip tests for missing API keys automatically. Keep outputs for
inspection with:
    CHRONO_KEEP_TEST_OUTPUTS=1 pytest ... -m api -v
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Ensure project root is on sys.path before any modules.* imports
# (avoids circular import in modules.documents <-> modules.transcribe).
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import project modules at module level (order matters for circular deps).
from main.unified_transcriber import process_documents  # noqa: E402
from modules.config.service import ConfigService, get_config_service  # noqa: E402
from modules.documents.page_range import parse_page_range  # noqa: E402
from modules.transcribe.user_config import UserConfiguration  # noqa: E402
import modules.config.config_loader as _cl_mod  # noqa: E402
STAGING_DIR = PROJECT_ROOT / "staging"
STAGING_PDF_DIR = STAGING_DIR / "pdfs"
STAGING_PDF = STAGING_PDF_DIR / "Antonio Franco.pdf"
STAGING_IMAGES_DIR = STAGING_DIR / "images"
CONFIG_DIR = PROJECT_ROOT / "config"
SCHEMAS_DIR = PROJECT_ROOT / "schemas"

KEEP_OUTPUTS = os.environ.get(
    "CHRONO_KEEP_TEST_OUTPUTS", ""
).lower() in ("1", "true", "yes")

PROVIDERS = {
    "openai": {
        "model": "gpt-5-nano",
        "env_var": "OPENAI_API_KEY",
        "reasoning_effort": "low",
        "temperature": 1,
    },
    "anthropic": {
        "model": "claude-haiku-4-5-20251001",
        "env_var": "ANTHROPIC_API_KEY",
        "reasoning_effort": "low",
        "temperature": 1,
    },
    "google": {
        "model": "gemini-2.0-flash",
        "env_var": "GOOGLE_API_KEY",
        "reasoning_effort": None,
        "temperature": 0.0,
    },
    "custom": {
        "model": "kristaller486/dots.ocr-1.5",
        "env_var": "SUZ_API_KEY",
        "reasoning_effort": None,
        "temperature": 0.0,
    },
}

CUSTOM_ENDPOINT_URL = "https://ai.soziologie.uzh.ch/v1/chat/completions"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_key(env_var: str, label: str) -> None:
    if not os.environ.get(env_var):
        pytest.skip(f"{label} test skipped: {env_var} not set")


TESSERACT_CMD = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"


def _require_tesseract() -> None:
    if not Path(TESSERACT_CMD).exists():
        pytest.skip(f"Tesseract not found at {TESSERACT_CMD}")
    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def _require_custom_endpoint() -> None:
    _require_key("SUZ_API_KEY", "Custom (UZH)")
    try:
        import httpx

        resp = httpx.head(CUSTOM_ENDPOINT_URL, timeout=5)
        if resp.status_code >= 500:
            pytest.skip(
                f"Custom endpoint unreachable (HTTP {resp.status_code})"
            )
    except Exception as exc:
        pytest.skip(f"Custom endpoint unreachable: {exc}")


def _make_model_config_yaml(
    provider: str,
    model: str,
    reasoning_effort: str | None = None,
    temperature: float = 0.0,
) -> str:
    if provider == "custom":
        return (
            "transcription_model:\n"
            "  provider: custom\n"
            f'  name: "{model}"\n'
            "  custom_endpoint:\n"
            f'    base_url: "{CUSTOM_ENDPOINT_URL}"\n'
            '    api_key_env_var: "SUZ_API_KEY"\n'
            "    use_plain_text_prompt: true\n"
            "    capabilities:\n"
            "      supports_vision: true\n"
            "      supports_structured_output: false\n"
            "  max_output_tokens: 4096\n"
            f"  temperature: {temperature}\n"
            "  expects_image_inputs: true\n"
        )

    lines = [
        "transcription_model:",
        f"  provider: {provider}",
        f"  name: {model}",
        "  max_output_tokens: 16384",
        f"  temperature: {temperature}",
        "  expects_image_inputs: true",
    ]
    if reasoning_effort:
        lines.append("  reasoning:")
        lines.append(f"    effort: {reasoning_effort}")
    return "\n".join(lines) + "\n"


def _make_paths_config_yaml(output_dir: Path) -> str:
    out = str(output_dir).replace("\\", "/")
    return textwrap.dedent(f"""\
        general:
          interactive_mode: false
          retain_temporary_jsonl: true
          keep_preprocessed_images: false
          input_paths_is_output_path: false
          output_format: 'txt'
          resume_mode: 'skip'
          logs_dir: ''
          auto_mode_pdf_use_ocr_for_scanned: true
          auto_mode_pdf_use_ocr_for_searchable: true
          auto_mode_pdf_ocr_method: 'gpt'
          auto_mode_image_ocr_method: 'gpt'

        file_paths:
          PDFs:
            input: ''
            output: '{out}'
          Images:
            input: ''
            output: '{out}'
          EPUBs:
            input: ''
            output: '{out}'
          MOBIs:
            input: ''
            output: '{out}'
          Auto:
            input: ''
            output: '{out}'
    """)


def _make_concurrency_config_yaml() -> str:
    src = CONFIG_DIR / "concurrency_config.yaml"
    with src.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["daily_token_limit"] = {"enabled": False, "daily_tokens": 999_999_999}
    return yaml.dump(cfg, default_flow_style=False, allow_unicode=True)


def _make_temp_config_dir(
    base_tmp: Path,
    provider: str,
    model: str,
    output_dir: Path,
    reasoning_effort: str | None = None,
    temperature: float = 0.0,
) -> Path:
    config_tmp = base_tmp / "config"
    config_tmp.mkdir(parents=True, exist_ok=True)

    (config_tmp / "model_config.yaml").write_text(
        _make_model_config_yaml(provider, model, reasoning_effort, temperature),
        encoding="utf-8",
    )
    (config_tmp / "paths_config.yaml").write_text(
        _make_paths_config_yaml(output_dir),
        encoding="utf-8",
    )
    (config_tmp / "concurrency_config.yaml").write_text(
        _make_concurrency_config_yaml(),
        encoding="utf-8",
    )

    shutil.copy2(
        CONFIG_DIR / "image_processing_config.yaml",
        config_tmp / "image_processing_config.yaml",
    )
    return config_tmp


def _run_cli(
    args: list[str],
    config_dir: Path,
    timeout: int = 900,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["CHRONO_CONFIG_DIR"] = str(config_dir)
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "main" / "unified_transcriber.py")]
        + args,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _assert_output_valid(
    output_dir: Path,
    ext: str = ".txt",
) -> Path:
    files = list(output_dir.rglob(f"*{ext}"))
    assert len(files) > 0, (
        f"No {ext} output file found in {output_dir}. "
        f"Contents: {list(output_dir.rglob('*'))}"
    )
    output_file = files[0]
    assert output_file.stat().st_size > 0, (
        f"Output file {output_file} is empty"
    )
    content = output_file.read_text(encoding="utf-8")
    all_lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    assert len(all_lines) > 0, "Output file contains only whitespace"
    error_markers = [
        "[transcription error:",
        "[Transcription not possible]",
        "[No transcribable text]",
    ]
    non_error = [
        ln
        for ln in all_lines
        if not any(m in ln for m in error_markers)
    ]
    assert len(non_error) > 0, (
        f"All {len(all_lines)} output lines are error markers"
    )
    return output_file


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_config_service():
    ConfigService.reset()
    yield
    ConfigService.reset()


@pytest.fixture()
def output_dir(tmp_path: Path) -> Path:
    d = tmp_path / "output"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# CLI Mode Tests
# ---------------------------------------------------------------------------


@pytest.mark.api
class TestCLIMode:

    def test_01_openai_cli_pdf(self, tmp_path: Path, output_dir: Path) -> None:
        p = PROVIDERS["openai"]
        _require_key(p["env_var"], "OpenAI")
        config_dir = _make_temp_config_dir(
            tmp_path, "openai", p["model"], output_dir,
            reasoning_effort=p["reasoning_effort"],
            temperature=p["temperature"],
        )
        result = _run_cli(
            [
                "--input", str(STAGING_PDF_DIR),
                "--output", str(output_dir),
                "--type", "pdfs",
                "--method", "gpt",
                "--model", p["model"],
                "--provider", "openai",
                "--reasoning-effort", "low",
                "--pages", "first:1",
                "--force",
            ],
            config_dir,
        )
        assert result.returncode == 0, (
            f"CLI exited with {result.returncode}.\n"
            f"STDOUT:\n{result.stdout[-2000:]}\n"
            f"STDERR:\n{result.stderr[-2000:]}"
        )
        _assert_output_valid(output_dir)

    def test_02_anthropic_cli_pdf(self, tmp_path: Path, output_dir: Path) -> None:
        p = PROVIDERS["anthropic"]
        _require_key(p["env_var"], "Anthropic")
        config_dir = _make_temp_config_dir(
            tmp_path, "anthropic", p["model"], output_dir,
            reasoning_effort=p["reasoning_effort"],
            temperature=p["temperature"],
        )
        result = _run_cli(
            [
                "--input", str(STAGING_PDF_DIR),
                "--output", str(output_dir),
                "--type", "pdfs",
                "--method", "gpt",
                "--model", p["model"],
                "--provider", "anthropic",
                "--reasoning-effort", "low",
                "--pages", "first:1",
                "--force",
            ],
            config_dir,
        )
        assert result.returncode == 0, (
            f"CLI exited with {result.returncode}.\n"
            f"STDOUT:\n{result.stdout[-2000:]}\n"
            f"STDERR:\n{result.stderr[-2000:]}"
        )
        _assert_output_valid(output_dir)

    def test_03_google_cli_pdf(self, tmp_path: Path, output_dir: Path) -> None:
        p = PROVIDERS["google"]
        _require_key(p["env_var"], "Google")
        config_dir = _make_temp_config_dir(
            tmp_path, "google", p["model"], output_dir,
            temperature=p["temperature"],
        )
        result = _run_cli(
            [
                "--input", str(STAGING_PDF_DIR),
                "--output", str(output_dir),
                "--type", "pdfs",
                "--method", "gpt",
                "--model", p["model"],
                "--provider", "google",
                "--pages", "first:1",
                "--force",
            ],
            config_dir,
        )
        assert result.returncode == 0, (
            f"CLI exited with {result.returncode}.\n"
            f"STDOUT:\n{result.stdout[-2000:]}\n"
            f"STDERR:\n{result.stderr[-2000:]}"
        )
        _assert_output_valid(output_dir)

    def test_04_custom_cli_pdf(self, tmp_path: Path, output_dir: Path) -> None:
        _require_custom_endpoint()
        p = PROVIDERS["custom"]
        config_dir = _make_temp_config_dir(
            tmp_path, "custom", p["model"], output_dir,
            temperature=p["temperature"],
        )
        result = _run_cli(
            [
                "--input", str(STAGING_PDF_DIR),
                "--output", str(output_dir),
                "--type", "pdfs",
                "--method", "gpt",
                "--pages", "first:1",
                "--force",
            ],
            config_dir,
        )
        assert result.returncode == 0, (
            f"CLI exited with {result.returncode}.\n"
            f"STDOUT:\n{result.stdout[-2000:]}\n"
            f"STDERR:\n{result.stderr[-2000:]}"
        )
        _assert_output_valid(output_dir)

    def test_05_tesseract_cli_pdf(self, tmp_path: Path, output_dir: Path) -> None:
        _require_tesseract()

        config_dir = _make_temp_config_dir(
            tmp_path, "openai", "gpt-5-nano", output_dir,
        )
        result = _run_cli(
            [
                "--input", str(STAGING_PDF_DIR),
                "--output", str(output_dir),
                "--type", "pdfs",
                "--method", "tesseract",
                "--pages", "first:1",
                "--force",
            ],
            config_dir,
        )
        assert result.returncode == 0, (
            f"CLI exited with {result.returncode}.\n"
            f"STDOUT:\n{result.stdout[-2000:]}\n"
            f"STDERR:\n{result.stderr[-2000:]}"
        )
        _assert_output_valid(output_dir)

    def test_06_openai_cli_output_format_md(
        self, tmp_path: Path, output_dir: Path
    ) -> None:
        p = PROVIDERS["openai"]
        _require_key(p["env_var"], "OpenAI")
        config_dir = _make_temp_config_dir(
            tmp_path, "openai", p["model"], output_dir,
            reasoning_effort=p["reasoning_effort"],
            temperature=p["temperature"],
        )
        result = _run_cli(
            [
                "--input", str(STAGING_PDF_DIR),
                "--output", str(output_dir),
                "--type", "pdfs",
                "--method", "gpt",
                "--model", p["model"],
                "--provider", "openai",
                "--reasoning-effort", "low",
                "--pages", "first:1",
                "--force",
                "--output-format", "md",
            ],
            config_dir,
        )
        assert result.returncode == 0, (
            f"CLI exited with {result.returncode}.\n"
            f"STDOUT:\n{result.stdout[-2000:]}\n"
            f"STDERR:\n{result.stderr[-2000:]}"
        )
        _assert_output_valid(output_dir, ext=".md")

    def test_07_resume_skip_behavior(
        self, tmp_path: Path, output_dir: Path
    ) -> None:
        p = PROVIDERS["openai"]
        _require_key(p["env_var"], "OpenAI")
        config_dir = _make_temp_config_dir(
            tmp_path, "openai", p["model"], output_dir,
            reasoning_effort=p["reasoning_effort"],
            temperature=p["temperature"],
        )
        cli_args = [
            "--input", str(STAGING_PDF_DIR),
            "--output", str(output_dir),
            "--type", "pdfs",
            "--method", "gpt",
            "--model", p["model"],
            "--provider", "openai",
            "--reasoning-effort", "low",
            "--pages", "first:1",
            "--force",
        ]
        r1 = _run_cli(cli_args, config_dir)
        assert r1.returncode == 0, (
            f"First run failed ({r1.returncode}).\n"
            f"STDERR:\n{r1.stderr[-2000:]}"
        )
        out_file = _assert_output_valid(output_dir)
        mtime_after_first = out_file.stat().st_mtime

        time.sleep(1.1)

        skip_args = [a for a in cli_args if a != "--force"] + ["--resume"]
        r2 = _run_cli(skip_args, config_dir)
        assert r2.returncode == 0, (
            f"Second run failed ({r2.returncode}).\n"
            f"STDERR:\n{r2.stderr[-2000:]}"
        )
        mtime_after_second = out_file.stat().st_mtime
        assert mtime_after_first == mtime_after_second, (
            "Output file was rewritten during resume/skip run"
        )

    def test_08_postprocess(self, tmp_path: Path, output_dir: Path) -> None:
        p = PROVIDERS["openai"]
        _require_key(p["env_var"], "OpenAI")
        config_dir = _make_temp_config_dir(
            tmp_path, "openai", p["model"], output_dir,
            reasoning_effort=p["reasoning_effort"],
            temperature=p["temperature"],
        )
        r1 = _run_cli(
            [
                "--input", str(STAGING_PDF_DIR),
                "--output", str(output_dir),
                "--type", "pdfs",
                "--method", "gpt",
                "--model", p["model"],
                "--provider", "openai",
                "--reasoning-effort", "low",
                "--pages", "first:1",
                "--force",
            ],
            config_dir,
        )
        assert r1.returncode == 0, f"Transcription failed: {r1.stderr[-2000:]}"
        out_file = _assert_output_valid(output_dir)

        env = os.environ.copy()
        env["CHRONO_CONFIG_DIR"] = str(config_dir)
        env["PYTHONPATH"] = str(PROJECT_ROOT)
        pp_result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "main" / "postprocess_transcriptions.py"),
                "--input", str(out_file),
                "--in-place",
                "--max-blank-lines", "2",
            ],
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert pp_result.returncode == 0, (
            f"Postprocess failed ({pp_result.returncode}).\n"
            f"STDERR:\n{pp_result.stderr[-2000:]}"
        )
        assert out_file.stat().st_size > 0, (
            "Output file empty after postprocessing"
        )

    def test_09_auto_mode(self, tmp_path: Path, output_dir: Path) -> None:
        p = PROVIDERS["openai"]
        _require_key(p["env_var"], "OpenAI")

        auto_input = tmp_path / "auto_input"
        auto_input.mkdir()
        shutil.copy2(STAGING_PDF, auto_input / STAGING_PDF.name)
        staging_in = str(auto_input).replace("\\", "/")
        staging_out = str(output_dir).replace("\\", "/")

        config_tmp = tmp_path / "config"
        config_tmp.mkdir(parents=True, exist_ok=True)

        (config_tmp / "model_config.yaml").write_text(
            _make_model_config_yaml(
                "openai", p["model"],
                reasoning_effort=p["reasoning_effort"],
                temperature=p["temperature"],
            ),
            encoding="utf-8",
        )

        paths_yaml = textwrap.dedent(f"""\
            general:
              interactive_mode: false
              retain_temporary_jsonl: true
              keep_preprocessed_images: false
              input_paths_is_output_path: false
              output_format: 'txt'
              resume_mode: 'skip'
              logs_dir: ''
              auto_mode_pdf_use_ocr_for_scanned: true
              auto_mode_pdf_use_ocr_for_searchable: true
              auto_mode_pdf_ocr_method: 'gpt'
              auto_mode_image_ocr_method: 'gpt'

            file_paths:
              PDFs:
                input: ''
                output: '{staging_out}'
              Images:
                input: ''
                output: '{staging_out}'
              EPUBs:
                input: ''
                output: '{staging_out}'
              MOBIs:
                input: ''
                output: '{staging_out}'
              Auto:
                input: '{staging_in}'
                output: '{staging_out}'
        """)
        (config_tmp / "paths_config.yaml").write_text(
            paths_yaml, encoding="utf-8"
        )
        (config_tmp / "concurrency_config.yaml").write_text(
            _make_concurrency_config_yaml(), encoding="utf-8"
        )
        shutil.copy2(
            CONFIG_DIR / "image_processing_config.yaml",
            config_tmp / "image_processing_config.yaml",
        )

        result = _run_cli(
            [
                "--auto",
                "--pages", "first:1",
                "--force",
            ],
            config_tmp,
        )
        assert result.returncode == 0, (
            f"Auto mode exited with {result.returncode}.\n"
            f"STDOUT:\n{result.stdout[-2000:]}\n"
            f"STDERR:\n{result.stderr[-2000:]}"
        )
        output_files = list(output_dir.rglob("*.txt"))
        assert len(output_files) > 0, (
            f"Auto mode produced no output. Dir contents: "
            f"{list(output_dir.rglob('*'))}"
        )


# ---------------------------------------------------------------------------
# Programmatic Mode Tests (exercises same pipeline as interactive mode)
# ---------------------------------------------------------------------------


@pytest.mark.api
class TestProgrammaticMode:

    @staticmethod
    def _setup_config_env(config_dir: Path) -> str | None:
        old = os.environ.get("CHRONO_CONFIG_DIR")
        os.environ["CHRONO_CONFIG_DIR"] = str(config_dir)
        _cl_mod.CONFIG_DIR = _cl_mod._compute_config_dir()
        _cl_mod.DEFAULT_CONFIG_PATH = _cl_mod.CONFIG_DIR / "model_config.yaml"
        _cl_mod.DEFAULT_PATHS_CONFIG_PATH = (
            _cl_mod.CONFIG_DIR / "paths_config.yaml"
        )
        _cl_mod.DEFAULT_CONCURRENCY_CONFIG_PATH = (
            _cl_mod.CONFIG_DIR / "concurrency_config.yaml"
        )
        _cl_mod.DEFAULT_IMAGE_PROCESSING_CONFIG_PATH = (
            _cl_mod.CONFIG_DIR / "image_processing_config.yaml"
        )
        ConfigService.reset()
        return old

    @staticmethod
    def _teardown_config_env(old_val: str | None) -> None:
        if old_val is None:
            os.environ.pop("CHRONO_CONFIG_DIR", None)
        else:
            os.environ["CHRONO_CONFIG_DIR"] = old_val
        _cl_mod.CONFIG_DIR = _cl_mod._compute_config_dir()
        _cl_mod.DEFAULT_CONFIG_PATH = _cl_mod.CONFIG_DIR / "model_config.yaml"
        _cl_mod.DEFAULT_PATHS_CONFIG_PATH = (
            _cl_mod.CONFIG_DIR / "paths_config.yaml"
        )
        _cl_mod.DEFAULT_CONCURRENCY_CONFIG_PATH = (
            _cl_mod.CONFIG_DIR / "concurrency_config.yaml"
        )
        _cl_mod.DEFAULT_IMAGE_PROCESSING_CONFIG_PATH = (
            _cl_mod.CONFIG_DIR / "image_processing_config.yaml"
        )
        ConfigService.reset()

    @pytest.mark.asyncio
    async def test_10_openai_programmatic_pdf(
        self, tmp_path: Path, output_dir: Path
    ) -> None:
        p = PROVIDERS["openai"]
        _require_key(p["env_var"], "OpenAI")

        config_dir = _make_temp_config_dir(
            tmp_path, "openai", p["model"], output_dir,
            reasoning_effort=p["reasoning_effort"],
            temperature=p["temperature"],
        )
        old = self._setup_config_env(config_dir)
        try:
            svc = get_config_service()
            user_config = UserConfiguration(
                processing_type="pdfs",
                transcription_method="gpt",
                selected_items=[STAGING_PDF],
                resume_mode="overwrite",
                page_range=parse_page_range("first:1"),
                output_format="txt",
                selected_schema_name="markdown_transcription_schema",
                selected_schema_path=(
                    SCHEMAS_DIR / "markdown_transcription_schema.json"
                ),
            )
            paths_cfg = svc.get_paths_config()
            paths_cfg.setdefault("file_paths", {}).setdefault(
                "PDFs", {}
            )["output"] = str(output_dir)

            await process_documents(
                user_config,
                paths_cfg,
                svc.get_model_config(),
                svc.get_concurrency_config(),
                svc.get_image_processing_config(),
            )
        finally:
            self._teardown_config_env(old)

        _assert_output_valid(output_dir)

    @pytest.mark.asyncio
    async def test_11_anthropic_programmatic_pdf(
        self, tmp_path: Path, output_dir: Path
    ) -> None:
        p = PROVIDERS["anthropic"]
        _require_key(p["env_var"], "Anthropic")

        config_dir = _make_temp_config_dir(
            tmp_path, "anthropic", p["model"], output_dir,
            reasoning_effort=p["reasoning_effort"],
            temperature=p["temperature"],
        )
        old = self._setup_config_env(config_dir)
        try:
            svc = get_config_service()
            user_config = UserConfiguration(
                processing_type="pdfs",
                transcription_method="gpt",
                selected_items=[STAGING_PDF],
                resume_mode="overwrite",
                page_range=parse_page_range("first:1"),
                output_format="txt",
                selected_schema_name="markdown_transcription_schema",
                selected_schema_path=(
                    SCHEMAS_DIR / "markdown_transcription_schema.json"
                ),
            )
            paths_cfg = svc.get_paths_config()
            paths_cfg.setdefault("file_paths", {}).setdefault(
                "PDFs", {}
            )["output"] = str(output_dir)

            await process_documents(
                user_config,
                paths_cfg,
                svc.get_model_config(),
                svc.get_concurrency_config(),
                svc.get_image_processing_config(),
            )
        finally:
            self._teardown_config_env(old)

        _assert_output_valid(output_dir)

    @pytest.mark.asyncio
    async def test_12_google_programmatic_pdf(
        self, tmp_path: Path, output_dir: Path
    ) -> None:
        p = PROVIDERS["google"]
        _require_key(p["env_var"], "Google")

        config_dir = _make_temp_config_dir(
            tmp_path, "google", p["model"], output_dir,
            temperature=p["temperature"],
        )
        old = self._setup_config_env(config_dir)
        try:
            svc = get_config_service()
            user_config = UserConfiguration(
                processing_type="pdfs",
                transcription_method="gpt",
                selected_items=[STAGING_PDF],
                resume_mode="overwrite",
                page_range=parse_page_range("first:1"),
                output_format="txt",
                selected_schema_name="markdown_transcription_schema",
                selected_schema_path=(
                    SCHEMAS_DIR / "markdown_transcription_schema.json"
                ),
            )
            paths_cfg = svc.get_paths_config()
            paths_cfg.setdefault("file_paths", {}).setdefault(
                "PDFs", {}
            )["output"] = str(output_dir)

            await process_documents(
                user_config,
                paths_cfg,
                svc.get_model_config(),
                svc.get_concurrency_config(),
                svc.get_image_processing_config(),
            )
        finally:
            self._teardown_config_env(old)

        _assert_output_valid(output_dir)

    @pytest.mark.asyncio
    async def test_13_custom_programmatic_pdf(
        self, tmp_path: Path, output_dir: Path
    ) -> None:
        _require_custom_endpoint()
        p = PROVIDERS["custom"]

        config_dir = _make_temp_config_dir(
            tmp_path, "custom", p["model"], output_dir,
            temperature=p["temperature"],
        )
        old = self._setup_config_env(config_dir)
        try:
            svc = get_config_service()
            user_config = UserConfiguration(
                processing_type="pdfs",
                transcription_method="gpt",
                selected_items=[STAGING_PDF],
                resume_mode="overwrite",
                page_range=parse_page_range("first:1"),
                output_format="txt",
            )
            paths_cfg = svc.get_paths_config()
            paths_cfg.setdefault("file_paths", {}).setdefault(
                "PDFs", {}
            )["output"] = str(output_dir)

            await process_documents(
                user_config,
                paths_cfg,
                svc.get_model_config(),
                svc.get_concurrency_config(),
                svc.get_image_processing_config(),
            )
        finally:
            self._teardown_config_env(old)

        _assert_output_valid(output_dir)

    @pytest.mark.asyncio
    async def test_14_tesseract_programmatic_pdf(
        self, tmp_path: Path, output_dir: Path
    ) -> None:
        _require_tesseract()

        config_dir = _make_temp_config_dir(
            tmp_path, "openai", "gpt-5-nano", output_dir,
        )
        old = self._setup_config_env(config_dir)
        try:
            svc = get_config_service()
            user_config = UserConfiguration(
                processing_type="pdfs",
                transcription_method="tesseract",
                selected_items=[STAGING_PDF],
                resume_mode="overwrite",
                page_range=parse_page_range("first:1"),
                output_format="txt",
            )
            paths_cfg = svc.get_paths_config()
            paths_cfg.setdefault("file_paths", {}).setdefault(
                "PDFs", {}
            )["output"] = str(output_dir)

            await process_documents(
                user_config,
                paths_cfg,
                svc.get_model_config(),
                svc.get_concurrency_config(),
                svc.get_image_processing_config(),
            )
        finally:
            self._teardown_config_env(old)

        _assert_output_valid(output_dir)


# ---------------------------------------------------------------------------
# Interactive Mode Tests (monkeypatched ui_input)
# ---------------------------------------------------------------------------


def _make_interactive_paths_config_yaml(
    pdf_input: Path,
    output_dir: Path,
) -> str:
    inp = str(pdf_input).replace("\\", "/")
    out = str(output_dir).replace("\\", "/")
    return textwrap.dedent(f"""\
        general:
          interactive_mode: true
          retain_temporary_jsonl: true
          keep_preprocessed_images: false
          input_paths_is_output_path: false
          output_format: 'txt'
          resume_mode: 'skip'
          logs_dir: ''
          auto_mode_pdf_use_ocr_for_scanned: true
          auto_mode_pdf_use_ocr_for_searchable: true
          auto_mode_pdf_ocr_method: 'gpt'
          auto_mode_image_ocr_method: 'gpt'

        file_paths:
          PDFs:
            input: '{inp}'
            output: '{out}'
          Images:
            input: ''
            output: '{out}'
          EPUBs:
            input: ''
            output: '{out}'
          MOBIs:
            input: ''
            output: '{out}'
          Auto:
            input: ''
            output: '{out}'
    """)


class _InputFeeder:
    """Replaces ui_input() with a queue of scripted responses."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._idx = 0

    def __call__(self, prompt: str, style: str = "") -> str:
        if self._idx >= len(self._responses):
            raise RuntimeError(
                f"Interactive prompt ran out of scripted responses "
                f"at index {self._idx}. Prompt was: {prompt!r}"
            )
        val = self._responses[self._idx]
        self._idx += 1
        return val


@pytest.mark.api
class TestInteractiveMode:

    @staticmethod
    def _make_interactive_config_dir(
        base_tmp: Path,
        provider: str,
        model: str,
        pdf_input: Path,
        output_dir: Path,
        reasoning_effort: str | None = None,
        temperature: float = 0.0,
    ) -> Path:
        config_tmp = base_tmp / "config"
        config_tmp.mkdir(parents=True, exist_ok=True)
        (config_tmp / "model_config.yaml").write_text(
            _make_model_config_yaml(
                provider, model, reasoning_effort, temperature
            ),
            encoding="utf-8",
        )
        (config_tmp / "paths_config.yaml").write_text(
            _make_interactive_paths_config_yaml(pdf_input, output_dir),
            encoding="utf-8",
        )
        (config_tmp / "concurrency_config.yaml").write_text(
            _make_concurrency_config_yaml(),
            encoding="utf-8",
        )
        shutil.copy2(
            CONFIG_DIR / "image_processing_config.yaml",
            config_tmp / "image_processing_config.yaml",
        )
        return config_tmp

    def _run_interactive(
        self,
        tmp_path: Path,
        output_dir: Path,
        provider: str,
        model: str,
        responses: list[str],
        reasoning_effort: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        config_dir = self._make_interactive_config_dir(
            tmp_path, provider, model, STAGING_PDF_DIR, output_dir,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
        )

        old = os.environ.get("CHRONO_CONFIG_DIR")
        os.environ["CHRONO_CONFIG_DIR"] = str(config_dir)
        _cl_mod.CONFIG_DIR = _cl_mod._compute_config_dir()
        _cl_mod.DEFAULT_CONFIG_PATH = _cl_mod.CONFIG_DIR / "model_config.yaml"
        _cl_mod.DEFAULT_PATHS_CONFIG_PATH = (
            _cl_mod.CONFIG_DIR / "paths_config.yaml"
        )
        _cl_mod.DEFAULT_CONCURRENCY_CONFIG_PATH = (
            _cl_mod.CONFIG_DIR / "concurrency_config.yaml"
        )
        _cl_mod.DEFAULT_IMAGE_PROCESSING_CONFIG_PATH = (
            _cl_mod.CONFIG_DIR / "image_processing_config.yaml"
        )
        ConfigService.reset()

        import modules.ui.prompts as prompts_mod

        original_ui_input = prompts_mod.ui_input
        prompts_mod.ui_input = _InputFeeder(responses)
        try:
            from main.unified_transcriber import (
                UnifiedTranscriberScript,
            )

            script = UnifiedTranscriberScript()
            script.execute()
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
        finally:
            prompts_mod.ui_input = original_ui_input
            if old is None:
                os.environ.pop("CHRONO_CONFIG_DIR", None)
            else:
                os.environ["CHRONO_CONFIG_DIR"] = old
            _cl_mod.CONFIG_DIR = _cl_mod._compute_config_dir()
            _cl_mod.DEFAULT_CONFIG_PATH = (
                _cl_mod.CONFIG_DIR / "model_config.yaml"
            )
            _cl_mod.DEFAULT_PATHS_CONFIG_PATH = (
                _cl_mod.CONFIG_DIR / "paths_config.yaml"
            )
            _cl_mod.DEFAULT_CONCURRENCY_CONFIG_PATH = (
                _cl_mod.CONFIG_DIR / "concurrency_config.yaml"
            )
            _cl_mod.DEFAULT_IMAGE_PROCESSING_CONFIG_PATH = (
                _cl_mod.CONFIG_DIR / "image_processing_config.yaml"
            )
            ConfigService.reset()

    # Interactive prompt sequence for PDF + GPT (synchronous):
    #   1. processing_type:  "3" = pdfs
    #   2. method:           "3" = gpt
    #   3. batch:            "2" = synchronous
    #   4. schema:           "1" = first schema (markdown)
    #   5. context:          "1" = hierarchical/auto
    #   6. item_selection:   "1" = process all PDFs
    #   7. page_range:       "2" = yes, specify
    #   8. range text:       "first:1"
    #   9. resume_mode:      "2" = overwrite
    #  10. summary confirm:  "y"
    GPT_PDF_RESPONSES = [
        "3", "3", "2", "1", "1", "1", "2", "first:1", "2", "y"
    ]

    # Interactive prompt sequence for PDF + Tesseract:
    #   1. processing_type:  "3" = pdfs
    #   2. method:           "2" = tesseract
    #   3. (batch skipped for non-gpt)
    #   4. item_selection:   "1" = process all PDFs
    #   5. page_range:       "2" = yes
    #   6. range text:       "first:1"
    #   7. resume_mode:      "2" = overwrite
    #   8. summary confirm:  "y"
    TESSERACT_PDF_RESPONSES = [
        "3", "2", "1", "2", "first:1", "2", "y"
    ]

    def test_15_openai_interactive_pdf(
        self, tmp_path: Path, output_dir: Path
    ) -> None:
        p = PROVIDERS["openai"]
        _require_key(p["env_var"], "OpenAI")
        self._run_interactive(
            tmp_path, output_dir,
            "openai", p["model"],
            self.GPT_PDF_RESPONSES,
            reasoning_effort=p["reasoning_effort"],
            temperature=p["temperature"],
        )
        _assert_output_valid(output_dir)

    def test_16_anthropic_interactive_pdf(
        self, tmp_path: Path, output_dir: Path
    ) -> None:
        p = PROVIDERS["anthropic"]
        _require_key(p["env_var"], "Anthropic")
        self._run_interactive(
            tmp_path, output_dir,
            "anthropic", p["model"],
            self.GPT_PDF_RESPONSES,
            reasoning_effort=p["reasoning_effort"],
            temperature=p["temperature"],
        )
        _assert_output_valid(output_dir)

    def test_17_google_interactive_pdf(
        self, tmp_path: Path, output_dir: Path
    ) -> None:
        p = PROVIDERS["google"]
        _require_key(p["env_var"], "Google")
        self._run_interactive(
            tmp_path, output_dir,
            "google", p["model"],
            self.GPT_PDF_RESPONSES,
            temperature=p["temperature"],
        )
        _assert_output_valid(output_dir)

    def test_18_custom_interactive_pdf(
        self, tmp_path: Path, output_dir: Path
    ) -> None:
        _require_custom_endpoint()
        p = PROVIDERS["custom"]
        self._run_interactive(
            tmp_path, output_dir,
            "custom", p["model"],
            self.GPT_PDF_RESPONSES,
            temperature=p["temperature"],
        )
        _assert_output_valid(output_dir)

    def test_19_tesseract_interactive_pdf(
        self, tmp_path: Path, output_dir: Path
    ) -> None:
        _require_tesseract()

        self._run_interactive(
            tmp_path, output_dir,
            "openai", "gpt-5-nano",
            self.TESSERACT_PDF_RESPONSES,
        )
        _assert_output_valid(output_dir)


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------


def _run_standalone() -> None:
    """Run all tests and print a summary table."""
    test_cells: list[
        tuple[str, str, str, str]
    ] = [
        ("01", "OpenAI",     "CLI",          "test_01_openai_cli_pdf"),
        ("02", "Anthropic",  "CLI",          "test_02_anthropic_cli_pdf"),
        ("03", "Google",     "CLI",          "test_03_google_cli_pdf"),
        ("04", "Custom",     "CLI",          "test_04_custom_cli_pdf"),
        ("05", "Tesseract",  "CLI",          "test_05_tesseract_cli_pdf"),
        ("06", "OpenAI(md)", "CLI",          "test_06_openai_cli_output_format_md"),
        ("07", "OpenAI",     "CLI(resume)",  "test_07_resume_skip_behavior"),
        ("08", "OpenAI",     "CLI(postproc)","test_08_postprocess"),
        ("09", "OpenAI",     "CLI(auto)",    "test_09_auto_mode"),
        ("10", "OpenAI",     "Programmatic", "test_10_openai_programmatic_pdf"),
        ("11", "Anthropic",  "Programmatic", "test_11_anthropic_programmatic_pdf"),
        ("12", "Google",     "Programmatic", "test_12_google_programmatic_pdf"),
        ("13", "Custom",     "Programmatic", "test_13_custom_programmatic_pdf"),
        ("14", "Tesseract",  "Programmatic", "test_14_tesseract_programmatic_pdf"),
        ("15", "OpenAI",     "Interactive",  "test_15_openai_interactive_pdf"),
        ("16", "Anthropic",  "Interactive",  "test_16_anthropic_interactive_pdf"),
        ("17", "Google",     "Interactive",  "test_17_google_interactive_pdf"),
        ("18", "Custom",     "Interactive",  "test_18_custom_interactive_pdf"),
        ("19", "Tesseract",  "Interactive",  "test_19_tesseract_interactive_pdf"),
    ]

    print("\n=== ChronoTranscriber Live API Smoke Test ===\n")
    print(f" {'#':>2}  {'Provider':<14} {'Mode':<16} {'Result':<8} {'Time':>6}")
    print("-" * 55)

    pass_count = skip_count = fail_count = 0

    for num, provider, mode, test_name in test_cells:
        t0 = time.time()
        try:
            result = subprocess.run(
                [
                    sys.executable, "-m", "pytest",
                    str(Path(__file__)),
                    "-k", test_name,
                    "-x", "--tb=short", "--no-header", "-q",
                ],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=920,
            )
            elapsed = time.time() - t0

            if "SKIPPED" in result.stdout or "skipped" in result.stdout:
                status = "SKIP"
                skip_count += 1
            elif result.returncode == 0:
                status = "PASS"
                pass_count += 1
            else:
                status = "FAIL"
                fail_count += 1
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            status = "TIMEOUT"
            fail_count += 1
        except Exception as exc:
            elapsed = time.time() - t0
            status = "ERROR"
            fail_count += 1

        print(
            f" {num:>2}  {provider:<14} {mode:<16} {status:<8} "
            f"{elapsed:5.1f}s"
        )

    print(f"\nResults: {pass_count} PASS, {skip_count} SKIP, {fail_count} FAIL")
    sys.exit(1 if fail_count > 0 else 0)


if __name__ == "__main__":
    _run_standalone()
