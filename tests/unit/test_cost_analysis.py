"""Tests for modules.operations.cost_analysis."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from modules.operations.cost_analysis import (
    MODEL_PRICING,
    TokenUsage,
    FileStats,
    CostAnalysis,
    normalize_model_name,
    extract_token_usage_from_record,
    calculate_cost,
    analyze_jsonl_file,
    find_jsonl_files,
    perform_cost_analysis,
    save_analysis_to_csv,
)


# ---------------------------------------------------------------------------
# normalize_model_name
# ---------------------------------------------------------------------------

class TestNormalizeModelName:
    def test_exact_match(self):
        assert normalize_model_name("gpt-4o") == "gpt-4o"

    def test_date_suffix_stripped(self):
        assert normalize_model_name("gpt-4o-2024-05-13") == "gpt-4o-2024-05-13"

    def test_unknown_model_returned_as_is(self):
        assert normalize_model_name("totally-unknown-model") == "totally-unknown-model"

    def test_anthropic_model(self):
        assert normalize_model_name("claude-3-5-sonnet") == "claude-3-5-sonnet"

    def test_gemini_model(self):
        assert normalize_model_name("gemini-2.5-pro") == "gemini-2.5-pro"

    def test_prefix_matching_with_date_suffix(self):
        # normalize_model_name iterates MODEL_PRICING keys and returns
        # the first key whose name is a prefix; "gpt-5" matches first.
        result = normalize_model_name("gpt-5-mini-2025-08-07")
        assert result in MODEL_PRICING


# ---------------------------------------------------------------------------
# extract_token_usage_from_record
# ---------------------------------------------------------------------------

class TestExtractTokenUsageFromRecord:
    def test_raw_response_format(self):
        record = {
            "raw_response": {
                "model": "gpt-4o",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "total_tokens": 150,
                },
            }
        }
        usage = extract_token_usage_from_record(record)
        assert usage is not None
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
        assert usage.model == "gpt-4o"

    def test_response_data_format(self):
        record = {
            "response_data": {
                "model": "claude-3-5-sonnet",
                "usage": {
                    "prompt_tokens": 200,
                    "completion_tokens": 100,
                    "total_tokens": 300,
                },
            }
        }
        usage = extract_token_usage_from_record(record)
        assert usage is not None
        assert usage.prompt_tokens == 200
        assert usage.completion_tokens == 100
        assert usage.model == "claude-3-5-sonnet"

    def test_cached_tokens_extracted(self):
        record = {
            "raw_response": {
                "model": "gpt-4o",
                "usage": {
                    "input_tokens": 500,
                    "output_tokens": 100,
                    "total_tokens": 600,
                    "input_tokens_details": {"cached_tokens": 200},
                },
            }
        }
        usage = extract_token_usage_from_record(record)
        assert usage.cached_tokens == 200

    def test_reasoning_tokens_extracted(self):
        record = {
            "raw_response": {
                "model": "o1",
                "usage": {
                    "input_tokens": 500,
                    "output_tokens": 300,
                    "total_tokens": 800,
                    "output_tokens_details": {"reasoning_tokens": 200},
                },
            }
        }
        usage = extract_token_usage_from_record(record)
        assert usage.reasoning_tokens == 200

    def test_total_tokens_computed_when_missing(self):
        record = {
            "raw_response": {
                "model": "gpt-4o",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "total_tokens": 0,
                },
            }
        }
        usage = extract_token_usage_from_record(record)
        assert usage.total_tokens == 150

    def test_model_from_request_metadata(self):
        record = {
            "request_metadata": {
                "payload": {"model": "gpt-5"}
            }
        }
        usage = extract_token_usage_from_record(record)
        # The function always tries to parse usage data from raw_response/response_data;
        # when those are empty dicts, it still returns a TokenUsage with zero tokens.
        assert usage is not None
        assert usage.model == "gpt-5"
        assert usage.prompt_tokens == 0

    def test_model_from_request_context(self):
        record = {
            "request_context": {"model": "gemini-2.5-pro"},
            "raw_response": {
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                },
            },
        }
        usage = extract_token_usage_from_record(record)
        assert usage is not None
        assert usage.model == "gemini-2.5-pro"

    def test_no_usage_data_returns_zero_usage(self):
        record = {"some_other_key": "value"}
        usage = extract_token_usage_from_record(record)
        # Function returns TokenUsage with all zeros when raw_response/response_data
        # are empty dicts (which is the default for dict.get with {}).
        assert usage is not None
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0

    def test_empty_record_returns_zero_usage(self):
        usage = extract_token_usage_from_record({})
        assert usage is not None
        assert usage.total_tokens == 0

    def test_none_values_treated_as_zero(self):
        record = {
            "raw_response": {
                "model": "gpt-4o",
                "usage": {
                    "input_tokens": None,
                    "output_tokens": None,
                    "total_tokens": None,
                },
            }
        }
        usage = extract_token_usage_from_record(record)
        assert usage is not None
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0

    def test_prompt_tokens_details_fallback(self):
        record = {
            "raw_response": {
                "model": "gpt-4o",
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                    "prompt_tokens_details": {"cached_tokens": 30},
                    "completion_tokens_details": {"reasoning_tokens": 10},
                },
            }
        }
        usage = extract_token_usage_from_record(record)
        assert usage.cached_tokens == 30
        assert usage.reasoning_tokens == 10


# ---------------------------------------------------------------------------
# calculate_cost
# ---------------------------------------------------------------------------

class TestCalculateCost:
    def test_known_model_cost(self):
        # gpt-4o: input=2.50, cached=1.25, output=10.00 per million
        cost = calculate_cost(1_000_000, 0, 1_000_000, "gpt-4o")
        assert cost == pytest.approx(12.50)

    def test_with_cached_tokens(self):
        # 500k uncached at 2.50/M + 500k cached at 1.25/M + 0 output
        cost = calculate_cost(1_000_000, 500_000, 0, "gpt-4o")
        assert cost == pytest.approx(1.25 + 0.625)

    def test_unknown_model_returns_zero(self):
        cost = calculate_cost(100, 0, 100, "unknown-model-xyz")
        assert cost == 0.0

    def test_discount_applied(self):
        full_cost = calculate_cost(1_000_000, 0, 0, "gpt-4o")
        half_cost = calculate_cost(1_000_000, 0, 0, "gpt-4o", discount=0.5)
        assert half_cost == pytest.approx(full_cost * 0.5)

    def test_zero_tokens(self):
        cost = calculate_cost(0, 0, 0, "gpt-4o")
        assert cost == 0.0


# ---------------------------------------------------------------------------
# analyze_jsonl_file
# ---------------------------------------------------------------------------

class TestAnalyzeJsonlFile:
    def test_analyze_valid_file(self, tmp_path):
        jsonl = tmp_path / "test.jsonl"
        records = [
            {
                "status": "success",
                "raw_response": {
                    "model": "gpt-4o",
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 50,
                        "total_tokens": 150,
                    },
                },
            },
            {
                "status": "success",
                "raw_response": {
                    "model": "gpt-4o",
                    "usage": {
                        "input_tokens": 200,
                        "output_tokens": 80,
                        "total_tokens": 280,
                    },
                },
            },
        ]
        jsonl.write_text(
            "\n".join(json.dumps(r) for r in records),
            encoding="utf-8",
        )
        stats = analyze_jsonl_file(jsonl)
        assert stats.total_chunks == 2
        assert stats.successful_chunks == 2
        assert stats.failed_chunks == 0
        assert stats.prompt_tokens == 300
        assert stats.completion_tokens == 130
        assert stats.model == "gpt-4o"
        assert stats.cost_standard > 0

    def test_analyze_file_with_failures(self, tmp_path):
        jsonl = tmp_path / "test.jsonl"
        records = [
            {"status": "success", "raw_response": {"model": "gpt-4o", "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}},
            {"status": "error"},
        ]
        jsonl.write_text(
            "\n".join(json.dumps(r) for r in records),
            encoding="utf-8",
        )
        stats = analyze_jsonl_file(jsonl)
        assert stats.successful_chunks == 1
        assert stats.failed_chunks == 1

    def test_analyze_empty_file(self, tmp_path):
        jsonl = tmp_path / "empty.jsonl"
        jsonl.write_text("", encoding="utf-8")
        stats = analyze_jsonl_file(jsonl)
        assert stats.total_chunks == 0

    def test_analyze_file_with_invalid_json(self, tmp_path):
        jsonl = tmp_path / "bad.jsonl"
        jsonl.write_text("not json\n{\"status\":\"success\"}\n", encoding="utf-8")
        stats = analyze_jsonl_file(jsonl)
        assert stats.total_chunks == 1

    def test_analyze_nonexistent_file(self, tmp_path):
        jsonl = tmp_path / "missing.jsonl"
        stats = analyze_jsonl_file(jsonl)
        assert stats.total_chunks == 0

    def test_blank_lines_skipped(self, tmp_path):
        jsonl = tmp_path / "blanks.jsonl"
        jsonl.write_text("\n\n{\"status\":\"success\"}\n\n", encoding="utf-8")
        stats = analyze_jsonl_file(jsonl)
        assert stats.total_chunks == 1


# ---------------------------------------------------------------------------
# find_jsonl_files
# ---------------------------------------------------------------------------

class TestFindJsonlFiles:
    def test_finds_files_in_input_dir(self, tmp_path):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "test.jsonl").write_text("{}", encoding="utf-8")

        paths_config = {"general": {"input_paths_is_output_path": True}}
        schemas_paths = {"schema1": {"input": str(input_dir)}}
        result = find_jsonl_files(paths_config, schemas_paths)
        assert len(result) == 1

    def test_finds_files_in_output_dir(self, tmp_path):
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "result.jsonl").write_text("{}", encoding="utf-8")

        paths_config = {"general": {"input_paths_is_output_path": False}}
        schemas_paths = {"schema1": {"output": str(output_dir)}}
        result = find_jsonl_files(paths_config, schemas_paths)
        assert len(result) == 1

    def test_no_duplicates(self, tmp_path):
        dir_ = tmp_path / "data"
        dir_.mkdir()
        (dir_ / "test.jsonl").write_text("{}", encoding="utf-8")

        paths_config = {"general": {"input_paths_is_output_path": True}}
        schemas_paths = {
            "s1": {"input": str(dir_)},
            "s2": {"input": str(dir_)},
        }
        result = find_jsonl_files(paths_config, schemas_paths)
        assert len(result) == 1

    def test_nonexistent_directory_skipped(self):
        paths_config = {"general": {"input_paths_is_output_path": True}}
        schemas_paths = {"s1": {"input": "/nonexistent/path"}}
        result = find_jsonl_files(paths_config, schemas_paths)
        assert result == []

    def test_finds_files_in_subdirectories(self, tmp_path):
        base = tmp_path / "input"
        sub = base / "sub"
        sub.mkdir(parents=True)
        (sub / "deep.jsonl").write_text("{}", encoding="utf-8")

        paths_config = {"general": {"input_paths_is_output_path": True}}
        schemas_paths = {"s1": {"input": str(base)}}
        result = find_jsonl_files(paths_config, schemas_paths)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# perform_cost_analysis
# ---------------------------------------------------------------------------

class TestPerformCostAnalysis:
    def test_aggregates_multiple_files(self, tmp_path):
        for i in range(2):
            path = tmp_path / f"file{i}.jsonl"
            record = {
                "status": "success",
                "raw_response": {
                    "model": "gpt-4o",
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 50,
                        "total_tokens": 150,
                    },
                },
            }
            path.write_text(json.dumps(record), encoding="utf-8")

        files = [tmp_path / f"file{i}.jsonl" for i in range(2)]
        analysis = perform_cost_analysis(files)
        assert analysis.total_files == 2
        assert analysis.total_chunks == 2
        assert analysis.total_prompt_tokens == 200
        assert analysis.total_completion_tokens == 100
        assert "gpt-4o" in analysis.models_used

    def test_empty_file_list(self):
        analysis = perform_cost_analysis([])
        assert analysis.total_files == 0
        assert analysis.total_chunks == 0


# ---------------------------------------------------------------------------
# save_analysis_to_csv
# ---------------------------------------------------------------------------

class TestSaveAnalysisToCsv:
    def test_saves_valid_csv(self, tmp_path):
        analysis = CostAnalysis(
            file_stats=[
                FileStats(
                    file_path=Path("test.jsonl"),
                    model="gpt-4o",
                    total_chunks=5,
                    successful_chunks=4,
                    failed_chunks=1,
                    prompt_tokens=1000,
                    cached_tokens=100,
                    completion_tokens=500,
                    reasoning_tokens=0,
                    total_tokens=1500,
                    cost_standard=0.015,
                    cost_discounted=0.0075,
                )
            ],
            total_files=1,
            total_chunks=5,
            total_prompt_tokens=1000,
            total_cached_tokens=100,
            total_completion_tokens=500,
            total_reasoning_tokens=0,
            total_tokens=1500,
            total_cost_standard=0.015,
            total_cost_discounted=0.0075,
            models_used={"gpt-4o": 1},
        )
        csv_path = tmp_path / "analysis.csv"
        save_analysis_to_csv(analysis, csv_path)
        assert csv_path.exists()

        with csv_path.open("r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
        # Header + 1 data row + blank + summary
        assert len(reader) >= 3
        assert reader[0][0] == "File"


# ---------------------------------------------------------------------------
# TokenUsage / FileStats / CostAnalysis dataclasses
# ---------------------------------------------------------------------------

class TestTokenUsageDataclass:
    def test_defaults(self):
        usage = TokenUsage()
        assert usage.prompt_tokens == 0
        assert usage.cached_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.reasoning_tokens == 0
        assert usage.total_tokens == 0
        assert usage.model == ""


class TestFileStatsDataclass:
    def test_defaults(self, tmp_path):
        stats = FileStats(file_path=tmp_path / "x.jsonl")
        assert stats.total_chunks == 0
        assert stats.cost_standard == 0.0


class TestCostAnalysisDataclass:
    def test_defaults(self):
        analysis = CostAnalysis()
        assert analysis.file_stats == []
        assert analysis.models_used == {}
