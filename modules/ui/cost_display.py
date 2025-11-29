# modules/ui/cost_display.py

"""
Display functions for cost analysis results.
"""

from __future__ import annotations

from typing import Optional

from modules.operations.cost_analysis import CostAnalysis
from modules.ui import print_info, print_success, print_header, ui_print, PromptStyle


def display_analysis(analysis: CostAnalysis, interactive_mode: bool = True) -> None:
    """
    Display cost analysis results.
    
    Args:
        analysis: CostAnalysis object
        interactive_mode: Whether to use interactive UI formatting
    """
    if interactive_mode:
        print_header("Token Cost Analysis")
        print_info(f"Total files analyzed: {analysis.total_files}")
        print_info(f"Total chunks processed: {analysis.total_chunks}")
        print_info("")
        
        print_info("Token Usage:")
        print_info(f"  Prompt tokens: {analysis.total_prompt_tokens:,}")
        print_info(f"  Cached tokens: {analysis.total_cached_tokens:,}")
        print_info(f"  Completion tokens: {analysis.total_completion_tokens:,}")
        if analysis.total_reasoning_tokens > 0:
            print_info(f"  Reasoning tokens: {analysis.total_reasoning_tokens:,}")
        print_info(f"  Total tokens: {analysis.total_tokens:,}")
        print_info("")
        
        print_info("Models Used:")
        for model, count in analysis.models_used.items():
            print_info(f"  {model}: {count} file(s)")
        print_info("")
        
        print_header("Cost Estimates")
        print_info(f"Standard pricing: ${analysis.total_cost_standard:.4f}")
        print_success(f"Discounted pricing (50% off): ${analysis.total_cost_discounted:.4f}")
        print_info(f"Potential savings: ${analysis.total_cost_standard - analysis.total_cost_discounted:.4f}")
        print_info("")
        
        if analysis.file_stats:
            print_header("Per-File Breakdown")
            for stats in analysis.file_stats:
                print_info(f"\n{stats.file_path.name}:")
                print_info(f"  Model: {stats.model or 'Unknown'}")
                print_info(f"  Chunks: {stats.successful_chunks}/{stats.total_chunks} successful")
                print_info(f"  Tokens: {stats.total_tokens:,}")
                print_info(f"  Cost (standard): ${stats.cost_standard:.4f}")
                print_info(f"  Cost (discounted): ${stats.cost_discounted:.4f}")
    else:
        # CLI mode output
        print("\n=== Token Cost Analysis ===")
        print(f"Total files analyzed: {analysis.total_files}")
        print(f"Total chunks processed: {analysis.total_chunks}")
        print("")
        
        print("Token Usage:")
        print(f"  Prompt tokens: {analysis.total_prompt_tokens:,}")
        print(f"  Cached tokens: {analysis.total_cached_tokens:,}")
        print(f"  Completion tokens: {analysis.total_completion_tokens:,}")
        if analysis.total_reasoning_tokens > 0:
            print(f"  Reasoning tokens: {analysis.total_reasoning_tokens:,}")
        print(f"  Total tokens: {analysis.total_tokens:,}")
        print("")
        
        print("Models Used:")
        for model, count in analysis.models_used.items():
            print(f"  {model}: {count} file(s)")
        print("")
        
        print("=== Cost Estimates ===")
        print(f"Standard pricing: ${analysis.total_cost_standard:.4f}")
        print(f"Discounted pricing (50% off): ${analysis.total_cost_discounted:.4f}")
        print(f"Potential savings: ${analysis.total_cost_standard - analysis.total_cost_discounted:.4f}")
        print("")
        
        if analysis.file_stats:
            print("=== Per-File Breakdown ===")
            for stats in analysis.file_stats:
                print(f"\n{stats.file_path.name}:")
                print(f"  Model: {stats.model or 'Unknown'}")
                print(f"  Chunks: {stats.successful_chunks}/{stats.total_chunks} successful")
                print(f"  Tokens: {stats.total_tokens:,}")
                print(f"  Cost (standard): ${stats.cost_standard:.4f}")
                print(f"  Cost (discounted): ${stats.cost_discounted:.4f}")
