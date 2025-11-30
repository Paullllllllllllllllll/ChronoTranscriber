"""
Evaluation metrics for transcription quality assessment.

This module provides functions to compute Character Error Rate (CER) and 
Word Error Rate (WER) using Levenshtein distance, enabling comparison of
model outputs against ground truth transcriptions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class TranscriptionMetrics:
    """Container for transcription evaluation metrics."""
    
    cer: float  # Character Error Rate (0.0 to 1.0+)
    wer: float  # Word Error Rate (0.0 to 1.0+)
    char_distance: int  # Raw Levenshtein distance (characters)
    word_distance: int  # Raw Levenshtein distance (words)
    ref_char_count: int  # Total characters in reference
    ref_word_count: int  # Total words in reference
    hyp_char_count: int  # Total characters in hypothesis
    hyp_word_count: int  # Total words in hypothesis
    
    def to_dict(self) -> dict:
        """Convert metrics to dictionary for serialization."""
        return {
            "cer": round(self.cer, 4),
            "wer": round(self.wer, 4),
            "cer_percent": round(self.cer * 100, 2),
            "wer_percent": round(self.wer * 100, 2),
            "char_distance": self.char_distance,
            "word_distance": self.word_distance,
            "ref_char_count": self.ref_char_count,
            "ref_word_count": self.ref_word_count,
            "hyp_char_count": self.hyp_char_count,
            "hyp_word_count": self.hyp_word_count,
        }


def levenshtein_distance(s1: List[str], s2: List[str]) -> int:
    """
    Compute the Levenshtein distance between two sequences.
    
    Uses dynamic programming with O(min(m,n)) space complexity.
    
    Args:
        s1: First sequence (reference)
        s2: Second sequence (hypothesis)
        
    Returns:
        Minimum number of insertions, deletions, and substitutions
        required to transform s2 into s1.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # j+1 instead of j since previous_row and current_row are 1 longer
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def normalize_text(text: str, lowercase: bool = False) -> str:
    """
    Normalize text for comparison.
    
    Args:
        text: Input text to normalize
        lowercase: Whether to convert to lowercase
        
    Returns:
        Normalized text with standardized whitespace
    """
    if text is None:
        return ""
    
    # Normalize whitespace (collapse multiple spaces/newlines to single space)
    normalized = re.sub(r'\s+', ' ', text.strip())
    
    if lowercase:
        normalized = normalized.lower()
    
    return normalized


def tokenize_words(text: str) -> List[str]:
    """
    Tokenize text into words.
    
    Args:
        text: Input text
        
    Returns:
        List of word tokens
    """
    # Split on whitespace, filter empty strings
    return [w for w in text.split() if w]


def compute_cer(reference: str, hypothesis: str, normalize: bool = True) -> Tuple[float, int, int, int]:
    """
    Compute Character Error Rate.
    
    CER = (S + D + I) / N
    where S = substitutions, D = deletions, I = insertions, N = reference length
    
    Args:
        reference: Ground truth text
        hypothesis: Model output text
        normalize: Whether to normalize whitespace before comparison
        
    Returns:
        Tuple of (CER, distance, ref_length, hyp_length)
    """
    if normalize:
        reference = normalize_text(reference)
        hypothesis = normalize_text(hypothesis)
    
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    
    if len(ref_chars) == 0:
        # If reference is empty, CER is 0 if hypothesis is also empty, else undefined (use hyp length)
        if len(hyp_chars) == 0:
            return 0.0, 0, 0, 0
        return float(len(hyp_chars)), len(hyp_chars), 0, len(hyp_chars)
    
    distance = levenshtein_distance(ref_chars, hyp_chars)
    cer = distance / len(ref_chars)
    
    return cer, distance, len(ref_chars), len(hyp_chars)


def compute_wer(reference: str, hypothesis: str, normalize: bool = True) -> Tuple[float, int, int, int]:
    """
    Compute Word Error Rate.
    
    WER = (S + D + I) / N
    where S = substitutions, D = deletions, I = insertions, N = reference word count
    
    Args:
        reference: Ground truth text
        hypothesis: Model output text
        normalize: Whether to normalize whitespace before comparison
        
    Returns:
        Tuple of (WER, distance, ref_word_count, hyp_word_count)
    """
    if normalize:
        reference = normalize_text(reference)
        hypothesis = normalize_text(hypothesis)
    
    ref_words = tokenize_words(reference)
    hyp_words = tokenize_words(hypothesis)
    
    if len(ref_words) == 0:
        if len(hyp_words) == 0:
            return 0.0, 0, 0, 0
        return float(len(hyp_words)), len(hyp_words), 0, len(hyp_words)
    
    distance = levenshtein_distance(ref_words, hyp_words)
    wer = distance / len(ref_words)
    
    return wer, distance, len(ref_words), len(hyp_words)


def compute_metrics(reference: str, hypothesis: str, normalize: bool = True) -> TranscriptionMetrics:
    """
    Compute both CER and WER for a transcription.
    
    Args:
        reference: Ground truth text
        hypothesis: Model output text
        normalize: Whether to normalize whitespace before comparison
        
    Returns:
        TranscriptionMetrics object with all computed values
    """
    cer, char_dist, ref_chars, hyp_chars = compute_cer(reference, hypothesis, normalize)
    wer, word_dist, ref_words, hyp_words = compute_wer(reference, hypothesis, normalize)
    
    return TranscriptionMetrics(
        cer=cer,
        wer=wer,
        char_distance=char_dist,
        word_distance=word_dist,
        ref_char_count=ref_chars,
        ref_word_count=ref_words,
        hyp_char_count=hyp_chars,
        hyp_word_count=hyp_words,
    )


def aggregate_metrics(metrics_list: List[TranscriptionMetrics]) -> TranscriptionMetrics:
    """
    Aggregate metrics across multiple documents/pages.
    
    Uses micro-averaging: sum all distances and divide by sum of reference lengths.
    This weights longer documents more heavily, which is typically desired.
    
    Args:
        metrics_list: List of per-document/page metrics
        
    Returns:
        Aggregated TranscriptionMetrics
    """
    if not metrics_list:
        return TranscriptionMetrics(
            cer=0.0, wer=0.0,
            char_distance=0, word_distance=0,
            ref_char_count=0, ref_word_count=0,
            hyp_char_count=0, hyp_word_count=0,
        )
    
    total_char_dist = sum(m.char_distance for m in metrics_list)
    total_word_dist = sum(m.word_distance for m in metrics_list)
    total_ref_chars = sum(m.ref_char_count for m in metrics_list)
    total_ref_words = sum(m.ref_word_count for m in metrics_list)
    total_hyp_chars = sum(m.hyp_char_count for m in metrics_list)
    total_hyp_words = sum(m.hyp_word_count for m in metrics_list)
    
    agg_cer = total_char_dist / total_ref_chars if total_ref_chars > 0 else 0.0
    agg_wer = total_word_dist / total_ref_words if total_ref_words > 0 else 0.0
    
    return TranscriptionMetrics(
        cer=agg_cer,
        wer=agg_wer,
        char_distance=total_char_dist,
        word_distance=total_word_dist,
        ref_char_count=total_ref_chars,
        ref_word_count=total_ref_words,
        hyp_char_count=total_hyp_chars,
        hyp_word_count=total_hyp_words,
    )


def format_metrics_table(
    model_metrics: dict[str, dict[str, TranscriptionMetrics]],
    categories: Optional[List[str]] = None,
) -> str:
    """
    Format metrics as a Markdown table for display.
    
    Args:
        model_metrics: Dict mapping model_name -> category -> metrics
        categories: List of category names (if None, derived from data)
        
    Returns:
        Markdown-formatted table string
    """
    if not model_metrics:
        return "No metrics to display."
    
    # Collect all categories
    if categories is None:
        categories = set()
        for cat_metrics in model_metrics.values():
            categories.update(cat_metrics.keys())
        categories = sorted(categories)
    
    # Build header
    lines = ["| Model | Category | CER (%) | WER (%) | Ref Chars | Ref Words |"]
    lines.append("|-------|----------|---------|---------|-----------|-----------|")
    
    # Add rows
    for model_name in sorted(model_metrics.keys()):
        for category in categories:
            if category in model_metrics[model_name]:
                m = model_metrics[model_name][category]
                lines.append(
                    f"| {model_name} | {category} | {m.cer*100:.2f} | {m.wer*100:.2f} | "
                    f"{m.ref_char_count:,} | {m.ref_word_count:,} |"
                )
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Quick test
    ref = "The quick brown fox jumps over the lazy dog."
    hyp = "The quikc brown fox jumps over the lazy dog"
    
    metrics = compute_metrics(ref, hyp)
    print(f"Reference: {ref}")
    print(f"Hypothesis: {hyp}")
    print(f"CER: {metrics.cer:.4f} ({metrics.cer*100:.2f}%)")
    print(f"WER: {metrics.wer:.4f} ({metrics.wer*100:.2f}%)")
    print(f"Character distance: {metrics.char_distance}")
    print(f"Word distance: {metrics.word_distance}")
