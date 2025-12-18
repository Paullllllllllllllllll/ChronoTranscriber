"""
Evaluation metrics for transcription quality assessment.

This module provides functions to compute Character Error Rate (CER) and 
Word Error Rate (WER) using Levenshtein distance, enabling comparison of
model outputs against ground truth transcriptions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict


# =============================================================================
# Regex patterns for markdown formatting and page markers
# =============================================================================

# Page markers: <page_number>123</page_number> or <page_number>123<page_number>
PAGE_MARKER_PATTERN = re.compile(r'<page_number>.*?</?page_number>', re.IGNORECASE)

# Markdown formatting patterns
MARKDOWN_PATTERNS = {
    'bold': re.compile(r'\*\*[^*]+\*\*'),
    'italic': re.compile(r'(?<!\*)\*[^*]+\*(?!\*)'),
    'heading': re.compile(r'^#{1,6}\s+.+$', re.MULTILINE),
    'footnote': re.compile(r'\[\^\d+\]:?'),
    'latex': re.compile(r'\$\$[^$]+\$\$'),
    'image_desc': re.compile(r'\[Image:[^\]]+\]'),
}

# Combined pattern for all formatting elements
ALL_FORMATTING_PATTERN = re.compile(
    r'(<page_number>.*?</?page_number>)|'  # Page markers
    r'(\*\*[^*]+\*\*)|'  # Bold
    r'(?<!\*)(\*[^*]+\*)(?!\*)|'  # Italic
    r'(^#{1,6}\s+)|'  # Heading markers (just the # symbols)
    r'(\[\^\d+\]:?)|'  # Footnotes
    r'(\$\$[^$]+\$\$)|'  # LaTeX
    r'(\[Image:[^\]]+\])',  # Image descriptions
    re.MULTILINE | re.IGNORECASE
)


@dataclass
class FormattingMetrics:
    """Container for formatting-specific evaluation metrics."""
    
    # Page marker metrics
    page_marker_cer: float = 0.0
    page_marker_distance: int = 0
    ref_page_marker_count: int = 0
    hyp_page_marker_count: int = 0
    ref_page_marker_chars: int = 0
    hyp_page_marker_chars: int = 0
    
    # Markdown formatting metrics
    markdown_cer: float = 0.0
    markdown_distance: int = 0
    ref_markdown_count: int = 0
    hyp_markdown_count: int = 0
    ref_markdown_chars: int = 0
    hyp_markdown_chars: int = 0
    
    def to_dict(self) -> dict:
        """Convert metrics to dictionary for serialization."""
        return {
            "page_marker_cer": round(self.page_marker_cer, 4),
            "page_marker_cer_percent": round(self.page_marker_cer * 100, 2),
            "page_marker_distance": self.page_marker_distance,
            "ref_page_marker_count": self.ref_page_marker_count,
            "hyp_page_marker_count": self.hyp_page_marker_count,
            "ref_page_marker_chars": self.ref_page_marker_chars,
            "hyp_page_marker_chars": self.hyp_page_marker_chars,
            "markdown_cer": round(self.markdown_cer, 4),
            "markdown_cer_percent": round(self.markdown_cer * 100, 2),
            "markdown_distance": self.markdown_distance,
            "ref_markdown_count": self.ref_markdown_count,
            "hyp_markdown_count": self.hyp_markdown_count,
            "ref_markdown_chars": self.ref_markdown_chars,
            "hyp_markdown_chars": self.hyp_markdown_chars,
        }


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
    
    # Content-only metrics (excluding formatting)
    content_cer: float = 0.0  # CER for text content only
    content_wer: float = 0.0  # WER for text content only
    content_char_distance: int = 0
    content_word_distance: int = 0
    ref_content_chars: int = 0
    ref_content_words: int = 0
    
    # Formatting-specific metrics
    formatting: Optional[FormattingMetrics] = None
    
    def to_dict(self) -> dict:
        """Convert metrics to dictionary for serialization."""
        result = {
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
            # Content-only metrics
            "content_cer": round(self.content_cer, 4),
            "content_wer": round(self.content_wer, 4),
            "content_cer_percent": round(self.content_cer * 100, 2),
            "content_wer_percent": round(self.content_wer * 100, 2),
            "content_char_distance": self.content_char_distance,
            "content_word_distance": self.content_word_distance,
            "ref_content_chars": self.ref_content_chars,
            "ref_content_words": self.ref_content_words,
        }
        if self.formatting:
            result["formatting"] = self.formatting.to_dict()
        return result


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


def extract_formatting_elements(text: str) -> Tuple[List[str], List[str]]:
    """
    Extract page markers and markdown formatting elements from text.
    
    Args:
        text: Input text
        
    Returns:
        Tuple of (page_markers, markdown_elements)
    """
    if text is None:
        return [], []
    
    page_markers = PAGE_MARKER_PATTERN.findall(text)
    
    markdown_elements = []
    for pattern_name, pattern in MARKDOWN_PATTERNS.items():
        matches = pattern.findall(text)
        markdown_elements.extend(matches)
    
    return page_markers, markdown_elements


def strip_formatting(text: str) -> str:
    """
    Remove all formatting elements (page markers and markdown) from text.
    
    Args:
        text: Input text
        
    Returns:
        Text with formatting elements removed
    """
    if text is None:
        return ""
    
    # Remove page markers
    result = PAGE_MARKER_PATTERN.sub('', text)
    
    # Remove markdown formatting symbols but keep content
    # Bold: **text** -> text
    result = re.sub(r'\*\*([^*]+)\*\*', r'\1', result)
    # Italic: *text* -> text
    result = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'\1', result)
    # Headings: # Heading -> Heading
    result = re.sub(r'^#{1,6}\s+', '', result, flags=re.MULTILINE)
    # Footnotes: [^1]: -> (keep reference text)
    result = re.sub(r'\[\^(\d+)\]:?', r'[\1]', result)
    # LaTeX: $$equation$$ -> equation
    result = re.sub(r'\$\$([^$]+)\$\$', r'\1', result)
    # Image descriptions: keep as-is (they're content descriptors)
    
    return result


def compute_formatting_metrics(
    reference: str,
    hypothesis: str,
    normalize: bool = True,
) -> FormattingMetrics:
    """
    Compute metrics specifically for formatting elements.
    
    Args:
        reference: Ground truth text
        hypothesis: Model output text
        normalize: Whether to normalize whitespace
        
    Returns:
        FormattingMetrics object
    """
    ref_page_markers, ref_markdown = extract_formatting_elements(reference)
    hyp_page_markers, hyp_markdown = extract_formatting_elements(hypothesis)
    
    # Page marker metrics
    ref_pm_text = ' '.join(ref_page_markers)
    hyp_pm_text = ' '.join(hyp_page_markers)
    ref_pm_chars = len(ref_pm_text.replace(' ', ''))
    hyp_pm_chars = len(hyp_pm_text.replace(' ', ''))
    
    if ref_pm_chars > 0:
        pm_dist = levenshtein_distance(list(ref_pm_text), list(hyp_pm_text))
        pm_cer = pm_dist / ref_pm_chars
    else:
        pm_dist = len(hyp_pm_text)
        pm_cer = float(pm_dist) if pm_dist > 0 else 0.0
    
    # Markdown metrics
    ref_md_text = ' '.join(ref_markdown)
    hyp_md_text = ' '.join(hyp_markdown)
    ref_md_chars = len(ref_md_text.replace(' ', ''))
    hyp_md_chars = len(hyp_md_text.replace(' ', ''))
    
    if ref_md_chars > 0:
        md_dist = levenshtein_distance(list(ref_md_text), list(hyp_md_text))
        md_cer = md_dist / ref_md_chars
    else:
        md_dist = len(hyp_md_text)
        md_cer = float(md_dist) if md_dist > 0 else 0.0
    
    return FormattingMetrics(
        page_marker_cer=pm_cer,
        page_marker_distance=pm_dist,
        ref_page_marker_count=len(ref_page_markers),
        hyp_page_marker_count=len(hyp_page_markers),
        ref_page_marker_chars=ref_pm_chars,
        hyp_page_marker_chars=hyp_pm_chars,
        markdown_cer=md_cer,
        markdown_distance=md_dist,
        ref_markdown_count=len(ref_markdown),
        hyp_markdown_count=len(hyp_markdown),
        ref_markdown_chars=ref_md_chars,
        hyp_markdown_chars=hyp_md_chars,
    )


def compute_metrics(
    reference: str,
    hypothesis: str,
    normalize: bool = True,
    compute_formatting: bool = True,
) -> TranscriptionMetrics:
    """
    Compute both CER and WER for a transcription.
    
    Computes three levels of metrics:
    1. Overall: Full text including all formatting
    2. Content-only: Text with formatting elements stripped
    3. Formatting: Separate metrics for page markers and markdown
    
    Args:
        reference: Ground truth text
        hypothesis: Model output text
        normalize: Whether to normalize whitespace before comparison
        compute_formatting: Whether to compute separate formatting metrics
        
    Returns:
        TranscriptionMetrics object with all computed values
    """
    # Overall metrics (full text)
    cer, char_dist, ref_chars, hyp_chars = compute_cer(reference, hypothesis, normalize)
    wer, word_dist, ref_words, hyp_words = compute_wer(reference, hypothesis, normalize)
    
    # Content-only metrics (formatting stripped)
    ref_content = strip_formatting(reference)
    hyp_content = strip_formatting(hypothesis)
    
    content_cer, content_char_dist, ref_content_chars, _ = compute_cer(
        ref_content, hyp_content, normalize
    )
    content_wer, content_word_dist, ref_content_words, _ = compute_wer(
        ref_content, hyp_content, normalize
    )
    
    # Formatting-specific metrics
    formatting_metrics = None
    if compute_formatting:
        formatting_metrics = compute_formatting_metrics(reference, hypothesis, normalize)
    
    return TranscriptionMetrics(
        cer=cer,
        wer=wer,
        char_distance=char_dist,
        word_distance=word_dist,
        ref_char_count=ref_chars,
        ref_word_count=ref_words,
        hyp_char_count=hyp_chars,
        hyp_word_count=hyp_words,
        content_cer=content_cer,
        content_wer=content_wer,
        content_char_distance=content_char_dist,
        content_word_distance=content_word_dist,
        ref_content_chars=ref_content_chars,
        ref_content_words=ref_content_words,
        formatting=formatting_metrics,
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
    
    # Overall metrics
    total_char_dist = sum(m.char_distance for m in metrics_list)
    total_word_dist = sum(m.word_distance for m in metrics_list)
    total_ref_chars = sum(m.ref_char_count for m in metrics_list)
    total_ref_words = sum(m.ref_word_count for m in metrics_list)
    total_hyp_chars = sum(m.hyp_char_count for m in metrics_list)
    total_hyp_words = sum(m.hyp_word_count for m in metrics_list)
    
    agg_cer = total_char_dist / total_ref_chars if total_ref_chars > 0 else 0.0
    agg_wer = total_word_dist / total_ref_words if total_ref_words > 0 else 0.0
    
    # Content-only metrics
    total_content_char_dist = sum(m.content_char_distance for m in metrics_list)
    total_content_word_dist = sum(m.content_word_distance for m in metrics_list)
    total_ref_content_chars = sum(m.ref_content_chars for m in metrics_list)
    total_ref_content_words = sum(m.ref_content_words for m in metrics_list)
    
    agg_content_cer = total_content_char_dist / total_ref_content_chars if total_ref_content_chars > 0 else 0.0
    agg_content_wer = total_content_word_dist / total_ref_content_words if total_ref_content_words > 0 else 0.0
    
    # Aggregate formatting metrics
    formatting_metrics_list = [m.formatting for m in metrics_list if m.formatting is not None]
    agg_formatting = None
    
    if formatting_metrics_list:
        total_pm_dist = sum(f.page_marker_distance for f in formatting_metrics_list)
        total_ref_pm_chars = sum(f.ref_page_marker_chars for f in formatting_metrics_list)
        total_hyp_pm_chars = sum(f.hyp_page_marker_chars for f in formatting_metrics_list)
        total_ref_pm_count = sum(f.ref_page_marker_count for f in formatting_metrics_list)
        total_hyp_pm_count = sum(f.hyp_page_marker_count for f in formatting_metrics_list)
        
        total_md_dist = sum(f.markdown_distance for f in formatting_metrics_list)
        total_ref_md_chars = sum(f.ref_markdown_chars for f in formatting_metrics_list)
        total_hyp_md_chars = sum(f.hyp_markdown_chars for f in formatting_metrics_list)
        total_ref_md_count = sum(f.ref_markdown_count for f in formatting_metrics_list)
        total_hyp_md_count = sum(f.hyp_markdown_count for f in formatting_metrics_list)
        
        agg_pm_cer = total_pm_dist / total_ref_pm_chars if total_ref_pm_chars > 0 else 0.0
        agg_md_cer = total_md_dist / total_ref_md_chars if total_ref_md_chars > 0 else 0.0
        
        agg_formatting = FormattingMetrics(
            page_marker_cer=agg_pm_cer,
            page_marker_distance=total_pm_dist,
            ref_page_marker_count=total_ref_pm_count,
            hyp_page_marker_count=total_hyp_pm_count,
            ref_page_marker_chars=total_ref_pm_chars,
            hyp_page_marker_chars=total_hyp_pm_chars,
            markdown_cer=agg_md_cer,
            markdown_distance=total_md_dist,
            ref_markdown_count=total_ref_md_count,
            hyp_markdown_count=total_hyp_md_count,
            ref_markdown_chars=total_ref_md_chars,
            hyp_markdown_chars=total_hyp_md_chars,
        )
    
    return TranscriptionMetrics(
        cer=agg_cer,
        wer=agg_wer,
        char_distance=total_char_dist,
        word_distance=total_word_dist,
        ref_char_count=total_ref_chars,
        ref_word_count=total_ref_words,
        hyp_char_count=total_hyp_chars,
        hyp_word_count=total_hyp_words,
        content_cer=agg_content_cer,
        content_wer=agg_content_wer,
        content_char_distance=total_content_char_dist,
        content_word_distance=total_content_word_dist,
        ref_content_chars=total_ref_content_chars,
        ref_content_words=total_ref_content_words,
        formatting=agg_formatting,
    )


def format_metrics_table(
    model_metrics: dict[str, dict[str, TranscriptionMetrics]],
    categories: Optional[List[str]] = None,
    include_formatting: bool = False,
) -> str:
    """
    Format metrics as a Markdown table for display.
    
    Args:
        model_metrics: Dict mapping model_name -> category -> metrics
        categories: List of category names (if None, derived from data)
        include_formatting: Whether to include formatting-specific metrics
        
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
    
    # Build header - show overall and content-only metrics
    lines = ["| Model | Category | CER (%) | WER (%) | Content CER (%) | Content WER (%) | Ref Chars |"]
    lines.append("|-------|----------|---------|---------|-----------------|-----------------|-----------|")
    
    # Add rows
    for model_name in sorted(model_metrics.keys()):
        for category in categories:
            if category in model_metrics[model_name]:
                m = model_metrics[model_name][category]
                lines.append(
                    f"| {model_name} | {category} | {m.cer*100:.2f} | {m.wer*100:.2f} | "
                    f"{m.content_cer*100:.2f} | {m.content_wer*100:.2f} | {m.ref_char_count:,} |"
                )
    
    result = "\n".join(lines)
    
    # Optionally add formatting metrics table
    if include_formatting:
        fmt_lines = ["\n\n### Formatting Metrics\n"]
        fmt_lines.append("| Model | Category | Page Marker CER (%) | Markdown CER (%) | Page Markers | Markdown Elements |")
        fmt_lines.append("|-------|----------|---------------------|------------------|--------------|-------------------|")
        
        for model_name in sorted(model_metrics.keys()):
            for category in categories:
                if category in model_metrics[model_name]:
                    m = model_metrics[model_name][category]
                    if m.formatting:
                        f = m.formatting
                        fmt_lines.append(
                            f"| {model_name} | {category} | {f.page_marker_cer*100:.2f} | "
                            f"{f.markdown_cer*100:.2f} | {f.ref_page_marker_count} | {f.ref_markdown_count} |"
                        )
                    else:
                        fmt_lines.append(f"| {model_name} | {category} | N/A | N/A | N/A | N/A |")
        
        result += "\n".join(fmt_lines)
    
    return result


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
