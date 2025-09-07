# modules/repair_utils.py

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Regex patterns used to detect failure placeholders in final text
FAILURE_PATTERNS = [
    re.compile(r"^\[transcription error:\s*.+\]$", re.IGNORECASE),
    re.compile(r"^\[Transcription not possible.*\]$", re.IGNORECASE),
]
NO_TEXT_PATTERN = re.compile(r"^\[No transcribable text.*\]$", re.IGNORECASE)


@dataclass
class ImageEntry:
    order_index: int
    image_name: str
    pre_processed_image: Optional[str]
    custom_id: Optional[str]
    page_number: Optional[int] = None


def extract_image_name_from_failure_line(line: str) -> Optional[str]:
    """
    Extract the image file name from a placeholder line such as:
      - "[Transcription not possible: IMG_0001.png]"
      - "[No transcribable text: page_12.jpg]"
      - "[transcription error: scan_03.png; status 400; code invalid_image]"
    Returns the extracted image name if found, else None.
    """
    pattern = re.compile(
        r"^\[(?:transcription error|Transcription not possible|No transcribable text):\s*([^;]+)",
        re.IGNORECASE,
    )
    m = pattern.match(line.strip())
    if m:
        return m.group(1).strip()
    return None


def is_failure_line(line: str) -> bool:
    for pat in FAILURE_PATTERNS:
        if pat.match(line.strip()):
            return True
    return False


def collect_image_entries_from_jsonl(temp_jsonl_path: Optional[Path]) -> List[ImageEntry]:
    """
    Parse local transcription JSONL to reconstruct page ordering and metadata.
    Returns a list of ImageEntry sorted by order_index.
    """
    entries: Dict[int, ImageEntry] = {}
    if temp_jsonl_path is None or not temp_jsonl_path.exists():
        return []

    try:
        with temp_jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                # Preferred: image_metadata
                if "image_metadata" in obj and isinstance(obj["image_metadata"], dict):
                    meta = obj["image_metadata"]
                    oi = meta.get("order_index")
                    if isinstance(oi, int):
                        entries[oi] = ImageEntry(
                            order_index=oi,
                            image_name=str(meta.get("image_name") or "").strip(),
                            pre_processed_image=str(meta.get("pre_processed_image") or "").strip() or None,
                            custom_id=str(meta.get("custom_id") or "").strip() or None,
                            page_number=meta.get("page_number"),
                        )
                    continue

                # Fallback: batch_request lines
                if "batch_request" in obj and isinstance(obj["batch_request"], dict):
                    br = obj["batch_request"]
                    custom_id = str(br.get("custom_id") or "").strip() or None
                    ii = br.get("image_info") or {}
                    oi = ii.get("order_index")
                    if isinstance(oi, int):
                        entries[oi] = ImageEntry(
                            order_index=oi,
                            image_name=str(ii.get("image_name") or "").strip(),
                            pre_processed_image=None,
                            custom_id=custom_id,
                            page_number=ii.get("page_number"),
                        )
                    continue

                # Synchronous GPT JSONL records
                if "method" in obj and obj.get("method") == "gpt" and "order_index" in obj:
                    try:
                        oi = int(obj.get("order_index"))
                        entries.setdefault(
                            oi,
                            ImageEntry(
                                order_index=oi,
                                image_name=str(obj.get("image_name") or "").strip(),
                                pre_processed_image=str(obj.get("pre_processed_image") or "").strip() or None,
                                custom_id=None,
                                page_number=None,
                            ),
                        )
                    except Exception:
                        pass

    except Exception as e:
        logger.error("Error reading JSONL %s: %s", temp_jsonl_path, e)

    return [entries[k] for k in sorted(entries.keys())]


def find_failure_indices(lines: List[str], include_no_text: bool) -> List[int]:
    idxs: List[int] = []
    for i, line in enumerate(lines):
        if is_failure_line(line):
            idxs.append(i)
        elif include_no_text and NO_TEXT_PATTERN.match(line.strip()):
            idxs.append(i)
    return idxs


def resolve_image_path(parent_folder: Path, entry: ImageEntry) -> Optional[Path]:
    if entry.pre_processed_image:
        p = Path(entry.pre_processed_image)
        if p.exists():
            return p
        rel = parent_folder / Path(entry.pre_processed_image).name
        if rel.exists():
            return rel

    # Fallback search by name in common subfolders
    for sub in ("preprocessed_images", "preprocessed_images_tesseract"):
        d = parent_folder / sub
        if d.exists():
            cand = d / entry.image_name
            if cand.exists():
                return cand
    return None


def backup_file(path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = path.with_suffix(f".bak.{ts}.txt")
    backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return backup


def write_repair_jsonl_line(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
