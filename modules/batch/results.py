"""Batch result downloading, transcription sorting, and output finalization.

Provides functions to download and parse batch results from OpenAI and
non-OpenAI providers, sort transcriptions into page order, and write the
final transcription output files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openai import OpenAI, OpenAIError

from modules.batch.backends import BatchHandle, BatchStatus, get_batch_backend
from modules.batch.mapping import extract_custom_id_mapping
from modules.infra.logger import setup_logger
from modules.llm.openai_sdk_utils import coerce_file_id, sdk_to_dict
from modules.llm.response_parsing import (
    extract_transcribed_text,
    process_batch_output,
)
from modules.postprocess.writer import write_transcription_output
from modules.ui import print_error, print_info, print_success, print_warning
from modules.ui.batch_display import (
    display_batch_processing_progress,
    display_page_error_summary,
    display_transcription_not_possible_summary,
    print_no_transcribable_text,
    print_transcription_item_error,
    print_transcription_not_possible,
)

logger = setup_logger(__name__)


def append_finalized_marker(temp_file: Path) -> None:
    """Append a ``finalized`` batch_session marker to a retained temp JSONL.

    Written AFTER the final output so the results are safely on disk first. On
    the next ``check_batches`` run, ``_parse_temp_file_metadata`` surfaces this
    status and ``process_all_batches`` skips the file; without it a retained
    temp JSONL (the shipped ``retain_temporary_jsonl: true`` default) is
    re-downloaded and re-finalized on every run, silently reverting any manual
    ``repair_transcriptions`` edits to the final ``.txt`` and — once the
    provider's output files expire — reporting long-finalized jobs as "pending"
    forever. The marker carries only a ``status`` (no provider), and
    ``batch_session`` is a recognized metadata key, so it is inert to every
    downstream JSONL reader (custom_id mapping, transcription extraction, repair
    discovery). No-op when the temp file was already deleted (``retain`` False),
    so it never resurrects a cleaned-up file. Best-effort: a write failure is
    warned and swallowed.
    """
    if not temp_file.exists():
        return
    try:
        with temp_file.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {"batch_session": {"status": "finalized"}}, ensure_ascii=False
                )
                + "\n"
            )
    except OSError as e:
        logger.warning("Could not append finalized marker to %s: %s", temp_file.name, e)


def _process_non_openai_batch(
    temp_file: Path,
    batch_ids: set[str],
    batch_provider: str,
    batch_tracking_records: list[dict[str, Any]],
    processing_settings: dict[str, Any],
    postprocessing_config: dict[str, Any] | None = None,
    output_format: str = "txt",
) -> str:
    """Process batch results for non-OpenAI providers (Anthropic, Google).

    Uses the BatchBackend abstraction to check status and download results.

    Returns:
        Outcome string for summary accounting (CT-4): ``"finalized"`` when the
        output file was written, ``"failed"`` on a terminal batch failure or
        local write error, ``"pending"`` otherwise (still running or
        retryable download problem).
    """

    identifier = temp_file.stem.replace("_transcription", "")

    print_info(f"Processing {batch_provider.upper()} batch for {temp_file.name}...")

    try:
        backend = get_batch_backend(batch_provider)
    except (ValueError, ImportError) as e:
        print_error(f"Failed to get batch backend for {batch_provider}: {e}")
        return "failed"

    # Check status of all batches
    all_completed = True
    completed_count = 0
    failed_count = 0

    for batch_id in batch_ids:
        # Reconstruct handle from tracking record
        tracking = next(
            (t for t in batch_tracking_records if t.get("batch_id") == batch_id), {}
        )
        handle = BatchHandle(
            provider=batch_provider,
            batch_id=batch_id,
            metadata=tracking.get("metadata", {}),
        )

        try:
            status_info = backend.get_status(handle)
        except (OSError, ValueError, KeyError, AttributeError, RuntimeError) as e:
            print_warning(f"Failed to get status for batch {batch_id}: {e}")
            all_completed = False
            continue

        if status_info.status == BatchStatus.COMPLETED:
            completed_count += 1
        elif status_info.status == BatchStatus.FAILED:
            failed_count += 1
            all_completed = False
            err_msg = status_info.error_message or "Unknown error"
            print_warning(f"Batch {batch_id} failed: {err_msg}")
        elif status_info.status in (BatchStatus.CANCELLED, BatchStatus.EXPIRED):
            # Terminal non-success states: they never progress, so count them as
            # failures rather than leaving the file to report "pending" forever
            # on every run (matches the OpenAI-side B6 semantics in status.py).
            failed_count += 1
            all_completed = False
            print_warning(f"Batch {batch_id} was {status_info.status.value}")
        else:
            all_completed = False
            sv = status_info.status.value
            print_info(f"Batch {batch_id} has status '{sv}' - not yet completed.")

    # Display progress
    display_batch_processing_progress(
        temp_file=temp_file,
        batch_ids=list(batch_ids),
        completed_count=completed_count,
        missing_count=0,
    )

    if not all_completed:
        if failed_count > 0:
            print_warning(
                f"{failed_count} batches have failed."
                f" Check the {batch_provider} dashboard for details."
            )
            return "failed"
        return "pending"

    # All batches completed - download results
    print_info(
        f"All batches for {temp_file.name} report 'completed'. "
        f"Attempting to download and process results..."
    )

    # Extract custom_id mapping
    custom_id_map, batch_order = extract_custom_id_mapping(temp_file)

    # Collect all transcriptions
    all_transcriptions: list[dict[str, Any]] = []

    for batch_id in batch_ids:
        tracking = next(
            (t for t in batch_tracking_records if t.get("batch_id") == batch_id), {}
        )
        handle = BatchHandle(
            provider=batch_provider,
            batch_id=batch_id,
            metadata=tracking.get("metadata", {}),
        )

        try:
            for result in backend.download_results(handle):
                image_info = custom_id_map.get(result.custom_id, {})
                image_name = (
                    image_info.get("image_name")
                    or result.custom_id
                    or "[unknown image]"
                )
                page_number = image_info.get("page_number")

                if not result.success:
                    print_transcription_item_error(
                        image_name=image_name,
                        page_number=page_number,
                        err_code=result.error_code,
                        err_message=result.error,
                    )
                    all_transcriptions.append(
                        {
                            "custom_id": result.custom_id,
                            "transcription": f"[transcription error: {result.error}]",
                            "order_info": batch_order.get(result.custom_id),
                            "image_info": image_info,
                            "error": True,
                        }
                    )
                    continue

                # Extract transcription text
                transcription_text = result.content
                if result.parsed_output and isinstance(result.parsed_output, dict):
                    transcription_text = extract_transcribed_text(result.parsed_output)

                # Handle special cases
                if result.no_transcribable_text:
                    print_no_transcribable_text(
                        image_name=image_name, page_number=page_number
                    )
                    transcription_text = "[No transcribable text]"
                    all_transcriptions.append(
                        {
                            "custom_id": result.custom_id,
                            "transcription": transcription_text,
                            "order_info": batch_order.get(result.custom_id),
                            "image_info": image_info,
                            "warning": True,
                            "warning_type": "no_transcribable_text",
                        }
                    )
                elif result.transcription_not_possible:
                    print_transcription_not_possible(
                        image_name=image_name, page_number=page_number
                    )
                    transcription_text = "[Transcription not possible]"
                    all_transcriptions.append(
                        {
                            "custom_id": result.custom_id,
                            "transcription": transcription_text,
                            "order_info": batch_order.get(result.custom_id),
                            "image_info": image_info,
                            "warning": True,
                            "warning_type": "transcription_not_possible",
                        }
                    )
                else:
                    all_transcriptions.append(
                        {
                            "custom_id": result.custom_id,
                            "transcription": transcription_text,
                            "order_info": batch_order.get(result.custom_id),
                            "image_info": image_info,
                        }
                    )

        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            RuntimeError,
        ) as e:
            print_error(f"Failed to download results for batch {batch_id}: {e}")
            logger.exception(f"Error downloading batch {batch_id}: {e}")
            return "pending"

    if not all_transcriptions:
        print_warning(f"No transcriptions extracted for {temp_file.name}. Skipping.")
        return "pending"

    # Collapse duplicate custom_ids from repeated submissions of the same item
    # before reconciliation, so a doubled submission cannot double every page.
    all_transcriptions = _dedupe_transcriptions_by_custom_id(all_transcriptions)

    # Completeness reconciliation (parity with the OpenAI path, decision 2): emit
    # placeholders for any expected page (from the image_metadata custom_id map)
    # that produced neither an output nor an error entry, so a page dropped by
    # backend.download_results() does not silently vanish from the final output.
    if custom_id_map:
        seen = {cid for cid in (t.get("custom_id") for t in all_transcriptions) if cid}
        all_transcriptions.extend(
            _reconcile_missing_custom_ids(custom_id_map, batch_order, seen)
        )

    # Sort transcriptions by order
    def get_sorting_key(entry: dict[str, Any]) -> tuple[int, Any]:
        order_info = entry.get("order_info")
        if order_info is not None:
            return (0, order_info)
        custom_id = entry.get("custom_id")
        if custom_id in batch_order:
            return (1, batch_order[custom_id])
        if isinstance(custom_id, str) and custom_id.startswith("req-"):
            try:
                req_idx = int(custom_id.split("-", 1)[1]) - 1
                if req_idx >= 0:
                    return (2, req_idx)
            except (ValueError, IndexError):
                logger.debug("Could not parse req index from custom_id: %s", custom_id)
        return (3, 999999)

    all_transcriptions.sort(key=get_sorting_key)

    # Build final output
    final_txt_path = temp_file.parent / f"{identifier}.txt"

    pages = []
    for t in all_transcriptions:
        tx = t.get("transcription", "")
        info = t.get("image_info", {}) or {}
        pn = info.get("page_number")
        iname = info.get("image_name")
        pages.append({"text": tx, "page_number": pn, "image_name": iname})

    # Write output
    try:
        actual_path = write_transcription_output(
            pages,
            final_txt_path,
            output_format=output_format,
            postprocess=bool(postprocessing_config),
            postprocessing_config=postprocessing_config,
        )
        print_success(f"Saved transcription to {actual_path.name}")
        logger.info(f"Wrote final transcription: {actual_path}")
    except (OSError, ValueError, TypeError) as e:
        print_error(f"Failed to write transcription file: {e}")
        logger.exception(f"Error writing {final_txt_path}: {e}")
        return "failed"

    # Optionally delete temp JSONL
    if not processing_settings.get("retain_temporary_jsonl", True):
        try:
            temp_file.unlink()
            print_info(f"Deleted temporary file: {temp_file.name}")
        except OSError as e:
            logger.warning(f"Could not delete temp file {temp_file.name}: {e}")

    # Mark the retained temp JSONL finalized so a later check_batches run skips
    # it instead of re-downloading and re-finalizing (which would clobber repair
    # edits). No-op when the file was just deleted above.
    append_finalized_marker(temp_file)

    return "finalized"


def _resolve_error_file_id(batch: dict[str, Any]) -> str | None:
    """Resolve an OpenAI batch's error file id across known field shapes."""
    for key in (
        "error_file_id",
        "error_file",
        "errors_file_id",
        "error_file_ids",
    ):
        error_file_id = coerce_file_id(batch.get(key))
        if error_file_id:
            return error_file_id
    return None


def _save_error_file(
    client: OpenAI,
    error_file_id: str,
    batch_id: str,
    temp_file: Path,
) -> str | None:
    """Download an OpenAI batch error file to disk; return its text or None."""
    try:
        resp = client.files.content(error_file_id)
        err_bytes = resp.read()
    except (OpenAIError, OSError, ValueError, TypeError) as e:
        logger.warning("Could not download error_file_id for %s: %s", batch_id, e)
        return None

    try:
        err_text = (
            err_bytes.decode("utf-8")
            if isinstance(err_bytes, (bytes, bytearray))
            else str(err_bytes)
        )
    except (UnicodeDecodeError, ValueError):
        logger.debug("Could not decode error bytes as UTF-8, falling back to str()")
        err_text = str(err_bytes)

    # Short, path-safe filename to avoid MAX_PATH issues on Windows
    short_batch_id = batch_id.replace("batch_", "")[:16]
    errors_path = temp_file.parent / f"errors_{short_batch_id}.jsonl"
    try:
        errors_path.parent.mkdir(parents=True, exist_ok=True)
        with errors_path.open("w", encoding="utf-8") as ef:
            ef.write(err_text)
        print_info(f"Saved error details for batch {batch_id} to {errors_path.name}")
        logger.info("Saved error file for %s -> %s", batch_id, errors_path)
    except OSError as e:
        logger.warning("Could not persist error file for %s: %s", batch_id, e)
    return err_text


def _parse_error_file_entries(
    err_text: str,
    custom_id_map: dict[str, Any],
    batch_order: dict[str, Any],
    seen: set[str],
) -> list[dict[str, Any]]:
    """Build placeholder transcription entries from an OpenAI error-file body.

    Each error line names a ``custom_id`` whose page produced no output; we emit
    a ``[transcription error]`` placeholder so the page survives into the final
    output (and can be located by ``repair --errors-only``) instead of silently
    vanishing (B1). Custom ids already present in *seen* are skipped.
    """
    entries: list[dict[str, Any]] = []
    for line in err_text.strip().split("\n"):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        custom_id = obj.get("custom_id")
        if not custom_id or custom_id in seen:
            continue
        image_info = custom_id_map.get(custom_id, {})
        image_name = image_info.get("image_name") or custom_id or "[unknown image]"
        page_number = image_info.get("page_number")
        err_obj = obj.get("error") or {}
        resp_obj = obj.get("response")
        if not err_obj and isinstance(resp_obj, dict):
            body = resp_obj.get("body")
            if isinstance(body, dict):
                err_obj = body.get("error") or {}
        err_code = None
        err_message = None
        if isinstance(err_obj, dict):
            err_code = err_obj.get("code") or err_obj.get("type")
            err_message = err_obj.get("message") or err_obj.get("error")
        diag_text = "[transcription error"
        if err_code:
            diag_text += f"; code {err_code}"
        if err_message:
            diag_text += f"; {err_message}"
        diag_text += "]"
        print_transcription_item_error(
            image_name=image_name,
            page_number=page_number,
            err_code=err_code,
            err_message=err_message,
        )
        seen.add(custom_id)
        entries.append(
            {
                "custom_id": custom_id,
                "transcription": diag_text,
                "order_info": batch_order.get(custom_id),
                "image_info": image_info,
                "error": True,
                "error_details": {"code": err_code, "message": err_message},
            }
        )
    return entries


def _reconcile_missing_custom_ids(
    custom_id_map: dict[str, Any],
    batch_order: dict[str, Any],
    seen: set[str],
) -> list[dict[str, Any]]:
    """Emit placeholders for expected custom_ids that produced no result at all.

    Reconciles the delivered results against the ``image_metadata`` custom_id
    map so a page dropped from both the output and error files still appears as
    a ``[transcription error]`` placeholder in the final output (completeness
    contract, decision 2).
    """
    entries: list[dict[str, Any]] = []
    for custom_id, image_info in custom_id_map.items():
        if custom_id in seen:
            continue
        image_name = (image_info or {}).get("image_name") or custom_id
        page_number = (image_info or {}).get("page_number")
        print_transcription_item_error(
            image_name=image_name,
            page_number=page_number,
            err_message="no result returned by provider",
        )
        seen.add(custom_id)
        entries.append(
            {
                "custom_id": custom_id,
                "transcription": "[transcription error: no result returned]",
                "order_info": batch_order.get(custom_id),
                "image_info": image_info or {},
                "error": True,
                "error_details": {"code": "missing", "message": "no result returned"},
            }
        )
    return entries


def _build_fallback_entry(
    idx: int,
    transcription: str,
    line_custom_ids: list[str | None],
    batch_order: dict[str, Any],
) -> dict[str, Any]:
    """Build a fallback transcription entry, attaching the line's custom_id.

    When direct parsing fails and we fall back to ``process_batch_output``, we
    still know each JSONL line's ``custom_id`` by position. Attaching it (and
    the matching ``order_info``) keeps the entry in page order and stops the
    completeness reconciliation from emitting a duplicate error placeholder for
    the same page (B3).
    """
    entry: dict[str, Any] = {"transcription": transcription, "fallback_index": idx}
    custom_id = line_custom_ids[idx] if idx < len(line_custom_ids) else None
    if custom_id:
        entry["custom_id"] = custom_id
        order_info = batch_order.get(custom_id)
        if order_info is not None:
            entry["order_info"] = order_info
    return entry


def _download_and_parse_openai_results(
    batch_ids: set[str],
    batch_dict: dict[str, dict[str, Any]],
    client: OpenAI,
    custom_id_map: dict[str, Any],
    batch_order: dict[str, Any],
    temp_file: Path,
) -> tuple[list[dict[str, Any]], bool]:
    """Download results from OpenAI and parse them into transcription entries.

    Iterates over each completed batch, resolves its output file ID, downloads
    the JSONL result, and parses each response line (Responses API and legacy
    Chat Completions formats).  Falls back to ``process_batch_output`` when
    direct parsing fails.

    Returns:
        (all_transcriptions, all_completed) where *all_completed* may be set
        to ``False`` if any batch could not be downloaded or lacked an output file.
    """
    all_transcriptions: list[dict[str, Any]] = []
    all_completed = True

    for batch_id in batch_ids:
        batch = batch_dict[batch_id]

        # Resolve the output file id robustly
        output_file_id = coerce_file_id(batch.get("output_file_id"))

        if not output_file_id:
            # Refresh batch details (list endpoint may omit some fields)
            try:
                refreshed = sdk_to_dict(client.batches.retrieve(batch_id))
                if isinstance(refreshed, dict) and refreshed.get("id"):
                    batch = refreshed
                    batch_dict[batch_id] = refreshed
                    output_file_id = coerce_file_id(batch.get("output_file_id"))
            except (OpenAIError, OSError, ValueError, TypeError) as e:
                logger.warning("Could not refresh batch %s: %s", batch_id, e)

        if not output_file_id:
            # Try alternate field shapes just in case
            for key in (
                "output_file_id",
                "output_file",
                "output_file_ids",
                "response_file_id",
                "result_file_id",
                "results_file_id",
                "result_file_ids",
            ):
                output_file_id = coerce_file_id(batch.get(key))
                if output_file_id:
                    break

        if not output_file_id:
            # No output at all: the whole batch's pages are error/missing.
            # Parse the error file (if any) into placeholders so those pages
            # are not silently dropped (B1); reconciliation below covers the
            # rest.
            error_file_id = _resolve_error_file_id(batch)

            print_error(
                f"Batch {batch_id} is marked completed but no output_file_id"
                f" was found. Emitting placeholders for its pages."
            )
            logger.warning("Batch %s completed without output_file_id.", batch_id)

            if error_file_id:
                err_text = _save_error_file(client, error_file_id, batch_id, temp_file)
                if err_text:
                    seen = {
                        cid
                        for cid in (e.get("custom_id") for e in all_transcriptions)
                        if cid
                    }
                    all_transcriptions.extend(
                        _parse_error_file_entries(
                            err_text, custom_id_map, batch_order, seen
                        )
                    )

            all_completed = False
            continue

        try:
            file_stream = client.files.content(output_file_id)
            file_content = file_stream.read()
        except (OpenAIError, OSError, ValueError) as e:
            logger.exception(f"Error downloading batch {batch_id}: {e}")
            print_error(f"Failed to download output for Batch ID {batch_id}: {e}")
            # If any batch fails to download, skip writing the output
            all_completed = False
            break

        # Track entries produced by *this* batch's lines (not the global list)
        # so the no-entry fallback below can fire per batch, and remember each
        # line's custom_id so fallback entries stay ordered. Defined before the
        # try so the except-branch fallback can reference them safely (B3).
        batch_start_len = len(all_transcriptions)
        batch_line_custom_ids: list[str | None] = []

        # Parse the batch output file content (Responses API aware)
        try:
            batch_response_text = (
                file_content.decode("utf-8")
                if isinstance(file_content, bytes)
                else str(file_content)
            )
            batch_response_lines = batch_response_text.strip().split("\n")
            for line in batch_response_lines:
                if not line.strip():
                    continue
                try:
                    response_obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                custom_id = response_obj.get("custom_id")
                batch_line_custom_ids.append(custom_id)
                image_info = custom_id_map.get(custom_id, {}) if custom_id else {}
                image_name = image_info.get("image_name") or (
                    custom_id or "[unknown image]"
                )
                page_number = image_info.get("page_number")
                transcription_text = None

                # Primary: Responses API body with status/diagnostics
                resp = response_obj.get("response")
                if isinstance(resp, dict):
                    status_code = resp.get("status_code")
                    body = resp.get("body")
                    if isinstance(body, dict):
                        # Detect explicit API errors per item
                        error_obj = body.get("error")
                        if isinstance(status_code, int) and status_code != 200:
                            err_code = (error_obj or {}).get("code") or (
                                error_obj or {}
                            ).get("type")
                            err_message = (error_obj or {}).get("message") or (
                                error_obj or {}
                            ).get("error")
                            diag_text = f"[transcription error; status {status_code}"
                            if err_code:
                                diag_text += f"; code {err_code}"
                            if err_message:
                                diag_text += f"; {err_message}"
                            diag_text += "]"
                            print_transcription_item_error(
                                image_name=image_name,
                                page_number=page_number,
                                status_code=status_code,
                                err_code=err_code,
                                err_message=err_message,
                            )
                            all_transcriptions.append(
                                {
                                    "custom_id": custom_id,
                                    "transcription": diag_text,
                                    "order_info": batch_order.get(custom_id),
                                    "image_info": image_info,
                                    "error": True,
                                    "error_details": {
                                        "status_code": status_code,
                                        "code": err_code,
                                        "message": err_message,
                                    },
                                }
                            )
                            continue
                        if isinstance(error_obj, dict):
                            err_code = error_obj.get("code") or error_obj.get("type")
                            err_message = error_obj.get("message") or error_obj.get(
                                "error"
                            )
                            diag_text = "[transcription error"
                            if err_code:
                                diag_text += f"; code {err_code}"
                            if err_message:
                                diag_text += f"; {err_message}"
                            diag_text += "]"
                            print_transcription_item_error(
                                image_name=image_name,
                                page_number=page_number,
                                err_code=err_code,
                                err_message=err_message,
                            )
                            all_transcriptions.append(
                                {
                                    "custom_id": custom_id,
                                    "transcription": diag_text,
                                    "order_info": batch_order.get(custom_id),
                                    "image_info": image_info,
                                    "error": True,
                                    "error_details": {
                                        "status_code": status_code,
                                        "code": err_code,
                                        "message": err_message,
                                    },
                                }
                            )
                            continue
                        # Success path: extract transcription text
                        transcription_text = extract_transcribed_text(body, "")

                # Secondary: legacy Chat Completions style at body level
                if not transcription_text and isinstance(resp, dict):
                    body = resp.get("body")
                    if isinstance(body, dict) and "choices" in body:
                        transcription_text = extract_transcribed_text(body, "")

                if transcription_text:
                    # Provide page-aware placeholders for special cases
                    if transcription_text == "[Transcription not possible]":
                        print_transcription_not_possible(
                            image_name=image_name, page_number=page_number
                        )
                        transcription_text = "[Transcription not possible]"
                        entry = {
                            "custom_id": custom_id,
                            "transcription": transcription_text,
                            "order_info": batch_order.get(custom_id),
                            "image_info": image_info,
                            "warning": True,
                            "warning_type": "transcription_not_possible",
                        }
                    elif transcription_text == "[No transcribable text]":
                        print_no_transcribable_text(
                            image_name=image_name, page_number=page_number
                        )
                        transcription_text = "[No transcribable text]"
                        entry = {
                            "custom_id": custom_id,
                            "transcription": transcription_text,
                            "order_info": batch_order.get(custom_id),
                            "image_info": image_info,
                            "warning": True,
                            "warning_type": "no_transcribable_text",
                        }
                    else:
                        entry = {
                            "custom_id": custom_id,
                            "transcription": transcription_text,
                            "order_info": batch_order.get(custom_id),
                            "image_info": image_info,
                        }
                    all_transcriptions.append(entry)

            # If this batch's lines produced no entries, fall back to the
            # generic processor. Attach each line's custom_id (by position) so
            # the fallback entries keep their page order instead of being
            # reconciled away as duplicate error placeholders (B3).
            if len(all_transcriptions) == batch_start_len:
                transcriptions = process_batch_output(file_content)
                for idx, transcription in enumerate(transcriptions):
                    all_transcriptions.append(
                        _build_fallback_entry(
                            idx, transcription, batch_line_custom_ids, batch_order
                        )
                    )
        except (
            json.JSONDecodeError,
            KeyError,
            TypeError,
            IndexError,
            AttributeError,
            ValueError,
            UnicodeDecodeError,
        ) as json_parse_error:
            # Fall back to the process_batch_output function if direct parsing fails
            logger.warning(
                "Could not directly parse batch result,"
                " falling back to process_batch_output: %s",
                json_parse_error,
            )
            transcriptions = process_batch_output(file_content)

            # Attach each line's custom_id (by position) where known so entries
            # keep their page order instead of collapsing to fallback_index (B3).
            for idx, transcription in enumerate(transcriptions):
                all_transcriptions.append(
                    _build_fallback_entry(
                        idx, transcription, batch_line_custom_ids, batch_order
                    )
                )

        # A completed batch can carry BOTH an output file and an error file;
        # parse the latter so pages routed to it are not silently dropped (B1).
        error_file_id = _resolve_error_file_id(batch)
        if error_file_id:
            err_text = _save_error_file(client, error_file_id, batch_id, temp_file)
            if err_text:
                seen = {
                    cid
                    for cid in (e.get("custom_id") for e in all_transcriptions)
                    if cid
                }
                all_transcriptions.extend(
                    _parse_error_file_entries(
                        err_text, custom_id_map, batch_order, seen
                    )
                )

    # Completeness reconciliation: emit placeholders for any expected page
    # (from the image_metadata custom_id map) that produced neither an output
    # nor an error entry (decision 2). Skipped when a batch download failed
    # outright (all_completed already False) so we do not mark genuinely
    # pending pages as errors.
    if all_completed and custom_id_map:
        seen = {cid for cid in (e.get("custom_id") for e in all_transcriptions) if cid}
        all_transcriptions.extend(
            _reconcile_missing_custom_ids(custom_id_map, batch_order, seen)
        )

    return all_transcriptions, all_completed


def _dedupe_transcriptions_by_custom_id(
    all_transcriptions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse duplicate ``custom_id`` entries to one entry per page.

    Duplicates arise when the same item was submitted more than once (e.g. a
    re-run before finalization resubmitted the whole document): every
    submission reuses the same ``req-<n>`` ids, each downloaded batch yields
    one entry per page, and the final output doubles every page. Keep a
    single entry per custom_id, preferring a successful transcription over an
    error placeholder; among equals the later entry wins. Entries without a
    custom_id are kept as-is.
    """
    by_id: dict[str, int] = {}
    deduped: list[dict[str, Any]] = []
    for entry in all_transcriptions:
        cid = entry.get("custom_id")
        if not isinstance(cid, str) or not cid:
            deduped.append(entry)
            continue
        pos = by_id.get(cid)
        if pos is None:
            by_id[cid] = len(deduped)
            deduped.append(entry)
            continue
        if entry.get("error") and not deduped[pos].get("error"):
            continue
        deduped[pos] = entry
    removed = len(all_transcriptions) - len(deduped)
    if removed:
        print_warning(
            f"Collapsed {removed} duplicate page result(s) from repeated batch"
            f" submissions of the same document."
        )
        logger.warning(
            "Deduplicated %d duplicate custom_id entries across batches.", removed
        )
    return deduped


def _sort_transcriptions(
    all_transcriptions: list[dict[str, Any]],
    batch_order: dict[str, Any],
    temp_file: Path,
) -> None:
    """Sort transcriptions in-place using the multi-level sorting strategy.

    Duplicate ``custom_id`` entries (repeated submissions of the same item)
    are collapsed first, so a doubled submission cannot double every page in
    the final output.

    The strategy uses four priority tiers:
    1. ``order_info`` attached during parsing (authoritative).
    2. ``batch_order`` mapping derived from JSONL metadata.
    3. Index parsed from a ``req-<n>`` custom_id pattern.
    4. Fallback enumeration index for degraded entries.

    Also logs the final sort order for debugging.
    """
    all_transcriptions[:] = _dedupe_transcriptions_by_custom_id(all_transcriptions)
    print_info("Arranging transcriptions in the correct order...")
    print_info(f"Found {len(all_transcriptions)} transcriptions to combine.")
    print_info(
        f"Order tracking data: {len(batch_order)} custom IDs mapped to order indices."
    )
    print_info("Using multi-level sorting to ensure correct page order.")

    def get_sorting_key(transcription_entry: dict[str, Any]) -> tuple[int, Any]:
        """Return a sorting key tuple for stable page ordering."""
        # 1) order_info attached during parsing (authoritative)
        order_info = transcription_entry.get("order_info")
        if order_info is not None:
            return (0, order_info)

        # 2) order_index from batch_order mapping derived from JSONL metadata
        custom_id = transcription_entry.get("custom_id")
        if custom_id in batch_order:
            return (1, batch_order[custom_id])

        # 3) Derive index from a custom_id pattern like 'req-<n>'
        if isinstance(custom_id, str) and custom_id.startswith("req-"):
            try:
                req_idx = int(custom_id.split("-", 1)[1]) - 1
                if req_idx >= 0:
                    return (2, req_idx)
            except (ValueError, IndexError):
                logger.debug("Could not parse req index from custom_id: %s", custom_id)

        # 4) Fallback to the enumeration index used when we had to degrade
        return (3, transcription_entry.get("fallback_index", 999999))

    # Sort the transcriptions using the multi-level key
    all_transcriptions.sort(key=get_sorting_key)

    # Log the sorting order for debugging
    logger.info(f"Sorted {len(all_transcriptions)} transcriptions for {temp_file.name}")
    for idx, entry in enumerate(all_transcriptions):
        custom_id = entry.get("custom_id", "unknown")
        order_info = entry.get("order_info")
        image_name = entry.get("image_info", {}).get("image_name", "unknown")
        logger.info(
            "Position %d: custom_id=%s, order_info=%s, image=%s",
            idx,
            custom_id,
            order_info,
            image_name,
        )


def _finalize_batch_output(
    all_transcriptions: list[dict[str, Any]],
    temp_file: Path,
    identifier: str,
    processing_settings: dict[str, Any],
    postprocessing_config: dict[str, Any] | None,
    output_format: str,
) -> None:
    """Display diagnostics, write output file, and optionally delete the temp file.

    Summarizes per-page errors and transcription-not-possible warnings to the
    console, builds the page list in sorted order, writes the final output via
    ``write_transcription_output``, and removes the temporary JSONL when
    configured to do so.
    """
    final_txt_path = temp_file.parent / f"{identifier}.txt"

    # Summarize per-page diagnostics to console
    error_entries = [e for e in all_transcriptions if e.get("error")]
    if error_entries:
        display_page_error_summary(error_entries)

    np_entries = [
        e
        for e in all_transcriptions
        if e.get("warning_type") == "transcription_not_possible"
    ]
    if np_entries:
        display_transcription_not_possible_summary(len(np_entries))

    # Build page dicts in sorted order
    pages = []
    for t in all_transcriptions:
        tx = t.get("transcription", "")
        info = t.get("image_info", {}) or {}
        pn = info.get("page_number")
        iname = info.get("image_name")
        pages.append({"text": tx, "page_number": pn, "image_name": iname})

    processing_success = False
    try:
        actual_path = write_transcription_output(
            pages,
            final_txt_path,
            output_format=output_format,
            postprocess=bool(postprocessing_config),
            postprocessing_config=postprocessing_config,
        )
        logger.info(
            f"All batches for {temp_file.name} processed and saved to {actual_path}"
        )
        print_success(
            f"Processed all batches for {temp_file.name}. Results saved to "
            f"{actual_path.name}"
        )
        processing_success = True
    except (OSError, ValueError, TypeError) as e:
        logger.exception(f"Error writing final output for {temp_file.name}: {e}")
        print_error(f"Failed to write final output for {temp_file.name}: {e}")

    # Delete temporary JSONL file if we successfully wrote the final output
    if processing_success and not processing_settings.get(
        "retain_temporary_jsonl", True
    ):
        try:
            temp_file.unlink()
            logger.info(f"Deleted temporary file after processing: {temp_file}")
            print_info(f"Deleted temporary file: {temp_file.name}")
        except OSError as e:
            logger.exception(f"Error deleting temporary file {temp_file}: {e}")
            print_error(f"Could not delete temporary file {temp_file.name}: {e}")
