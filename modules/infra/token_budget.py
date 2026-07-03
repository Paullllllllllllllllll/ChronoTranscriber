"""Unified token budget: tracking, persistence, and async wait-for-reset.

Consolidates the former modules.infra.token_budget (state + persistence +
thread-safe accounting) and modules.infra.token_budget (async wait-for-reset
loop) into a single module. One feature, one place.

Exports:
    DailyTokenTracker       persistent, thread-safe token counter
    get_token_tracker       singleton accessor (reads concurrency_config.yaml)
    check_and_wait_for_token_limit
                            async wait until midnight reset when limit reached
    TokenBudget             alias of DailyTokenTracker (new preferred name)
    get_token_budget        alias of get_token_tracker (new preferred name)
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import json
import os
import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from modules.config.service import get_config_service
from modules.infra.logger import setup_logger
from modules.ui import print_info, print_success, print_warning

logger = setup_logger(__name__)

# Legacy location (pre-v1.15): a per-CWD dotfile. Kept only for one-time
# adoption so a run started from the old directory does not lose today's count.
_LEGACY_TOKEN_TRACKER_FILE = Path.cwd() / ".chronotranscriber_token_state.json"
_TOKEN_STATE_FILENAME = "token_state.json"

# Minimum seconds between on-disk state writes. add_tokens() fires per API call
# (up to ~20/s under concurrency); without debouncing that rewrites the state
# file that often, ON THE EVENT LOOP. In-memory counts stay exact; a background
# daemon thread performs the throttled write off the loop, and an atexit flush
# plus explicit flush() guarantee the last change is persisted.
_STATE_WRITE_DEBOUNCE_S = 1.0
# How often the background writer wakes to check for a pending, debounced write.
_WRITER_POLL_INTERVAL_S = 0.5

_tracker_instance: DailyTokenTracker | None = None
_tracker_lock = threading.Lock()


def _default_state_dir() -> Path:
    """User-level state directory: ``~/.chronotranscriber`` (decision 4)."""
    return Path.home() / ".chronotranscriber"


def resolve_token_state_file() -> Path:
    """Resolve the token-state file path.

    Order (decision 4): ``paths_config.general.state_dir`` override, else the
    user-level ``~/.chronotranscriber/`` directory. The previous per-CWD dotfile
    is adopted once when the resolved file does not yet exist, so today's running
    total survives the move.
    """
    state_dir: Path | None = None
    try:
        general = get_config_service().get_paths_config().get("general", {})
        override = general.get("state_dir")
        # Only honor a real string path; a mocked config (tests) can yield a
        # truthy non-string that would create a junk directory.
        if isinstance(override, str) and override.strip():
            state_dir = Path(override).expanduser()
    except (KeyError, AttributeError, TypeError, OSError):
        state_dir = None
    if state_dir is None:
        state_dir = _default_state_dir()

    state_file = state_dir / _TOKEN_STATE_FILENAME
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
        if not state_file.exists() and _LEGACY_TOKEN_TRACKER_FILE.exists():
            import shutil

            shutil.copy2(_LEGACY_TOKEN_TRACKER_FILE, state_file)
            logger.info(
                "Adopted legacy token-state file %s -> %s",
                _LEGACY_TOKEN_TRACKER_FILE,
                state_file,
            )
    except OSError as e:
        logger.warning("Could not prepare token-state dir %s: %s", state_dir, e)
    return state_file


def _blocking_retry_sleep(seconds: float) -> None:
    """Block briefly between synchronous state-save retries.

    A short, deliberately blocking sleep (<=0.1 s) used only inside the
    ``_save_state`` retry loop, which runs on the background writer thread (or,
    rarely, at day rollover / flush), never on the event loop.
    """
    time.sleep(seconds)


class DailyTokenTracker:
    """Thread-safe daily token usage tracker with persistent state.

    Tracks token usage across API calls and enforces daily limits with
    automatic reset at midnight in the local timezone.

    Disk persistence is debounced and performed off the event loop: ``add_tokens``
    only marks state dirty in memory, and a background daemon thread writes the
    throttled snapshot. ``flush`` and an ``atexit`` hook guarantee the final state
    reaches disk.
    """

    def __init__(
        self,
        daily_limit: int,
        enabled: bool = True,
        state_file: Path | None = None,
        chunk_estimate_seed: int = 25_000,
        estimate_smoothing: float = 0.3,
    ) -> None:
        self.daily_limit = daily_limit
        self.enabled = enabled
        self.state_file = state_file or resolve_token_state_file()

        self._lock = threading.Lock()
        self._current_date: str = ""  # Format: YYYY-MM-DD
        self._tokens_used_today: int = 0

        # Debounced off-loop write bookkeeping. add_tokens() only marks state
        # dirty (the in-memory count stays exact); a lazily-started background
        # daemon thread performs the throttled disk write so no I/O or sleep runs
        # on the event loop. flush()/atexit guarantee the final state is
        # persisted.
        self._pending_write: bool = False
        self._last_write_monotonic: float = 0.0
        self._writer_thread: threading.Thread | None = None
        self._writer_stop = threading.Event()
        self._atexit_registered: bool = False

        # Page-level reservation state (in-memory only; transient per run).
        # _tokens_reserved is headroom claimed by in-flight calls that have not
        # yet committed actual usage via add_tokens(); the admission check in
        # try_reserve() subtracts both committed and reserved tokens so that
        # concurrent workers cannot collectively overshoot the daily limit.
        self._tokens_reserved: int = 0
        self._seed: int = max(1, int(chunk_estimate_seed))
        self._alpha: float = min(1.0, max(0.0, float(estimate_smoothing)))
        self._ewma: float = float(self._seed)

        self._load_state()

        logger.info(
            f"Token tracker initialized: enabled={enabled}, "
            f"daily_limit={daily_limit:,}, "
            f"current_usage={self._tokens_used_today:,}"
        )

    def _get_current_date_str(self) -> str:
        return datetime.now().strftime("%Y-%m-%d")

    def _load_state(self) -> None:
        if not self.state_file.exists():
            self._current_date = self._get_current_date_str()
            self._tokens_used_today = 0
            logger.debug("No existing token state file found, starting fresh")
            return

        try:
            with open(self.state_file, encoding="utf-8") as f:
                state = json.load(f)

            saved_date = state.get("date", "")
            saved_tokens = state.get("tokens_used", 0)

            current_date = self._get_current_date_str()

            if saved_date == current_date:
                self._current_date = saved_date
                self._tokens_used_today = saved_tokens
                logger.info(
                    f"Loaded token state for {current_date}: "
                    f"{self._tokens_used_today:,} tokens used"
                )
            else:
                self._current_date = current_date
                self._tokens_used_today = 0
                logger.info(
                    f"New day detected (was {saved_date}, now {current_date}). "
                    "Token counter reset to 0."
                )
                self._save_state()

        except Exception as e:
            logger.warning(f"Error loading token state from {self.state_file}: {e}")
            self._current_date = self._get_current_date_str()
            self._tokens_used_today = 0

    def _save_state(self) -> None:
        """Persist state to disk via a per-process-unique temp file + atomic
        replace. Must be called under ``self._lock``.

        The temp name embeds the pid and a short random token so concurrent
        processes never collide on it (a fixed ``.tmp`` name previously let one
        process's write clobber another's mid-flight). The ``replace()`` is
        retried on transient Windows file locks (PermissionError) AND on a
        FileNotFoundError from a lost replace() race (another process moved our
        temp first); only after exhausting retries does it fall back to a direct
        non-atomic write. The temp file is always removed in ``finally``.
        """
        # If the target directory is gone (e.g. a test temp dir torn down while
        # the background writer was still polling), there is nothing we can do;
        # clear the pending flag and skip silently rather than logging an error.
        if not self.state_file.parent.exists():
            self._pending_write = False
            return

        temp_file = self.state_file.with_name(
            f"{self.state_file.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
        )
        try:
            state = {
                "date": self._current_date,
                "tokens_used": self._tokens_used_today,
                "last_updated": datetime.now().isoformat(),
            }

            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)

            max_retries = 3
            retry_delay = 0.1
            for attempt in range(max_retries):
                try:
                    temp_file.replace(self.state_file)
                    self._last_write_monotonic = time.monotonic()
                    self._pending_write = False
                    return
                except (PermissionError, FileNotFoundError) as exc:
                    if attempt < max_retries - 1:
                        logger.debug(
                            "Transient error replacing %s (%s), retrying in "
                            "%.0f ms (attempt %d/%d)",
                            self.state_file,
                            type(exc).__name__,
                            retry_delay * 1000,
                            attempt + 1,
                            max_retries,
                        )
                        _blocking_retry_sleep(retry_delay)
                    else:
                        logger.warning(
                            "Could not atomically replace %s after %d attempts; "
                            "falling back to direct write",
                            self.state_file,
                            max_retries,
                        )
                        with open(self.state_file, "w", encoding="utf-8") as f:
                            json.dump(state, f, indent=2)
                        self._last_write_monotonic = time.monotonic()
                        self._pending_write = False

        except Exception as e:
            logger.error(f"Error saving token state to {self.state_file}: {e}")

        finally:
            with contextlib.suppress(Exception):
                if temp_file.exists():
                    temp_file.unlink()

    def _ensure_writer_thread(self) -> None:
        """Start the background debounced-write thread once (call under lock).

        Also registers the atexit flush lazily here (rather than in ``__init__``)
        so only trackers that actually record usage register a hook; trackers
        that merely answer queries in tests never accumulate one.
        """
        if not self._atexit_registered:
            atexit.register(self._flush_on_exit)
            self._atexit_registered = True
        if self._writer_thread is not None:
            return
        thread = threading.Thread(
            target=self._writer_loop,
            name="token-budget-writer",
            daemon=True,
        )
        self._writer_thread = thread
        thread.start()

    def _writer_loop(self) -> None:
        """Daemon loop: persist a pending write once the debounce interval elapses.

        Runs off the event loop so ``add_tokens`` never performs disk I/O on it.
        Wakes every ``_WRITER_POLL_INTERVAL_S`` and, if a write is pending and at
        least ``_STATE_WRITE_DEBOUNCE_S`` has passed since the last write, saves.
        """
        while not self._writer_stop.wait(_WRITER_POLL_INTERVAL_S):
            with self._lock:
                if self._pending_write and (
                    time.monotonic() - self._last_write_monotonic
                    >= _STATE_WRITE_DEBOUNCE_S
                ):
                    self._save_state()

    def _flush_on_exit(self) -> None:
        """atexit hook: stop the writer and persist any pending state.

        Fully silent: skips the write when nothing is pending or the target
        directory has gone (e.g. a test temp dir already removed), and swallows
        any error so interpreter shutdown is never disrupted.
        """
        self._writer_stop.set()
        try:
            if not self._pending_write:
                return
            if not self.state_file.parent.exists():
                return
            with self._lock:
                if self._pending_write:
                    self._save_state()
        except Exception:
            pass

    def flush(self) -> None:
        """Force-persist any pending debounced state write."""
        with self._lock:
            if self._pending_write:
                self._save_state()

    def set_daily_limit(self, new_limit: int) -> None:
        """Update the daily token limit at runtime.

        Used by the wait loop so a user editing ``concurrency_config.yaml``
        mid-wait (raising ``daily_token_limit.daily_tokens``) lifts the cap
        without a restart. A no-op when the value is unchanged.
        """
        new_limit = int(new_limit)
        with self._lock:
            if new_limit != self.daily_limit:
                logger.info(
                    "Daily token limit updated: %s -> %s",
                    f"{self.daily_limit:,}",
                    f"{new_limit:,}",
                )
                self.daily_limit = new_limit

    def _check_and_reset_if_new_day(self) -> None:
        current_date = self._get_current_date_str()

        if current_date != self._current_date:
            logger.info(
                f"New day detected: {current_date} (was {self._current_date}). "
                f"Resetting token counter from {self._tokens_used_today:,} to 0."
            )
            self._current_date = current_date
            self._tokens_used_today = 0
            # Persist the rollover immediately (rare event, once per day) so the
            # reset survives even if the process exits before the next write.
            self._save_state()

    def add_tokens(self, tokens: int) -> None:
        if not self.enabled or tokens <= 0:
            return

        with self._lock:
            self._check_and_reset_if_new_day()
            self._tokens_used_today += tokens
            # Update the rolling per-call estimate used by try_reserve().
            self._ewma = self._alpha * tokens + (1.0 - self._alpha) * self._ewma
            # Mark dirty and let the background writer persist off the event
            # loop; the in-memory count is already exact.
            self._pending_write = True
            self._ensure_writer_thread()

            logger.debug(
                f"Added {tokens:,} tokens. "
                f"Daily total: {self._tokens_used_today:,}/{self.daily_limit:,}"
            )

    def try_reserve(self, estimate: int | None = None) -> int | None:
        """Reserve estimated tokens for one page before launching it.

        The estimate is the larger of the caller-supplied hint and the rolling
        EWMA of observed per-call usage, so the reservation tracks reality and
        never drops below the average. Image pages have no cheap pre-count, so
        callers typically pass no hint and rely on the EWMA.

        Returns the reserved amount, ``0`` when limiting is disabled (admit
        freely, nothing to release), or ``None`` when the remaining budget
        cannot cover the estimate (caller should stop admitting new work). A
        non-zero reservation must be matched by a later :meth:`release` of the
        same amount once the call completes.
        """
        if not self.enabled:
            return 0

        with self._lock:
            self._check_and_reset_if_new_day()
            est = max(int(estimate or 0), max(1, round(self._ewma)))
            available = (
                self.daily_limit - self._tokens_used_today - self._tokens_reserved
            )
            if est > available:
                return None
            self._tokens_reserved += est
            return est

    def release(self, amount: int) -> None:
        """Release a reservation made by :meth:`try_reserve` after the call.

        Actual usage is committed separately via :meth:`add_tokens`; releasing
        only frees the transient headroom the reservation was holding.
        """
        if not self.enabled or amount <= 0:
            return

        with self._lock:
            self._tokens_reserved = max(0, self._tokens_reserved - amount)

    def get_tokens_used_today(self) -> int:
        with self._lock:
            self._check_and_reset_if_new_day()
            return self._tokens_used_today

    def get_tokens_remaining(self) -> int:
        if not self.enabled:
            return self.daily_limit

        with self._lock:
            self._check_and_reset_if_new_day()
            remaining = self.daily_limit - self._tokens_used_today
            return max(0, remaining)

    def is_limit_reached(self) -> bool:
        if not self.enabled:
            return False

        return self.get_tokens_remaining() == 0

    def can_use_tokens(self, estimated_tokens: int = 0) -> bool:
        if not self.enabled:
            return True

        remaining = self.get_tokens_remaining()

        if estimated_tokens > 0:
            return remaining >= estimated_tokens
        else:
            return remaining > 0

    def get_seconds_until_reset(self) -> int:
        now = datetime.now()
        tomorrow = now.date() + timedelta(days=1)
        midnight = datetime.combine(tomorrow, datetime.min.time())

        delta = midnight - now
        return int(delta.total_seconds())

    def get_reset_time(self) -> datetime:
        now = datetime.now()
        tomorrow = now.date() + timedelta(days=1)
        return datetime.combine(tomorrow, datetime.min.time())

    def get_usage_percentage(self) -> float:
        if not self.enabled or self.daily_limit == 0:
            return 0.0

        used = self.get_tokens_used_today()
        return (used / self.daily_limit) * 100.0

    def get_stats(self) -> dict[str, Any]:
        used = self.get_tokens_used_today()
        remaining = self.get_tokens_remaining()
        percentage = self.get_usage_percentage()
        seconds_until_reset = self.get_seconds_until_reset()
        reset_time = self.get_reset_time()

        return {
            "enabled": self.enabled,
            "daily_limit": self.daily_limit,
            "tokens_used_today": used,
            "tokens_remaining": remaining,
            "usage_percentage": round(percentage, 2),
            "limit_reached": self.is_limit_reached(),
            "seconds_until_reset": seconds_until_reset,
            "reset_time": reset_time.isoformat(),
            "current_date": self._current_date,
        }


def get_token_tracker() -> DailyTokenTracker:
    """Get the singleton token tracker instance."""
    global _tracker_instance

    if _tracker_instance is None:
        with _tracker_lock:
            if _tracker_instance is None:
                conc_cfg = get_config_service().get_concurrency_config()
                token_cfg = conc_cfg.get("daily_token_limit", {})

                enabled = bool(token_cfg.get("enabled", False))
                _daily_tokens_raw = token_cfg.get("daily_tokens", 10_000_000)
                daily_limit = int(str(_daily_tokens_raw).replace("_", ""))
                seed = int(
                    str(token_cfg.get("chunk_estimate_seed", 25_000)).replace("_", "")
                )
                smoothing = float(token_cfg.get("estimate_smoothing", 0.3))

                _tracker_instance = DailyTokenTracker(
                    daily_limit=daily_limit,
                    enabled=enabled,
                    state_file=resolve_token_state_file(),
                    chunk_estimate_seed=seed,
                    estimate_smoothing=smoothing,
                )

    return _tracker_instance


def _read_configured_daily_limit() -> int | None:
    """Read the configured daily token limit fresh from disk.

    Uses a throwaway ``ConfigLoader`` (its per-instance cache is empty) so a
    mid-wait edit to ``concurrency_config.yaml`` raising
    ``daily_token_limit.daily_tokens`` is observed without a restart. Returns
    ``None`` when the value is absent or the config cannot be read, so callers
    keep the current limit on failure.
    """
    from modules.config.config_loader import ConfigLoader

    concurrency_config = ConfigLoader().get_concurrency_config() or {}
    token_cfg = concurrency_config.get("daily_token_limit", {}) or {}
    raw = token_cfg.get("daily_tokens")
    if raw is None:
        return None
    return int(str(raw).replace("_", ""))


async def check_and_wait_for_token_limit(
    concurrency_config: dict[str, Any],
) -> bool:
    """Check if daily token limit is reached and wait until next day if needed.

    Args:
        concurrency_config: Concurrency configuration dictionary.

    Returns:
        True if processing can continue, False if user cancelled wait.
    """
    token_cfg = concurrency_config.get("daily_token_limit", {})
    enabled = bool(token_cfg.get("enabled", False))

    if not enabled:
        return True

    token_tracker = get_token_tracker()

    if not token_tracker.is_limit_reached():
        return True

    stats = token_tracker.get_stats()
    reset_time = token_tracker.get_reset_time()
    seconds_until_reset = token_tracker.get_seconds_until_reset()

    logger.warning(
        f"Daily token limit reached: "
        f"{stats['tokens_used_today']:,}/{stats['daily_limit']:,} tokens used"
    )
    print_warning(
        f"Daily token limit reached: "
        f"{stats['tokens_used_today']:,}/{stats['daily_limit']:,} tokens used"
    )
    print_info(
        f"Waiting until {reset_time.strftime('%Y-%m-%d %H:%M:%S')} "
        f"({seconds_until_reset // 3600}h {(seconds_until_reset % 3600) // 60}m) "
        "for token limit reset..."
    )
    print_info("Press Ctrl+C to cancel and exit.")

    try:
        sleep_interval = 1
        elapsed = 0

        while elapsed < seconds_until_reset:
            interval = min(sleep_interval, max(0, seconds_until_reset - elapsed))
            await asyncio.sleep(interval)
            elapsed += interval

            # Live re-read of the configured daily limit: a user raising
            # daily_token_limit.daily_tokens mid-wait lifts the cap without a
            # restart. A read failure keeps the current limit (debug-logged).
            try:
                new_limit = _read_configured_daily_limit()
                if new_limit is not None:
                    token_tracker.set_daily_limit(new_limit)
            except Exception as exc:
                logger.debug("Could not refresh daily token limit during wait: %s", exc)

            if not token_tracker.is_limit_reached():
                logger.info("Token limit has been reset. Resuming processing.")
                print_success("Token limit has been reset. Resuming processing.")
                return True

        logger.info("Token limit has been reset. Resuming processing.")
        print_success("Token limit has been reset. Resuming processing.")
        return True

    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Wait cancelled by user.")
        print_info("Wait cancelled by user.")
        return False


TokenBudget = DailyTokenTracker
get_token_budget = get_token_tracker


__all__ = [
    "DailyTokenTracker",
    "TokenBudget",
    "get_token_tracker",
    "get_token_budget",
    "check_and_wait_for_token_limit",
]
