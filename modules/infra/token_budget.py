"""Unified token budget: tracking, persistence, and async wait-for-reset.

Consolidates the former modules.infra.token_budget (state + persistence +
thread-safe accounting) and modules.infra.token_budget (async wait-for-reset
loop) into a single module. One feature, one place.

Exports:
    DailyTokenTracker       persistent, thread-safe token counter
    get_token_tracker       singleton accessor (reads concurrency_config.yaml)
    check_and_wait_for_token_limit
                            async wait until the 00:01 UTC reset when limit reached
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
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modules.config.service import get_config_service
from modules.infra.logger import setup_logger
from modules.ui import print_info, print_success, print_warning

if TYPE_CHECKING:
    from modules.infra.shared_ledger import SharedTokenLedger

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

# This tool's field name in the shared cross-tool ledger.
_LEDGER_TOOL_NAME = "chronotranscriber"

# One-minute safety buffer past OpenAI's 00:00 UTC free-tier reset, so the
# budget day never rolls over before the upstream quota has actually reset.
_RESET_BUFFER = timedelta(minutes=1)

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

    Tracks token usage across API calls and enforces daily limits with an
    automatic reset at 00:01 UTC (one minute after OpenAI's 00:00 UTC
    free-tier reset).

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
        shared_enabled: bool = False,
        shared_ledger_dir: str | Path | None = None,
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

        # Shared cross-tool ledger state (only touched when shared_enabled).
        # The ledger is constructed lazily on first use so a disabled tracker
        # performs zero ledger I/O. _unsynced_delta accumulates committed tokens
        # not yet pushed to the ledger; _combined_total caches the last-known
        # combined usage across all tools. Budget math while enabled uses
        # (_combined_total + _unsynced_delta) as the effective usage. The ledger
        # sync rides the existing background writer: add_tokens marks a pending
        # write, and the writer thread pushes ledger.sync(delta) on its debounce
        # ticks INSTEAD of writing the private state file.
        self._shared_enabled: bool = bool(shared_enabled)
        self._shared_ledger_dir: str | Path | None = shared_ledger_dir or None
        self._ledger: SharedTokenLedger | None = None
        self._ledger_construct_failed: bool = False
        self._ledger_tool_name: str = _LEDGER_TOOL_NAME
        self._unsynced_delta: int = 0
        self._combined_total: int = 0
        self._seeded: bool = False
        self._ledger_degraded: bool = False
        self._ledger_sync_in_flight: bool = False

        self._load_state()

        # Seed the shared ledger once at init so the combined baseline (this
        # tool's prior same-day usage plus any concurrent tools) is known before
        # the first admission check. Best-effort: a degraded ledger simply leaves
        # the tracker in standalone mode. Init is off the event loop, so this
        # runs inline and the combined total is visible immediately.
        if self._shared_enabled:
            with contextlib.suppress(Exception):
                self.sync_ledger_now()

        logger.info(
            f"Token tracker initialized: enabled={enabled}, "
            f"daily_limit={daily_limit:,}, "
            f"current_usage={self._tokens_used_today:,}, "
            f"shared_budget={self._shared_enabled}"
        )

    def _get_current_date_str(self) -> str:
        """Get the current budget-day key (YYYY-MM-DD), buffered UTC.

        The day rolls over at 00:01 UTC rather than at exact UTC midnight, so
        the reset never fires before OpenAI's own 00:00 UTC free-tier reset
        has actually happened.
        """
        return (datetime.now(UTC) - _RESET_BUFFER).strftime("%Y-%m-%d")

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
        least ``_STATE_WRITE_DEBOUNCE_S`` has passed since the last write, either
        pushes the accumulated delta to the shared ledger (when the shared budget
        is enabled) or saves the private state file. The ledger sync is dispatched
        with the tracker lock RELEASED so the OS file lock never stalls callers.
        """
        while not self._writer_stop.wait(_WRITER_POLL_INTERVAL_S):
            with self._lock:
                due = self._pending_write and (
                    time.monotonic() - self._last_write_monotonic
                    >= _STATE_WRITE_DEBOUNCE_S
                )
                shared = self._shared_enabled
                if due and not shared:
                    self._save_state()
            if due and shared:
                # Ledger I/O outside the lock: the writer thread is already off
                # the event loop, so run the sync inline here.
                self.sync_ledger_now()

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
        """Force-persist any pending debounced state write.

        When the shared budget is enabled this forces a final ledger sync so the
        last accumulated delta lands before exit; if a background sync is in
        flight, wait briefly (bounded) for it to clear so the final push is not
        skipped by the in-flight guard. The private state file is not written
        while the ledger is the active persistence.
        """
        if self._shared_enabled:
            for _ in range(50):
                with self._lock:
                    busy = self._ledger_sync_in_flight
                if not busy:
                    break
                time.sleep(0.02)
            with contextlib.suppress(Exception):
                self.sync_ledger_now()
            return
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

    # ------------------------------------------------------------------
    # Shared cross-tool ledger integration
    # ------------------------------------------------------------------

    @staticmethod
    def _running_on_event_loop() -> bool:
        """True when called from a thread with a running asyncio event loop.

        Ledger I/O takes an OS file lock and must never block the event loop, so
        on-loop callers dispatch the sync to a background thread while off-loop
        callers (init, background writer, tests, atexit) run it inline for
        determinism.
        """
        try:
            asyncio.get_running_loop()
            return True
        except RuntimeError:
            return False

    def _get_or_create_ledger_locked(self) -> SharedTokenLedger | None:
        """Construct the shared ledger lazily. Must hold ``self._lock``.

        Construction touches no filesystem (the ledger defers all I/O to its
        locked merge), so a bad directory cannot fail here; a genuinely invalid
        tool name is the only ValueError, and it is latched so we do not retry
        forever.
        """
        if self._ledger is None and not self._ledger_construct_failed:
            try:
                from modules.infra.shared_ledger import SharedTokenLedger

                self._ledger = SharedTokenLedger(
                    self._ledger_tool_name,
                    ledger_dir=self._shared_ledger_dir,
                )
            except Exception as exc:  # pragma: no cover - defensive
                self._ledger_construct_failed = True
                logger.warning("Could not construct shared token ledger: %s", exc)
        return self._ledger

    def _effective_used_locked(self) -> int:
        """Return the usage figure the budget is enforced against. Hold the lock.

        Enabled and healthy: the last-known combined total across all tools plus
        this tool's not-yet-synced delta, so our own in-flight usage is never
        undercounted. Disabled or degraded: the private per-tool count, i.e.
        exactly today's standalone semantics.
        """
        if self._shared_enabled and not self._ledger_degraded:
            return self._combined_total + self._unsynced_delta
        return self._tokens_used_today

    def _perform_ledger_sync(self) -> None:
        """Run a ledger sync off the event loop where one is running.

        On an event loop, dispatch to a daemon thread so the OS file lock never
        blocks the loop. Off-loop (init, background writer, tests, atexit), run
        inline so the refreshed combined total is visible to the immediately
        following check.
        """
        if self._running_on_event_loop():
            threading.Thread(target=self.sync_ledger_now, daemon=True).start()
        else:
            self.sync_ledger_now()

    def sync_ledger_now(self) -> None:
        """Seed-or-sync the shared ledger, writing back the combined total.

        Discipline: snapshot the delta under the tracker lock, call the ledger
        (seed or sync) with the lock RELEASED, then write the returned combined
        total back under the lock. The ledger has its own internal mutex; we
        never hold the tracker lock across a ledger call so the hot path cannot
        stall on ledger I/O. Degradation (ledger returns None) leaves the tracker
        in standalone mode with the unsynced delta preserved so a transient
        failure self-heals and the full accumulated amount replays on a later
        sync.
        """
        if not self._shared_enabled:
            return

        with self._lock:
            if self._ledger_sync_in_flight:
                return
            self._ledger_sync_in_flight = True
            ledger = self._get_or_create_ledger_locked()
            need_seed = not self._seeded
            own_committed = self._tokens_used_today
            delta = self._unsynced_delta

        try:
            if ledger is None:
                with self._lock:
                    self._ledger_degraded = True
                return

            own_field: int | None = None
            if need_seed:
                combined = ledger.seed(own_committed)
                if combined is not None:
                    breakdown = ledger.read_breakdown()
                    if breakdown is not None:
                        own_field = int(
                            breakdown.get(self._ledger_tool_name, own_committed)
                        )
            else:
                combined = ledger.sync(delta)

            with self._lock:
                if combined is None:
                    # Degraded: keep the unsynced delta so the full accumulated
                    # amount is pushed once the ledger recovers.
                    self._ledger_degraded = True
                else:
                    self._ledger_degraded = False
                    self._combined_total = combined
                    if need_seed:
                        self._seeded = True
                        baseline = own_field if own_field is not None else own_committed
                        if baseline > self._tokens_used_today:
                            self._tokens_used_today = baseline
                        # Any delta committed during the seed round is preserved
                        # for the next sync; the baseline is now in the ledger.
                        self._unsynced_delta = max(
                            0, self._tokens_used_today - baseline
                        )
                    else:
                        # Subtract only what we pushed; deltas that arrived
                        # mid-sync remain queued for the next push.
                        self._unsynced_delta = max(0, self._unsynced_delta - delta)
        finally:
            with self._lock:
                self._ledger_sync_in_flight = False
                # Reset the writer debounce clock and re-evaluate the dirty flag:
                # a residual (or still-degraded) delta keeps the writer retrying;
                # a fully-pushed delta clears it.
                self._last_write_monotonic = time.monotonic()
                self._pending_write = self._unsynced_delta > 0

    def _maybe_forced_refresh_before_admit(self) -> None:
        """Force a ledger refresh before a reservation when it matters.

        Triggers a forced (debounce-bypassing) sync when the shared budget is not
        yet seeded, or when the cached combined total already exceeds 80% of the
        daily limit, so admission near the cap sees the freshest cross-tool usage.
        Off-loop this runs inline (fresh value visible to the caller); on-loop it
        dispatches and the value converges shortly after.
        """
        if not self._shared_enabled:
            return
        trigger = False
        with self._lock:
            near_limit = (
                self.daily_limit > 0 and self._combined_total > 0.8 * self.daily_limit
            )
            if (not self._seeded or near_limit) and not self._ledger_sync_in_flight:
                trigger = True
        if trigger:
            self._perform_ledger_sync()

    def _check_and_reset_if_new_day(self) -> None:
        current_date = self._get_current_date_str()

        if current_date != self._current_date:
            logger.info(
                f"New day detected: {current_date} (was {self._current_date}). "
                f"Resetting token counter from {self._tokens_used_today:,} to 0."
            )
            self._current_date = current_date
            self._tokens_used_today = 0
            if self._shared_enabled:
                # The ledger rolls over internally; reset the local mirror and
                # force a re-seed on the next sync. The private file is left
                # untouched while the shared budget is the active persistence.
                self._unsynced_delta = 0
                self._combined_total = 0
                self._seeded = False
            else:
                # Persist the rollover immediately (rare event, once per day) so
                # the reset survives even if the process exits before the next
                # write.
                self._save_state()

    def add_tokens(self, tokens: int) -> None:
        if not self.enabled or tokens <= 0:
            return

        with self._lock:
            self._check_and_reset_if_new_day()
            self._tokens_used_today += tokens
            # Update the rolling per-call estimate used by try_reserve().
            self._ewma = self._alpha * tokens + (1.0 - self._alpha) * self._ewma
            if self._shared_enabled:
                # Ledger is the active persistence: accumulate the delta and let
                # the background writer push it to the ledger on its debounce
                # tick (INSTEAD of writing the private file). The in-memory count
                # is already exact.
                self._unsynced_delta += tokens
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

        # Forced pre-admission refresh when near the combined cap (or not yet
        # seeded). Runs outside the tracker lock; inline off-loop so the fresh
        # combined total is visible to the admission check below. A no-op when
        # the shared budget is disabled.
        self._maybe_forced_refresh_before_admit()

        with self._lock:
            self._check_and_reset_if_new_day()
            est = max(int(estimate or 0), max(1, round(self._ewma)))
            available = (
                self.daily_limit - self._effective_used_locked() - self._tokens_reserved
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
        """Tokens used today.

        With the shared budget enabled this is the COMBINED usage across all
        participating tools (the figure the daily limit is enforced against);
        otherwise it is this tool's private count. See
        :meth:`get_own_tokens_used_today` for the per-tool figure.
        """
        with self._lock:
            self._check_and_reset_if_new_day()
            return self._effective_used_locked()

    def get_own_tokens_used_today(self) -> int:
        """Return this tool's private token count for today (never combined)."""
        with self._lock:
            self._check_and_reset_if_new_day()
            return self._tokens_used_today

    def get_tokens_remaining(self) -> int:
        if not self.enabled:
            return self.daily_limit

        with self._lock:
            self._check_and_reset_if_new_day()
            remaining = self.daily_limit - self._effective_used_locked()
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
        """Seconds until the counter resets (the next 00:01 UTC)."""
        now = datetime.now(UTC)
        delta = self.get_reset_time() - now
        return max(0, int(delta.total_seconds()))

    def get_reset_time(self) -> datetime:
        """Timezone-aware UTC datetime of the next 00:01 UTC reset.

        One minute after OpenAI's 00:00 UTC free-tier reset.
        """
        now = datetime.now(UTC)
        anchor = now - _RESET_BUFFER
        # datetime.min.time() is time(0, 0), i.e. midnight, without importing
        # the datetime.time class (which would shadow the stdlib time module
        # imported above for time.monotonic()/time.sleep()).
        next_midnight = datetime.combine(
            anchor.date() + timedelta(days=1), datetime.min.time(), tzinfo=UTC
        )
        return next_midnight + _RESET_BUFFER

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

        stats = {
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
        stats.update(self._shared_stats())
        return stats

    def _shared_stats(self) -> dict[str, Any]:
        """Shared-budget stats: combined total, own count, and per-tool split.

        Returns an empty dict when the shared budget is disabled so callers see
        no change. ``read_breakdown`` is a lock-free ledger read run outside the
        tracker lock; ``tokens_used_today`` above already reflects the combined
        figure when enabled.
        """
        if not self._shared_enabled:
            return {}
        with self._lock:
            ledger = self._ledger
            own = self._tokens_used_today
            combined = self._effective_used_locked()
            degraded = self._ledger_degraded
        breakdown: dict[str, int] | None = None
        if ledger is not None:
            breakdown = ledger.read_breakdown()
        return {
            "shared_budget_enabled": True,
            "shared_budget_degraded": degraded,
            "own_tokens_used_today": own,
            "combined_tokens_used_today": combined,
            "shared_breakdown": breakdown or {},
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

                # Optional opt-in cross-tool combined budget (default off).
                shared_cfg = conc_cfg.get("shared_token_budget", {}) or {}
                shared_enabled = bool(shared_cfg.get("enabled", False))
                shared_ledger_dir = shared_cfg.get("ledger_dir", "") or None

                _tracker_instance = DailyTokenTracker(
                    daily_limit=daily_limit,
                    enabled=enabled,
                    state_file=resolve_token_state_file(),
                    chunk_estimate_seed=seed,
                    estimate_smoothing=smoothing,
                    shared_enabled=shared_enabled,
                    shared_ledger_dir=shared_ledger_dir,
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


def _describe_reset_time(reset_time: datetime) -> str:
    """Render an aware-UTC reset instant for user-facing messages.

    The actual reset always happens at 00:01 UTC regardless of the local
    offset, so the UTC anchor is always shown alongside the more readable
    local wall-clock time.
    """
    local = reset_time.astimezone()
    return f"{local.strftime('%Y-%m-%d %H:%M:%S')} local (00:01 UTC)"


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
        f"Waiting until {_describe_reset_time(reset_time)} "
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

            # Forced ledger refresh each poll so another tool's usage or its
            # midnight reset is observed while we wait. Runs off the event loop
            # via to_thread; a no-op (and skipped) when the shared budget is
            # disabled, so single-tool waits are unchanged.
            if getattr(token_tracker, "_shared_enabled", False):
                try:
                    await asyncio.to_thread(token_tracker.sync_ledger_now)
                except Exception as exc:
                    logger.debug("Shared ledger refresh during wait failed: %s", exc)

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
