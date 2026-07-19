"""Unified token budget: tracking, persistence, and async wait-for-reset.

Consolidates the former modules.infra.token_budget (state + persistence +
thread-safe accounting) and modules.infra.token_budget (async wait-for-reset
loop) into a single module. One feature, one place.

Schema v2 (per-provider-key accounting): usage is attributed to a
``BucketKey(provider, key_env, pool)`` where ``key_env`` is the NAME of the
env var that served the call (never the key value) and ``pool`` is the daily-
allowance pool label derived from the model. Pools are definable per provider
in config (``per_key_pool_caps.<provider>.<label>.models``); providers without
configured pools fall back to the vendored built-ins (currently OpenAI's
complimentary-token program), and unmatched models derive ``None``. Two guards
enforce the budget: a PER-KEY-POOL cap (primary; only for buckets with a pool
AND a resolvable cap -- a pool without a cap is tracked but uncapped) and the
legacy COMBINED daily cap (secondary). Under the default ``scope: pooled`` the
combined cap applies only to pooled calls and to un-stamped (unattributed)
usage, so a pool-less bucket (a self-hosted/local endpoint or any provider
without a pool program) is counted but NEVER blocked by other providers'
usage. ``scope: all`` restores the legacy block-everything behaviour.

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
from modules.infra.shared_ledger import (
    DEFAULT_POOL_CAPS,
    UNATTRIBUTED,
    UNATTRIBUTED_BUCKET,
    BucketKey,
    CompiledPools,
    compile_pools,
    derive_pool,
)
from modules.ui import print_info, print_success, print_warning

if TYPE_CHECKING:
    from modules.infra.shared_ledger import SharedTokenLedger, UsageSnapshot

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

# Budget scopes for the secondary combined cap.
_SCOPE_POOLED = "pooled"
_SCOPE_ALL = "all"
_VALID_SCOPES = (_SCOPE_POOLED, _SCOPE_ALL)

_tracker_instance: DailyTokenTracker | None = None
_tracker_lock = threading.Lock()


def _default_state_dir() -> Path:
    """User-level state directory: ``~/.chronotranscriber`` (decision 4)."""
    return Path.home() / ".chronotranscriber"


def _resolve_bucket(
    provider: str | None,
    key_env: str | None,
    model: str | None,
    pools: CompiledPools | None = None,
) -> BucketKey:
    """Resolve a call's accounting bucket from its stamp.

    A missing provider OR key_env yields the sentinel ``UNATTRIBUTED_BUCKET``
    (un-stamped call sites keep today's combined-only semantics); otherwise the
    daily-allowance pool is derived from the model via ``pools`` (configured
    pools replace the built-ins per provider; uncovered providers fall back to
    the built-ins; no match derives ``None``). The key value is never seen
    here -- only the env var NAME.
    """
    if not provider or not key_env:
        return UNATTRIBUTED_BUCKET
    return BucketKey(
        str(provider), str(key_env), derive_pool(provider, model, pools=pools)
    )


def _bucket_to_str(bucket: BucketKey) -> str:
    """Serialize a bucket to the private-state key ``provider|key_env|pool``."""
    return f"{bucket.provider}|{bucket.key_env}|{bucket.pool or ''}"


def _bucket_from_str(text: str) -> BucketKey | None:
    """Parse a ``provider|key_env|pool`` key back to a bucket; None if malformed."""
    parts = text.split("|")
    if len(parts) != 3:
        return None
    provider, key_env, pool = parts
    if not provider or not key_env:
        return None
    return BucketKey(provider, key_env, pool or None)


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

    Usage is attributed per ``BucketKey(provider, key_env, pool)``. A per-key
    pool cap (primary guard) and the combined daily cap (secondary guard) admit
    or defer each call; the combined guard is scoped so a pool-less bucket is
    never blocked by OpenAI usage under the default ``scope: pooled``.

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
        daily_scope: str = _SCOPE_POOLED,
        per_key_pool_caps_enabled: bool = True,
        pool_caps: dict[tuple[str, str], int] | None = None,
        provider_pools: dict[str, dict[str, list[str]]] | None = None,
    ) -> None:
        self.daily_limit = daily_limit
        self.enabled = enabled
        self.state_file = state_file or resolve_token_state_file()

        # Budget policy (secondary combined cap scope + primary per-key caps).
        # ``_pool_caps`` is keyed by (provider, pool label); a bucket whose
        # (provider, label) pair is absent falls back to DEFAULT_POOL_CAPS by
        # label, and a label unknown there is tracked but uncapped.
        # ``_compiled_pools`` holds configured pool definitions (provider ->
        # longest-prefix-first table); derive_pool replaces the built-ins for
        # providers it covers and falls back to them for the rest.
        self._scope = daily_scope if daily_scope in _VALID_SCOPES else _SCOPE_POOLED
        self._per_key_pool_caps_enabled = bool(per_key_pool_caps_enabled)
        self._pool_caps: dict[tuple[str, str], int] = dict(pool_caps or {})
        self._compiled_pools: CompiledPools = compile_pools(provider_pools or {})

        self._lock = threading.Lock()
        self._current_date: str = ""  # Format: YYYY-MM-DD
        self._tokens_used_today: int = 0
        # Private per-bucket committed counts for today. Always maintained (even
        # standalone) so the private state file can persist them and the per-key
        # pool gate has a figure in standalone/degraded mode.
        self._bucket_used_today: dict[BucketKey, int] = {}

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
        # yet committed actual usage via add_tokens(), kept PER BUCKET so a
        # reservation counts against its own key pool. The admission check in
        # try_reserve() subtracts committed and reserved tokens so concurrent
        # workers cannot collectively overshoot a pool cap or the daily limit.
        self._tokens_reserved: dict[BucketKey, int] = {}
        self._seed: int = max(1, int(chunk_estimate_seed))
        self._alpha: float = min(1.0, max(0.0, float(estimate_smoothing)))
        self._ewma: float = float(self._seed)

        # Shared cross-tool ledger state (only touched when shared_enabled).
        # The ledger is constructed lazily on first use so a disabled tracker
        # performs zero ledger I/O. _unsynced_deltas accumulates committed tokens
        # per bucket not yet pushed to the ledger; _combined_total caches the
        # last-known combined usage across all tools and _bucket_totals caches
        # the cross-tool per-bucket aggregate (the per-key enforcement figure).
        # Budget math while enabled uses (_combined_total + sum(_unsynced_deltas))
        # as the effective combined usage. The ledger sync rides the existing
        # background writer: add_tokens marks a pending write, and the writer
        # thread pushes ledger.sync_usage(deltas) on its debounce ticks INSTEAD
        # of writing the private state file.
        self._shared_enabled: bool = bool(shared_enabled)
        self._shared_ledger_dir: str | Path | None = shared_ledger_dir or None
        self._ledger: SharedTokenLedger | None = None
        self._ledger_construct_failed: bool = False
        self._ledger_tool_name: str = _LEDGER_TOOL_NAME
        self._unsynced_deltas: dict[BucketKey, int] = {}
        self._combined_total: int = 0
        self._bucket_totals: dict[BucketKey, int] = {}
        self._seeded: bool = False
        self._ledger_degraded: bool = False
        self._ledger_sync_in_flight: bool = False

        # One-time warning latch for a mid-wait OpenAI key remap.
        self._warned_key_remap: bool = False

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
            f"scope={self._scope}, "
            f"per_key_pool_caps={self._per_key_pool_caps_enabled}, "
            f"shared_budget={self._shared_enabled}"
        )

    def _get_current_date_str(self) -> str:
        """Get the current budget-day key (YYYY-MM-DD), buffered UTC.

        The day rolls over at 00:01 UTC rather than at exact UTC midnight, so
        the reset never fires before OpenAI's own 00:00 UTC free-tier reset
        has actually happened.
        """
        return (datetime.now(UTC) - _RESET_BUFFER).strftime("%Y-%m-%d")

    def _parse_saved_buckets(self, raw: Any) -> dict[BucketKey, int]:
        """Parse the private-state ``buckets`` object into per-bucket counts.

        Malformed entries are dropped silently (never-crash contract).
        """
        result: dict[BucketKey, int] = {}
        if not isinstance(raw, dict):
            return result
        for key, value in raw.items():
            if not isinstance(key, str) or not isinstance(value, (int, float)):
                continue
            bucket = _bucket_from_str(key)
            if bucket is None:
                continue
            result[bucket] = result.get(bucket, 0) + int(value)
        return result

    def _load_state(self) -> None:
        if not self.state_file.exists():
            self._current_date = self._get_current_date_str()
            self._tokens_used_today = 0
            self._bucket_used_today = {}
            logger.debug("No existing token state file found, starting fresh")
            return

        try:
            with open(self.state_file, encoding="utf-8") as f:
                state = json.load(f)

            saved_date = state.get("date", "")
            saved_tokens = int(state.get("tokens_used", 0) or 0)
            saved_buckets = state.get("buckets")

            current_date = self._get_current_date_str()

            if saved_date == current_date:
                self._current_date = saved_date
                self._tokens_used_today = saved_tokens
                if isinstance(saved_buckets, dict):
                    self._bucket_used_today = self._parse_saved_buckets(saved_buckets)
                elif saved_tokens > 0:
                    # Legacy state without per-bucket attribution: adopt the
                    # whole day's count as the unattributed bucket.
                    self._bucket_used_today = {UNATTRIBUTED_BUCKET: saved_tokens}
                else:
                    self._bucket_used_today = {}
                logger.info(
                    f"Loaded token state for {current_date}: "
                    f"{self._tokens_used_today:,} tokens used"
                )
            else:
                self._current_date = current_date
                self._tokens_used_today = 0
                self._bucket_used_today = {}
                logger.info(
                    f"New day detected (was {saved_date}, now {current_date}). "
                    "Token counter reset to 0."
                )
                self._save_state()

        except Exception as e:
            logger.warning(f"Error loading token state from {self.state_file}: {e}")
            self._current_date = self._get_current_date_str()
            self._tokens_used_today = 0
            self._bucket_used_today = {}

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
                "buckets": {
                    _bucket_to_str(bucket): tokens
                    for bucket, tokens in self._bucket_used_today.items()
                    if tokens > 0
                },
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
        pushes the accumulated per-bucket deltas to the shared ledger (when the
        shared budget is enabled) or saves the private state file. The ledger
        sync is dispatched with the tracker lock RELEASED so the OS file lock
        never stalls callers.
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
        """atexit hook: stop the writer and flush pending state or ledger delta.

        Shared mode delegates to :meth:`flush`, which pushes the accumulated
        unsynced ledger delta (a bare ``_save_state`` would write the private
        file — which shared mode deliberately avoids — and drop the delta).
        atexit runs on the main thread with no running loop, so the final
        ledger sync runs inline. Fully silent: skips the private write when
        nothing is pending or the target directory has gone (e.g. a test temp
        dir already removed), and swallows any error so interpreter shutdown
        is never disrupted.
        """
        self._writer_stop.set()
        try:
            if self._shared_enabled:
                self.flush()
                return
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

    def set_token_policy(
        self,
        scope: str | None = None,
        per_key_pool_caps_enabled: bool | None = None,
        pool_caps: dict[tuple[str, str], int] | None = None,
        provider_pools: dict[str, dict[str, list[str]]] | None = None,
    ) -> None:
        """Update the budget policy at runtime (scope, per-key caps, pools).

        Used by the wait loop's fresh config re-read so a mid-wait edit to
        ``daily_token_limit.scope`` or ``per_key_pool_caps`` (caps AND pool
        model lists) takes effect without a restart. Each argument is optional;
        ``None`` leaves that facet unchanged. Invalid scopes are ignored.
        """
        compiled = compile_pools(provider_pools) if provider_pools is not None else None
        with self._lock:
            if scope is not None and scope in _VALID_SCOPES and scope != self._scope:
                logger.info("Token scope updated: %s -> %s", self._scope, scope)
                self._scope = scope
            if per_key_pool_caps_enabled is not None:
                self._per_key_pool_caps_enabled = bool(per_key_pool_caps_enabled)
            if pool_caps is not None:
                self._pool_caps = dict(pool_caps)
            if compiled is not None:
                self._compiled_pools = compiled

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
        """Return the combined-usage figure the budget is enforced against.

        Hold the lock. Enabled and healthy: the last-known combined total across
        all tools plus this tool's not-yet-synced deltas, so our own in-flight
        usage is never undercounted. Disabled or degraded: the private per-tool
        count, i.e. exactly today's standalone semantics.
        """
        if self._shared_enabled and not self._ledger_degraded:
            return self._combined_total + sum(self._unsynced_deltas.values())
        return self._tokens_used_today

    def _bucket_usage_locked(self, bucket: BucketKey) -> int:
        """Cross-tool + local usage for one bucket (per-key pool enforcement).

        Hold the lock. Healthy shared mode: the cross-tool aggregate for this
        bucket (which already includes our synced rows) plus our not-yet-synced
        delta plus outstanding reservations. Standalone/degraded: our private
        per-bucket count plus reservations.
        """
        reserved = self._tokens_reserved.get(bucket, 0)
        if self._shared_enabled and not self._ledger_degraded:
            return (
                self._bucket_totals.get(bucket, 0)
                + self._unsynced_deltas.get(bucket, 0)
                + reserved
            )
        return self._bucket_used_today.get(bucket, 0) + reserved

    def _total_reserved_locked(self) -> int:
        """Sum of all outstanding per-bucket reservations. Hold the lock."""
        return sum(self._tokens_reserved.values())

    def _pool_cap_for(self, provider: str, pool: str | None) -> int | None:
        """Per-key daily cap (tokens) for a (provider, pool label), or None.

        Resolution: the configured cap for (provider, label), else the vendored
        ``DEFAULT_POOL_CAPS`` by label, else ``None`` -- the bucket is then
        tracked but uncapped (a configured pool that only lists models).
        """
        if pool is None:
            return None
        cap = self._pool_caps.get((str(provider).strip().lower(), pool))
        if cap is None:
            cap = DEFAULT_POOL_CAPS.get(pool)
        return int(cap) if cap is not None else None

    def _pool_gate_applies(self, bucket: BucketKey) -> bool:
        """True when the primary per-key pool cap governs this bucket.

        Requires per-key caps enabled, a derived pool, AND a resolvable cap;
        a pool without any cap is tracked but never blocked.
        """
        return (
            self._per_key_pool_caps_enabled
            and bucket.pool is not None
            and self._pool_cap_for(bucket.provider, bucket.pool) is not None
        )

    def _combined_gate_applies(self, bucket: BucketKey) -> bool:
        """True when the secondary combined cap governs this bucket.

        ``scope: all`` gates every call (legacy block-everything). ``scope:
        pooled`` (default) gates only pooled OpenAI calls and un-stamped
        (unattributed) usage; a stamped pool-less bucket (custom/local endpoint
        or any non-OpenAI provider) is counted but NEVER blocked -- the bug fix.
        """
        if self._scope == _SCOPE_ALL:
            return True
        if bucket.provider == UNATTRIBUTED:
            return True
        return bucket.pool is not None

    def sync_ledger_now(self) -> None:
        """Seed-or-sync the shared ledger, writing back the combined snapshot.

        Discipline: snapshot the per-bucket deltas under the tracker lock, call
        the ledger (seed_usage or sync_usage) with the lock RELEASED, then write
        the returned snapshot back under the lock. The ledger has its own
        internal mutex; we never hold the tracker lock across a ledger call so
        the hot path cannot stall on ledger I/O. Degradation (ledger returns
        None) leaves the tracker in standalone mode with the unsynced deltas
        preserved so a transient failure self-heals and the full accumulated
        amount replays on a later sync.
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
            own_buckets = dict(self._bucket_used_today)
            deltas = {b: n for b, n in self._unsynced_deltas.items() if n > 0}

        try:
            if ledger is None:
                with self._lock:
                    self._ledger_degraded = True
                return

            snap: UsageSnapshot | None
            if need_seed:
                snap = ledger.seed_usage(own_committed, own_buckets)
            else:
                snap = ledger.sync_usage(deltas)

            with self._lock:
                if snap is None:
                    # Degraded: keep the unsynced deltas so the full accumulated
                    # amount is pushed once the ledger recovers.
                    self._ledger_degraded = True
                else:
                    self._ledger_degraded = False
                    self._combined_total = snap.combined
                    self._bucket_totals = dict(snap.buckets)
                    if need_seed:
                        self._seeded = True
                        if snap.own_total > self._tokens_used_today:
                            self._tokens_used_today = snap.own_total
                    # Subtract only what we pushed/seeded, per bucket; deltas
                    # that arrived mid-sync remain queued for the next push.
                    for bucket, amount in deltas.items():
                        remaining = self._unsynced_deltas.get(bucket, 0) - amount
                        if remaining > 0:
                            self._unsynced_deltas[bucket] = remaining
                        else:
                            self._unsynced_deltas.pop(bucket, None)
        finally:
            with self._lock:
                self._ledger_sync_in_flight = False
                # Reset the writer debounce clock and re-evaluate the dirty flag:
                # a residual (or still-degraded) delta keeps the writer retrying;
                # a fully-pushed delta clears it.
                self._last_write_monotonic = time.monotonic()
                self._pending_write = any(n > 0 for n in self._unsynced_deltas.values())

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

    def _check_and_reset_if_new_day(self) -> None:
        current_date = self._get_current_date_str()

        if current_date != self._current_date:
            logger.info(
                f"New day detected: {current_date} (was {self._current_date}). "
                f"Resetting token counter from {self._tokens_used_today:,} to 0."
            )
            self._current_date = current_date
            self._tokens_used_today = 0
            self._bucket_used_today = {}
            if self._shared_enabled:
                # The ledger rolls over internally; reset the local mirror and
                # force a re-seed on the next sync. The private file is left
                # untouched while the shared budget is the active persistence.
                self._unsynced_deltas = {}
                self._combined_total = 0
                self._bucket_totals = {}
                self._seeded = False
            else:
                # Persist the rollover immediately (rare event, once per day) so
                # the reset survives even if the process exits before the next
                # write.
                self._save_state()

    def add_tokens(
        self,
        tokens: int,
        provider: str | None = None,
        key_env: str | None = None,
        model: str | None = None,
    ) -> None:
        """Commit ``tokens`` of usage, attributed to the call's key/pool bucket.

        The stamp (provider, key_env, model) resolves the ``BucketKey``; a
        missing stamp lands in the sentinel unattributed bucket. The plain daily
        sum, the private per-bucket count, and (in shared mode) the per-bucket
        unsynced delta are all updated in one locked step.
        """
        if not self.enabled or tokens <= 0:
            return

        bucket = _resolve_bucket(provider, key_env, model, self._compiled_pools)
        with self._lock:
            self._check_and_reset_if_new_day()
            self._tokens_used_today += tokens
            self._bucket_used_today[bucket] = (
                self._bucket_used_today.get(bucket, 0) + tokens
            )
            # Update the rolling per-call estimate used by try_reserve().
            self._ewma = self._alpha * tokens + (1.0 - self._alpha) * self._ewma
            if self._shared_enabled:
                # Ledger is the active persistence: accumulate the delta and let
                # the background writer push it to the ledger on its debounce
                # tick (INSTEAD of writing the private file). The in-memory count
                # is already exact.
                self._unsynced_deltas[bucket] = (
                    self._unsynced_deltas.get(bucket, 0) + tokens
                )
            # Mark dirty and let the background writer persist off the event
            # loop; the in-memory count is already exact.
            self._pending_write = True
            self._ensure_writer_thread()

            logger.debug(
                f"Added {tokens:,} tokens ({_bucket_to_str(bucket)}). "
                f"Daily total: {self._tokens_used_today:,}/{self.daily_limit:,}"
            )

    def try_reserve(
        self,
        estimate: int | None = None,
        provider: str | None = None,
        key_env: str | None = None,
        model: str | None = None,
    ) -> int | None:
        """Reserve estimated tokens for one page before launching it.

        The estimate is the larger of the caller-supplied hint and the rolling
        EWMA of observed per-call usage, so the reservation tracks reality and
        never drops below the average. Image pages have no cheap pre-count, so
        callers typically pass no hint and rely on the EWMA.

        Admission applies the primary per-key pool cap (only for pooled OpenAI
        buckets) and the secondary combined cap (scoped per :meth:`
        _combined_gate_applies`), so a pool-less bucket is never blocked by
        OpenAI usage under ``scope: pooled``.

        Returns the reserved amount, ``0`` when limiting is disabled (admit
        freely, nothing to release), or ``None`` when a governing cap cannot
        cover the estimate (caller should stop admitting new work). A non-zero
        reservation must be matched by a later :meth:`release` of the same
        amount and stamp once the call completes.
        """
        if not self.enabled:
            return 0

        bucket = _resolve_bucket(provider, key_env, model, self._compiled_pools)

        # Forced pre-admission refresh when near the combined cap (or not yet
        # seeded). Runs outside the tracker lock; inline off-loop so the fresh
        # combined total is visible to the admission check below. A no-op when
        # the shared budget is disabled.
        self._maybe_forced_refresh_before_admit()

        with self._lock:
            self._check_and_reset_if_new_day()
            est = max(int(estimate or 0), max(1, round(self._ewma)))

            # Primary guard: per-key pool cap (skipped when the pool has no
            # resolvable cap -- tracked but uncapped).
            cap = (
                self._pool_cap_for(bucket.provider, bucket.pool)
                if self._per_key_pool_caps_enabled
                else None
            )
            if cap is not None and est > cap - self._bucket_usage_locked(bucket):
                return None

            # Secondary guard: combined daily cap (scoped).
            if self._combined_gate_applies(bucket):
                available = (
                    self.daily_limit
                    - self._effective_used_locked()
                    - self._total_reserved_locked()
                )
                if est > available:
                    return None

            self._tokens_reserved[bucket] = self._tokens_reserved.get(bucket, 0) + est
            return est

    def release(
        self,
        amount: int,
        provider: str | None = None,
        key_env: str | None = None,
        model: str | None = None,
    ) -> None:
        """Release a reservation made by :meth:`try_reserve` after the call.

        Frees the transient headroom the reservation held on the call's bucket;
        actual usage is committed separately via :meth:`add_tokens`.
        """
        if not self.enabled or amount <= 0:
            return

        bucket = _resolve_bucket(provider, key_env, model, self._compiled_pools)
        with self._lock:
            new_value = self._tokens_reserved.get(bucket, 0) - amount
            if new_value > 0:
                self._tokens_reserved[bucket] = new_value
            else:
                self._tokens_reserved.pop(bucket, None)

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

    def would_block_next_page(
        self,
        provider: str | None = None,
        key_env: str | None = None,
        model: str | None = None,
    ) -> bool:
        """True when the next page cannot be admitted for the given bucket.

        Mirrors :meth:`try_reserve` admission (both the per-key pool cap and the
        combined cap) without mutating reservation state, so the wait loop can
        treat "reservation-blocked near a cap" as limit-reached instead of
        consulting :meth:`is_limit_reached` alone (which would return instantly
        and spin the caller's re-pass loop). With no stamp the bucket is
        unattributed, preserving today's combined-only semantics. A disabled
        tracker never blocks.
        """
        if not self.enabled:
            return False

        bucket = _resolve_bucket(provider, key_env, model, self._compiled_pools)
        with self._lock:
            self._check_and_reset_if_new_day()
            est = max(1, round(self._ewma))

            cap = (
                self._pool_cap_for(bucket.provider, bucket.pool)
                if self._per_key_pool_caps_enabled
                else None
            )
            if cap is not None and est > cap - self._bucket_usage_locked(bucket):
                return True

            if self._combined_gate_applies(bucket):
                available = (
                    self.daily_limit
                    - self._effective_used_locked()
                    - self._total_reserved_locked()
                )
                if est > available:
                    return True

            return False

    def estimate_exceeds_daily_limit(
        self,
        provider: str | None = None,
        key_env: str | None = None,
        model: str | None = None,
    ) -> bool:
        """True when the per-page estimate alone exceeds a governing cap.

        Even a full daily reset cannot admit the next page in that case (used and
        reserved drop to 0, but the estimate still exceeds the cap). Checks the
        per-key pool cap (for pooled buckets) and the combined daily limit (when
        the combined gate governs the bucket). With no stamp this reduces to the
        legacy ``est > daily_limit`` check. A disabled tracker never blocks.
        """
        if not self.enabled:
            return False

        bucket = _resolve_bucket(provider, key_env, model, self._compiled_pools)
        with self._lock:
            self._check_and_reset_if_new_day()
            est = max(1, round(self._ewma))
            cap = (
                self._pool_cap_for(bucket.provider, bucket.pool)
                if self._per_key_pool_caps_enabled
                else None
            )
            if cap is not None and est > cap:
                return True
            return self._combined_gate_applies(bucket) and est > self.daily_limit

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
            "scope": self._scope,
            "per_key_pool_caps_enabled": self._per_key_pool_caps_enabled,
            "pool_caps": {
                f"{provider}|{label}": cap
                for (provider, label), cap in self._pool_caps.items()
            },
            "default_pool_caps": dict(DEFAULT_POOL_CAPS),
            "bucket_usage": self._bucket_usage_stats(),
        }
        stats.update(self._shared_stats())
        return stats

    def _bucket_usage_stats(self) -> dict[str, dict[str, Any]]:
        """Per-bucket used/remaining (own + cross-tool where available).

        For a pooled bucket ``remaining`` is against its per-key pool cap;
        pool-less buckets report the cap/remaining as ``None`` (no per-key cap).
        """
        with self._lock:
            buckets: set[BucketKey] = set(self._bucket_used_today)
            buckets.update(self._bucket_totals)
            buckets.update(self._tokens_reserved)
            buckets.update(self._unsynced_deltas)
            result: dict[str, dict[str, Any]] = {}
            for bucket in buckets:
                used = self._bucket_usage_locked(bucket)
                cap = (
                    self._pool_cap_for(bucket.provider, bucket.pool)
                    if self._per_key_pool_caps_enabled
                    else None
                )
                remaining = max(0, cap - used) if cap is not None else None
                result[_bucket_to_str(bucket)] = {
                    "used": used,
                    "cap": cap,
                    "remaining": remaining,
                    "pool": bucket.pool,
                }
            return result

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


def _coerce_cap(raw: Any) -> int | None:
    """Coerce a config cap value (int/float or underscore string) to int."""
    try:
        return int(str(raw).replace("_", ""))
    except (ValueError, TypeError):
        return None


def _parse_pool_caps(
    token_cfg: dict[str, Any],
) -> tuple[bool, dict[tuple[str, str], int], dict[str, dict[str, list[str]]]]:
    """Parse ``per_key_pool_caps`` into (enabled, caps, provider_pools).

    Each pool entry under ``per_key_pool_caps.<provider>.<label>`` is either a
    bare cap value (models come from the vendored built-ins -- the original
    form) or a mapping with optional ``cap`` and ``models`` keys, letting any
    provider define arbitrary pool labels with their own model prefix lists.
    ``caps`` is keyed by (provider, label); entries carrying ``models`` land in
    ``provider_pools`` (compiled later; they REPLACE the built-ins for that
    provider). An absent block means per-key caps apply with the vendored
    defaults; ``enabled: false`` disables them. Malformed entries are dropped.
    """
    caps: dict[tuple[str, str], int] = {}
    pools: dict[str, dict[str, list[str]]] = {}
    pkpc = token_cfg.get("per_key_pool_caps")
    if not isinstance(pkpc, dict):
        return True, caps, pools
    enabled = bool(pkpc.get("enabled", True))
    for provider, entries in pkpc.items():
        if provider == "enabled" or not isinstance(provider, str):
            continue
        if not isinstance(entries, dict):
            continue
        provider_key = provider.strip().lower()
        for label, entry in entries.items():
            if not isinstance(label, str) or not label.strip():
                continue
            label_key = label.strip()
            if isinstance(entry, dict):
                cap = _coerce_cap(entry.get("cap")) if "cap" in entry else None
                if cap is not None:
                    caps[(provider_key, label_key)] = cap
                models = entry.get("models")
                if isinstance(models, list):
                    prefixes = [m for m in models if isinstance(m, str) and m.strip()]
                    if prefixes:
                        pools.setdefault(provider_key, {})[label_key] = prefixes
            else:
                # Bare form: the value is the cap; models from the built-ins.
                cap = _coerce_cap(entry)
                if cap is not None:
                    caps[(provider_key, label_key)] = cap
    return enabled, caps, pools


def _parse_scope(token_cfg: dict[str, Any]) -> str:
    """Parse the combined-cap ``scope`` knob, defaulting to ``pooled``."""
    scope = str(token_cfg.get("scope", _SCOPE_POOLED)).strip().lower()
    return scope if scope in _VALID_SCOPES else _SCOPE_POOLED


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

                scope = _parse_scope(token_cfg)
                caps_enabled, pool_caps, provider_pools = _parse_pool_caps(token_cfg)

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
                    daily_scope=scope,
                    per_key_pool_caps_enabled=caps_enabled,
                    pool_caps=pool_caps,
                    provider_pools=provider_pools,
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


def _refresh_token_policy(tracker: DailyTokenTracker) -> None:
    """Re-read the budget policy (scope + per-key pool caps) fresh from disk.

    A mid-wait edit to ``daily_token_limit.scope`` or ``per_key_pool_caps`` is
    then observed without a restart. Fully guarded: any read failure leaves the
    current policy untouched.
    """
    try:
        from modules.config.config_loader import ConfigLoader

        concurrency_config = ConfigLoader().get_concurrency_config() or {}
        token_cfg = concurrency_config.get("daily_token_limit", {}) or {}
        scope = _parse_scope(token_cfg)
        caps_enabled, pool_caps, provider_pools = _parse_pool_caps(token_cfg)
        tracker.set_token_policy(
            scope=scope,
            per_key_pool_caps_enabled=caps_enabled,
            pool_caps=pool_caps,
            provider_pools=provider_pools,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not refresh token policy during wait: %s", exc)


def _reresolve_openai_key_env(tracker: DailyTokenTracker, stamp_key_env: str) -> None:
    """Warn once if the OpenAI key env var was remapped mid-wait.

    Provider instances resolve and cache their key env var at construction, and
    this repo builds the transcriber/provider once per run (reused across all
    items and pages), so a mid-wait remap of ``api_keys_config.openai`` to a
    different env var cannot switch the active bucket without a restart. Rather
    than silently keep waiting on the exhausted key, log a single clear warning
    so the operator knows a restart is needed to pick up the new key's fresh
    pool. Guarded: any resolution failure is ignored.
    """
    try:
        from modules.llm.providers.factory import ProviderType, resolve_api_key_env_var

        fresh = resolve_api_key_env_var(ProviderType.OPENAI)
        if fresh and fresh != stamp_key_env and not tracker._warned_key_remap:
            tracker._warned_key_remap = True
            logger.warning(
                "OpenAI key env var was remapped mid-wait (%s -> %s); the active "
                "provider still uses %s. Restart to transcribe on the new key's "
                "fresh pool.",
                stamp_key_env,
                fresh,
                stamp_key_env,
            )
            print_warning(
                f"OpenAI key remapped to {fresh} mid-wait; restart to use its "
                f"fresh daily pool (the current run stays on {stamp_key_env})."
            )
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not re-resolve OpenAI key env during wait: %s", exc)


def _describe_reset_time(reset_time: datetime) -> str:
    """Render an aware-UTC reset instant for user-facing messages.

    The actual reset always happens at 00:01 UTC regardless of the local
    offset, so the UTC anchor is always shown alongside the more readable
    local wall-clock time.
    """
    local = reset_time.astimezone()
    return f"{local.strftime('%Y-%m-%d %H:%M:%S')} local (00:01 UTC)"


def _log_per_key_exhaustion(
    tracker: DailyTokenTracker,
    stamp: tuple[str | None, str | None, str | None] | None,
) -> None:
    """Log a per-key exhaustion message naming the key_env and its sibling pool.

    Only meaningful for a pooled OpenAI bucket. Reads the shared ledger's
    lock-free snapshot to show the OTHER OpenAI key's remaining pool where
    available; degraded/standalone mode says the cross-tool view is
    unavailable.
    """
    if not stamp:
        return
    provider, key_env, model = stamp
    bucket = _resolve_bucket(provider, key_env, model, tracker._compiled_pools)
    if bucket.pool is None:
        return
    cap = tracker._pool_cap_for(bucket.provider, bucket.pool)
    if cap is None:
        # A cap-less pool is tracked but uncapped; it cannot be exhausted.
        return
    with tracker._lock:
        used = tracker._bucket_usage_locked(bucket)
        degraded = tracker._ledger_degraded
        shared = tracker._shared_enabled
        ledger = tracker._ledger
    logger.warning(
        "Per-key pool exhausted: %s pool '%s' at %s/%s tokens.",
        key_env,
        bucket.pool,
        f"{used:,}",
        f"{cap:,}",
    )
    print_warning(
        f"API key {key_env} ({bucket.provider}) has exhausted its "
        f"'{bucket.pool}' daily pool ({used:,}/{cap:,})."
    )
    if not shared or ledger is None or degraded:
        print_info(
            "Cross-tool view unavailable (standalone/degraded); the other key's "
            "remaining pool cannot be shown."
        )
        return
    snapshot = ledger.read_usage()
    if snapshot is None:
        print_info("Cross-tool view unavailable; the ledger could not be read.")
        return
    shown = False
    for other, other_used in snapshot.buckets.items():
        if (
            other.provider == bucket.provider
            and other.pool == bucket.pool
            and other.key_env != key_env
        ):
            other_cap = tracker._pool_cap_for(other.provider, other.pool)
            if other_cap is None:
                continue
            print_info(
                f"Other {bucket.provider} key {other.key_env} '{other.pool}' "
                f"pool: {max(0, other_cap - other_used):,} tokens remaining."
            )
            shown = True
    if not shown:
        print_info(
            f"No other {bucket.provider} key has used the '{bucket.pool}' pool "
            f"today. Remap the provider's key env var (e.g. to OPENAI_API_KEY_2) "
            f"and restart to use a second key's fresh pool."
        )


async def check_and_wait_for_token_limit(
    concurrency_config: dict[str, Any],
    reservation_aware: bool = False,
    stamp: tuple[str | None, str | None, str | None] | None = None,
) -> bool:
    """Check if the daily token limit is reached and wait until next day if needed.

    Args:
        concurrency_config: Concurrency configuration dictionary.
        reservation_aware: When True, treat "remaining budget < the current
            per-page reservation estimate" as limit-reached (mirrors admission
            control via :meth:`DailyTokenTracker.would_block_next_page`), so the
            wait actually waits while pages are reservation-blocked near the cap
            rather than returning instantly. The default (False) keeps the plain
            :meth:`is_limit_reached` semantics used by the per-item pre-gate.
        stamp: Optional (provider, key_env, model) of the active call so the
            per-key pool cap governs the wait and messages can name the key.
            ``None`` keeps today's combined-only semantics.

    Returns:
        True if processing can continue. False if the wait cannot help: either
        the user cancelled (Ctrl+C) or — for reservation-aware callers — the
        per-page estimate exceeds the governing cap, so no reset frees enough
        budget. Callers must treat False as an honest give-up (mark the item
        failed), never as success.
    """
    token_cfg = concurrency_config.get("daily_token_limit", {})
    enabled = bool(token_cfg.get("enabled", False))

    if not enabled:
        return True

    token_tracker = get_token_tracker()
    provider, key_env, model = stamp if stamp else (None, None, None)

    def _still_blocked() -> bool:
        # Reservation-aware callers (the mid-document re-pass loop) must keep
        # waiting while admission control defers pages near the cap; plain
        # callers only wait once the budget is fully spent.
        if reservation_aware:
            return token_tracker.would_block_next_page(
                provider=provider, key_env=key_env, model=model
            )
        return token_tracker.is_limit_reached()

    if not _still_blocked():
        return True

    # Fast-fail: if a single page's estimate exceeds a governing cap, a reset
    # cannot admit it — waiting would burn ~48 h (two useless resets) before the
    # caller's stalled-resets safeguard fires. Give up now (CT-11).
    if reservation_aware and token_tracker.estimate_exceeds_daily_limit(
        provider=provider, key_env=key_env, model=model
    ):
        daily_limit = int(token_tracker.get_stats().get("daily_limit", 0))
        logger.warning(
            "Per-page token estimate exceeds the governing cap (daily limit %s); "
            "a daily reset cannot admit the next page. Not waiting.",
            f"{daily_limit:,}",
        )
        print_warning(
            "A single page's token estimate exceeds the governing budget; "
            "not waiting. Raise the cap to process the remaining pages."
        )
        return False

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
    # Per-key exhaustion detail (names the key_env and the sibling key's pool).
    _log_per_key_exhaustion(token_tracker, stamp)
    print_info(
        f"Waiting until {_describe_reset_time(reset_time)} "
        f"({seconds_until_reset // 3600}h {(seconds_until_reset % 3600) // 60}m) "
        "for token limit reset..."
    )
    print_info("Press Ctrl+C to cancel and exit.")

    try:
        # Poll once a second so cancellation (Ctrl+C) and the reset check stay
        # responsive, but only re-read the YAML-backed config every ~30 s: each
        # config re-read builds a fresh ConfigLoader over two YAML files, and a
        # daily-reset wait can last hours, so doing it every second churned the
        # filesystem needlessly.
        sleep_interval = 1
        config_refresh_interval = 30
        elapsed = 0
        since_config_refresh = config_refresh_interval  # refresh on the first pass

        while elapsed < seconds_until_reset:
            interval = min(sleep_interval, max(0, seconds_until_reset - elapsed))
            await asyncio.sleep(interval)
            elapsed += interval
            since_config_refresh += interval

            # Forced ledger refresh each poll so another tool's usage or its
            # midnight reset is observed while we wait. Runs off the event loop
            # via to_thread; a no-op (and skipped) when the shared budget is
            # disabled, so single-tool waits are unchanged.
            if getattr(token_tracker, "_shared_enabled", False):
                try:
                    await asyncio.to_thread(token_tracker.sync_ledger_now)
                except Exception as exc:
                    logger.debug("Shared ledger refresh during wait failed: %s", exc)

            # Live re-read of the configured daily limit and budget policy, at
            # most every ~30 s: a user raising daily_token_limit.daily_tokens (or
            # changing scope / per_key_pool_caps) mid-wait takes effect without a
            # restart. A read failure keeps the current values (debug-logged).
            if since_config_refresh >= config_refresh_interval:
                since_config_refresh = 0
                try:
                    new_limit = _read_configured_daily_limit()
                    if new_limit is not None:
                        token_tracker.set_daily_limit(new_limit)
                except Exception as exc:
                    logger.debug(
                        "Could not refresh daily token limit during wait: %s", exc
                    )
                _refresh_token_policy(token_tracker)

                # Re-resolve the OpenAI key mapping fresh so a mid-wait remap to a
                # second key is at least surfaced (the provider caches its key, so
                # a restart is needed to actually switch buckets).
                if key_env:
                    _reresolve_openai_key_env(token_tracker, key_env)

            if not _still_blocked():
                logger.info("Token limit has been reset. Resuming processing.")
                print_success("Token limit has been reset. Resuming processing.")
                return True

        # Countdown expired: the reset moment has passed. Re-check rather than
        # assume the budget is free — if the per-page estimate exceeds the whole
        # daily limit, would_block_next_page() stays True even after reset, and
        # returning True here would spin the caller's re-pass loop into a second
        # full-day wait (CT-11). Return False so the caller gives up honestly.
        if _still_blocked():
            logger.warning(
                "Token limit reset elapsed but the next page is still blocked "
                "(per-page estimate exceeds the governing cap). Giving up."
            )
            print_warning(
                "Token limit reset elapsed but the next page still cannot be "
                "admitted; giving up. Raise the cap to continue."
            )
            return False

        logger.info("Token limit has been reset. Resuming processing.")
        print_success("Token limit has been reset. Resuming processing.")
        return True

    except KeyboardInterrupt:
        logger.info("Wait cancelled by user.")
        print_info("Wait cancelled by user.")
        return False
    except asyncio.CancelledError:
        # Task cancellation (e.g. Ctrl+C propagated as a cancel, or an
        # enclosing timeout) must actually stop the run. Swallowing it here
        # returned False and let the caller continue as if the user had merely
        # declined to wait, so the run never halted. Re-raise to propagate.
        logger.info("Token-limit wait cancelled; propagating cancellation.")
        raise


TokenBudget = DailyTokenTracker
get_token_budget = get_token_tracker


__all__ = [
    "DailyTokenTracker",
    "TokenBudget",
    "get_token_tracker",
    "get_token_budget",
    "check_and_wait_for_token_limit",
]
