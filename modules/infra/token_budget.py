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
import json
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from modules.config.service import get_config_service
from modules.infra.logger import setup_logger
from modules.ui import print_info, print_success, print_warning

logger = setup_logger(__name__)

_TOKEN_TRACKER_FILE = Path.cwd() / ".chronotranscriber_token_state.json"

_tracker_instance: Optional["DailyTokenTracker"] = None
_tracker_lock = threading.Lock()


def _sleep_compat(seconds: float) -> None:
    """Sleep without blocking the asyncio event loop if one is running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        time.sleep(seconds)


class DailyTokenTracker:
    """Thread-safe daily token usage tracker with persistent state.

    Tracks token usage across API calls and enforces daily limits with
    automatic reset at midnight in the local timezone.
    """

    def __init__(
        self,
        daily_limit: int,
        enabled: bool = True,
        state_file: Optional[Path] = None,
    ) -> None:
        self.daily_limit = daily_limit
        self.enabled = enabled
        self.state_file = state_file or _TOKEN_TRACKER_FILE

        self._lock = threading.Lock()
        self._current_date: str = ""  # Format: YYYY-MM-DD
        self._tokens_used_today: int = 0

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
            with open(self.state_file, "r", encoding="utf-8") as f:
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
        temp_file = self.state_file.with_suffix(".tmp")
        try:
            state = {
                "date": self._current_date,
                "tokens_used": self._tokens_used_today,
                "last_updated": datetime.now().isoformat(),
            }

            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)

            # Atomic replace with retry for transient Windows file locks
            # (antivirus, indexer, etc.)
            max_retries = 3
            retry_delay = 0.1
            for attempt in range(max_retries):
                try:
                    temp_file.replace(self.state_file)
                    return
                except PermissionError:
                    if attempt < max_retries - 1:
                        logger.debug(
                            f"Transient lock on {self.state_file}, "
                            f"retrying in {retry_delay * 1000:.0f} ms "
                            f"(attempt {attempt + 1}/{max_retries})"
                        )
                        _sleep_compat(retry_delay)
                    else:
                        logger.warning(
                            f"Could not atomically replace "
                            f"{self.state_file} after {max_retries} "
                            f"attempts; falling back to direct write"
                        )
                        with open(
                            self.state_file, "w", encoding="utf-8"
                        ) as f:
                            json.dump(state, f, indent=2)

        except Exception as e:
            logger.error(
                f"Error saving token state to {self.state_file}: {e}"
            )

        finally:
            try:
                if temp_file.exists():
                    os.remove(temp_file)
            except OSError:
                pass

    def _check_and_reset_if_new_day(self) -> None:
        current_date = self._get_current_date_str()

        if current_date != self._current_date:
            logger.info(
                f"New day detected: {current_date} (was {self._current_date}). "
                f"Resetting token counter from {self._tokens_used_today:,} to 0."
            )
            self._current_date = current_date
            self._tokens_used_today = 0
            self._save_state()

    def add_tokens(self, tokens: int) -> None:
        if not self.enabled or tokens <= 0:
            return

        with self._lock:
            self._check_and_reset_if_new_day()
            self._tokens_used_today += tokens
            self._save_state()

            logger.debug(
                f"Added {tokens:,} tokens. "
                f"Daily total: {self._tokens_used_today:,}/{self.daily_limit:,}"
            )

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

                _tracker_instance = DailyTokenTracker(
                    daily_limit=daily_limit,
                    enabled=enabled,
                )

    return _tracker_instance


async def check_and_wait_for_token_limit(
    concurrency_config: Dict[str, Any],
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
