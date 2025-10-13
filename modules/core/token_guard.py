"""Token limit guard for daily usage enforcement.

Provides async wait functionality when daily token limits are reached,
allowing graceful handling of rate limits without failing jobs.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict

from modules.infra.logger import setup_logger
from modules.core.utils import console_print

logger = setup_logger(__name__)


async def check_and_wait_for_token_limit(concurrency_config: Dict[str, Any]) -> bool:
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
    
    from modules.token_tracker import get_token_tracker
    token_tracker = get_token_tracker()
    
    if not token_tracker.is_limit_reached():
        return True
    
    # Token limit reached - need to wait until next day
    stats = token_tracker.get_stats()
    reset_time = token_tracker.get_reset_time()
    seconds_until_reset = token_tracker.get_seconds_until_reset()
    
    logger.warning(
        f"Daily token limit reached: {stats['tokens_used_today']:,}/{stats['daily_limit']:,} tokens used"
    )
    console_print(
        f"\n[WARNING] Daily token limit reached: {stats['tokens_used_today']:,}/{stats['daily_limit']:,} tokens used"
    )
    console_print(
        f"[INFO] Waiting until {reset_time.strftime('%Y-%m-%d %H:%M:%S')} "
        f"({seconds_until_reset // 3600}h {(seconds_until_reset % 3600) // 60}m) "
        "for token limit reset..."
    )
    console_print("[INFO] Press Ctrl+C to cancel and exit.")
    
    try:
        # Sleep in smaller intervals to allow for interruption
        sleep_interval = 1  # Check every second for responsiveness
        elapsed = 0
        
        while elapsed < seconds_until_reset:
            interval = min(sleep_interval, max(0, seconds_until_reset - elapsed))
            await asyncio.sleep(interval)
            elapsed += interval
            
            # Re-check if it's a new day
            if not token_tracker.is_limit_reached():
                logger.info("Token limit has been reset. Resuming processing.")
                console_print("[SUCCESS] Token limit has been reset. Resuming processing.")
                return True
        
        logger.info("Token limit has been reset. Resuming processing.")
        console_print("[SUCCESS] Token limit has been reset. Resuming processing.")
        return True
        
    except KeyboardInterrupt:
        logger.info("Wait cancelled by user (KeyboardInterrupt).")
        console_print("\n[INFO] Wait cancelled by user.")
        return False
