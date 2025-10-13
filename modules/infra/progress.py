"""Progress tracking utilities for async operations.

Provides progress tracking and reporting for long-running async tasks.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Optional

from modules.infra.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class ProgressState:
    """Track progress of an async operation."""
    
    total: int
    completed: int = 0
    failed: int = 0
    start_time: Optional[datetime] = None
    
    def __post_init__(self):
        if self.start_time is None:
            self.start_time = datetime.now()
    
    @property
    def remaining(self) -> int:
        """Get number of remaining items."""
        return self.total - self.completed - self.failed
    
    @property
    def percent_complete(self) -> float:
        """Get completion percentage."""
        if self.total == 0:
            return 100.0
        return (self.completed + self.failed) / self.total * 100.0
    
    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        return (datetime.now() - self.start_time).total_seconds()
    
    @property
    def estimated_remaining_seconds(self) -> Optional[float]:
        """Estimate remaining time in seconds."""
        processed = self.completed + self.failed
        if processed == 0 or self.remaining == 0:
            return None
        
        rate = processed / self.elapsed_seconds
        return self.remaining / rate if rate > 0 else None
    
    def increment_completed(self) -> None:
        """Increment completed count."""
        self.completed += 1
    
    def increment_failed(self) -> None:
        """Increment failed count."""
        self.failed += 1
    
    def format_summary(self) -> str:
        """Format progress summary string."""
        elapsed_min = int(self.elapsed_seconds / 60)
        elapsed_sec = int(self.elapsed_seconds % 60)
        
        summary = (
            f"{self.completed}/{self.total} completed "
            f"({self.percent_complete:.1f}%) "
            f"[{elapsed_min}m {elapsed_sec}s elapsed"
        )
        
        if self.failed > 0:
            summary += f", {self.failed} failed"
        
        est_remaining = self.estimated_remaining_seconds
        if est_remaining is not None and self.remaining > 0:
            est_min = int(est_remaining / 60)
            est_sec = int(est_remaining % 60)
            summary += f", ~{est_min}m {est_sec}s remaining"
        
        summary += "]"
        return summary


class ProgressTracker:
    """Async-safe progress tracker with callback support."""
    
    def __init__(
        self,
        total: int,
        on_update: Optional[Callable[[ProgressState], None]] = None,
        update_interval: int = 10,
    ):
        """Initialize progress tracker.
        
        Args:
            total: Total number of items to process.
            on_update: Optional callback called on progress updates.
            update_interval: Number of items between progress reports.
        """
        self.state = ProgressState(total=total)
        self.on_update = on_update
        self.update_interval = update_interval
        self._lock = asyncio.Lock()
    
    async def increment_completed(self) -> None:
        """Increment completed count (thread-safe)."""
        async with self._lock:
            self.state.increment_completed()
            await self._maybe_report()
    
    async def increment_failed(self) -> None:
        """Increment failed count (thread-safe)."""
        async with self._lock:
            self.state.increment_failed()
            await self._maybe_report()
    
    async def _maybe_report(self) -> None:
        """Report progress if interval reached."""
        processed = self.state.completed + self.state.failed
        if processed % self.update_interval == 0 or processed == self.state.total:
            if self.on_update:
                try:
                    self.on_update(self.state)
                except Exception as e:
                    logger.error(f"Progress callback failed: {e}")
    
    async def finalize(self) -> None:
        """Force final progress report."""
        if self.on_update:
            try:
                self.on_update(self.state)
            except Exception as e:
                logger.error(f"Final progress callback failed: {e}")
