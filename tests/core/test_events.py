"""Tests for events.py: the null sink drops every event."""

from __future__ import annotations

from chronotranscriber.events import (
    ItemFinished,
    ItemStarted,
    NullEvents,
    PageDone,
    Waiting,
)


def test_null_events_drop_everything() -> None:
    events = NullEvents()

    events.item_started(ItemStarted("talk.mp3", "audio", 1, 1))
    events.page_done(PageDone("talk.mp3", 0, "ok", 1, 2, unit="chunk"))
    events.waiting(Waiting("batch pending", seconds=30.0))
    events.warning("chunk 2 failed")
    events.item_finished(ItemFinished("talk.mp3", 1, 1, False, ("talk.txt",)))
