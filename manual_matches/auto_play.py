"""Launch a GUI match and automatically advance past each round's end screen.

This is intentionally a viewing-only wrapper: game rules, agent code, model
weights and actions are untouched.  The synthetic Return key is only consumed
by the framework's GUI event loop to continue between rounds.
"""

from __future__ import annotations

import sys
from pathlib import Path

# When invoked from ``manual_matches/``, make the project root importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pygame

from main import main


_real_event_get = pygame.event.get


def _events_with_auto_continue(*args, **kwargs):
    events = _real_event_get(*args, **kwargs)
    if events:
        return events
    return [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN)]


pygame.event.get = _events_with_auto_continue


if __name__ == "__main__":
    main()
