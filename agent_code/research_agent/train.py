"""Thin official training adapter; route behaviour lives in ExperimentRuntime."""

from __future__ import annotations

from typing import List


def setup_training(self):
    """The runtime is constructed by callbacks.setup before this hook runs."""


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    self.runtime.observe(old_game_state, self_action, new_game_state, events)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    self.runtime.end_round(last_game_state, last_action, events)
