"""Official Bomberman training callbacks for the initial R01 pipeline."""

from __future__ import annotations

from typing import List

import events as e

from .callbacks import _ARTIFACT_PATH
from .config import ACTIONS
from .learning import update_q_learning
from .state import encode_state, legal_action_mask


OFFICIAL_REWARDS = {
    e.COIN_COLLECTED: 1.0,
    e.KILLED_OPPONENT: 5.0,
}


def setup_training(self):
    self.training_updates = 0


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: List[str],
):
    if old_game_state is None or self.model is None:
        return
    _update_from_transition(self, old_game_state, self_action, new_game_state, events, terminal=False)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    if last_game_state is not None and self.model is not None:
        _update_from_transition(self, last_game_state, last_action, None, events, terminal=True)
        self.model.save(_ARTIFACT_PATH)
        self.logger.info("Saved model after %d updates", self.training_updates)


def _update_from_transition(
    self,
    old_game_state: dict,
    action: str,
    new_game_state: dict | None,
    events: List[str],
    *,
    terminal: bool,
):
    state = encode_state(old_game_state, self.config.state_encoder)
    next_state = None if terminal else encode_state(new_game_state, self.config.state_encoder)
    next_mask = None if terminal else legal_action_mask(new_game_state)
    reward = sum(OFFICIAL_REWARDS.get(event, 0.0) for event in events)
    td_error = update_q_learning(
        self.model,
        self.config,
        state,
        ACTIONS.index(action),
        reward,
        next_state,
        next_mask,
    )
    self.training_updates += 1
    self.logger.debug("reward=%+.2f td_error=%+.4f", reward, td_error)
