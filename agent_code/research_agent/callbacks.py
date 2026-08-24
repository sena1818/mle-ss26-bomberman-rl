"""Official Bomberman evaluation callbacks for the shared research agent."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .config import ACTIONS, active_config
from .learning import choose_action_index
from .networks import LinearQNetwork, build_network
from .state import encode_state, legal_action_mask


_ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "active_model.npz"


def setup(self):
    self.config = active_config()
    self.rng = np.random.default_rng(0)
    # The feature dimension is obtained once from a synthetic legal game-state
    # only during setup-free testing; production setup defers it until first act.
    self.model = None
    self.logger.info("Loaded experiment configuration %s", self.config.name)


def act(self, game_state: dict) -> str:
    state = encode_state(game_state, self.config.state_encoder)
    if self.model is None:
        self.model = _load_or_create_model(self, state.shape[0])

    epsilon = self.config.epsilon if self.train else 0.0
    action_index = choose_action_index(
        self.model,
        state,
        legal_action_mask(game_state),
        epsilon,
        self.rng,
    )
    return ACTIONS[action_index]


def state_to_features(game_state: dict) -> np.ndarray | None:
    """Compatibility helper for training code and framework examples."""
    return encode_state(game_state, active_config().state_encoder)


def _load_or_create_model(self, input_dim: int) -> LinearQNetwork:
    continue_training = self.train and os.environ.get("BOMBERMAN_CONTINUE", "0") == "1"
    if (not self.train or continue_training) and _ARTIFACT_PATH.exists():
        self.logger.info("Loading trained model from %s", _ARTIFACT_PATH)
        return LinearQNetwork.load(_ARTIFACT_PATH)
    self.logger.info("Creating a fresh %s model", self.config.network)
    return build_network(self.config, input_dim)
