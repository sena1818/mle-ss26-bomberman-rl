"""Thin official-framework adapter; route behaviour lives in ExperimentRuntime."""

from __future__ import annotations

import os

import numpy as np

from .config import active_config
from .runtime import ExperimentRuntime
from .state import encode_state


def setup(self):
    self.runtime = ExperimentRuntime(
        active_config(),
        train=self.train,
        agent_seed=int(os.environ.get("BOMBERMAN_AGENT_SEED", "0")),
        logger=self.logger,
    )
    # setup is not subject to the official 0.5 s per-step timeout, so this is
    # where model construction and the first forward pass belong.
    self.runtime.warm_up()
    # Retained only for framework examples and compatibility helpers. New code
    # uses ``self.runtime`` rather than reaching into model internals.
    self.config = self.runtime.config


def act(self, game_state: dict) -> str:
    return self.runtime.select_action(game_state)


def state_to_features(game_state: dict) -> np.ndarray | None:
    """Compatibility helper for framework examples."""
    return encode_state(game_state, active_config().state_encoder)
