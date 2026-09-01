"""rule_based_agent with a fraction of its actions replaced by random ones.

A stand-in for a middling tournament entrant: an agent that mostly knows what
it is doing and sometimes does not.  ``rule_based_agent`` is the best dodger
in this framework (docs/01 section 7.32) and the scripted agents that are
weaker than it do not dodge at all, so the pool of proxies had nothing in
between.  The fraction is read from ``BOMBERMAN_NOISY_RULE_BASED_EPSILON`` and
defaults to 0.15; the random draw is over the four moves and BOMB, never WAIT,
so the noise is mistakes rather than hesitation.
"""

from __future__ import annotations

import os

import numpy as np

from ..rule_based_agent import callbacks as rule_based

RANDOM_ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "BOMB")


def setup(self):
    rule_based.setup(self)
    raw = os.environ.get("BOMBERMAN_NOISY_RULE_BASED_EPSILON", "0.15")
    self.noise_epsilon = float(raw)
    if not 0.0 <= self.noise_epsilon <= 1.0:
        raise ValueError(f"BOMBERMAN_NOISY_RULE_BASED_EPSILON must lie in [0, 1], got {raw!r}")
    self.noise_generator = np.random.default_rng()


def act(self, game_state: dict):
    action = rule_based.act(self, game_state)
    if self.noise_generator.random() < self.noise_epsilon:
        return str(self.noise_generator.choice(RANDOM_ACTIONS))
    return action
