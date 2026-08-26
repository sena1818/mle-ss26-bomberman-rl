"""The shared learner interface and frozen transition fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Transition:
    """Algorithm-independent post-encoding transition passed by the runtime.

    ``reward`` is the accumulated, already discounted ``n_step``-step return
    ``sum_{i<n} gamma^i r_{t+i}``, and ``next_state`` is the state reached after
    those ``n`` steps.  A learner therefore bootstraps with ``gamma ** n_step``,
    never with ``gamma``.  ``n_step = 1`` is the one-step case and keeps the
    historical behaviour exactly.
    """

    state: np.ndarray
    action_index: int
    reward: float
    next_state: np.ndarray | None
    next_legal_mask: np.ndarray | None
    terminal: bool
    n_step: int = 1


class Learner(Protocol):
    """Owns algorithm-specific state such as a replay buffer or target network."""

    def select_action(self, state: np.ndarray, legal_mask: np.ndarray, epsilon: float, generator: np.random.Generator) -> int: ...

    def observe(self, transition: Transition) -> float:
        """Consume one transition and return a TD-error magnitude for logging.

        An online learner returns the signed TD error of the single update it
        performed.  A batch learner returns the mean absolute TD error of the
        batch it drew, or 0.0 for a step that only filled the buffer.
        """

    def end_round(self) -> None: ...
