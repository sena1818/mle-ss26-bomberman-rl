"""The shared learner interface and frozen transition fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Transition:
    """Algorithm-independent post-encoding transition passed by the runtime."""

    state: np.ndarray
    action_index: int
    reward: float
    next_state: np.ndarray | None
    next_legal_mask: np.ndarray | None
    terminal: bool


class Learner(Protocol):
    """Owns algorithm-specific state such as SARSA cache or DQN replay."""

    def select_action(self, state: np.ndarray, legal_mask: np.ndarray, epsilon: float, generator: np.random.Generator) -> int: ...

    def observe(self, transition: Transition) -> float: ...

    def end_round(self) -> None: ...
