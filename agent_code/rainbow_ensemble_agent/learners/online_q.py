"""Online one-step (and n-step) Q-learning, used by the M1--M3 lines."""

from __future__ import annotations

import numpy as np

from ..config import ExperimentConfig
from ..models.base import QModel
from .base import Transition


class OnlineQLearner:
    """Fully online Q-learning: one gradient step per transition, no buffer."""

    def __init__(self, config: ExperimentConfig, model: QModel):
        self.config = config
        self.model = model
        self.gradient_steps = 0

    def select_action(self, state: np.ndarray, legal_mask: np.ndarray, epsilon: float, generator: np.random.Generator) -> int:
        legal_indices = np.flatnonzero(legal_mask)
        if len(legal_indices) == 0:
            raise ValueError("No legal action was available.")
        if generator.random() < epsilon:
            return int(generator.choice(legal_indices))
        return int(np.argmax(np.where(legal_mask, self.model.q_values(state), -np.inf)))

    def step_diagnostics(self) -> dict:
        """Online updating has no buffer, so every observed step is a step."""
        return {"gradient_applied": True, "gradient_steps": self.gradient_steps}

    def observe(self, transition: Transition) -> float:
        if not hasattr(self.model, "q_learning_update"):
            raise TypeError("OnlineQLearner requires a QModel with q_learning_update.")
        # An n-step return already carries n discount factors, so the bootstrap
        # term is discounted by gamma**n.  n = 1 leaves this exactly as it was.
        self.gradient_steps += 1
        return self.model.q_learning_update(
            transition.state,
            transition.action_index,
            transition.reward,
            transition.next_state,
            transition.next_legal_mask,
            self.config.learning_rate,
            self.config.discount ** transition.n_step,
        )

    def end_round(self) -> None:
        """Nothing is cached between rounds; the n-step window lives in the runtime."""
