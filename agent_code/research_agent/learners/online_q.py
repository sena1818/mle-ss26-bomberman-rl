"""R01 learner adapter: online one-step Q-learning."""

from __future__ import annotations

import numpy as np

from ..config import ExperimentConfig
from ..models.base import QModel
from .base import Transition


class OnlineQLearner:
    """Online Q-learning; R02 can reuse this learner with an MLP QModel."""

    def __init__(self, config: ExperimentConfig, model: QModel):
        self.config = config
        self.model = model

    def select_action(self, state: np.ndarray, legal_mask: np.ndarray, epsilon: float, generator: np.random.Generator) -> int:
        legal_indices = np.flatnonzero(legal_mask)
        if len(legal_indices) == 0:
            raise ValueError("No legal action was available.")
        if generator.random() < epsilon:
            return int(generator.choice(legal_indices))
        return int(np.argmax(np.where(legal_mask, self.model.q_values(state), -np.inf)))

    def observe(self, transition: Transition) -> float:
        if not hasattr(self.model, "q_learning_update"):
            raise TypeError("OnlineQLearner requires a QModel with q_learning_update.")
        return self.model.q_learning_update(
            transition.state,
            transition.action_index,
            transition.reward,
            transition.next_state,
            transition.next_legal_mask,
            self.config.learning_rate,
            self.config.discount,
        )

    def end_round(self) -> None:
        """R01 has no per-round cache; future learners clear private state here."""
