"""Action selection and the algorithm dispatch shared by experiment callbacks."""

from __future__ import annotations

import numpy as np

from .config import ExperimentConfig
from .networks import LinearQNetwork


def choose_action_index(
    model: LinearQNetwork,
    state: np.ndarray,
    legal_mask: np.ndarray,
    epsilon: float,
    generator: np.random.Generator,
) -> int:
    """Use epsilon-greedy exploration, always restricted to legal actions."""
    legal_indices = np.flatnonzero(legal_mask)
    if len(legal_indices) == 0:
        raise ValueError("No legal action was available.")
    if generator.random() < epsilon:
        return int(generator.choice(legal_indices))
    q_values = model.q_values(state)
    masked_q = np.where(legal_mask, q_values, -np.inf)
    return int(np.argmax(masked_q))


def update_q_learning(
    model: LinearQNetwork,
    config: ExperimentConfig,
    state: np.ndarray,
    action_index: int,
    reward: float,
    next_state: np.ndarray | None,
    next_legal_mask: np.ndarray | None,
) -> float:
    if config.algorithm != "q_learning":
        raise NotImplementedError(f"Algorithm {config.algorithm!r} has not been implemented yet.")
    return model.q_learning_update(
        state,
        action_index,
        reward,
        next_state,
        next_legal_mask,
        config.learning_rate,
        config.discount,
    )
