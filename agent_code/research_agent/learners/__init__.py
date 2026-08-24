"""Learner adapters selected by the shared experiment runtime."""

from __future__ import annotations

from ..config import ExperimentConfig
from ..models.base import QModel
from .base import Learner, Transition
from .online_q import OnlineQLearner


def build_learner(config: ExperimentConfig, model: QModel) -> Learner:
    if config.algorithm == "q_learning":
        return OnlineQLearner(config, model)
    raise NotImplementedError(f"Learner adapter {config.algorithm!r} has not been implemented yet.")


__all__ = ("Learner", "Transition", "OnlineQLearner", "build_learner")
