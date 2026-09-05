"""Learner adapters selected by the shared experiment runtime.

The update rule and the data pipeline are separate declarations.  A route
declares ``algorithm``; whether the updates are drawn from a replay buffer with
a target network is declared by ``replay``.  Classic DQN is therefore
``q_learning`` plus a replay block, and Double DQN requires one.
"""

from __future__ import annotations

from ..config import ExperimentConfig
from ..models.base import QModel
from .base import Learner, Transition
from .online_q import OnlineQLearner
from .replay_q import ReplayQLearner


def build_learner(config: ExperimentConfig, model: QModel, *, seed: int = 0, training: bool = True) -> Learner:
    # An evaluation job never receives a transition: the framework only calls
    # ``act``.  Giving it the online adapter keeps greedy action selection
    # identical while avoiding a replay allocation of hundreds of megabytes per
    # job, multiplied by every checkpoint and seed in an evaluation sweep.
    if not training:
        return OnlineQLearner(config, model)
    if config.replay is not None:
        return ReplayQLearner(config, model, seed=seed)
    if config.algorithm == "q_learning":
        return OnlineQLearner(config, model)
    raise NotImplementedError(
        f"Learner adapter {config.algorithm!r} without replay has not been implemented yet."
    )


__all__ = ("Learner", "Transition", "OnlineQLearner", "ReplayQLearner", "build_learner")
