"""Experiment selection and deliberately small first-pass hyperparameters."""

from dataclasses import dataclass


ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")
ACTIVE_EXPERIMENT = "R01"


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    state_encoder: str
    network: str
    algorithm: str
    learning_rate: float
    discount: float
    epsilon: float
    safety_filter: str


EXPERIMENTS = {
    "R01": ExperimentConfig(
        name="R01",
        state_encoder="handcrafted_v1",
        network="linear_q",
        algorithm="q_learning",
        learning_rate=0.02,
        discount=0.95,
        epsilon=0.15,
        safety_filter="legality_only",
    ),
}


def active_config() -> ExperimentConfig:
    """Return the one selected experiment, failing early on a spelling mistake."""
    return EXPERIMENTS[ACTIVE_EXPERIMENT]
