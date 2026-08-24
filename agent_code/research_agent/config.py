"""Route configurations selected by the shared ExperimentRuntime."""

from dataclasses import dataclass
import os


ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")
ACTIVE_EXPERIMENT = "R01"
FEATURE_DIMENSION = 44
FEATURE_VERSION = "handcrafted_v1"
REWARD_VERSION = "A00"

# Verified against the unmodified SS26 settings.py.  They are deliberately
# local constants: a submitted agent cannot rely on imports outside its folder.
MAX_STEPS = 400
BOMB_POWER = 3
BOMB_TIMER = 4


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
    feature_version: str
    reward_version: str


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
        feature_version=FEATURE_VERSION,
        reward_version=REWARD_VERSION,
    ),
}


def active_config() -> ExperimentConfig:
    """Select the requested route without exposing it to official callbacks."""
    selected = os.environ.get("BOMBERMAN_EXPERIMENT", ACTIVE_EXPERIMENT)
    try:
        return EXPERIMENTS[selected]
    except KeyError as exc:
        raise ValueError(f"Unknown experiment route {selected!r}; declared routes: {sorted(EXPERIMENTS)}") from exc
