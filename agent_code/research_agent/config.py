"""Route configurations selected by the shared ExperimentRuntime."""

from dataclasses import dataclass, replace
import os


ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")
ACTIVE_EXPERIMENT = "R01"
FEATURE_DIMENSION = 44
FEATURE_VERSION = "handcrafted_v1"
REWARD_VERSION = "A00"
EXPLORATION_VERSION = "E00"
# A03 and A05 are the two additional death-penalty levels of the D dose-response
# study (A02 = 5.0, A03 = 1.0, A05 = 0.0).  They change nothing else.  A04
# (SAFE_BOMB) is deliberately absent: it needs per-step runtime state and is
# specified but not implemented.  See docs/01 section 4.2.
REWARD_VERSIONS = frozenset({"A00", "A01", "A02", "A03", "A05"})
# Exploration is versioned independently from the route and reward.  E01 is
# deliberately the only non-constant schedule currently registered: it is the
# predeclared, one-variable comparison against the historical E00 baseline.
EXPLORATION_VERSIONS = frozenset({"E00", "E01"})
# How a curriculum indexes its exploration schedule.  A curriculum segment is a
# separate game process whose round counter restarts at 1, so the schedule needs
# an explicit choice instead of an accidental one.
CURRICULUM_ANNEAL_MODES = frozenset({"global_round_offset", "per_segment"})
# A round that ends because the agent survived to the step limit is a time-limit
# truncation, not a terminal state: the MDP would have continued.  Bootstrapping
# is the correct target there (Pardo et al. 2018).  A round that ends because the
# agent died is a real terminal state and always uses ``target = r``.  This is a
# declared, ablatable choice rather than a constant hidden in the runtime.
TERMINAL_ON_TRUNCATION = False

EXPLORATION_SCHEDULES = {
    "E00": {
        "kind": "constant",
        "epsilon": 0.15,
        "description": "epsilon is 0.15 throughout training",
    },
    "E01": {
        "kind": "hold_then_linear",
        "initial_epsilon": 0.30,
        "hold_fraction": 0.20,
        "final_epsilon": 0.05,
        "description": "epsilon is 0.30 for the first 20% of training rounds, then linearly decays to 0.05",
    },
}

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
    exploration_version: str
    terminal_on_truncation: bool = TERMINAL_ON_TRUNCATION


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
        exploration_version=EXPLORATION_VERSION,
        terminal_on_truncation=TERMINAL_ON_TRUNCATION,
    ),
}


def _boolean_environment(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes"}:
        return True
    if lowered in {"0", "false", "no"}:
        return False
    raise ValueError(f"{name} must be one of 1/0/true/false/yes/no, got {raw!r}")


def exploration_specification(exploration_version: str) -> dict:
    """Return the serializable, versioned training-exploration definition."""
    try:
        return {"exploration_version": exploration_version, **EXPLORATION_SCHEDULES[exploration_version]}
    except KeyError as exc:
        raise ValueError(
            f"Unknown exploration version {exploration_version!r}; declared versions: {sorted(EXPLORATION_VERSIONS)}"
        ) from exc


def epsilon_for_training_round(config: ExperimentConfig, round_number: int, training_rounds: int) -> float:
    """Return the declared epsilon for one *training* round.

    E01 is indexed by the predeclared total training budget, not wall-clock
    time or steps.  With 500 rounds its first 100 rounds use 0.30; rounds
    101--500 interpolate from just below 0.30 to exactly 0.05.  Evaluation
    never calls this function: it always uses greedy epsilon 0.
    """
    if training_rounds < 1:
        raise ValueError("BOMBERMAN_TRAINING_ROUNDS must be positive.")
    if not 1 <= round_number <= training_rounds:
        raise ValueError(
            f"Training round {round_number} is outside the declared budget 1..{training_rounds}."
        )
    if config.exploration_version == "E00":
        return config.epsilon
    if config.exploration_version == "E01":
        hold_rounds = max(1, int(training_rounds * EXPLORATION_SCHEDULES["E01"]["hold_fraction"]))
        if round_number <= hold_rounds or hold_rounds == training_rounds:
            return float(EXPLORATION_SCHEDULES["E01"]["initial_epsilon"])
        if round_number == training_rounds:
            return float(EXPLORATION_SCHEDULES["E01"]["final_epsilon"])
        progress = (round_number - hold_rounds) / (training_rounds - hold_rounds)
        initial = float(EXPLORATION_SCHEDULES["E01"]["initial_epsilon"])
        final = float(EXPLORATION_SCHEDULES["E01"]["final_epsilon"])
        return initial + progress * (final - initial)
    raise ValueError(
        f"Unknown exploration version {config.exploration_version!r}; declared versions: {sorted(EXPLORATION_VERSIONS)}"
    )


def active_config() -> ExperimentConfig:
    """Select the requested route, reward, and exploration version for one job."""
    selected = os.environ.get("BOMBERMAN_EXPERIMENT", ACTIVE_EXPERIMENT)
    try:
        route_config = EXPERIMENTS[selected]
    except KeyError as exc:
        raise ValueError(f"Unknown experiment route {selected!r}; declared routes: {sorted(EXPERIMENTS)}") from exc
    reward_version = os.environ.get("BOMBERMAN_REWARD_VERSION", route_config.reward_version)
    if reward_version not in REWARD_VERSIONS:
        raise ValueError(f"Unknown reward version {reward_version!r}; declared versions: {sorted(REWARD_VERSIONS)}")
    exploration_version = os.environ.get("BOMBERMAN_EXPLORATION_VERSION", route_config.exploration_version)
    if exploration_version not in EXPLORATION_VERSIONS:
        raise ValueError(
            f"Unknown exploration version {exploration_version!r}; declared versions: {sorted(EXPLORATION_VERSIONS)}"
        )
    return replace(
        route_config,
        reward_version=reward_version,
        exploration_version=exploration_version,
        terminal_on_truncation=_boolean_environment(
            "BOMBERMAN_TERMINAL_ON_TRUNCATION", route_config.terminal_on_truncation
        ),
    )
