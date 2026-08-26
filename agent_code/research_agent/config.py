"""Route configurations selected by the shared ExperimentRuntime.

One *route* is a frozen agent design: state representation, Q-model and update
rule.  Routes are grouped into the four *main lines* of docs/05.  Everything a
line varies on top of its route -- reward version, exploration schedule,
potential shaping, n-step length, replay -- is a separate, explicitly declared
dimension so that any single-factor comparison stays a single factor.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
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
# specified but not implemented.  See docs/01 section 4.2.  A06 is A03 plus
# potential-based shaping (docs/05 section 4); its event table is identical to
# A03 on purpose, so the shaping term is the only variable.
REWARD_VERSIONS = frozenset({"A00", "A01", "A02", "A03", "A05", "A06"})
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

# The four main lines of docs/05.  A line is a research question; a route is the
# concrete agent design that answers it.  Naming is fixed here so that configs,
# snapshots and reports all use the same identifier for the same thing.
MAIN_LINES = {
    "M1": "minimal interpretable baseline: handcrafted features, linear Q, online Q-learning",
    "M2": "M1 plus potential shaping, n-step returns and replay; the model is unchanged",
    "M3": "M2 with a small MLP replacing the linear head; tests for feature interactions",
    "M4": "egocentric board tensor with a (Dueling) Double DQN; learned spatial representation",
}

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

# Potential-based shaping, keyed by the reward version that switches it on.  The
# weights live here rather than in the shaping module so that one reward version
# label always names one exact set of numbers, recorded in the run snapshot.
SHAPING_SPECIFICATIONS = {
    "A06": {
        "name": "potential_v1",
        "coin_weight": 0.05,
        "distance_cap": 20,
        "danger_weight": 0.30,
        "terminal_potential": 0.0,
        "notes": (
            "phi(s) = -coin_weight * min(BFS distance to the nearest reachable collection target, "
            "distance_cap) - danger_weight * [s lies in a future blast]; phi(terminal) = 0. "
            "Shaping is gamma * phi(s') - phi(s) with the learner's own gamma, so the optimal "
            "policy is unchanged (Ng, Harada & Russell 1999)."
        ),
    },
}

# Verified against the unmodified SS26 settings.py.  They are deliberately
# local constants: a submitted agent cannot rely on imports outside its folder.
MAX_STEPS = 400
BOMB_POWER = 3
BOMB_TIMER = 4


@dataclass(frozen=True)
class ReplayConfig:
    """Experience replay and target network settings for one route.

    ``None`` instead of an instance means fully online updating, which is what
    M1 uses.  Every field is declared rather than defaulted inside a learner so
    that the run snapshot records the values that actually produced a result.
    """

    capacity: int = 10_000
    batch_size: int = 32
    min_size: int = 1_000
    train_every: int = 1
    target_update_every: int = 500
    # "none" or "d4": the eight board symmetries, only valid for a spatial,
    # agent-centred state representation.  See docs/05 section 5.4.
    augmentation: str = "none"

    def __post_init__(self) -> None:
        if min(self.capacity, self.batch_size, self.min_size, self.train_every, self.target_update_every) < 1:
            raise ValueError("Every replay setting must be a positive integer.")
        if self.batch_size > self.capacity:
            raise ValueError("replay.batch_size cannot exceed replay.capacity.")
        if self.min_size > self.capacity:
            raise ValueError("replay.min_size cannot exceed replay.capacity.")
        if self.batch_size > self.min_size:
            raise ValueError("replay.min_size must be at least replay.batch_size.")
        if self.augmentation not in {"none", "d4"}:
            raise ValueError(f"replay.augmentation must be 'none' or 'd4', got {self.augmentation!r}")

    @classmethod
    def parse(cls, value: dict | None) -> "ReplayConfig | None":
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError("replay must be null or an object of replay settings.")
        unknown = sorted(set(value) - {field for field in cls.__dataclass_fields__})
        if unknown:
            raise ValueError(f"Unknown replay settings: {', '.join(unknown)}")
        return cls(
            capacity=int(value.get("capacity", cls.capacity)),
            batch_size=int(value.get("batch_size", cls.batch_size)),
            min_size=int(value.get("min_size", cls.min_size)),
            train_every=int(value.get("train_every", cls.train_every)),
            target_update_every=int(value.get("target_update_every", cls.target_update_every)),
            augmentation=str(value.get("augmentation", cls.augmentation)),
        )


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
    # Which of the four main lines of docs/05 this route serves.  M1 and M2
    # share one route on purpose: M2 changes reward shaping, n-step and replay,
    # none of which are part of an agent *design*.
    lines: tuple[str, ...] = ("M1",)
    terminal_on_truncation: bool = TERMINAL_ON_TRUNCATION
    # Bootstrap length of the TD target.  n = 1 is the historical behaviour.
    n_step: int = 1
    # Hidden widths of an MLP head; empty for a purely linear or convolutional model.
    hidden_layers: tuple[int, ...] = ()
    # None means online updating without a target network.
    replay: ReplayConfig | None = None


EXPERIMENTS = {
    # M1 -- the frozen minimal baseline.  Never change these numbers: every
    # published R01 result was produced by exactly this configuration.
    "R01": ExperimentConfig(
        name="R01",
        lines=("M1", "M2"),
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
    # M3 -- identical to R01 except for the function approximator.  Whether
    # shaping, n-step or replay are switched on is declared per experiment, so
    # that M2 and M3 can be compared at matching training recipes.
    "R02": ExperimentConfig(
        name="R02",
        lines=("M3",),
        state_encoder="handcrafted_v1",
        network="mlp_q",
        algorithm="q_learning",
        learning_rate=0.02,
        discount=0.95,
        epsilon=0.15,
        safety_filter="legality_only",
        feature_version=FEATURE_VERSION,
        reward_version=REWARD_VERSION,
        exploration_version=EXPLORATION_VERSION,
        terminal_on_truncation=TERMINAL_ON_TRUNCATION,
        hidden_layers=(64, 32),
    ),
    # M4 anchor -- egocentric board tensor, CNN plus global-scalar MLP, Double
    # DQN.  docs/05 section 5.4 requires this to learn from scratch before any
    # further increment (D4 augmentation, behaviour cloning, dueling) is added.
    "R07": ExperimentConfig(
        name="R07",
        lines=("M4",),
        state_encoder="board_egocentric_v1",
        network="cnn_mlp_q",
        algorithm="double_dqn",
        learning_rate=2.5e-4,
        discount=0.95,
        epsilon=0.15,
        safety_filter="legality_only",
        feature_version="board_egocentric_v1",
        reward_version=REWARD_VERSION,
        exploration_version=EXPLORATION_VERSION,
        terminal_on_truncation=TERMINAL_ON_TRUNCATION,
        hidden_layers=(256,),
        replay=ReplayConfig(),
    ),
    # M4 dueling increment -- identical to R07 apart from the value/advantage
    # split in the head.  It exists as its own route so the increment is a
    # single declared factor rather than a flag buried in the model.
    "R08": ExperimentConfig(
        name="R08",
        lines=("M4",),
        state_encoder="board_egocentric_v1",
        network="dueling_cnn_mlp_q",
        algorithm="double_dqn",
        learning_rate=2.5e-4,
        discount=0.95,
        epsilon=0.15,
        safety_filter="legality_only",
        feature_version="board_egocentric_v1",
        reward_version=REWARD_VERSION,
        exploration_version=EXPLORATION_VERSION,
        terminal_on_truncation=TERMINAL_ON_TRUNCATION,
        hidden_layers=(256,),
        replay=ReplayConfig(),
    ),
}

# Which route serves which main line, for reports and for the runner's checks.
ROUTES_BY_LINE = {
    line: tuple(sorted(name for name, config in EXPERIMENTS.items() if line in config.lines))
    for line in MAIN_LINES
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


def _integer_environment(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _replay_environment(name: str, default: ReplayConfig | None) -> ReplayConfig | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be JSON (an object of replay settings, or null).") from exc
    return ReplayConfig.parse(parsed)


def exploration_specification(exploration_version: str) -> dict:
    """Return the serializable, versioned training-exploration definition."""
    try:
        return {"exploration_version": exploration_version, **EXPLORATION_SCHEDULES[exploration_version]}
    except KeyError as exc:
        raise ValueError(
            f"Unknown exploration version {exploration_version!r}; declared versions: {sorted(EXPLORATION_VERSIONS)}"
        ) from exc


def shaping_specification(reward_version: str) -> dict | None:
    """Return the potential-shaping definition a reward version switches on.

    Shaping is derived from the reward version rather than declared separately,
    so a config can never say A03 and silently train with a shaped reward.
    """
    if reward_version not in REWARD_VERSIONS:
        raise ValueError(f"Unknown reward version {reward_version!r}; declared versions: {sorted(REWARD_VERSIONS)}")
    specification = SHAPING_SPECIFICATIONS.get(reward_version)
    return dict(specification) if specification is not None else None


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


def validate_config(config: ExperimentConfig) -> ExperimentConfig:
    """Fail closed on a combination no learner or model adapter implements."""
    if config.n_step < 1:
        raise ValueError(f"n_step must be at least 1, got {config.n_step}.")
    if config.algorithm == "double_dqn" and config.replay is None:
        raise ValueError("double_dqn requires a replay buffer and a target network; replay must not be null.")
    if config.replay is not None and config.replay.augmentation == "d4" and config.state_encoder != "board_egocentric_v1":
        raise ValueError(
            "replay.augmentation 'd4' requires the agent-centred board_egocentric_v1 representation: "
            f"the board symmetries are not label-preserving for {config.state_encoder!r}."
        )
    return config


def active_config() -> ExperimentConfig:
    """Select the requested route and its declared dimensions for one job."""
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
    return validate_config(replace(
        route_config,
        reward_version=reward_version,
        exploration_version=exploration_version,
        terminal_on_truncation=_boolean_environment(
            "BOMBERMAN_TERMINAL_ON_TRUNCATION", route_config.terminal_on_truncation
        ),
        n_step=_integer_environment("BOMBERMAN_N_STEP", route_config.n_step),
        replay=_replay_environment("BOMBERMAN_REPLAY", route_config.replay),
    ))
