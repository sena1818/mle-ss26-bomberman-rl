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
# handcrafted_v2 appends ten escape entries, v3 another eight routing entries.
# Every v1 index keeps its meaning, so a v1 model's weights stay readable, but
# the vectors are different lengths and arms across versions are not comparable.
FEATURE_DIMENSION_V2 = 54
FEATURE_DIMENSION_V3 = 62
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
# Exploration is versioned independently from the route and reward.  E01 defines
# its hold as a *fraction* of the training budget, which means changing the
# budget silently changes the schedule too: every budget comparison in
# docs/01 sections 7.13 and 7.18 is confounded by it (docs/05 section 0.14).
# The E02 family states both phases in absolute rounds so budget and schedule
# are independent.  Each point of the ablation is its own label rather than a
# parameter, so a finished run's snapshot names the exact schedule it saw --
# the same discipline as A00--A06 and R02_1--R02_3.
EXPLORATION_VERSIONS = frozenset({"E00", "E01", "E02", "E03", "E04", "E05", "E06", "E07", "E08"})
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
    # The E02 family: hold and anneal are both counted in rounds, so a longer
    # budget no longer stretches the schedule.  E02 is the absolute spelling of
    # E01 at a 2000-round budget and is bit-identical to it there, which is what
    # lets runs/m3_3lx_oppeval_20260826 serve as the control for E03--E06
    # without being re-run.  E03/E04 vary only the hold; E05/E06 vary only the
    # floor.  hold_rounds + anneal_rounds is 2000 in all five, so every arm
    # reaches its floor exactly at the end of the budget and the only thing that
    # differs is how the schedule is spent.
    "E02": {
        "kind": "hold_then_linear_absolute",
        "initial_epsilon": 0.30,
        "hold_rounds": 400,
        "anneal_rounds": 1600,
        "final_epsilon": 0.05,
        "description": "epsilon is 0.30 for 400 rounds, then linearly decays to 0.05 over 1600 rounds"
                       " (identical to E01 at a 2000-round budget)",
    },
    "E03": {
        "kind": "hold_then_linear_absolute",
        "initial_epsilon": 0.30,
        "hold_rounds": 100,
        "anneal_rounds": 1900,
        "final_epsilon": 0.05,
        "description": "epsilon is 0.30 for 100 rounds, then linearly decays to 0.05 over 1900 rounds",
    },
    "E04": {
        "kind": "hold_then_linear_absolute",
        "initial_epsilon": 0.30,
        "hold_rounds": 1000,
        "anneal_rounds": 1000,
        "final_epsilon": 0.05,
        "description": "epsilon is 0.30 for 1000 rounds, then linearly decays to 0.05 over 1000 rounds",
    },
    "E05": {
        "kind": "hold_then_linear_absolute",
        "initial_epsilon": 0.30,
        "hold_rounds": 400,
        "anneal_rounds": 1600,
        "final_epsilon": 0.02,
        "description": "E02 with the floor lowered to 0.02; the hold and anneal lengths are unchanged",
    },
    "E06": {
        "kind": "hold_then_linear_absolute",
        "initial_epsilon": 0.30,
        "hold_rounds": 400,
        "anneal_rounds": 1600,
        "final_epsilon": 0.10,
        "description": "E02 with the floor raised to 0.10; the hold and anneal lengths are unchanged",
    },
    # E07 / E08 exist for the M4 line and differ from each other in exactly one
    # number, the starting epsilon.
    #
    # Every E00--E06 schedule starts at 0.30, which was never questioned because
    # it did not need to be: with handcrafted features and a linear or small MLP
    # head, an untrained greedy action is close to arbitrary, so 0.30 explores
    # fine.  A randomly initialised CNN is different in kind -- its argmax is
    # near-constant across states, so 0.30 leaves 70% of steps executing one
    # fixed action and the buffer fills with a single trajectory.  E07 starts at
    # 1.0 and holds there while the buffer fills, which is the ordinary DQN
    # spelling of the same idea.
    #
    # E08 is for an arm that starts from a behaviour-cloned policy.  There, a
    # high epsilon would spend the clone before it can be used, so the start is
    # low.  Warm start and exploration are coupled -- a BC arm changes both on
    # purpose -- and the report has to say so rather than call it one factor.
    "E07": {
        "kind": "hold_then_linear_absolute",
        "initial_epsilon": 1.00,
        "hold_rounds": 100,
        "anneal_rounds": 900,
        "final_epsilon": 0.05,
        "description": "epsilon is 1.00 for 100 rounds, then linearly decays to 0.05 over 900 rounds"
                       " (cold start for a randomly initialised deep network)",
    },
    "E08": {
        "kind": "hold_then_linear_absolute",
        "initial_epsilon": 0.20,
        "hold_rounds": 100,
        "anneal_rounds": 900,
        "final_epsilon": 0.05,
        "description": "E07 with the start lowered to 0.20; hold and anneal lengths are unchanged"
                       " (warm start from a behaviour-cloned policy)",
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
EXPLOSION_TIMER = 2


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
    # The MLP routes make their optimization recipe explicit.  Keeping these
    # in the route declaration (rather than as hidden model defaults) makes a
    # stabilized M3 baseline reproducible and leaves R02/M3.0 untouched.
    optimizer: str = "sgd"
    td_loss: str = "mse"
    gradient_clip_norm: float | None = None


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
    # M3.1 -- same handcrafted MLP architecture as R02/M3.0, but with a
    # numerically stable fitted-Q recipe.  It is deliberately a new route so
    # the historical online-SGD control remains immutable and attributable.
    "R02_1": ExperimentConfig(
        name="R02_1",
        lines=("M3",),
        state_encoder="handcrafted_v1",
        network="mlp_q",
        algorithm="q_learning",
        learning_rate=1e-3,
        discount=0.95,
        epsilon=0.15,
        safety_filter="legality_only",
        feature_version=FEATURE_VERSION,
        reward_version=REWARD_VERSION,
        exploration_version=EXPLORATION_VERSION,
        terminal_on_truncation=TERMINAL_ON_TRUNCATION,
        hidden_layers=(64, 32),
        optimizer="adam",
        td_loss="huber",
        gradient_clip_norm=10.0,
    ),
    # M3.2 / M3.3 -- the M3.1 recipe untouched, with the state representation as
    # the only changed factor.  docs/01 section 7.10 located M3.1's remaining
    # suicides in the features rather than in the reward or the task, so these
    # two routes are the controlled test of that finding: R02_2 adds only the
    # escape entries, R02_3 adds the routing entries on top.  Keeping them apart
    # is the point -- bundled, neither result would be attributable.
    "R02_2": ExperimentConfig(
        name="R02_2",
        lines=("M3",),
        state_encoder="handcrafted_v2",
        network="mlp_q",
        algorithm="q_learning",
        learning_rate=1e-3,
        discount=0.95,
        epsilon=0.15,
        safety_filter="legality_only",
        feature_version="handcrafted_v2",
        reward_version=REWARD_VERSION,
        exploration_version=EXPLORATION_VERSION,
        terminal_on_truncation=TERMINAL_ON_TRUNCATION,
        hidden_layers=(64, 32),
        optimizer="adam",
        td_loss="huber",
        gradient_clip_norm=10.0,
    ),
    "R02_3": ExperimentConfig(
        name="R02_3",
        lines=("M3",),
        state_encoder="handcrafted_v3",
        network="mlp_q",
        algorithm="q_learning",
        learning_rate=1e-3,
        discount=0.95,
        epsilon=0.15,
        safety_filter="legality_only",
        feature_version="handcrafted_v3",
        reward_version=REWARD_VERSION,
        exploration_version=EXPLORATION_VERSION,
        terminal_on_truncation=TERMINAL_ON_TRUNCATION,
        hidden_layers=(64, 32),
        optimizer="adam",
        td_loss="huber",
        gradient_clip_norm=10.0,
    ),
    # R02_4 and R02_5 are the two single-factor arms docs/01 section 7.16.6
    # pointed at: the remaining self-inflicted deaths are a value-function
    # failure, and what acts on a value function is the algorithm or its
    # capacity, not more features.  Each is R02_3 with exactly one field moved,
    # so a result names the one thing that changed.
    "R02_4": ExperimentConfig(
        name="R02_4",
        lines=("M3",),
        state_encoder="handcrafted_v3",
        network="mlp_q",
        # The only change from R02_3.  Overestimation by the max operator is the
        # textbook cause of a stalling action ranking above a visible escape,
        # which is what section 7.16.5 measured.
        algorithm="double_dqn",
        learning_rate=1e-3,
        discount=0.95,
        epsilon=0.15,
        safety_filter="legality_only",
        feature_version="handcrafted_v3",
        reward_version=REWARD_VERSION,
        exploration_version=EXPLORATION_VERSION,
        terminal_on_truncation=TERMINAL_ON_TRUNCATION,
        hidden_layers=(64, 32),
        replay=ReplayConfig(),
        optimizer="adam",
        td_loss="huber",
        gradient_clip_norm=10.0,
    ),
    "R02_5": ExperimentConfig(
        name="R02_5",
        lines=("M3",),
        state_encoder="handcrafted_v3",
        network="mlp_q",
        algorithm="q_learning",
        learning_rate=1e-3,
        discount=0.95,
        epsilon=0.15,
        safety_filter="legality_only",
        feature_version="handcrafted_v3",
        reward_version=REWARD_VERSION,
        exploration_version=EXPLORATION_VERSION,
        terminal_on_truncation=TERMINAL_ON_TRUNCATION,
        # The only change from R02_3.  mean_hidden_zero_fraction rose from 0.625
        # at 2000 rounds to 0.745 at 5000 (docs/01 sections 7.12.4 and 7.18):
        # three quarters of the units are silent on a typical input.
        hidden_layers=(128, 64),
        optimizer="adam",
        td_loss="huber",
        gradient_clip_norm=10.0,
    ),
    # R02_6 and R02_7 exist because R02_5 answered a narrower question than it
    # looked like.  It widened the net while holding a recipe that was tuned
    # around (64, 32), so its result is "widening does not help at this recipe",
    # not "this width is worse".  R02_6 keeps the width and halves the learning
    # rate -- the wider net carries about 2.6x the parameters for the same
    # ~111k updates -- and R02_7 keeps the width and takes the algorithm that
    # cut deaths by a third at (64, 32).
    "R02_6": ExperimentConfig(
        name="R02_6",
        lines=("M3",),
        state_encoder="handcrafted_v3",
        network="mlp_q",
        algorithm="q_learning",
        # The only change from R02_5.
        learning_rate=5e-4,
        discount=0.95,
        epsilon=0.15,
        safety_filter="legality_only",
        feature_version="handcrafted_v3",
        reward_version=REWARD_VERSION,
        exploration_version=EXPLORATION_VERSION,
        terminal_on_truncation=TERMINAL_ON_TRUNCATION,
        hidden_layers=(128, 64),
        optimizer="adam",
        td_loss="huber",
        gradient_clip_norm=10.0,
    ),
    "R02_7": ExperimentConfig(
        name="R02_7",
        lines=("M3",),
        state_encoder="handcrafted_v3",
        network="mlp_q",
        # The only change from R02_5.
        algorithm="double_dqn",
        learning_rate=1e-3,
        discount=0.95,
        epsilon=0.15,
        safety_filter="legality_only",
        feature_version="handcrafted_v3",
        reward_version=REWARD_VERSION,
        exploration_version=EXPLORATION_VERSION,
        terminal_on_truncation=TERMINAL_ON_TRUNCATION,
        hidden_layers=(128, 64),
        replay=ReplayConfig(),
        optimizer="adam",
        td_loss="huber",
        gradient_clip_norm=10.0,
    ),
    # M4 anchor -- egocentric board tensor, CNN plus global-scalar MLP, Double
    # DQN.  docs/05 section 5.4 requires this to learn from scratch before any
    # further increment (behaviour cloning, dueling) is added.
    #
    # The replay settings are the route's, not a learner default, because they
    # are the recipe: capacity 100k (about 330 rounds of history, affordable
    # only because the spatial states are stored as uint8 codes), batch 32 every
    # fourth environment step.  That last pair is the throughput lever for the
    # whole line -- a gradient step costs ~50x an inference, so the replay ratio
    # of 8 samples per environment step, not the game, decides how many
    # environment steps a day of compute buys.  8 is the ratio DQN and Rainbow
    # both use; 16 was measured at twice the cost for no argued benefit, and is
    # left as a single-factor arm rather than paid for up front.  D4 augmentation is on from the
    # anchor rather than added later: it is a property of the representation
    # (the arena's symmetry group acts on an agent-centred window), it costs
    # nothing at sample time, and holding it back would only make the anchor a
    # weaker base for every increment measured against it.
    "R07": ExperimentConfig(
        name="R07",
        lines=("M4",),
        state_encoder="board_egocentric_v2",
        network="cnn_mlp_q",
        algorithm="double_dqn",
        learning_rate=2.5e-4,
        discount=0.95,
        epsilon=0.15,
        safety_filter="legality_only",
        feature_version="board_egocentric_v2",
        reward_version=REWARD_VERSION,
        exploration_version=EXPLORATION_VERSION,
        terminal_on_truncation=TERMINAL_ON_TRUNCATION,
        hidden_layers=(256,),
        replay=ReplayConfig(
            capacity=100_000,
            batch_size=32,
            min_size=10_000,
            train_every=4,
            target_update_every=500,
            augmentation="d4",
        ),
        optimizer="adam",
        td_loss="huber",
        gradient_clip_norm=10.0,
    ),
    # M4 dueling increment -- identical to R07 apart from the value/advantage
    # split in the head.  It exists as its own route so the increment is a
    # single declared factor rather than a flag buried in the model.
    "R08": ExperimentConfig(
        name="R08",
        lines=("M4",),
        state_encoder="board_egocentric_v2",
        network="dueling_cnn_mlp_q",
        algorithm="double_dqn",
        learning_rate=2.5e-4,
        discount=0.95,
        epsilon=0.15,
        safety_filter="legality_only",
        feature_version="board_egocentric_v2",
        reward_version=REWARD_VERSION,
        exploration_version=EXPLORATION_VERSION,
        terminal_on_truncation=TERMINAL_ON_TRUNCATION,
        hidden_layers=(256,),
        replay=ReplayConfig(
            capacity=100_000,
            batch_size=32,
            min_size=10_000,
            train_every=4,
            target_update_every=500,
            augmentation="d4",
        ),
        optimizer="adam",
        td_loss="huber",
        gradient_clip_norm=10.0,
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

    The E02 family answers the same question in absolute rounds.  It reads the
    budget only to reject an out-of-range round, so a schedule and a budget can
    finally be varied one at a time.  A budget longer than
    ``hold_rounds + anneal_rounds`` simply stays at the floor for the remainder;
    a budget shorter than that never reaches the floor, which ``Experiment``
    rejects at config time rather than discovering mid-run.
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
    schedule = EXPLORATION_SCHEDULES.get(config.exploration_version)
    if schedule is not None and schedule["kind"] == "hold_then_linear_absolute":
        hold_rounds = int(schedule["hold_rounds"])
        anneal_rounds = int(schedule["anneal_rounds"])
        initial = float(schedule["initial_epsilon"])
        final = float(schedule["final_epsilon"])
        if round_number <= hold_rounds or anneal_rounds < 1:
            return initial
        if round_number >= hold_rounds + anneal_rounds:
            return final
        progress = (round_number - hold_rounds) / anneal_rounds
        return initial + progress * (final - initial)
    raise ValueError(
        f"Unknown exploration version {config.exploration_version!r}; declared versions: {sorted(EXPLORATION_VERSIONS)}"
    )


def validate_config(config: ExperimentConfig) -> ExperimentConfig:
    """Fail closed on a combination no learner or model adapter implements."""
    if config.n_step < 1:
        raise ValueError(f"n_step must be at least 1, got {config.n_step}.")
    if config.optimizer not in {"sgd", "adam"}:
        raise ValueError(f"optimizer must be 'sgd' or 'adam', got {config.optimizer!r}.")
    if config.td_loss not in {"mse", "huber"}:
        raise ValueError(f"td_loss must be 'mse' or 'huber', got {config.td_loss!r}.")
    if config.gradient_clip_norm is not None and config.gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive when declared.")
    if config.algorithm == "double_dqn" and config.replay is None:
        raise ValueError("double_dqn requires a replay buffer and a target network; replay must not be null.")
    if config.replay is not None and config.replay.augmentation == "d4" and not config.state_encoder.startswith("board_egocentric_"):
        raise ValueError(
            "replay.augmentation 'd4' requires an agent-centred board_egocentric representation: "
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
