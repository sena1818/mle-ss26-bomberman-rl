"""Dependency-free shared machinery for reproducible Bomberman experiments."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / "runs"
SCENARIOS = {"empty", "coin-heaven", "loot-crate", "classic"}
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
# Every implemented route, keyed by the id used in a config's ``route`` field.
# A route is an agent *design*; the main line it serves is recorded so that a
# run directory says which of the four research questions it belongs to.  What
# a line varies on top of its route -- reward version, exploration schedule,
# shaping, n-step, replay -- is declared per experiment, not baked in here.
#
# A04 (SAFE_BOMB) is specified in docs/01 but not implemented, so no route
# lists it.  A06 is A03 plus potential shaping and is only offered on routes
# whose state representation the potential is defined for.
_VECTOR_REWARD_VERSIONS = {"A00", "A01", "A02", "A03", "A05", "A06"}
IMPLEMENTED_ROUTES = {
    "R01": {
        "lines": ("M1", "M2"),
        "declaration": ("linear_q", "q_learning", "handcrafted_v1"),
        "reward_versions": _VECTOR_REWARD_VERSIONS,
        "exploration_versions": {"E00", "E01"},
    },
    "R02": {
        "lines": ("M3",),
        "declaration": ("mlp_q", "q_learning", "handcrafted_v1"),
        "reward_versions": _VECTOR_REWARD_VERSIONS,
        "exploration_versions": {"E00", "E01"},
    },
    "R07": {
        "lines": ("M4",),
        "declaration": ("cnn_mlp_q", "double_dqn", "board_egocentric_v1"),
        "reward_versions": _VECTOR_REWARD_VERSIONS,
        "exploration_versions": {"E00", "E01"},
    },
    "R08": {
        "lines": ("M4",),
        "declaration": ("dueling_cnn_mlp_q", "double_dqn", "board_egocentric_v1"),
        "reward_versions": _VECTOR_REWARD_VERSIONS,
        "exploration_versions": {"E00", "E01"},
    },
}
MAIN_LINES = ("M1", "M2", "M3", "M4")
CHECKPOINT_MODES = {"latest", "all"}
# How a run may be warm-started from weights produced outside it.  Only
# behaviour cloning is implemented; "checkpoint" is reserved and fails closed.
INITIAL_MODEL_KINDS = {"behaviour_cloning"}
# Mirrors agent_code.research_agent.config.CURRICULUM_ANNEAL_MODES.  A curriculum
# segment is a separate process with its own round counter, so the way the
# exploration schedule spans segments has to be declared rather than inferred.
CURRICULUM_ANNEAL_MODES = {"global_round_offset", "per_segment"}
SEED_ROLES = ("validation", "holdout")
# The only repository directories a running job imports from.  See copy_runtime.
RUNTIME_ROOT_DIRECTORIES = ("assets", "agent_code")
_RUNTIME_IGNORED = shutil.ignore_patterns("__pycache__", "*.pyc", "tests", "artifacts", "logs")
# The declarative vocabulary a config may use.  It is deliberately wider than
# ``IMPLEMENTED_ROUTES``: a value here parses, but only a registered route runs.
# ``dqn`` is absent on purpose -- classic DQN is ``q_learning`` plus a declared
# ``replay`` block, so having both would let one setup be spelled two ways.
DECLARATIVE_ROUTE_VALUES = {
    "model": {"linear_q", "mlp_q", "cnn_q", "cnn_mlp_q", "dueling_cnn_mlp_q"},
    "algorithm": {"q_learning", "sarsa", "double_dqn"},
    "state_representation": {
        "handcrafted_v1", "board_channels_v1", "board_channels_global_v1", "board_egocentric_v1",
    },
}


_REPLAY_SETTINGS = {"capacity", "batch_size", "min_size", "train_every", "target_update_every", "augmentation"}


class ConfigError(ValueError):
    """An experiment config is malformed or requests an unimplemented route."""


def _parse_replay(value: Any) -> dict[str, Any] | None:
    """Validate an ``agent.replay`` block without duplicating the agent's rules.

    Only the key set and the value types are checked here; the numeric
    invariants live in ``research_agent.config.ReplayConfig`` and are enforced
    when ``resolved_runtime_config`` instantiates it, so there is one authority.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ConfigError("agent.replay must be null or an object of replay settings")
    unknown = sorted(set(value) - _REPLAY_SETTINGS)
    if unknown:
        raise ConfigError(f"Unknown agent.replay settings: {', '.join(unknown)}")
    parsed: dict[str, Any] = {}
    for key, setting in value.items():
        if key == "augmentation":
            parsed[key] = safe_identifier(setting, "agent.replay.augmentation")
        else:
            try:
                parsed[key] = int(setting)
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"agent.replay.{key} must be an integer") from exc
    return parsed


def _parse_shaping(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or "name" not in value:
        raise ConfigError("shaping must be null or an object naming the shaping function")
    return dict(value)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or not SAFE_ID.fullmatch(value):
        raise ConfigError(f"{label} must contain only letters, digits, '-' and '_': {value!r}")
    return value


@dataclass(frozen=True)
class Budget:
    rounds: int
    checkpoint_every: int

    @classmethod
    def parse(cls, value: dict[str, Any], label: str) -> "Budget":
        try:
            parsed = cls(rounds=int(value["rounds"]), checkpoint_every=int(value["checkpoint_every"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(f"{label} must contain integer rounds and checkpoint_every") from exc
        if parsed.rounds < 1 or parsed.checkpoint_every < 1:
            raise ConfigError(f"{label}.rounds and {label}.checkpoint_every must be positive")
        return parsed


@dataclass(frozen=True)
class Phase:
    scenario: str
    opponents: tuple[str, ...]
    seeds: tuple[int, ...]
    budget: Budget

    @classmethod
    def parse(cls, value: dict[str, Any], label: str) -> "Phase":
        try:
            scenario = value["scenario"]
            opponents = tuple(value["opponents"])
            seeds = tuple(int(seed) for seed in value["seeds"])
            budget = Budget.parse(value["budget"], f"{label}.budget")
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(f"{label} must contain scenario, opponents, seeds, and budget") from exc
        if scenario not in SCENARIOS:
            raise ConfigError(f"{label}.scenario must be one of {sorted(SCENARIOS)}")
        if len(opponents) > 3 or any(not isinstance(name, str) or not name for name in opponents):
            raise ConfigError(f"{label}.opponents must contain at most three non-empty agent names")
        if not seeds or len(set(seeds)) != len(seeds):
            raise ConfigError(f"{label}.seeds must be a non-empty list of distinct integers")
        return cls(scenario, opponents, seeds, budget)


@dataclass(frozen=True)
class CurriculumSegment:
    scenario: str
    rounds: int


@dataclass(frozen=True)
class Curriculum:
    """A warm-started, sequential set of training scenarios for one model."""

    source_run_id: str
    segments: tuple[CurriculumSegment, ...]
    # How the exploration schedule is indexed across segments.  Declared, never
    # inferred: each segment is its own process with its own round counter.
    anneal_mode: str = "global_round_offset"

    def segment_round_offset(self, segment_index: int) -> int:
        """Rounds completed before a 1-based segment, under the declared mode."""
        if self.anneal_mode == "per_segment":
            return 0
        return sum(segment.rounds for segment in self.segments[: segment_index - 1])

    def segment_schedule_rounds(self, segment_index: int, training: Phase) -> int:
        """The schedule denominator a 1-based segment must anneal against."""
        if self.anneal_mode == "per_segment":
            return self.segments[segment_index - 1].rounds
        return training.budget.rounds

    @classmethod
    def parse(cls, value: dict[str, Any], training: Phase) -> "Curriculum":
        try:
            source_run_id = safe_identifier(value["source_run_id"], "curriculum.source_run_id")
            raw_segments = value["segments"]
        except (KeyError, TypeError) as exc:
            raise ConfigError("curriculum must contain source_run_id and segments") from exc
        anneal_mode = value.get("anneal_mode", "global_round_offset")
        if anneal_mode not in CURRICULUM_ANNEAL_MODES:
            raise ConfigError(
                f"curriculum.anneal_mode must be one of {sorted(CURRICULUM_ANNEAL_MODES)}, got {anneal_mode!r}"
            )
        if not isinstance(raw_segments, list) or not raw_segments:
            raise ConfigError("curriculum.segments must be a non-empty list")
        segments: list[CurriculumSegment] = []
        for index, segment in enumerate(raw_segments, start=1):
            try:
                scenario = segment["scenario"]
                rounds = int(segment["rounds"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ConfigError(f"curriculum.segments[{index}] must contain scenario and positive rounds") from exc
            if scenario not in SCENARIOS or rounds < 1:
                raise ConfigError(f"curriculum.segments[{index}] has an invalid scenario or rounds value")
            if rounds % training.budget.checkpoint_every:
                raise ConfigError(
                    "Each curriculum segment must end on checkpoint_every so continuation checkpoints are explicit."
                )
            segments.append(CurriculumSegment(scenario, rounds))
        if sum(segment.rounds for segment in segments) != training.budget.rounds:
            raise ConfigError("curriculum segment rounds must sum exactly to training.budget.rounds")
        return cls(source_run_id, tuple(segments), anneal_mode)


@dataclass(frozen=True)
class CheckpointEvaluation:
    """Which saved checkpoints an evaluation suite runs against, and why.

    ``latest`` reproduces the historical behaviour exactly: only each training
    seed's final ``latest_model.npz`` is evaluated.  It stays the default so
    older runs remain comparable and so a protocol change is never silent.

    ``all`` additionally evaluates every periodic checkpoint, which is what
    turns three final points into a learning curve.  Checkpoints are addressed
    by round number, not by file name, because the file name also encodes an
    update count that is unknown until the training job has run.

    ``holdout_seeds`` are never used to choose a checkpoint.  Selection happens
    on ``validation_seeds`` only; the holdout numbers are what a report quotes.
    """

    mode: str = "latest"
    validation_seeds: tuple[int, ...] = ()
    holdout_seeds: tuple[int, ...] = ()

    @classmethod
    def parse(cls, value: Any, label: str, evaluation: "Phase") -> "CheckpointEvaluation":
        if value is None:
            return cls(mode="latest", validation_seeds=evaluation.seeds, holdout_seeds=())
        if not isinstance(value, dict):
            raise ConfigError(f"{label} must be an object")
        mode = value.get("mode", "latest")
        if mode not in CHECKPOINT_MODES:
            raise ConfigError(f"{label}.mode must be one of {sorted(CHECKPOINT_MODES)}")

        def seed_list(key: str, default: tuple[int, ...]) -> tuple[int, ...]:
            if key not in value:
                return default
            try:
                seeds = tuple(int(seed) for seed in value[key])
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"{label}.{key} must be a list of integers") from exc
            if len(set(seeds)) != len(seeds):
                raise ConfigError(f"{label}.{key} must not repeat a seed")
            return seeds

        validation = seed_list("validation_seeds", evaluation.seeds)
        holdout = seed_list("holdout_seeds", ())
        if not validation:
            raise ConfigError(f"{label}.validation_seeds must not be empty")
        overlap = sorted(set(validation) & set(holdout))
        if overlap:
            raise ConfigError(
                f"{label}: holdout seeds must not also select checkpoints; overlapping seeds {overlap}"
            )
        return cls(mode=mode, validation_seeds=validation, holdout_seeds=holdout)

    def checkpoint_rounds(self, training: "Phase") -> tuple[int | None, ...]:
        """Return the checkpoint rounds to evaluate; ``None`` means latest."""
        if self.mode == "latest":
            return (None,)
        every = training.budget.checkpoint_every
        rounds = tuple(range(every, training.budget.rounds + 1, every))
        if not rounds:
            raise ConfigError("checkpoint_evaluation.mode 'all' needs checkpoint_every <= rounds")
        # The final periodic checkpoint and ``latest_model.npz`` are the same
        # model when rounds is a multiple of checkpoint_every, so ``None`` is
        # only added when the run would otherwise never evaluate the final one.
        return rounds if training.budget.rounds % every == 0 else rounds + (None,)


@dataclass(frozen=True)
class EvaluationSuite:
    name: str
    phase: Phase
    checkpoints: CheckpointEvaluation | None = None


@dataclass(frozen=True)
class InitialModel:
    """A checkpoint every training seed starts from, instead of fresh weights.

    This is how the M4 behaviour-cloning warm start enters a run (docs/05
    section 5.4).  It is deliberately not the curriculum mechanism: a curriculum
    warm-starts each seed from *its own* earlier run, whereas this starts every
    seed from one shared, externally produced file.

    ``sha256`` is not optional decoration.  A warm start moves the result's
    origin outside the run directory, so without a recorded digest a run could
    never prove which weights it actually began from.
    """

    kind: str
    path: str
    sha256: str

    @classmethod
    def parse(cls, value: dict[str, Any] | None) -> "InitialModel | None":
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ConfigError("initial_model must be null or an object")
        unknown = sorted(set(value) - {"kind", "path", "sha256"})
        if unknown:
            raise ConfigError(f"Unknown initial_model keys: {', '.join(unknown)}")
        try:
            parsed = cls(kind=str(value["kind"]), path=str(value["path"]), sha256=str(value["sha256"]))
        except KeyError as exc:
            raise ConfigError("initial_model requires kind, path and sha256") from exc
        if parsed.kind not in INITIAL_MODEL_KINDS:
            raise ConfigError(f"initial_model.kind must be one of {sorted(INITIAL_MODEL_KINDS)}, got {parsed.kind!r}")
        if Path(parsed.path).is_absolute() or ".." in Path(parsed.path).parts:
            raise ConfigError("initial_model.path must be a repository-relative path without '..'")
        if len(parsed.sha256) != 64 or not all(character in "0123456789abcdef" for character in parsed.sha256):
            raise ConfigError("initial_model.sha256 must be a lowercase hex SHA-256 digest")
        return parsed

    def resolve(self) -> Path:
        """Return the checkpoint, verifying that it is the declared one."""
        resolved = (ROOT / self.path).resolve()
        if not resolved.is_file():
            raise ConfigError(
                f"initial_model.path does not exist: {resolved}. "
                "Produce it with scripts/pretrain_behaviour_cloning.py before preparing this run."
            )
        digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
        if digest != self.sha256:
            raise ConfigError(
                f"initial_model {self.path} has digest {digest}, but the config declares {self.sha256}. "
                "The warm-start weights are not the ones this experiment was written against."
            )
        return resolved


@dataclass(frozen=True)
class Experiment:
    schema_version: int
    experiment_id: str
    route: str
    agent_name: str
    model: str
    algorithm: str
    reward_version: str
    exploration_version: str
    state_representation: str
    training: Phase
    evaluation: Phase
    promotion_primary_metric: str
    curriculum: Curriculum | None = None
    evaluation_suites: tuple[EvaluationSuite, ...] = ()
    checkpoint_evaluation: CheckpointEvaluation = CheckpointEvaluation()
    design_note: str = ""
    predeclared_design_numbers: dict[str, Any] = field(default_factory=dict)
    # A surviving agent's last step is a time-limit truncation, not a terminal
    # state.  Declared here so the choice is snapshotted and ablatable.
    terminal_on_truncation: bool = False
    # Bootstrap length of the TD target.  1 is the historical one-step case.
    n_step: int = 1
    # None means fully online updating; an object switches on experience replay
    # and a target network.  See agent_code.research_agent.config.ReplayConfig.
    replay: dict[str, Any] | None = None
    # Informational echo of the shaping the reward version switches on.  It is
    # validated against the derived specification rather than trusted, so a
    # config can never declare one shaping and train with another.
    shaping: dict[str, Any] | None = None
    # Weights every training seed starts from; None means fresh initialization.
    initial_model: InitialModel | None = None

    def suite_checkpoints(self, suite: str) -> CheckpointEvaluation:
        """Return the checkpoint policy for one suite name.

        Diagnostic suites default to ``latest`` even when the primary suite
        sweeps every checkpoint: a transfer diagnostic answers "did the final
        model keep this ability", not "how did it get there".
        """
        if suite == "primary":
            return self.checkpoint_evaluation
        for declared in self.evaluation_suites:
            if declared.name == suite:
                if declared.checkpoints is not None:
                    return declared.checkpoints
                return CheckpointEvaluation(mode="latest", validation_seeds=declared.phase.seeds)
        raise ConfigError(f"Unknown evaluation suite {suite!r}")

    @classmethod
    def load(cls, path: Path) -> "Experiment":
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Invalid JSON in {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError("The experiment config must be a JSON object")
        required = {"schema_version", "experiment_id", "route", "agent", "reward_version", "training", "evaluation", "promotion"}
        missing = sorted(required - raw.keys())
        if missing:
            raise ConfigError(f"Missing config keys: {', '.join(missing)}")
        agent = raw["agent"]
        try:
            training = Phase.parse(raw["training"], "training")
            evaluation = Phase.parse(raw["evaluation"], "evaluation")
            raw_suites = raw.get("evaluation_suites", {})
            if not isinstance(raw_suites, dict):
                raise ConfigError("evaluation_suites must be an object keyed by suite name")
            suites = []
            for name, phase in raw_suites.items():
                suite_name = safe_identifier(name, "evaluation_suites name")
                suite_phase = Phase.parse(phase, f"evaluation_suites.{suite_name}")
                raw_checkpoints = phase.get("checkpoint_evaluation") if isinstance(phase, dict) else None
                suites.append(EvaluationSuite(
                    suite_name,
                    suite_phase,
                    CheckpointEvaluation.parse(
                        raw_checkpoints, f"evaluation_suites.{suite_name}.checkpoint_evaluation", suite_phase
                    ) if raw_checkpoints is not None else None,
                ))
            suites = tuple(suites)
            if any(suite.name == "primary" for suite in suites):
                raise ConfigError("evaluation_suites may not redefine the reserved primary suite")
            design_note = raw.get("_design_note", "")
            predeclared_design_numbers = raw.get("_predeclared_design_numbers", {})
            if not isinstance(design_note, str):
                raise ConfigError("_design_note must be a string")
            if not isinstance(predeclared_design_numbers, dict):
                raise ConfigError("_predeclared_design_numbers must be an object")
            experiment = cls(
                schema_version=int(raw["schema_version"]),
                experiment_id=safe_identifier(raw["experiment_id"], "experiment_id"),
                route=safe_identifier(raw["route"], "route"),
                agent_name=safe_identifier(agent["name"], "agent.name"),
                model=agent["model"],
                algorithm=agent["algorithm"],
                reward_version=safe_identifier(raw["reward_version"], "reward_version"),
                exploration_version=safe_identifier(raw.get("exploration_version", "E00"), "exploration_version"),
                state_representation=agent["state_representation"],
                training=training,
                evaluation=evaluation,
                promotion_primary_metric=raw["promotion"]["primary_metric"],
                curriculum=Curriculum.parse(raw["curriculum"], training) if "curriculum" in raw else None,
                evaluation_suites=suites,
                checkpoint_evaluation=CheckpointEvaluation.parse(
                    raw.get("checkpoint_evaluation"), "checkpoint_evaluation", evaluation
                ),
                design_note=design_note,
                predeclared_design_numbers=predeclared_design_numbers,
                terminal_on_truncation=bool(raw.get("terminal_on_truncation", False)),
                initial_model=InitialModel.parse(raw.get("initial_model")),
                n_step=int(agent.get("n_step", 1)),
                replay=_parse_replay(agent.get("replay")),
                shaping=_parse_shaping(raw.get("shaping")),
            )
        except (KeyError, TypeError) as exc:
            raise ConfigError("agent and promotion sections have invalid fields") from exc
        if experiment.schema_version != 1:
            raise ConfigError("Only schema_version 1 is supported")
        suite_names = [suite.name for suite in experiment.evaluation_suites]
        if len(suite_names) != len(set(suite_names)):
            raise ConfigError("evaluation_suites names must be distinct")
        for name, allowed in DECLARATIVE_ROUTE_VALUES.items():
            if getattr(experiment, name) not in allowed:
                raise ConfigError(f"agent.{name} is not a declared R01-R07 value")
        if experiment.promotion_primary_metric != "score":
            raise ConfigError("promotion.primary_metric is currently fixed to 'score'")
        if experiment.n_step < 1:
            raise ConfigError("agent.n_step must be a positive integer")
        if experiment.algorithm == "double_dqn" and experiment.replay is None:
            raise ConfigError("agent.algorithm 'double_dqn' requires an agent.replay block")
        return experiment

    @property
    def lines(self) -> tuple[str, ...]:
        """The main lines of docs/05 this experiment's route belongs to."""
        return tuple(IMPLEMENTED_ROUTES.get(self.route, {}).get("lines", ()))

    def require_implemented(self) -> None:
        """Fail closed on anything the agent code does not actually implement."""
        route = IMPLEMENTED_ROUTES.get(self.route)
        declared = (self.model, self.algorithm, self.state_representation)
        if route is None:
            raise ConfigError(
                f"Route {self.route!r} is not implemented. Implemented routes: {sorted(IMPLEMENTED_ROUTES)}."
            )
        if route["declaration"] != declared:
            raise ConfigError(
                f"Route {self.route} is implemented as {route['declaration']}, but this config declares {declared}."
            )
        if self.reward_version not in route["reward_versions"]:
            raise ConfigError(
                f"Route {self.route} does not implement reward version {self.reward_version!r}; "
                f"implemented: {sorted(route['reward_versions'])}."
            )
        if self.exploration_version not in route["exploration_versions"]:
            raise ConfigError(
                f"Route {self.route} does not implement exploration version {self.exploration_version!r}; "
                f"implemented: {sorted(route['exploration_versions'])}."
            )

    def snapshot(self) -> dict[str, Any]:
        """Return the canonical, reloadable config snapshot stored with a run."""
        snapshot = {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "route": self.route,
            "agent": {
                "name": self.agent_name,
                "model": self.model,
                "algorithm": self.algorithm,
                "state_representation": self.state_representation,
                "n_step": self.n_step,
                "replay": self.replay,
            },
            "main_lines": list(self.lines),
            "reward_version": self.reward_version,
            "exploration_version": self.exploration_version,
            "training": asdict(self.training),
            "evaluation": asdict(self.evaluation),
            "checkpoint_evaluation": asdict(self.checkpoint_evaluation),
            "terminal_on_truncation": self.terminal_on_truncation,
            "shaping": self.shaping,
            "promotion": {"primary_metric": self.promotion_primary_metric},
        }
        if self.design_note:
            snapshot["_design_note"] = self.design_note
        if self.predeclared_design_numbers:
            snapshot["_predeclared_design_numbers"] = self.predeclared_design_numbers
        if self.initial_model is not None:
            snapshot["initial_model"] = asdict(self.initial_model)
        if self.curriculum is not None:
            snapshot["curriculum"] = {
                "source_run_id": self.curriculum.source_run_id,
                "anneal_mode": self.curriculum.anneal_mode,
                "segments": [asdict(segment) for segment in self.curriculum.segments],
            }
        if self.evaluation_suites:
            snapshot["evaluation_suites"] = {}
            for suite in self.evaluation_suites:
                entry = asdict(suite.phase)
                if suite.checkpoints is not None:
                    # Nested inside the phase so the snapshot reloads through
                    # exactly the same parser that read the external config.
                    entry["checkpoint_evaluation"] = asdict(suite.checkpoints)
                snapshot["evaluation_suites"][suite.name] = entry
        return snapshot


def git_provenance() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    try:
        commit = git("rev-parse", "HEAD")
        status = git("status", "--porcelain")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("A git checkout is required for experiment provenance") from exc
    return {"git_commit": commit, "worktree_dirty": bool(status), "prepared_at_utc": utc_now()}


def verify_job_provenance(run_dir: Path) -> None:
    """Reject a worker whose source checkout differs from the prepared run."""
    try:
        expected = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Run provenance is unavailable or invalid: {run_dir / 'provenance.json'}") from exc
    current = git_provenance()
    if current["git_commit"] != expected.get("git_commit"):
        raise RuntimeError(
            "Worker checkout does not match the prepared experiment: "
            f"expected {expected.get('git_commit')}, found {current['git_commit']}."
        )
    if not expected.get("worktree_dirty", False) and current["worktree_dirty"]:
        raise RuntimeError(
            "Prepared experiment requires a clean checkout, but this worker has uncommitted changes."
        )


def resolved_runtime_config(experiment: Experiment) -> dict[str, Any]:
    """Freeze the full runtime configuration alongside the external JSON plan.

    The current runner supports only ``research_agent``/R01.  Importing its
    dependency-free config here records the values actually consumed by the
    callback runtime (including values not duplicated in experiment JSON).
    """
    if experiment.agent_name != "research_agent":
        raise ConfigError(f"No runtime config resolver is registered for {experiment.agent_name!r}")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from agent_code.research_agent.config import (
        ACTIONS,
        EXPERIMENTS,
        EXPLORATION_VERSIONS,
        REWARD_VERSIONS,
        ReplayConfig,
        exploration_specification,
        shaping_specification,
        validate_config,
    )
    from agent_code.research_agent.runtime.experiment import reward_specification
    from agent_code.research_agent.state import state_dimension

    try:
        config = EXPERIMENTS[experiment.route]
    except KeyError as exc:
        raise ConfigError(f"No runtime config is registered for route {experiment.route!r}") from exc
    if experiment.reward_version not in REWARD_VERSIONS:
        raise ConfigError(f"No runtime config is registered for reward version {experiment.reward_version!r}")
    if experiment.exploration_version not in EXPLORATION_VERSIONS:
        raise ConfigError(f"No runtime config is registered for exploration version {experiment.exploration_version!r}")
    try:
        config = validate_config(replace(
            config,
            reward_version=experiment.reward_version,
            exploration_version=experiment.exploration_version,
            terminal_on_truncation=experiment.terminal_on_truncation,
            n_step=experiment.n_step,
            # An absent ``agent.replay`` means no replay, never "whatever the
            # route happens to default to".  A route default exists only for a
            # manual invocation through environment variables; a config file
            # that wants a buffer has to say so, and ``Experiment.load``
            # enforces that for the algorithms which cannot run without one.
            replay=ReplayConfig.parse(experiment.replay),
        ))
    except ValueError as exc:
        # The agent package owns these invariants; surfacing them as a config
        # error keeps the runner's failure mode uniform.
        raise ConfigError(str(exc)) from exc
    derived_shaping = shaping_specification(experiment.reward_version)
    if experiment.shaping is not None and experiment.shaping.get("name") != (derived_shaping or {}).get("name"):
        raise ConfigError(
            f"shaping declares {experiment.shaping.get('name')!r} but reward version "
            f"{experiment.reward_version} implies {(derived_shaping or {}).get('name')!r}."
        )
    declared = (
        experiment.model,
        experiment.algorithm,
        experiment.state_representation,
        experiment.reward_version,
        experiment.exploration_version,
    )
    resolved = (
        config.network,
        config.algorithm,
        config.state_encoder,
        config.reward_version,
        config.exploration_version,
    )
    if declared != resolved:
        raise ConfigError(
            f"Experiment declaration {declared} does not match runtime configuration {resolved}."
        )
    return {
        "actions": list(ACTIONS),
        "main_lines": list(config.lines),
        "feature_dimension": state_dimension(config.state_encoder),
        "config": asdict(config),
        # Freezing the exact weights, not just the version label, means a run
        # directory stays interpretable even if a later commit edits the table.
        "reward_specification": reward_specification(experiment.reward_version),
        "exploration_specification": exploration_specification(experiment.exploration_version),
        "shaping_specification": derived_shaping,
    }


def copy_runtime(destination: Path) -> None:
    """Make a private framework copy so its fixed logger paths cannot collide.

    This is an allowlist, deliberately.  It used to be a deny-list, which meant
    every job silently copied whatever new thing appeared in the repository
    root.  A single ``.venv`` there cost 154 MiB per job and 95 GiB across one
    three-arm study, while the artifact those jobs exist to produce -- the
    trained linear model -- is 3.5 KiB.  A deny-list cannot be kept correct by
    review; an allowlist fails closed.

    The framework needs only the top-level modules it imports, ``assets`` (image
    files are loaded at import time), and ``agent_code``.  Nothing under
    ``scripts``, ``docs``, ``experiments`` or a virtualenv is ever imported by a
    running job.
    """
    destination.mkdir(parents=True, exist_ok=True)
    # Every top-level module, so a new framework file is picked up automatically
    # without anyone having to remember to extend this list.
    for module in sorted(ROOT.glob("*.py")):
        shutil.copy2(module, destination / module.name)
    for name in RUNTIME_ROOT_DIRECTORIES:
        source = ROOT / name
        if source.is_dir():
            shutil.copytree(source, destination / name, ignore=_RUNTIME_IGNORED)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
