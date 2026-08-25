"""Dependency-free shared machinery for reproducible Bomberman experiments."""

from __future__ import annotations

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
SUPPORTED_DECLARATIONS = {
    "R01": ("linear_q", "q_learning", "handcrafted_v1"),
}
SUPPORTED_REWARD_VERSIONS = {
    # A03 (death penalty 1.0) and A05 (death penalty 0.0) are the other two
    # levels of the D dose-response study; A02 (5.0) is the control arm.  A04
    # is specified in docs/01 but not implemented, so it is not listed here.
    "R01": {"A00", "A01", "A02", "A03", "A05"},
}
SUPPORTED_EXPLORATION_VERSIONS = {
    # E00 is the historical fixed epsilon=0.15 baseline. E01 is its only
    # currently implemented, predeclared single-variable comparison.
    "R01": {"E00", "E01"},
}
CHECKPOINT_MODES = {"latest", "all"}
# Mirrors agent_code.research_agent.config.CURRICULUM_ANNEAL_MODES.  A curriculum
# segment is a separate process with its own round counter, so the way the
# exploration schedule spans segments has to be declared rather than inferred.
CURRICULUM_ANNEAL_MODES = {"global_round_offset", "per_segment"}
SEED_ROLES = ("validation", "holdout")
# The only repository directories a running job imports from.  See copy_runtime.
RUNTIME_ROOT_DIRECTORIES = ("assets", "agent_code")
_RUNTIME_IGNORED = shutil.ignore_patterns("__pycache__", "*.pyc", "tests", "artifacts", "logs")
DECLARATIVE_ROUTE_VALUES = {
    "model": {"linear_q", "mlp_q", "cnn_q", "cnn_mlp_q", "dueling_cnn_mlp_q"},
    "algorithm": {"q_learning", "sarsa", "dqn", "double_dqn"},
    "state_representation": {"handcrafted_v1", "board_channels_v1", "board_channels_global_v1"},
}


class ConfigError(ValueError):
    """An experiment config is malformed or requests an unimplemented route."""


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
        return experiment

    def require_implemented(self) -> None:
        expected = SUPPORTED_DECLARATIONS.get(self.route)
        declared = (self.model, self.algorithm, self.state_representation)
        if (
            expected != declared
            or self.reward_version not in SUPPORTED_REWARD_VERSIONS.get(self.route, set())
            or self.exploration_version not in SUPPORTED_EXPLORATION_VERSIONS.get(self.route, set())
        ):
            raise ConfigError(
                f"{self.route} is declared for the shared infrastructure but is not implemented. "
                "Only the existing R01 linear Q-learning agent with A00, A01, A02, A03, or A05 and E00 or E01 may be run."
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
            },
            "reward_version": self.reward_version,
            "exploration_version": self.exploration_version,
            "training": asdict(self.training),
            "evaluation": asdict(self.evaluation),
            "checkpoint_evaluation": asdict(self.checkpoint_evaluation),
            "terminal_on_truncation": self.terminal_on_truncation,
            "promotion": {"primary_metric": self.promotion_primary_metric},
        }
        if self.design_note:
            snapshot["_design_note"] = self.design_note
        if self.predeclared_design_numbers:
            snapshot["_predeclared_design_numbers"] = self.predeclared_design_numbers
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
        FEATURE_DIMENSION,
        REWARD_VERSIONS,
        exploration_specification,
    )
    from agent_code.research_agent.runtime.experiment import reward_specification

    try:
        config = EXPERIMENTS[experiment.route]
    except KeyError as exc:
        raise ConfigError(f"No runtime config is registered for route {experiment.route!r}") from exc
    if experiment.reward_version not in REWARD_VERSIONS:
        raise ConfigError(f"No runtime config is registered for reward version {experiment.reward_version!r}")
    if experiment.exploration_version not in EXPLORATION_VERSIONS:
        raise ConfigError(f"No runtime config is registered for exploration version {experiment.exploration_version!r}")
    config = replace(
        config,
        reward_version=experiment.reward_version,
        exploration_version=experiment.exploration_version,
        terminal_on_truncation=experiment.terminal_on_truncation,
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
        "feature_dimension": FEATURE_DIMENSION,
        "config": asdict(config),
        # Freezing the exact weights, not just the version label, means a run
        # directory stays interpretable even if a later commit edits the table.
        "reward_specification": reward_specification(experiment.reward_version),
        "exploration_specification": exploration_specification(experiment.exploration_version),
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
