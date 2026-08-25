"""Dependency-free shared machinery for reproducible Bomberman experiments."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / "runs"
SCENARIOS = {"empty", "coin-heaven", "loot-crate", "classic"}
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
SUPPORTED_DECLARATIONS = {
    "R01": ("linear_q", "q_learning", "handcrafted_v1", "A00"),
}
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
class Experiment:
    schema_version: int
    experiment_id: str
    route: str
    agent_name: str
    model: str
    algorithm: str
    reward_version: str
    state_representation: str
    training: Phase
    evaluation: Phase
    promotion_primary_metric: str

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
            experiment = cls(
                schema_version=int(raw["schema_version"]),
                experiment_id=safe_identifier(raw["experiment_id"], "experiment_id"),
                route=safe_identifier(raw["route"], "route"),
                agent_name=safe_identifier(agent["name"], "agent.name"),
                model=agent["model"],
                algorithm=agent["algorithm"],
                reward_version=safe_identifier(raw["reward_version"], "reward_version"),
                state_representation=agent["state_representation"],
                training=Phase.parse(raw["training"], "training"),
                evaluation=Phase.parse(raw["evaluation"], "evaluation"),
                promotion_primary_metric=raw["promotion"]["primary_metric"],
            )
        except (KeyError, TypeError) as exc:
            raise ConfigError("agent and promotion sections have invalid fields") from exc
        if experiment.schema_version != 1:
            raise ConfigError("Only schema_version 1 is supported")
        for name, allowed in DECLARATIVE_ROUTE_VALUES.items():
            if getattr(experiment, name) not in allowed:
                raise ConfigError(f"agent.{name} is not a declared R01-R07 value")
        if experiment.promotion_primary_metric != "score":
            raise ConfigError("promotion.primary_metric is currently fixed to 'score'")
        return experiment

    def require_implemented(self) -> None:
        expected = SUPPORTED_DECLARATIONS.get(self.route)
        declared = (self.model, self.algorithm, self.state_representation, self.reward_version)
        if expected != declared:
            raise ConfigError(
                f"{self.route} is declared for the shared infrastructure but is not implemented. "
                "Only the existing R01 linear Q-learning/A00 agent may be run."
            )

    def snapshot(self) -> dict[str, Any]:
        """Return the canonical, reloadable config snapshot stored with a run."""
        return {
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
            "training": asdict(self.training),
            "evaluation": asdict(self.evaluation),
            "promotion": {"primary_metric": self.promotion_primary_metric},
        }


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
    from agent_code.research_agent.config import ACTIONS, EXPERIMENTS, FEATURE_DIMENSION

    try:
        config = EXPERIMENTS[experiment.route]
    except KeyError as exc:
        raise ConfigError(f"No runtime config is registered for route {experiment.route!r}") from exc
    declared = (experiment.model, experiment.algorithm, experiment.state_representation, experiment.reward_version)
    resolved = (config.network, config.algorithm, config.state_encoder, config.reward_version)
    if declared != resolved:
        raise ConfigError(
            f"Experiment declaration {declared} does not match runtime configuration {resolved}."
        )
    return {
        "actions": list(ACTIONS),
        "feature_dimension": FEATURE_DIMENSION,
        "config": asdict(config),
    }


def copy_runtime(destination: Path) -> None:
    """Make a private framework copy so its fixed logger paths cannot collide."""
    ignored = shutil.ignore_patterns(
        ".git", "runs", "logs", "__pycache__", "*.pyc", "artifacts", "screenshots", "replays", "results"
    )
    shutil.copytree(ROOT, destination, ignore=ignored)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
