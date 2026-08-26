"""Tests for the behaviour-cloning warm start and demonstration collection.

A warm start moves part of a result's origin outside the run directory, so the
tests here are mostly about provenance holding up: the declared checkpoint has
to exist, has to be the declared one, and has to reach every training job.
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import experiment_lib  # noqa: E402
from collect_demonstrations import collect_from_world  # noqa: E402
from experiment_lib import ConfigError, Experiment, write_json  # noqa: E402
from run_experiment import build_jobs, materialize_initial_model  # noqa: E402

from agent_code.research_agent.config import ACTIONS  # noqa: E402


def base_config() -> dict:
    return {
        "schema_version": 1,
        "experiment_id": "test_warm_start",
        "route": "R01",
        "agent": {"name": "research_agent", "model": "linear_q", "algorithm": "q_learning",
                  "state_representation": "handcrafted_v1"},
        "reward_version": "A00",
        "training": {"scenario": "coin-heaven", "opponents": [], "seeds": [11, 12],
                     "budget": {"rounds": 2, "checkpoint_every": 1}},
        "evaluation": {"scenario": "classic", "opponents": [], "seeds": [21],
                       "budget": {"rounds": 2, "checkpoint_every": 1}},
        "promotion": {"primary_metric": "score"},
    }


class StubAgent:
    def __init__(self, name: str):
        self.name = name
        self.last_game_state = None
        self.last_action = None


class StubWorld:
    """The minimum of the framework's world contract the collector relies on."""

    def __init__(self, actions: list[str | None], game_state: dict):
        self.agents = [StubAgent("rule_based_agent")]
        self._actions = list(actions)
        self._game_state = game_state
        self.running = False
        self.rounds_started = 0

    def new_round(self):
        self.rounds_started += 1
        self.running = True

    def do_step(self):
        agent = self.agents[0]
        agent.last_game_state = self._game_state
        agent.last_action = self._actions.pop(0) if self._actions else None
        if not self._actions:
            self.running = False

    def end_round(self):
        self.running = False

    def end(self):
        self.running = False


def game_state() -> dict:
    field = np.zeros((17, 17), dtype=int)
    field[[0, -1], :] = -1
    field[:, [0, -1]] = -1
    return {
        "round": 1, "step": 1, "field": field, "self": ("rule_based_agent", 0, True, (1, 1)),
        "others": [], "bombs": [], "coins": [(3, 3)],
        "explosion_map": np.zeros_like(field), "user_input": None,
    }


class DemonstrationCollectionTest(unittest.TestCase):
    def test_only_real_six_action_choices_become_demonstrations(self):
        # ``None`` before the first act and ``"ERROR"`` for a silenced exception
        # are the framework's substitutions, not the demonstrator's decisions.
        world = StubWorld([None, "UP", "ERROR", "BOMB"], game_state())
        states, actions, summary = collect_from_world(
            world, encoder="board_egocentric_v1", rounds=1, max_states=100, agent_name="rule_based_agent"
        )
        self.assertEqual(len(states), 2)
        self.assertEqual(list(actions), [ACTIONS.index("UP"), ACTIONS.index("BOMB")])
        self.assertEqual(summary["skipped_unusable_actions"], 2)

    def test_states_are_stored_in_the_declared_half_precision_encoding(self):
        world = StubWorld(["UP", "DOWN"], game_state())
        states, _, _ = collect_from_world(
            world, encoder="board_egocentric_v1", rounds=1, max_states=100, agent_name="rule_based_agent"
        )
        self.assertEqual(states.dtype, np.float16)

    def test_the_cap_stops_collection_rather_than_the_round_count(self):
        world = StubWorld(["UP"] * 50, game_state())
        states, _, _ = collect_from_world(
            world, encoder="board_egocentric_v1", rounds=10, max_states=3, agent_name="rule_based_agent"
        )
        self.assertEqual(len(states), 3)
        self.assertEqual(world.rounds_started, 1)

    def test_a_demonstrator_that_never_acted_is_an_error_not_an_empty_dataset(self):
        world = StubWorld([None, None], game_state())
        with self.assertRaises(RuntimeError):
            collect_from_world(world, encoder="board_egocentric_v1", rounds=1, max_states=10,
                               agent_name="rule_based_agent")


class InitialModelTest(unittest.TestCase):
    def _checkpoint(self, root: Path) -> tuple[Path, str]:
        path = root / "pretrained.npz"
        np.savez(path, weights=np.zeros((len(ACTIONS), 44), dtype=np.float32))
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def _config_with_initial_model(self, root: Path, *, digest: str, relative: str) -> Path:
        payload = base_config()
        payload["initial_model"] = {"kind": "behaviour_cloning", "path": relative, "sha256": digest}
        path = root / "config.json"
        write_json(path, payload)
        return path

    def test_a_declared_warm_start_reaches_every_training_job(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, digest = self._checkpoint(root)
            with patch.object(experiment_lib, "ROOT", root):
                experiment = Experiment.load(self._config_with_initial_model(root, digest=digest, relative=checkpoint.name))
                run_dir = root / "run"
                run_dir.mkdir()
                records = materialize_initial_model(experiment, run_dir)
            self.assertEqual([record["train_seed"] for record in records], [11, 12])
            for seed in (11, 12):
                self.assertTrue((run_dir / "inputs" / "initial_models" / f"train_seed{seed}.npz").is_file())
            training_jobs = [job for job in build_jobs(experiment, run_dir) if job["mode"] == "train"]
            self.assertEqual(len(training_jobs), 2)
            for job in training_jobs:
                self.assertEqual(job["initial_model"]["kind"], "behaviour_cloning")
                self.assertTrue(job["input_model_relpath"].endswith(".npz"))

    def test_a_checkpoint_that_is_not_the_declared_one_stops_the_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, _ = self._checkpoint(root)
            wrong = "0" * 64
            with patch.object(experiment_lib, "ROOT", root):
                experiment = Experiment.load(self._config_with_initial_model(root, digest=wrong, relative=checkpoint.name))
                # Silently training from different weights than the config
                # declares would make the result untraceable, so this must fail.
                with self.assertRaises(ConfigError):
                    experiment.initial_model.resolve()

    def test_a_missing_checkpoint_names_the_script_that_produces_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(experiment_lib, "ROOT", root):
                experiment = Experiment.load(
                    self._config_with_initial_model(root, digest="a" * 64, relative="absent.npz")
                )
                with self.assertRaises(ConfigError) as raised:
                    experiment.initial_model.resolve()
            self.assertIn("pretrain_behaviour_cloning", str(raised.exception))

    def test_an_escaping_or_absolute_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in ("../outside.npz", "/etc/passwd"):
                with self.subTest(path=relative), self.assertRaises(ConfigError):
                    Experiment.load(self._config_with_initial_model(root, digest="a" * 64, relative=relative))

    def test_an_unimplemented_warm_start_kind_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = base_config()
            payload["initial_model"] = {"kind": "distillation", "path": "x.npz", "sha256": "a" * 64}
            path = root / "config.json"
            write_json(path, payload)
            with self.assertRaises(ConfigError):
                Experiment.load(path)

    def test_the_warm_start_survives_a_snapshot_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, digest = self._checkpoint(root)
            with patch.object(experiment_lib, "ROOT", root):
                experiment = Experiment.load(self._config_with_initial_model(root, digest=digest, relative=checkpoint.name))
                snapshot = root / "snapshot.json"
                write_json(snapshot, experiment.snapshot())
                self.assertEqual(Experiment.load(snapshot), experiment)


if __name__ == "__main__":
    unittest.main()
