"""Tests for the frozen ExperimentRuntime seam."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from agent_code.research_agent.config import (
    EXPLORATION_VERSIONS,
    REWARD_VERSIONS,
    active_config,
    epsilon_for_training_round,
    exploration_specification,
)
from agent_code.research_agent.learners import OnlineQLearner
from agent_code.research_agent.models import LinearQModel, build_model
from agent_code.research_agent.runtime import ExperimentRuntime
from agent_code.research_agent.runtime.experiment import (
    DEATH_PENALTIES,
    REWARD_TABLES,
    reward_for_events,
    reward_specification,
)


def game_state() -> dict:
    field = np.zeros((9, 9), dtype=int)
    field[[0, -1], :] = -1
    field[:, [0, -1]] = -1
    return {
        "round": 1,
        "step": 1,
        "field": field,
        "self": ("research_agent", 0, True, (3, 3)),
        "others": [],
        "bombs": [],
        "coins": [(4, 3)],
        "explosion_map": np.zeros_like(field),
        "user_input": None,
    }


class ExperimentRuntimeTest(unittest.TestCase):
    def test_route_selection_is_runtime_configuration_not_callback_code(self):
        with patch.dict(os.environ, {"BOMBERMAN_EXPERIMENT": "R01"}, clear=False):
            self.assertEqual(active_config().name, "R01")
        with patch.dict(os.environ, {"BOMBERMAN_EXPERIMENT": "R99"}, clear=False):
            with self.assertRaises(ValueError):
                active_config()

    def test_e01_schedule_is_predeclared_and_evaluation_stays_greedy(self):
        config = replace(active_config(), exploration_version="E01")
        self.assertEqual(EXPLORATION_VERSIONS, frozenset({"E00", "E01"}))
        self.assertEqual(exploration_specification("E01")["hold_fraction"], 0.20)
        # 500 rounds: 1--100 at 0.30, then decay to exactly 0.05 at round 500.
        self.assertEqual(epsilon_for_training_round(config, 1, 500), 0.30)
        self.assertEqual(epsilon_for_training_round(config, 100, 500), 0.30)
        self.assertAlmostEqual(epsilon_for_training_round(config, 101, 500), 0.299375)
        self.assertEqual(epsilon_for_training_round(config, 500, 500), 0.05)
        with self.assertRaises(ValueError):
            epsilon_for_training_round(config, 501, 500)

        with tempfile.TemporaryDirectory() as temporary:
            environment = {
                "BOMBERMAN_ARTIFACT_DIR": str(Path(temporary) / "agent"),
                "BOMBERMAN_RUN_ID": "e01_eval_test",
                "BOMBERMAN_TRAINING_ROUNDS": "500",
            }
            with patch.dict(os.environ, environment, clear=False):
                runtime = ExperimentRuntime(config, train=False, agent_seed=3, logger=Mock())
                self.assertEqual(runtime._epsilon_for_game_state(game_state()), 0.0)

    def test_versioned_rewards_keep_one_death_penalty_and_a02_auxiliary_events(self):
        events = ["COIN_COLLECTED", "KILLED_SELF", "GOT_KILLED", "CRATE_DESTROYED", "COIN_FOUND"]
        self.assertEqual(reward_for_events("A00", events), 1.0)
        self.assertEqual(reward_for_events("A01", events), -4.0)
        self.assertAlmostEqual(reward_for_events("A02", events), -3.7)
        # A00-A02 are published baselines; these numbers may never drift.
        self.assertEqual(DEATH_PENALTIES["A00"], 0.0)
        self.assertEqual(DEATH_PENALTIES["A01"], -5.0)
        self.assertEqual(DEATH_PENALTIES["A02"], -5.0)
        with patch.dict(os.environ, {"BOMBERMAN_EXPERIMENT": "R01", "BOMBERMAN_REWARD_VERSION": "A02"}, clear=False):
            self.assertEqual(active_config().reward_version, "A02")

    def test_dose_response_arms_change_only_the_death_penalty(self):
        """A02, A03 and A05 must differ in exactly one number."""
        self.assertEqual(REWARD_TABLES["A03"], REWARD_TABLES["A02"])
        self.assertEqual(REWARD_TABLES["A05"], REWARD_TABLES["A02"])
        self.assertEqual(DEATH_PENALTIES["A03"], -1.0)
        self.assertEqual(DEATH_PENALTIES["A05"], 0.0)

        survived = ["COIN_COLLECTED", "CRATE_DESTROYED", "COIN_FOUND"]
        for version in ("A02", "A03", "A05"):
            with self.subTest(version=version):
                self.assertAlmostEqual(reward_for_events(version, survived), 1.3)

        died = ["CRATE_DESTROYED", "KILLED_SELF", "GOT_KILLED"]
        self.assertAlmostEqual(reward_for_events("A02", died), -4.9)
        self.assertAlmostEqual(reward_for_events("A03", died), -0.9)
        # The control arm carries no risk term at all, which is the point of it.
        self.assertAlmostEqual(reward_for_events("A05", died), 0.1)

    def test_one_death_is_penalised_once_even_though_two_events_fire(self):
        for version in ("A01", "A02", "A03"):
            with self.subTest(version=version):
                both = reward_for_events(version, ["KILLED_SELF", "GOT_KILLED"])
                self.assertEqual(both, reward_for_events(version, ["KILLED_SELF"]))
                self.assertEqual(both, DEATH_PENALTIES[version])

    def test_a04_is_specified_but_not_registered(self):
        """Do not let a documented-but-unimplemented version look runnable."""
        self.assertNotIn("A04", REWARD_VERSIONS)
        with self.assertRaises(ValueError):
            reward_for_events("A04", [])

    def test_every_declared_reward_version_is_fully_registered(self):
        for version in sorted(REWARD_VERSIONS):
            with self.subTest(version=version):
                specification = reward_specification(version)
                self.assertEqual(specification["reward_version"], version)
                self.assertEqual(specification["event_weights"], REWARD_TABLES[version])
                self.assertEqual(specification["death_penalty"], DEATH_PENALTIES[version])
                self.assertTrue(specification["notes"])
                self.assertEqual(specification["death_penalty_applications_per_death"], 1)
                with patch.dict(os.environ, {"BOMBERMAN_REWARD_VERSION": version}, clear=False):
                    self.assertEqual(active_config().reward_version, version)

    def test_r01_behaviour_is_hidden_behind_the_shared_runtime_interface(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = Path(temporary) / "job" / "agent"
            environment = {
                "BOMBERMAN_ARTIFACT_DIR": str(artifacts),
                "BOMBERMAN_RUN_ID": "runtime_test",
                "BOMBERMAN_SCENARIO": "classic",
                "BOMBERMAN_SEED": "3",
                "BOMBERMAN_CHECKPOINT_EVERY": "1",
            }
            with patch.dict(os.environ, environment, clear=False):
                runtime = ExperimentRuntime(active_config(), train=True, agent_seed=3, logger=Mock())
                action = runtime.select_action(game_state())
                self.assertIn(action, ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"))
                self.assertIsInstance(runtime.model, LinearQModel)
                self.assertIsInstance(runtime.learner, OnlineQLearner)
                runtime.observe(game_state(), action, game_state(), [], terminal=False)
                runtime.end_round(game_state(), action, [])
            self.assertTrue((artifacts / "latest_model.npz").is_file())
            self.assertTrue(any((artifacts / "checkpoints").glob("*.npz")))
            records = [json.loads(line) for line in (artifacts / "agent.jsonl").read_text(encoding="utf-8").splitlines()]
            # Evaluation jobs receive no game events, so the action record is the
            # only place a trajectory can be reconstructed from.
            action_records = [record for record in records if record["kind"] == "action"]
            self.assertTrue(action_records)
            self.assertEqual(action_records[0]["position"], [3, 3])
            setup = next(record for record in records if record["kind"] == "agent_setup")
            self.assertEqual(setup["reward_specification"]["reward_version"], setup["reward_version"])

    def test_unimplemented_route_adapter_fails_at_the_internal_seam(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = replace(active_config(), network="mlp_q")
            with patch.dict(os.environ, {"BOMBERMAN_ARTIFACT_DIR": str(Path(temporary) / "agent"), "BOMBERMAN_RUN_ID": "runtime_test"}, clear=False):
                runtime = ExperimentRuntime(config, train=True, agent_seed=1, logger=Mock())
                with self.assertRaises(NotImplementedError):
                    runtime.select_action(game_state())

    def test_model_initialization_is_reproducible_per_agent_seed(self):
        config = active_config()
        first = build_model(config, input_dim=44, seed=7)
        same_seed = build_model(config, input_dim=44, seed=7)
        other_seed = build_model(config, input_dim=44, seed=8)
        np.testing.assert_array_equal(first.weights, same_seed.weights)
        self.assertFalse(np.array_equal(first.weights, other_seed.weights))


if __name__ == "__main__":
    unittest.main()
