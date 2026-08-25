"""Tests for the frozen ExperimentRuntime seam."""

from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from agent_code.research_agent.config import active_config
from agent_code.research_agent.learners import OnlineQLearner
from agent_code.research_agent.models import LinearQModel, build_model
from agent_code.research_agent.runtime import ExperimentRuntime
from agent_code.research_agent.runtime.experiment import reward_for_events


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

    def test_versioned_rewards_keep_one_death_penalty_and_a02_auxiliary_events(self):
        events = ["COIN_COLLECTED", "KILLED_SELF", "GOT_KILLED", "CRATE_DESTROYED", "COIN_FOUND"]
        self.assertEqual(reward_for_events("A00", events), 1.0)
        self.assertEqual(reward_for_events("A01", events), -4.0)
        self.assertAlmostEqual(reward_for_events("A02", events), -3.7)
        with patch.dict(os.environ, {"BOMBERMAN_EXPERIMENT": "R01", "BOMBERMAN_REWARD_VERSION": "A02"}, clear=False):
            self.assertEqual(active_config().reward_version, "A02")

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
