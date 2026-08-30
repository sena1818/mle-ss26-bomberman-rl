"""Tests for the frozen opponent and the two exploration mechanisms at once.

Both features exist to make an arm interpretable, so both fail closed: an
opponent that quietly became something else, or a route that quietly ran two
exploration mechanisms, would produce a number nobody could read afterwards.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
ROOT_DIR = SCRIPTS.parent
for path in (str(SCRIPTS), str(ROOT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np  # noqa: E402

from agent_code.research_agent.config import EXPERIMENTS, validate_config  # noqa: E402
from experiment_lib import ROOT, ConfigError, Experiment, FROZEN_OPPONENT_AGENT  # noqa: E402

FROZEN_MODEL = "frozen_opponents/R02_9_seed1001_round05000.npz"
SELF_PLAY_CONFIG = ROOT / "experiments" / "m3_selfplay_r02_9_a06_e02_t02_n5_v3_5000_vs3self.json"


class NoisyAndEpsilonMustBeDeclaredTest(unittest.TestCase):
    def test_e12_without_noisy_is_refused(self):
        """E12 is epsilon 0, so a non-noisy route on it would not explore at all."""
        route = replace(EXPERIMENTS["R02_9"], exploration_version="E12")
        with self.assertRaises(ValueError) as caught:
            validate_config(route)
        self.assertIn("explore not at all", str(caught.exception))

    def test_noisy_plus_epsilon_is_refused_unless_declared(self):
        route = replace(EXPERIMENTS["R02_9"], noisy=True, exploration_version="E02")
        with self.assertRaises(ValueError) as caught:
            validate_config(route)
        self.assertIn("noisy_with_epsilon", str(caught.exception))

    def test_noisy_plus_epsilon_passes_once_declared(self):
        route = replace(EXPERIMENTS["R02_9"], noisy=True, exploration_version="E02",
                        noisy_with_epsilon=True)
        self.assertIs(validate_config(route), route)

    def test_the_declaration_alone_is_not_enough(self):
        route = replace(EXPERIMENTS["R02_9"], noisy_with_epsilon=True)
        with self.assertRaises(ValueError):
            validate_config(route)

    def test_r02_13_is_the_declared_combination(self):
        route = validate_config(EXPERIMENTS["R02_13"])
        self.assertTrue(route.noisy)
        self.assertTrue(route.noisy_with_epsilon)
        self.assertNotEqual(route.exploration_version, "E12")

    def test_r02_11_and_r02_12_still_refuse_epsilon(self):
        for name in ("R02_11", "R02_12"):
            with self.subTest(route=name):
                self.assertEqual(EXPERIMENTS[name].exploration_version, "E12")
                self.assertFalse(EXPERIMENTS[name].noisy_with_epsilon)


class FrozenOpponentIsADeclaredFactorTest(unittest.TestCase):
    def _shipped(self) -> dict:
        return json.loads(SELF_PLAY_CONFIG.read_text(encoding="utf-8"))

    def _load(self, raw: dict) -> Experiment:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            return Experiment.load(path)

    def test_the_shipped_self_play_config_loads(self):
        experiment = self._load(self._shipped())
        self.assertEqual(experiment.training.opponents,
                         (FROZEN_OPPONENT_AGENT,) * 3)
        self.assertIsNotNone(experiment.frozen_opponent)
        self.assertEqual(experiment.frozen_opponent.model_path, FROZEN_MODEL)

    def test_evaluation_stays_the_tournament_setting(self):
        """The point of the arm is a training factor; changing both ends measures nothing."""
        experiment = self._load(self._shipped())
        tournament = next(s for s in experiment.evaluation_suites
                          if s.name == "classic_versus_opponents")
        self.assertEqual(tournament.phase.opponents, ("rule_based_agent",) * 3)

    def test_frozen_opponents_without_a_block_are_refused(self):
        raw = self._shipped()
        del raw["frozen_opponent"]
        with self.assertRaises(ConfigError) as caught:
            self._load(raw)
        self.assertIn("no frozen_opponent block", str(caught.exception))

    def test_a_block_without_frozen_opponents_is_refused(self):
        raw = self._shipped()
        raw["training"]["opponents"] = ["rule_based_agent"] * 3
        for suite in raw["evaluation_suites"].values():
            suite["opponents"] = ["rule_based_agent"] * 3
        with self.assertRaises(ConfigError) as caught:
            self._load(raw)
        self.assertIn("no phase lists", str(caught.exception))

    def test_a_swapped_opponent_fails_the_run(self):
        """The opponent is the factor, so it may not change without the config saying so."""
        raw = self._shipped()
        raw["frozen_opponent"]["sha256"] = "0" * 64
        with self.assertRaises(ConfigError) as caught:
            self._load(raw)
        self.assertIn("may not change silently", str(caught.exception))

    def test_absolute_and_escaping_paths_are_refused(self):
        for bad in ("/etc/passwd", "../outside.npz"):
            with self.subTest(path=bad):
                raw = self._shipped()
                raw["frozen_opponent"]["model_path"] = bad
                with self.assertRaises(ConfigError):
                    self._load(raw)

    def test_the_snapshot_carries_the_block(self):
        """A job runs in another process; anything not snapshotted does not cross."""
        experiment = self._load(self._shipped())
        snapshot = experiment.snapshot()
        self.assertIn("frozen_opponent", snapshot)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.json"
            path.write_text(json.dumps(snapshot), encoding="utf-8")
            reloaded = Experiment.load(path)
        self.assertEqual(reloaded.frozen_opponent, experiment.frozen_opponent)


class FrozenAgentHasNoSideEffectsTest(unittest.TestCase):
    """The reason it does not reuse ExperimentRuntime.

    ``ExperimentRuntime.select_action`` appends one record per step to
    ``artifact_root()/agent.jsonl``, resolved from the environment at call
    time.  Three opponents inside the training process would interleave their
    steps into the trainer's own action log, which is the file sections 7.32,
    7.34 and 7.38 are computed from.
    """

    def _agent(self):
        from types import SimpleNamespace
        import logging
        from agent_code.frozen_agent import callbacks
        agent = SimpleNamespace(train=False, logger=logging.getLogger("frozen_test"))
        callbacks.setup(agent)
        return agent, callbacks

    def _state(self) -> dict:
        field = np.zeros((17, 17), dtype=np.int8)
        field[0, :] = field[-1, :] = field[:, 0] = field[:, -1] = -1
        return {
            "round": 1, "step": 1, "field": field, "bombs": [],
            "explosion_map": np.zeros((17, 17), dtype=np.int8),
            "coins": [(3, 1)], "self": ("frozen_agent", 0, True, (1, 1)),
            "others": [], "user_input": None,
        }

    def setUp(self):
        import os
        self._previous = dict(os.environ)
        os.environ["BOMBERMAN_FROZEN_MODEL_PATH"] = str(ROOT / FROZEN_MODEL)
        os.environ["BOMBERMAN_FROZEN_EXPERIMENT"] = "R02_9"
        self._artifacts = tempfile.TemporaryDirectory()
        os.environ["BOMBERMAN_ARTIFACT_DIR"] = self._artifacts.name

    def tearDown(self):
        import os
        os.environ.clear()
        os.environ.update(self._previous)
        self._artifacts.cleanup()

    def test_it_acts_legally_and_writes_nothing(self):
        agent, callbacks = self._agent()
        action = callbacks.act(agent, self._state())
        self.assertIn(action, callbacks.ACTIONS)
        self.assertEqual(sorted(Path(self._artifacts.name).iterdir()), [])

    def test_a_missing_checkpoint_is_loud(self):
        import os
        from types import SimpleNamespace
        import logging
        from agent_code.frozen_agent import callbacks
        os.environ["BOMBERMAN_FROZEN_MODEL_PATH"] = str(ROOT / "no_such_model.npz")
        with self.assertRaises(FileNotFoundError):
            callbacks.setup(SimpleNamespace(train=False, logger=logging.getLogger("t")))

    def test_no_checkpoint_at_all_is_loud(self):
        import os
        from types import SimpleNamespace
        import logging
        from agent_code.frozen_agent import callbacks
        del os.environ["BOMBERMAN_FROZEN_MODEL_PATH"]
        with self.assertRaises(ValueError) as caught:
            callbacks.setup(SimpleNamespace(train=False, logger=logging.getLogger("t")))
        self.assertIn("BOMBERMAN_FROZEN_MODEL_PATH", str(caught.exception))

    def test_it_is_greedy(self):
        """Same definition of greedy the reported numbers use."""
        agent, callbacks = self._agent()
        state = self._state()
        first = callbacks.act(agent, state)
        for _ in range(5):
            self.assertEqual(callbacks.act(agent, state), first)


if __name__ == "__main__":
    unittest.main()
