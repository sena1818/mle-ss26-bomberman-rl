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


def game_state(step: int = 1, round_number: int = 1) -> dict:
    field = np.zeros((9, 9), dtype=int)
    field[[0, -1], :] = -1
    field[:, [0, -1]] = -1
    return {
        "round": round_number,
        "step": step,
        "field": field,
        "self": ("research_agent", 0, True, (3, 3)),
        "others": [],
        "bombs": [],
        "coins": [(4, 3)],
        "explosion_map": np.zeros_like(field),
        "user_input": None,
    }


class RecordingLearner:
    """Wraps the real learner so a test can see every committed transition."""

    def __init__(self, inner):
        self.inner = inner
        self.transitions = []

    def select_action(self, *args, **kwargs):
        return self.inner.select_action(*args, **kwargs)

    def observe(self, transition):
        self.transitions.append(transition)
        return self.inner.observe(transition)

    def end_round(self):
        return self.inner.end_round()


def started_runtime(temporary: str, config=None, *, train: bool = True) -> ExperimentRuntime:
    """Return a runtime whose model and recording learner are initialized."""
    runtime = ExperimentRuntime(config or active_config(), train=train, agent_seed=3, logger=Mock())
    runtime.select_action(game_state())
    runtime.learner = RecordingLearner(runtime.learner)
    return runtime


def drive_round(runtime, *, steps: int, died: bool, final_events: list[str], round_number: int = 1) -> None:
    """Reproduce the official per-round callback sequence exactly.

    ``environment.do_step`` calls ``send_game_events`` -- which skips an agent
    that just died -- and only afterwards evaluates ``time_to_stop`` and calls
    ``end_of_round`` with the same stored state, action and event list.  A
    surviving agent therefore has its last step delivered twice; a dead agent
    never has its fatal step delivered through ``game_events_occurred``.
    """
    delivered = steps - 1 if died else steps
    for step in range(1, delivered + 1):
        events = final_events if (not died and step == steps) else []
        runtime.observe(
            game_state(step, round_number), "WAIT", game_state(step + 1, round_number), list(events)
        )
    runtime.end_round(game_state(steps, round_number), "WAIT", list(final_events))


class OfficialLifecycleTest(unittest.TestCase):
    """Each transition must reach the learner exactly once, with the right target.

    Regression cover for the terminal-transition defect: ``end_round`` used to
    commit unconditionally, so a surviving agent's last step was learned twice --
    once bootstrapped and once as a terminal target -- and its rewards were
    counted twice.  A ``coin-heaven`` round logged 51 COIN_COLLECTED events on a
    map that contains 50 coins.
    """

    def _environment(self, temporary: str, **extra: str) -> dict:
        return {
            "BOMBERMAN_ARTIFACT_DIR": str(Path(temporary) / "agent"),
            "BOMBERMAN_RUN_ID": "lifecycle_test",
            "BOMBERMAN_SCENARIO": "classic",
            "BOMBERMAN_SEED": "3",
            "BOMBERMAN_CHECKPOINT_EVERY": "1",
            **extra,
        }

    def test_surviving_round_learns_its_last_step_exactly_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(os.environ, self._environment(temporary), clear=False):
                runtime = started_runtime(temporary)
                drive_round(runtime, steps=5, died=False, final_events=["COIN_COLLECTED", "SURVIVED_ROUND"])

            transitions = runtime.learner.transitions
            self.assertEqual(len(transitions), 5, "one update per step, never two for the last one")
            self.assertEqual(runtime.round_updates, 5)
            # The decisive symptom: the final coin must be counted once.
            self.assertEqual(runtime.round_event_counts["COIN_COLLECTED"], 1)
            self.assertEqual(runtime.round_reward, 1.0)

    def test_time_limit_truncation_bootstraps_instead_of_cutting_the_return(self):
        """Surviving to the step limit is a truncation, not a terminal state."""
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(os.environ, self._environment(temporary), clear=False):
                runtime = started_runtime(temporary)
                self.assertFalse(runtime.config.terminal_on_truncation)
                drive_round(runtime, steps=4, died=False, final_events=["SURVIVED_ROUND"])

            final = runtime.learner.transitions[-1]
            self.assertFalse(final.terminal)
            self.assertIsNotNone(final.next_state)
            self.assertIsNotNone(final.next_legal_mask)

    def test_terminal_on_truncation_is_a_declared_switch(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = replace(active_config(), terminal_on_truncation=True)
            with patch.dict(os.environ, self._environment(temporary), clear=False):
                runtime = started_runtime(temporary, config)
                drive_round(runtime, steps=4, died=False, final_events=["SURVIVED_ROUND"])

            final = runtime.learner.transitions[-1]
            self.assertTrue(final.terminal)
            self.assertIsNone(final.next_state)
            self.assertEqual(len(runtime.learner.transitions), 4)

    def test_environment_variable_selects_the_truncation_target(self):
        with patch.dict(os.environ, {"BOMBERMAN_TERMINAL_ON_TRUNCATION": "1"}, clear=False):
            self.assertTrue(active_config().terminal_on_truncation)
        with patch.dict(os.environ, {"BOMBERMAN_TERMINAL_ON_TRUNCATION": "0"}, clear=False):
            self.assertFalse(active_config().terminal_on_truncation)
        with patch.dict(os.environ, {"BOMBERMAN_TERMINAL_ON_TRUNCATION": "maybe"}, clear=False):
            with self.assertRaises(ValueError):
                active_config()

    def test_fatal_step_is_learned_once_as_a_real_terminal_transition(self):
        """A dead agent's last step never reaches game_events_occurred."""
        with tempfile.TemporaryDirectory() as temporary:
            config = replace(active_config(), reward_version="A03")
            with patch.dict(os.environ, self._environment(temporary), clear=False):
                runtime = started_runtime(temporary, config)
                drive_round(runtime, steps=5, died=True, final_events=["KILLED_SELF", "GOT_KILLED"])

            transitions = runtime.learner.transitions
            self.assertEqual(len(transitions), 5, "the fatal step must not be dropped")
            self.assertTrue(transitions[-1].terminal)
            self.assertIsNone(transitions[-1].next_state)
            self.assertFalse(any(transition.terminal for transition in transitions[:-1]))
            # One death, one penalty, even though two death events fire.
            self.assertEqual(transitions[-1].reward, DEATH_PENALTIES["A03"])
            self.assertEqual(runtime.round_reward, DEATH_PENALTIES["A03"])

    def test_agent_killed_on_the_first_step_still_learns_that_step(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = replace(active_config(), reward_version="A03")
            with patch.dict(os.environ, self._environment(temporary), clear=False):
                runtime = started_runtime(temporary, config)
                drive_round(runtime, steps=1, died=True, final_events=["KILLED_SELF", "GOT_KILLED"])

            self.assertEqual(len(runtime.learner.transitions), 1)
            self.assertTrue(runtime.learner.transitions[0].terminal)
            self.assertEqual(runtime.round_reward, DEATH_PENALTIES["A03"])

    def test_no_transition_leaks_across_round_boundaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(os.environ, self._environment(temporary), clear=False):
                runtime = started_runtime(temporary)
                drive_round(runtime, steps=3, died=False, final_events=["SURVIVED_ROUND"], round_number=1)
                self.assertIsNone(runtime._pending)
                drive_round(runtime, steps=2, died=True, final_events=["KILLED_SELF"], round_number=2)
                self.assertIsNone(runtime._pending)

            self.assertEqual(len(runtime.learner.transitions), 5)
            self.assertEqual(runtime.round_updates, 2, "per-round metrics reset on a new round")
            self.assertEqual(runtime.training_updates, 5)

    def test_unusable_action_is_skipped_without_dropping_the_previous_step(self):
        """A timeout or silenced agent error has no six-action index."""
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(os.environ, self._environment(temporary), clear=False):
                runtime = started_runtime(temporary)
                runtime.observe(game_state(1), "WAIT", game_state(2), [])
                runtime.observe(game_state(2), "ERROR", game_state(3), [])
                runtime.end_round(game_state(3), None, ["SURVIVED_ROUND"])

            # Step 1 is still learned; the unusable step 2 is not invented.
            self.assertEqual(len(runtime.learner.transitions), 1)
            self.assertFalse(runtime.learner.transitions[0].terminal)


class NStepAndShapingLifecycleTest(unittest.TestCase):
    """The docs/05 lifecycle invariants must survive the M2 additions.

    Longer returns and a shaping term both change *what* is learned; neither
    may change *how many times* a step is learned from, or the count that the
    terminal-transition regression check relies on stops meaning anything.
    """

    def _environment(self, temporary: str, **extra: str) -> dict:
        return {
            "BOMBERMAN_ARTIFACT_DIR": str(Path(temporary) / "agent"),
            "BOMBERMAN_RUN_ID": "nstep_test",
            "BOMBERMAN_SCENARIO": "loot-crate",
            "BOMBERMAN_SEED": "3",
            "BOMBERMAN_CHECKPOINT_EVERY": "1",
            **extra,
        }

    def _run(self, temporary: str, *, n_step: int, reward_version: str, steps: int, died: bool,
             final_events: list[str]):
        config = replace(active_config(), n_step=n_step, reward_version=reward_version)
        runtime = started_runtime(temporary, config)
        drive_round(runtime, steps=steps, died=died, final_events=final_events)
        return runtime

    def test_a_longer_window_still_learns_from_every_step_exactly_once(self):
        for n_step in (1, 3, 5):
            for died in (False, True):
                with self.subTest(n_step=n_step, died=died):
                    with tempfile.TemporaryDirectory() as temporary:
                        with patch.dict(os.environ, self._environment(temporary), clear=False):
                            runtime = self._run(
                                temporary, n_step=n_step, reward_version="A03", steps=6, died=died,
                                final_events=["KILLED_SELF", "GOT_KILLED"] if died else ["SURVIVED_ROUND"],
                            )
                        transitions = runtime.learner.transitions
                        self.assertEqual(len(transitions), 6)
                        self.assertEqual(runtime.round_updates, 6)
                        # Every window that *reaches* the fatal state needs no
                        # bootstrap, so with n = 3 the last three returns are
                        # terminal.  A truncated round bootstraps all of them.
                        self.assertEqual(sum(t.terminal for t in transitions), min(n_step, 6) if died else 0)
                        self.assertTrue(all(t.terminal == (t.next_state is None) for t in transitions))

    def test_the_death_penalty_is_still_applied_exactly_once_under_n_step(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(os.environ, self._environment(temporary), clear=False):
                runtime = self._run(temporary, n_step=3, reward_version="A03", steps=4, died=True,
                                    final_events=["KILLED_SELF", "GOT_KILLED"])
            # The penalty enters exactly one one-step reward, so it appears in
            # the n-step returns that contain that step and nowhere else.
            self.assertAlmostEqual(runtime.round_reward, DEATH_PENALTIES["A03"], places=6)

    def test_a_truncated_round_bootstraps_every_leftover_window(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(os.environ, self._environment(temporary), clear=False):
                runtime = self._run(temporary, n_step=5, reward_version="A03", steps=3, died=False,
                                    final_events=["SURVIVED_ROUND"])
            transitions = runtime.learner.transitions
            self.assertEqual([t.n_step for t in transitions], [3, 2, 1])
            self.assertTrue(all(t.next_state is not None for t in transitions))

    def test_shaping_changes_the_learner_reward_but_not_the_reported_official_reward(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(os.environ, self._environment(temporary), clear=False):
                shaped = self._run(temporary, n_step=1, reward_version="A06", steps=4, died=False,
                                   final_events=["COIN_COLLECTED", "SURVIVED_ROUND"])
            with patch.dict(os.environ, self._environment(temporary), clear=False):
                plain = self._run(temporary, n_step=1, reward_version="A03", steps=4, died=False,
                                  final_events=["COIN_COLLECTED", "SURVIVED_ROUND"])
        # A03 and A06 have byte-identical event tables, so the reported reward
        # stays comparable; only the shaping column and the learner differ.
        self.assertAlmostEqual(shaped.round_reward, plain.round_reward, places=6)
        self.assertNotAlmostEqual(shaped.round_shaping_reward, 0.0)
        self.assertEqual(plain.round_shaping_reward, 0.0)
        shaped_rewards = [t.reward for t in shaped.learner.transitions]
        plain_rewards = [t.reward for t in plain.learner.transitions]
        self.assertFalse(np.allclose(shaped_rewards, plain_rewards))

    def test_shaping_uses_the_terminal_potential_on_a_fatal_transition(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(os.environ, self._environment(temporary), clear=False):
                runtime = self._run(temporary, n_step=1, reward_version="A06", steps=3, died=True,
                                    final_events=["KILLED_SELF", "GOT_KILLED"])
            fatal = runtime.learner.transitions[-1]
            self.assertTrue(fatal.terminal)
            # phi(terminal) = 0, so the shaping on the fatal step is -phi(s).
            expected = reward_for_events("A06", ["KILLED_SELF", "GOT_KILLED"]) - runtime.shaping.potential(
                game_state(3, 1)
            )
            self.assertAlmostEqual(fatal.reward, expected, places=6)


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
                runtime.observe(game_state(1), action, game_state(2), [])
                runtime.end_round(game_state(2), action, [])
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
            # ``cnn_q`` is a declared value in the config vocabulary that has no
            # QModel adapter: the seam must fail closed rather than substitute.
            config = replace(active_config(), network="cnn_q")
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
