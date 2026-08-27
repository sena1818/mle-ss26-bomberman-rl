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
    EXPERIMENTS,
    EXPLORATION_SCHEDULES,
    epsilon_for_training_step,
    EXPLORATION_VERSIONS,
    REWARD_VERSIONS,
    active_config,
    epsilon_for_training_round,
    exploration_specification,
)
from agent_code.research_agent.learners import OnlineQLearner
from agent_code.research_agent.models import LinearQModel, build_model
from agent_code.research_agent.runtime import ExperimentRuntime
from agent_code.research_agent.config import MAX_STEPS
from agent_code.research_agent.runtime.experiment import (
    DEATH_PENALTIES,
    REWARD_TABLES,
    reward_for_events,
    reward_specification,
)


def game_state(step: int = 1, round_number: int = 1, *, cleared: bool = False) -> dict:
    """One agent-visible state.

    ``cleared`` removes the last collectable coin from a board that already has
    no crates, no bombs and no opponents, which is exactly the condition
    ``environment.time_to_stop`` treats as "nothing left to do".
    """
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
        "coins": [] if cleared else [(4, 3)],
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


def drive_round(
    runtime,
    *,
    steps: int,
    died: bool,
    final_events: list[str],
    round_number: int = 1,
    end_reason: str | None = None,
) -> None:
    """Reproduce the official per-round callback sequence exactly.

    ``environment.do_step`` calls ``send_game_events`` -- which skips an agent
    that just died -- and only afterwards evaluates ``time_to_stop`` and calls
    ``end_of_round`` with the same stored state, action and event list.  A
    surviving agent therefore has its last step delivered twice; a dead agent
    never has its fatal step delivered through ``game_events_occurred``.

    Both states of one delivery carry the *same* step number: ``do_step`` stores
    ``last_game_state`` and then fetches the successor without changing
    ``self.step``.  ``time_to_stop`` compares that same counter against
    ``MAX_STEPS``, which is why the successor is what terminality is read from.

    A surviving round must say *why* it ended, because the runtime now reads that
    from the delivered successor state instead of being told afterwards:

    * ``"truncation"`` numbers the steps so the last delivery is ``MAX_STEPS``;
    * ``"task_complete"`` hands over a cleared board on the last delivery.
    """
    if not died and end_reason not in ("truncation", "task_complete"):
        raise ValueError("a surviving round must declare why it ended")
    delivered = steps - 1 if died else steps
    offset = MAX_STEPS - steps if end_reason == "truncation" else 0
    for step in range(1, delivered + 1):
        last = not died and step == steps
        runtime.observe(
            game_state(offset + step, round_number),
            "WAIT",
            game_state(offset + step, round_number, cleared=last and end_reason == "task_complete"),
            list(final_events) if last else [],
        )
    runtime.end_round(game_state(offset + steps, round_number), "WAIT", list(final_events))


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
                drive_round(runtime, steps=5, died=False, final_events=["COIN_COLLECTED", "SURVIVED_ROUND"],
                            end_reason="truncation")

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
                drive_round(runtime, steps=4, died=False, final_events=["SURVIVED_ROUND"], end_reason="truncation")

            final = runtime.learner.transitions[-1]
            self.assertFalse(final.terminal)
            self.assertIsNotNone(final.next_state)
            self.assertIsNotNone(final.next_legal_mask)

    def test_terminal_on_truncation_is_a_declared_switch(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = replace(active_config(), terminal_on_truncation=True)
            with patch.dict(os.environ, self._environment(temporary), clear=False):
                runtime = started_runtime(temporary, config)
                drive_round(runtime, steps=4, died=False, final_events=["SURVIVED_ROUND"], end_reason="truncation")

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
                drive_round(runtime, steps=3, died=False, final_events=["SURVIVED_ROUND"], round_number=1,
                            end_reason="truncation")
                self.assertIsNone(runtime._delivered_key)
                drive_round(runtime, steps=2, died=True, final_events=["KILLED_SELF"], round_number=2)
                self.assertIsNone(runtime._delivered_key)

            self.assertEqual(len(runtime.learner.transitions), 5)
            self.assertEqual(runtime.round_updates, 2, "per-round metrics reset on a new round")
            self.assertEqual(runtime.training_updates, 5)


    def test_a_step_is_learned_before_the_next_action_is_chosen(self):
        """One-step Q-learning must not lag the policy it is training.

        ``do_step`` chooses the next action in ``poll_and_run_agents`` *before*
        ``send_game_events`` delivers the previous step, so a runtime that holds
        a transition back until the next delivery makes every action come from
        parameters that exclude the step just observed.  That is a different
        algorithm, and it measurably changed results; see docs/05 section 1.10.
        """
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(os.environ, self._environment(temporary), clear=False):
                runtime = started_runtime(temporary)
                before = runtime.model.weights.copy()
                runtime.observe(game_state(1), "WAIT", game_state(1), ["COIN_COLLECTED"])
                # The update has already happened, so the parameters the next
                # act() will read differ from the ones that produced this step.
                self.assertEqual(len(runtime.learner.transitions), 1)
                self.assertFalse(np.array_equal(before, runtime.model.weights))
                runtime.select_action(game_state(2))

    def test_a_completed_task_is_terminal_regardless_of_the_truncation_switch(self):
        """Clearing the board ends the MDP; the step limit only truncates it.

        ``time_to_stop`` checks "nothing left to do" before it checks the step
        limit, and the two are not the same event: the first has a true
        remaining return of zero, the second does not.  Sharing one switch
        between them would cut the bootstrap on a round that merely ran out of
        time, or keep it on a round that genuinely finished.
        """
        for terminal_on_truncation in (False, True):
            with self.subTest(terminal_on_truncation=terminal_on_truncation):
                with tempfile.TemporaryDirectory() as temporary:
                    config = replace(active_config(), terminal_on_truncation=terminal_on_truncation)
                    with patch.dict(os.environ, self._environment(temporary), clear=False):
                        runtime = started_runtime(temporary, config)
                        drive_round(runtime, steps=3, died=False, final_events=["SURVIVED_ROUND"],
                                    end_reason="task_complete")

                    final = runtime.learner.transitions[-1]
                    self.assertEqual(len(runtime.learner.transitions), 3)
                    self.assertTrue(final.terminal, "a cleared board is a real terminal state")
                    self.assertIsNone(final.next_state)
                    self.assertEqual(runtime.round_end_mispredictions, 0)

    def test_a_round_end_the_runtime_did_not_predict_is_counted(self):
        """The smoke-stage approximation is measured, not assumed.

        ``round_end_reason`` cannot see an explosion that has decayed to smoke,
        so its "nothing left to do" test is necessary but can be early or late.
        Every disagreement with what the framework actually did is counted and
        written into ``round_end``.
        """
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(os.environ, self._environment(temporary), clear=False):
                runtime = started_runtime(temporary)
                # A board that still has a coin, at a step well below the limit:
                # nothing here says the round is about to end, yet it does.
                runtime.observe(game_state(7), "WAIT", game_state(7), [])
                self.assertEqual(runtime.round_end_mispredictions, 0)
                runtime.end_round(game_state(7), "WAIT", ["SURVIVED_ROUND"])

            self.assertEqual(runtime.round_end_mispredictions, 1)
            self.assertEqual(len(runtime.learner.transitions), 1, "still learned exactly once")
            record = json.loads((Path(temporary) / "agent" / "agent.jsonl").read_text().splitlines()[-1])
            self.assertEqual(record["round_end_mispredictions"], 1)

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
        drive_round(runtime, steps=steps, died=died, final_events=final_events,
                    end_reason=None if died else "truncation")
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
    def test_e02_is_e01_spelled_in_absolute_rounds_at_a_2000_round_budget(self):
        """The whole exploration ablation rests on this being an identity.

        E03--E06 are compared against runs/m3_3lx_oppeval_20260826, which was
        trained under E01.  That control is only legitimate if E02 -- the arm
        nobody needs to run -- produces the same epsilon on every one of the
        2000 rounds.  Anything less makes the ablation two changes, not one.
        """
        e01 = replace(active_config(), exploration_version="E01")
        e02 = replace(active_config(), exploration_version="E02")
        for round_number in range(1, 2001):
            self.assertEqual(
                epsilon_for_training_round(e01, round_number, 2000),
                epsilon_for_training_round(e02, round_number, 2000),
                f"schedules diverge at round {round_number}",
            )

    def test_the_absolute_schedules_do_not_move_when_the_budget_does(self):
        """The point of the family: budget and schedule are now independent."""
        config = replace(active_config(), exploration_version="E02")
        for budget in (2000, 3000, 5000):
            self.assertEqual(epsilon_for_training_round(config, 400, budget), 0.30)
            self.assertAlmostEqual(epsilon_for_training_round(config, 401, budget), 0.2998437, places=6)
            self.assertEqual(epsilon_for_training_round(config, 2000, budget), 0.05)
        # Past the anneal the floor simply holds; E01 would still be decaying.
        self.assertEqual(epsilon_for_training_round(config, 4000, 5000), 0.05)
        self.assertGreater(
            epsilon_for_training_round(replace(active_config(), exploration_version="E01"), 4000, 5000),
            0.05,
        )

    def test_each_ablation_point_varies_exactly_one_thing_from_e02(self):
        specifications = {version: exploration_specification(version) for version in
                          ("E02", "E03", "E04", "E05", "E06")}
        for version, specification in specifications.items():
            self.assertEqual(specification["kind"], "hold_then_linear_absolute")
            self.assertEqual(specification["initial_epsilon"], 0.30)
            self.assertEqual(
                specification["hold_rounds"] + specification["anneal_rounds"], 2000,
                f"{version} would not reach its floor exactly at the end of a 2000-round budget",
            )
        # E03/E04 move only the hold; E05/E06 move only the floor.
        for version in ("E03", "E04"):
            self.assertEqual(specifications[version]["final_epsilon"], 0.05)
            self.assertNotEqual(specifications[version]["hold_rounds"], 400)
        for version in ("E05", "E06"):
            self.assertEqual(specifications[version]["hold_rounds"], 400)
            self.assertNotEqual(specifications[version]["final_epsilon"], 0.05)

    def test_route_selection_is_runtime_configuration_not_callback_code(self):
        with patch.dict(os.environ, {"BOMBERMAN_EXPERIMENT": "R01"}, clear=False):
            self.assertEqual(active_config().name, "R01")
        with patch.dict(os.environ, {"BOMBERMAN_EXPERIMENT": "R99"}, clear=False):
            with self.assertRaises(ValueError):
                active_config()

    def test_e01_schedule_is_predeclared_and_evaluation_stays_greedy(self):
        config = replace(active_config(), exploration_version="E01")
        self.assertEqual(
            EXPLORATION_VERSIONS,
            frozenset({"E00", "E01", "E02", "E03", "E04", "E05", "E06", "E07", "E08",
                       "E09", "E10"}),
        )
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


class ColdAndWarmStartScheduleTest(unittest.TestCase):
    """E07 / E08: the two schedules the deep line needs, and why they differ."""

    def test_every_earlier_schedule_starts_at_the_same_epsilon(self):
        """The fact that makes E07 necessary rather than a preference."""
        starts = {
            name: schedule.get("initial_epsilon")
            for name, schedule in EXPLORATION_SCHEDULES.items()
            if name not in {"E00", "E07", "E08", "E09", "E10"}
        }
        self.assertEqual(set(starts.values()), {0.30})

    def test_e07_holds_at_one_then_reaches_the_floor_on_schedule(self):
        config = replace(active_config(), exploration_version="E07")
        self.assertEqual(epsilon_for_training_round(config, 1, 10000), 1.00)
        self.assertEqual(epsilon_for_training_round(config, 100, 10000), 1.00)
        self.assertAlmostEqual(epsilon_for_training_round(config, 550, 10000), 0.525)
        self.assertEqual(epsilon_for_training_round(config, 1000, 10000), 0.05)
        self.assertEqual(epsilon_for_training_round(config, 10000, 10000), 0.05)

    def test_e08_differs_from_e07_in_the_start_and_nothing_else(self):
        cold = EXPLORATION_SCHEDULES["E07"]
        warm = EXPLORATION_SCHEDULES["E08"]
        differing = {key for key in cold if key != "description" and cold[key] != warm.get(key)}
        self.assertEqual(differing, {"initial_epsilon"})
        self.assertEqual(warm["initial_epsilon"], 0.20)

    def test_an_absolute_schedule_makes_a_checkpoint_budget_independent(self):
        """Why the M4 line needs one long run instead of one arm per budget.

        E07 reads only its own hold and anneal lengths, so round N of a
        10000-round run sees exactly the epsilon round N of a 2000-round run
        would have seen.  Every checkpoint is therefore a shorter budget's
        final model, which E01 could never be.
        """
        config = replace(active_config(), exploration_version="E07")
        for round_number in (1, 100, 101, 500, 999, 1000, 1500, 2000):
            self.assertEqual(
                epsilon_for_training_round(config, round_number, 2000),
                epsilon_for_training_round(config, round_number, 10000),
            )


class StepCountedScheduleTest(unittest.TestCase):
    """E09 / E10: the schedule and the replay buffer in the same unit.

    The pilot measured a loot-crate round at 9.8 steps under epsilon 1.00 and
    210 steps once the agent had learned to survive.  A round is therefore not
    a unit of experience, and a round-counted hold cannot express "explore at
    full strength until the buffer is ready": under E07 the buffer held 982 of
    the 10,000 transitions it needed when the hold ended.
    """

    def test_the_hold_is_the_replay_min_size(self):
        schedule = EXPLORATION_SCHEDULES["E09"]
        self.assertEqual(schedule["hold_steps"], EXPERIMENTS["R07"].replay.min_size)

    def test_epsilon_follows_environment_steps(self):
        config = replace(active_config(), exploration_version="E09")
        self.assertEqual(epsilon_for_training_step(config, 0), 1.00)
        self.assertEqual(epsilon_for_training_step(config, 10_000), 1.00)
        self.assertAlmostEqual(epsilon_for_training_step(config, 110_000), 0.525)
        self.assertEqual(epsilon_for_training_step(config, 210_000), 0.05)
        self.assertEqual(epsilon_for_training_step(config, 3_000_000), 0.05)

    def test_e10_differs_from_e09_in_the_start_and_nothing_else(self):
        cold, warm = EXPLORATION_SCHEDULES["E09"], EXPLORATION_SCHEDULES["E10"]
        differing = {key for key in cold if key != "description" and cold[key] != warm.get(key)}
        self.assertEqual(differing, {"initial_epsilon"})

    def test_a_round_counted_query_on_a_step_schedule_is_refused(self):
        """Fail closed: a silent fallback would hand back the wrong epsilon."""
        config = replace(active_config(), exploration_version="E09")
        self.assertIsNone(epsilon_for_training_step(replace(config, exploration_version="E02"), 5))
        with self.assertRaises(ValueError):
            epsilon_for_training_round(config, 1, 10_000)
