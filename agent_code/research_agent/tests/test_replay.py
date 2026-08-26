"""Tests for the replay buffer and the replay/target-network learner."""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from agent_code.research_agent.config import EXPERIMENTS, ReplayConfig, validate_config
from agent_code.research_agent.learners import ReplayQLearner, build_learner
from agent_code.research_agent.learners.base import Transition
from agent_code.research_agent.models.linear_q import LinearQModel
from agent_code.research_agent.replay import ReplayBuffer


ACTION_COUNT = 6
# The learner sizes its buffer from the route's declared state encoder, so the
# test states must be the width handcrafted_v1 actually produces.
STATE_WIDTH = 44


def transition(value: float, action: int = 0, *, reward: float = 1.0, terminal: bool = False, n_step: int = 1) -> Transition:
    # Real handcrafted features are normalised into roughly [-1, 1]; keeping the
    # synthetic ones in that range means a test failure means a logic bug rather
    # than an SGD divergence the encoder would never have produced.
    value = value / 100.0
    return Transition(
        state=np.full(STATE_WIDTH, value, dtype=np.float32),
        action_index=action,
        reward=reward,
        next_state=None if terminal else np.full(STATE_WIDTH, value + 0.01, dtype=np.float32),
        next_legal_mask=None if terminal else np.ones(ACTION_COUNT, dtype=bool),
        terminal=terminal,
        n_step=n_step,
    )


class ReplayBufferTest(unittest.TestCase):
    def test_the_ring_overwrites_the_oldest_entry_and_the_size_saturates(self):
        buffer = ReplayBuffer(3, 4, ACTION_COUNT, seed=0)
        for index in range(5):
            buffer.append(np.full(4, index, dtype=np.float32), 0, float(index),
                          np.zeros(4, dtype=np.float32), np.ones(ACTION_COUNT, dtype=bool), False, 0.95)
        self.assertEqual(len(buffer), 3)
        self.assertEqual(sorted(buffer.rewards.tolist()), [2.0, 3.0, 4.0])

    def test_a_terminal_row_keeps_no_successor_to_bootstrap_from(self):
        buffer = ReplayBuffer(2, 4, ACTION_COUNT, seed=0)
        buffer.append(np.ones(4, dtype=np.float32), 1, 5.0, None, None, True, 0.95)
        self.assertTrue(buffer.terminals[0])
        np.testing.assert_array_equal(buffer.next_states[0], np.zeros(4, dtype=np.float32))
        self.assertFalse(buffer.next_legal_masks[0].any())

    def test_sampling_more_than_is_stored_is_refused_rather_than_padded(self):
        buffer = ReplayBuffer(10, 4, ACTION_COUNT, seed=0)
        buffer.append(np.ones(4, dtype=np.float32), 0, 1.0, np.ones(4, dtype=np.float32),
                      np.ones(ACTION_COUNT, dtype=bool), False, 0.95)
        with self.assertRaises(ValueError):
            buffer.sample(2)

    def test_the_stored_discount_travels_with_the_row(self):
        # n-step transitions of different lengths share one buffer, so gamma**n
        # cannot be recomputed at sampling time.
        buffer = ReplayBuffer(4, 4, ACTION_COUNT, seed=0)
        for n_step in (1, 3):
            buffer.append(np.ones(4, dtype=np.float32), 0, 1.0, np.ones(4, dtype=np.float32),
                          np.ones(ACTION_COUNT, dtype=bool), False, 0.95 ** n_step)
        np.testing.assert_allclose(sorted(buffer.discounts[:2].tolist()), [0.95 ** 3, 0.95], rtol=1e-6)


def replay_config(algorithm: str = "q_learning", **settings) -> object:
    base = {"capacity": 100, "batch_size": 4, "min_size": 4, "train_every": 1, "target_update_every": 2}
    return validate_config(replace(
        EXPERIMENTS["R01"],
        algorithm=algorithm,
        reward_version="A06",
        replay=ReplayConfig(**{**base, **settings}),
    ))


class ReplayQLearnerTest(unittest.TestCase):
    def _learner(self, algorithm: str = "q_learning", **settings) -> ReplayQLearner:
        config = replay_config(algorithm, **settings)
        return ReplayQLearner(config, LinearQModel(STATE_WIDTH, seed=1, learning_rate=0.05), seed=0)

    def test_nothing_is_learned_until_the_buffer_reaches_its_declared_minimum(self):
        learner = self._learner(min_size=8)
        before = learner.model.weights.copy()
        for index in range(7):
            self.assertEqual(learner.observe(transition(index)), 0.0)
        np.testing.assert_array_equal(learner.model.weights, before)
        learner.observe(transition(7))
        self.assertFalse(np.array_equal(learner.model.weights, before))

    def test_the_target_network_is_frozen_between_synchronisations(self):
        learner = self._learner(target_update_every=1000)
        target_before = learner.target_model.weights.copy()
        for index in range(40):
            learner.observe(transition(index))
        self.assertFalse(np.array_equal(learner.model.weights, target_before))
        np.testing.assert_array_equal(learner.target_model.weights, target_before)

    def test_the_target_network_is_synchronised_on_the_declared_cadence(self):
        learner = self._learner(target_update_every=3)
        for index in range(4):
            learner.observe(transition(index))
        self.assertEqual(learner.gradient_steps, 1)
        for index in range(2):
            learner.observe(transition(index))
        self.assertEqual(learner.gradient_steps, 3)
        np.testing.assert_array_equal(learner.target_model.weights, learner.model.weights)

    def test_a_terminal_transition_target_is_the_reward_alone(self):
        learner = self._learner(batch_size=1, min_size=1)
        learner.model.weights[:] = 0.0
        learner.model.bias[:] = 3.0
        learner.target_model.copy_parameters_from(learner.model)
        learner.observe(transition(1.0, action=2, reward=7.0, terminal=True))
        # Bootstrapping a terminal row would have produced 7 + 0.95 * 3.
        self.assertAlmostEqual(float(learner.model.bias[2]), 3.0 + 0.05 * (7.0 - 3.0), places=5)

    def test_an_illegal_next_action_never_enters_the_bootstrap(self):
        learner = self._learner(batch_size=1, min_size=1)
        learner.target_model.weights[:] = 0.0
        learner.target_model.bias[:] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 99.0], dtype=np.float32)
        mask = np.ones(ACTION_COUNT, dtype=bool)
        mask[5] = False
        learner.buffer.append(np.zeros(STATE_WIDTH, dtype=np.float32), 0, 0.0, np.zeros(STATE_WIDTH, dtype=np.float32), mask, False, 0.95)
        batch = learner.buffer.sample(1)
        self.assertAlmostEqual(float(learner._targets(batch)[0]), 0.0, places=5)

    def test_double_dqn_evaluates_the_online_argmax_with_the_target_network(self):
        learner = self._learner("double_dqn", batch_size=1, min_size=1)
        # Online picks action 1; the target network values that action at 2.0
        # while its own maximum is 9.0 at action 4.  Plain DQN would take 9.0.
        learner.model.weights[:] = 0.0
        learner.model.bias[:] = np.array([0.0, 5.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        learner.target_model.weights[:] = 0.0
        learner.target_model.bias[:] = np.array([0.0, 2.0, 0.0, 0.0, 9.0, 0.0], dtype=np.float32)
        learner.buffer.append(np.zeros(STATE_WIDTH, dtype=np.float32), 0, 1.0, np.zeros(STATE_WIDTH, dtype=np.float32),
                              np.ones(ACTION_COUNT, dtype=bool), False, 0.95)
        self.assertAlmostEqual(float(learner._targets(learner.buffer.sample(1))[0]), 1.0 + 0.95 * 2.0, places=5)

    def test_plain_q_learning_takes_the_target_networks_own_maximum(self):
        learner = self._learner("q_learning", batch_size=1, min_size=1)
        learner.model.weights[:] = 0.0
        learner.model.bias[:] = np.array([0.0, 5.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        learner.target_model.weights[:] = 0.0
        learner.target_model.bias[:] = np.array([0.0, 2.0, 0.0, 0.0, 9.0, 0.0], dtype=np.float32)
        learner.buffer.append(np.zeros(STATE_WIDTH, dtype=np.float32), 0, 1.0, np.zeros(STATE_WIDTH, dtype=np.float32),
                              np.ones(ACTION_COUNT, dtype=bool), False, 0.95)
        self.assertAlmostEqual(float(learner._targets(learner.buffer.sample(1))[0]), 1.0 + 0.95 * 9.0, places=5)

    def test_an_n_step_row_bootstraps_with_its_own_discount(self):
        learner = self._learner(batch_size=1, min_size=1)
        learner.target_model.weights[:] = 0.0
        learner.target_model.bias[:] = 4.0
        learner.buffer.append(np.zeros(STATE_WIDTH, dtype=np.float32), 0, 1.0, np.zeros(STATE_WIDTH, dtype=np.float32),
                              np.ones(ACTION_COUNT, dtype=bool), False, 0.95 ** 3)
        self.assertAlmostEqual(float(learner._targets(learner.buffer.sample(1))[0]), 1.0 + 0.95 ** 3 * 4.0, places=5)

    def test_evaluation_jobs_are_not_given_a_replay_allocation(self):
        # A greedy rollout never receives a transition; allocating the buffer
        # anyway would cost hundreds of megabytes per evaluation job.
        config = replay_config()
        model = LinearQModel(STATE_WIDTH, seed=1)
        self.assertIsInstance(build_learner(config, model, seed=0, training=True), ReplayQLearner)
        self.assertNotIsInstance(build_learner(config, model, seed=0, training=False), ReplayQLearner)

    def test_double_dqn_without_a_declared_buffer_is_refused(self):
        with self.assertRaises(ValueError):
            validate_config(replace(EXPERIMENTS["R01"], algorithm="double_dqn", replay=None))


class ReplayConfigTest(unittest.TestCase):
    def test_inconsistent_replay_settings_are_refused_at_construction(self):
        for settings in (
            {"capacity": 0},
            {"batch_size": 0},
            {"capacity": 10, "batch_size": 20, "min_size": 20},
            {"capacity": 10, "min_size": 20},
            {"batch_size": 64, "min_size": 32},
            {"augmentation": "rot90"},
        ):
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    ReplayConfig(**settings)

    def test_d4_augmentation_is_refused_for_a_representation_it_is_wrong_for(self):
        # Rotating a handcrafted feature vector is not a board symmetry: the
        # directional features would silently disagree with the action labels.
        with self.assertRaises(ValueError):
            validate_config(replace(
                EXPERIMENTS["R01"],
                replay=ReplayConfig(augmentation="d4"),
            ))
        validate_config(replace(
            EXPERIMENTS["R07"],
            replay=ReplayConfig(augmentation="d4"),
        ))


if __name__ == "__main__":
    unittest.main()
