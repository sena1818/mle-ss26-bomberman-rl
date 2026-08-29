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


class QuantisedBoardStorageTest(unittest.TestCase):
    """uint8 board storage: a memory layout, and only valid while it is exact."""

    def _board_state(self):
        from agent_code.research_agent import state as state_module

        field = np.zeros((9, 9), dtype=int)
        field[[0, -1], :] = -1
        field[:, [0, -1]] = -1
        field[2, 2] = 1
        return state_module.board_egocentric_v2({
            "round": 1,
            "step": 7,
            "field": field,
            "self": ("research_agent", 3, False, (3, 3)),
            "others": [("opponent", 1, True, (3, 5))],
            "bombs": [((3, 3), 2), ((5, 3), 0)],
            "coins": [(4, 4)],
            "explosion_map": np.zeros_like(field),
            "user_input": None,
        })

    def test_a_real_state_survives_the_round_trip_bit_for_bit(self):
        from agent_code.research_agent.state import quantised_board_spec

        vector = self._board_state()
        board_size, quantisation = quantised_board_spec("board_egocentric_v2")
        buffer = ReplayBuffer(
            4, vector.shape[0], ACTION_COUNT, seed=0,
            quantised_board=board_size, quantisation=quantisation,
        )
        buffer.append(vector, 0, 1.0, vector, np.ones(ACTION_COUNT, dtype=bool), False, 1.0)
        batch = buffer.sample(1)
        np.testing.assert_array_equal(batch["states"][0], vector)
        np.testing.assert_array_equal(batch["next_states"][0], vector)
        self.assertEqual(batch["states"].dtype, np.float32)

    def test_the_codes_are_a_quarter_of_the_float_footprint(self):
        from agent_code.research_agent.state import quantised_board_spec

        board_size, quantisation = quantised_board_spec("board_egocentric_v2")
        width = board_size + 6
        quantised = ReplayBuffer(
            100, width, ACTION_COUNT, quantised_board=board_size, quantisation=quantisation,
        )
        plain = ReplayBuffer(100, width, ACTION_COUNT)
        self.assertEqual(quantised.state_codes.dtype, np.uint8)
        self.assertIsNone(quantised.states)
        self.assertLess(quantised.state_codes.nbytes * 4, plain.states.nbytes + 1)

    def test_an_off_grid_value_is_refused_rather_than_rounded(self):
        buffer = ReplayBuffer(4, 5, ACTION_COUNT, quantised_board=4, quantisation=20)
        with self.assertRaises(ValueError) as raised:
            buffer.append(
                np.array([0.0, 0.5, 0.333, 1.0, 2.0], dtype=np.float32),
                0, 1.0, None, None, True, 1.0,
            )
        self.assertIn("0.333", str(raised.exception))

    def test_a_handcrafted_route_still_stores_plain_floats(self):
        from agent_code.research_agent.state import quantised_board_spec

        self.assertEqual(quantised_board_spec("handcrafted_v3"), (0, 0))
        buffer = ReplayBuffer(4, STATE_WIDTH, ACTION_COUNT)
        arbitrary = np.linspace(-0.7, 0.31, STATE_WIDTH).astype(np.float32)
        buffer.append(arbitrary, 0, 1.0, None, None, True, 1.0)
        np.testing.assert_array_equal(buffer.sample(1)["states"][0], arbitrary)

    def test_a_terminal_row_reads_back_as_zeros(self):
        from agent_code.research_agent.state import quantised_board_spec

        vector = self._board_state()
        board_size, quantisation = quantised_board_spec("board_egocentric_v2")
        buffer = ReplayBuffer(
            4, vector.shape[0], ACTION_COUNT,
            quantised_board=board_size, quantisation=quantisation,
        )
        buffer.append(vector, 0, 1.0, None, None, True, 1.0)
        batch = buffer.sample(1)
        np.testing.assert_array_equal(batch["next_states"][0], np.zeros_like(vector))
        self.assertTrue(bool(batch["terminals"][0]))


class GradientTelemetryTest(unittest.TestCase):
    """Telling "did not update" apart from "updated to a TD error of zero"."""

    def _learner(self, **overrides):
        settings = ReplayConfig(capacity=64, batch_size=4, min_size=8, train_every=4, **overrides)
        config = replace(validate_config(EXPERIMENTS["R01"]), replay=settings)
        return ReplayQLearner(config, LinearQModel(STATE_WIDTH, seed=0), seed=0)

    def test_a_step_below_min_size_reports_no_gradient_and_the_buffer_level(self):
        learner = self._learner()
        learner.observe(transition(1.0))
        step = learner.step_diagnostics()
        self.assertFalse(step["gradient_applied"])
        self.assertEqual(step["gradient_steps"], 0)
        self.assertEqual(step["replay_size"], 1)
        self.assertNotIn("mean_abs_td_error", step)

    def test_only_every_train_every_th_full_step_applies_a_gradient(self):
        learner = self._learner()
        applied = []
        for index in range(24):
            learner.observe(transition(float(index)))
            applied.append(learner.step_diagnostics()["gradient_applied"])
        # The buffer reaches min_size on the 8th observe, so the first seven
        # cannot update and the 8th is eligible (8 % train_every == 0).
        self.assertEqual(sum(applied[:7]), 0, "nothing may update below min_size")
        self.assertTrue(applied[7])
        self.assertEqual(learner.gradient_steps, sum(applied))
        self.assertEqual(learner.step_diagnostics()["gradient_steps"], learner.gradient_steps)
        self.assertEqual(sum(applied), 5)

    def test_a_real_gradient_step_reports_the_values_it_used(self):
        learner = self._learner()
        for index in range(24):
            learner.observe(transition(float(index)))
        step = learner.step_diagnostics()
        self.assertTrue(step["gradient_applied"])
        for key in ("replay_size", "mean_abs_td_error", "mean_target", "target_synchronised"):
            self.assertIn(key, step)
        self.assertGreaterEqual(step["replay_size"], 8)

    def test_the_target_network_synchronisation_is_visible(self):
        learner = self._learner(target_update_every=1)
        for index in range(24):
            learner.observe(transition(float(index)))
        self.assertTrue(learner.step_diagnostics()["target_synchronised"])

    def test_an_online_learner_calls_every_step_a_gradient_step(self):
        from agent_code.research_agent.learners import OnlineQLearner

        config = validate_config(EXPERIMENTS["R01"])
        learner = OnlineQLearner(config, LinearQModel(STATE_WIDTH, seed=0))
        learner.observe(transition(1.0))
        self.assertEqual(learner.step_diagnostics(), {"gradient_applied": True, "gradient_steps": 1})


class PrioritizedReplayTest(unittest.TestCase):
    """Proportional prioritized replay (Schaul et al. 2016).

    The motivation on this line is measured rather than general: KILLED_OPPONENT
    is 1.7% of the positive reward the agent sees and 191 times rarer than a
    coin (docs/01 section 7.27.2).  That is the situation prioritization exists
    for, so it is worth one arm -- but only if the arm is the thing it claims to
    be, which is what these tests pin.
    """

    def _draw(self, buffer: ReplayBuffer, draws: int, batch: int = 8, beta: float = 1.0):
        """Accumulate many small batches: the buffer refuses one larger than it holds."""
        indices, weights = [], []
        for _ in range(draws):
            sampled = buffer.sample(batch, beta=beta)
            indices.append(sampled["indices"])
            weights.append(sampled["weights"])
        return np.concatenate(indices), np.concatenate(weights)

    def _buffer(self, size: int = 8, **kwargs) -> ReplayBuffer:
        buffer = ReplayBuffer(size, 4, ACTION_COUNT, seed=0, **kwargs)
        for index in range(size):
            buffer.append(np.full(4, index / 100.0, dtype=np.float32), index % ACTION_COUNT, float(index),
                          np.zeros(4, dtype=np.float32), np.ones(ACTION_COUNT, dtype=bool), False, 0.95)
        return buffer

    def test_uniform_sampling_is_untouched_and_its_weights_are_exactly_one(self):
        """Every arm run before this existed has to keep the identical draw."""
        left = self._buffer()
        right = self._buffer()
        left.rng = np.random.default_rng(7)
        right.rng = np.random.default_rng(7)
        first, second = left.sample(4), right.sample(4)
        np.testing.assert_array_equal(first["indices"], second["indices"])
        np.testing.assert_array_equal(first["weights"], np.ones(4, dtype=np.float32))
        np.testing.assert_array_equal(first["rewards"], second["rewards"])

    def test_update_priorities_does_nothing_under_uniform_sampling(self):
        buffer = self._buffer()
        buffer.update_priorities(np.array([0, 1]), np.array([100.0, 100.0]))
        counts = np.bincount(self._draw(buffer, 250)[0], minlength=len(buffer))
        # A uniform draw over eight rows: no row may run away with the batch.
        self.assertLess(counts.max() / counts.min(), 1.6)

    def test_a_surprising_transition_is_replayed_far_more_often(self):
        buffer = self._buffer(sampling="prioritized", priority_exponent=1.0)
        buffer.update_priorities(np.arange(8), np.array([0.0] * 7 + [10.0]))
        counts = np.bincount(self._draw(buffer, 500)[0], minlength=8)
        # Priority 10.001 against 0.001: the last row should take essentially
        # the whole batch.  The exact ratio is the priority ratio, so this
        # asserts the mechanism rather than a magic number.
        self.assertGreater(counts[7] / counts.sum(), 0.98)

    def test_the_priority_exponent_interpolates_towards_uniform(self):
        flat = self._buffer(sampling="prioritized", priority_exponent=0.0)
        flat.update_priorities(np.arange(8), np.array([0.0] * 7 + [10.0]))
        counts = np.bincount(self._draw(flat, 500)[0], minlength=8)
        self.assertLess(counts[7] / counts.sum(), 0.20, "exponent 0 must be uniform sampling")

    def test_importance_weights_are_largest_for_the_rarest_draw_and_capped_at_one(self):
        buffer = self._buffer(sampling="prioritized", priority_exponent=1.0)
        # A mild contrast on purpose: at 9-against-0 the frequent row takes every
        # draw and there is no rare row left to compare its weight against.
        buffer.update_priorities(np.arange(8), np.array([3.0] + [1.0] * 7))
        indices, weights = self._draw(buffer, 25, beta=1.0)
        self.assertAlmostEqual(float(weights.max()), 1.0, places=6)
        # Row 0 is drawn most often, so it is the one the correction shrinks.
        heavy = weights[indices == 0]
        light = weights[indices != 0]
        self.assertTrue(len(heavy) and len(light))
        self.assertLess(heavy.max(), light.min())

    def test_beta_zero_switches_the_correction_off(self):
        buffer = self._buffer(sampling="prioritized", priority_exponent=1.0)
        buffer.update_priorities(np.arange(8), np.array([3.0] + [1.0] * 7))
        np.testing.assert_allclose(self._draw(buffer, 6, beta=0.0)[1], 1.0, atol=1e-6)

    def test_max_normalisation_shrinks_the_whole_update_and_mean_does_not(self):
        """The confound the first prioritized arm measured.

        At beta = 1 the raw weight (N * P(i))^-1 has expectation 1 under the
        sampling distribution, so dividing by the batch maximum scales the
        entire update down by a factor that depends on how skewed the priorities
        are.  runs/m3_per_5000_vs3rb_20260829 ran at a mean weight of 0.40.

        That is a shrink of the loss and NOT a smaller step: Adam divides by the
        running RMS of the gradient, so a uniform rescaling cancels, and the same
        arm's parameter norm matched R02_9's at every checkpoint (29.72 / 36.18 /
        39.16 against 29.19 / 35.79 / 38.63).  What is pinned here is the
        arithmetic -- the two normalisations differ in scale and agree in
        direction -- which is what makes a comparison of the two arms a single
        declared factor.  docs/01 section 7.38 records the withdrawn reading.
        """
        for normalisation, expected in (("max", 1.0), ("mean", None)):
            buffer = self._buffer(sampling="prioritized", priority_exponent=1.0,
                                  importance_normalisation=normalisation)
            buffer.update_priorities(np.arange(8), np.array([7.0] + [1.0] * 7))
            _, weights = self._draw(buffer, 60, beta=1.0)
            with self.subTest(normalisation=normalisation):
                if expected is None:
                    self.assertAlmostEqual(float(weights.mean()), 1.0, places=1)
                else:
                    self.assertAlmostEqual(float(weights.max()), expected, places=6)
                    self.assertLess(float(weights.mean()), 0.75,
                                    "max normalisation must visibly shrink the update")

    def test_the_two_normalisations_differ_only_by_a_positive_scale(self):
        """Same draw, same relative weighting; only the overall size differs."""
        weights = {}
        for normalisation in ("max", "mean"):
            buffer = self._buffer(sampling="prioritized", priority_exponent=1.0,
                                  importance_normalisation=normalisation)
            buffer.update_priorities(np.arange(8), np.array([7.0] + [1.0] * 7))
            buffer.rng = np.random.default_rng(11)
            batch = buffer.sample(8, beta=1.0)
            weights[normalisation] = batch["weights"]
        ratio = weights["mean"] / weights["max"]
        np.testing.assert_allclose(ratio, ratio[0], rtol=1e-5)
        self.assertGreater(ratio[0], 1.0, "mean normalisation must be the larger of the two")

    def test_an_undeclared_normalisation_is_refused(self):
        with self.assertRaises(ValueError):
            self._buffer(sampling="prioritized", importance_normalisation="median")

    def test_a_new_transition_enters_at_the_largest_priority(self):
        """Otherwise a transition could be evicted before it is ever replayed."""
        buffer = self._buffer(sampling="prioritized", priority_exponent=1.0)
        buffer.update_priorities(np.arange(8), np.full(8, 5.0))
        buffer.append(np.full(4, 0.5, dtype=np.float32), 0, 1.0,
                      np.zeros(4, dtype=np.float32), np.ones(ACTION_COUNT, dtype=bool), False, 0.95)
        self.assertEqual(buffer.priorities[0], buffer.priorities[:len(buffer)].max())

    def test_a_zero_td_error_stays_reachable(self):
        buffer = self._buffer(sampling="prioritized", priority_exponent=1.0)
        buffer.update_priorities(np.arange(8), np.zeros(8))
        self.assertGreater(buffer.priorities[:8].min(), 0.0)
        self.assertEqual(len(set(self._draw(buffer, 60)[0].tolist())), 8)


class PrioritizedLearnerTest(unittest.TestCase):
    def _learner(self, **replay) -> ReplayQLearner:
        settings = dict(capacity=64, batch_size=8, min_size=8, train_every=1, target_update_every=1000)
        settings.update(replay)
        config = validate_config(replace(
            EXPERIMENTS["R01"], algorithm="q_learning",
            replay=ReplayConfig(**settings)))
        return ReplayQLearner(config, LinearQModel(STATE_WIDTH, learning_rate=0.01), seed=3)

    def test_beta_anneals_from_the_declared_start_to_exactly_one(self):
        learner = self._learner(sampling="prioritized", importance_sampling_start=0.4,
                                importance_sampling_steps=100)
        self.assertAlmostEqual(learner._importance_sampling_exponent(), 0.4)
        learner.gradient_steps = 50
        self.assertAlmostEqual(learner._importance_sampling_exponent(), 0.7)
        learner.gradient_steps = 100
        self.assertAlmostEqual(learner._importance_sampling_exponent(), 1.0)
        learner.gradient_steps = 10_000
        self.assertAlmostEqual(learner._importance_sampling_exponent(), 1.0,
                               msg="beta must stay at 1, not overshoot")

    def test_a_uniform_learner_reports_no_prioritization_telemetry(self):
        learner = self._learner()
        for index in range(20):
            learner.observe(transition(index))
        self.assertNotIn("importance_sampling_exponent", learner.step_diagnostics())

    def test_the_prioritized_learner_records_that_it_was_live(self):
        """A declared axis nobody can see afterwards is the section 7.29 failure."""
        learner = self._learner(sampling="prioritized")
        for index in range(20):
            learner.observe(transition(index))
        diagnostics = learner.step_diagnostics()
        self.assertTrue(diagnostics["gradient_applied"])
        self.assertAlmostEqual(diagnostics["importance_sampling_exponent"], 0.4, places=3)
        self.assertLessEqual(diagnostics["mean_importance_weight"], 1.0)
        self.assertGreater(diagnostics["min_importance_weight"], 0.0)

    def test_training_moves_the_priorities_away_from_their_entry_value(self):
        learner = self._learner(sampling="prioritized")
        for index in range(40):
            learner.observe(transition(index, reward=5.0 if index == 3 else 0.0))
        priorities = learner.buffer.priorities[:len(learner.buffer)]
        self.assertGreater(len(set(np.round(priorities, 6).tolist())), 1,
                           "every priority is still the entry value; no error was written back")
