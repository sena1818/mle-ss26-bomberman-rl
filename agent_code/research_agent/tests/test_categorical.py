"""Tests for the C51 distributional head (Bellemare et al. 2017).

Distributional RL fails quietly.  A projection that loses mass, a support that
clips every target, or a gradient with the wrong sign all keep training and
produce a model that merely learns something else -- the first version of this
head walked its probability mass to the far end of the support and pinned the
cross-entropy at -log(1e-8) while reporting a perfectly ordinary-looking run.
Every test here pins one of those.
"""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from agent_code.research_agent.config import ACTIONS, EXPERIMENTS, ReplayConfig, validate_config
from agent_code.research_agent.learners import ReplayQLearner
from agent_code.research_agent.learners.base import Transition
from agent_code.research_agent.models import build_model, load_model
from agent_code.research_agent.models.categorical_mlp_q import CategoricalMLPQModel
from agent_code.research_agent.models.mlp_q import MLPQModel


# handcrafted_v3 is what R02_10 declares, and the learner sizes its replay
# buffer from the encoder rather than from the states it is handed.
STATE_WIDTH = 62


def model(**kwargs) -> CategoricalMLPQModel:
    settings = dict(atoms=21, value_min=-1.0, value_max=9.0, seed=0,
                    learning_rate=0.01, optimizer="adam")
    settings.update(kwargs)
    return CategoricalMLPQModel(4, (16,), **settings)


class ProjectionTest(unittest.TestCase):
    """The categorical Bellman operator: algorithm 1 of the paper."""

    def setUp(self):
        self.model = model()          # support -1, -0.5, ..., 9; delta_z 0.5
        self.probabilities = np.zeros((1, 21))
        self.probabilities[0, 10] = 1.0   # all mass at value 4.0

    def test_probability_mass_is_conserved(self):
        rewards = np.array([0.0, 1.3, -7.0, 40.0])
        target = self.model.project_targets(
            rewards, np.full(4, 0.9), np.zeros(4, bool), np.repeat(self.probabilities, 4, axis=0))
        np.testing.assert_allclose(target.sum(axis=1), 1.0, atol=1e-6)

    def test_a_terminal_transition_is_a_delta_at_its_reward(self):
        target = self.model.project_targets(
            np.array([2.0]), np.array([0.9]), np.array([True]), self.probabilities)
        self.assertAlmostEqual(float(target[0, 6]), 1.0, places=5)   # -1 + 6*0.5 = 2.0
        self.assertAlmostEqual(float(target.sum()), 1.0, places=5)

    def test_a_value_landing_exactly_on_an_atom_keeps_all_of_its_mass(self):
        """The case where both interpolation weights are zero."""
        # r = 0.5, gamma^n = 0.5, z = 4.0  ->  Tz = 2.5, exactly atom 7.
        target = self.model.project_targets(
            np.array([0.5]), np.array([0.5]), np.array([False]), self.probabilities)
        self.assertAlmostEqual(float(target[0, 7]), 1.0, places=5)

    def test_a_value_between_atoms_is_split_in_proportion(self):
        # r = 0.25, gamma^n = 0.5, z = 4.0 -> Tz = 2.25, one quarter of the way
        # from atom 6 (2.0) to atom 7 (2.5).
        target = self.model.project_targets(
            np.array([0.25]), np.array([0.5]), np.array([False]), self.probabilities)
        self.assertAlmostEqual(float(target[0, 6]), 0.5, places=5)
        self.assertAlmostEqual(float(target[0, 7]), 0.5, places=5)

    def test_values_beyond_the_support_are_clipped_to_its_ends(self):
        low = self.model.project_targets(
            np.array([-50.0]), np.array([0.9]), np.array([True]), self.probabilities)
        high = self.model.project_targets(
            np.array([50.0]), np.array([0.9]), np.array([True]), self.probabilities)
        self.assertAlmostEqual(float(low[0, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(high[0, -1]), 1.0, places=5)


class CategoricalHeadTest(unittest.TestCase):
    def test_q_values_are_the_expectation_of_the_predicted_distribution(self):
        head = model()
        states = np.random.default_rng(0).normal(size=(5, 4)).astype(np.float32)
        expected = head.distribution_batch(states) @ head.support
        np.testing.assert_allclose(head.q_values_batch(states), expected, rtol=1e-6)
        np.testing.assert_allclose(head.q_values(states[0]), expected[0], rtol=1e-6)

    def test_every_predicted_distribution_is_a_distribution(self):
        head = model()
        probabilities = head.distribution_batch(np.random.default_rng(1).normal(size=(7, 4)))
        self.assertEqual(probabilities.shape, (7, len(ACTIONS), 21))
        np.testing.assert_allclose(probabilities.sum(axis=-1), 1.0, rtol=1e-6)
        self.assertTrue((probabilities >= 0).all())

    def test_a_delta_target_is_learned(self):
        """The regression test for a gradient that ascended the wrong way.

        ``_apply_gradients`` adds, so the head must hand it (target - p), not
        the descent direction.  With the sign flipped this test's mass ends up
        at the opposite end of the support and the cross-entropy pins at
        -log(1e-8) = 18.42 instead of falling.
        """
        head = model()
        states = np.tile(np.array([0.5, -0.2, 0.1, 0.0], dtype=np.float32), (8, 1))
        actions = np.zeros(8, dtype=np.intp)
        target = np.zeros((8, 21), dtype=np.float32)
        target[:, 4] = 1.0                      # atom 4 is the value 1.0
        first = head.fit_batch_distribution(states, actions, target).mean()
        for _ in range(400):
            head.fit_batch_distribution(states, actions, target)
        final = head.fit_batch_distribution(states, actions, target).mean()
        self.assertLess(final, first / 100.0, "cross-entropy did not fall")
        probabilities = head.distribution_batch(states[:1])[0, 0]
        self.assertEqual(int(np.argmax(probabilities)), 4)
        self.assertAlmostEqual(float(head.q_values(states[0])[0]), 1.0, places=1)

    def test_the_gradient_enters_only_through_the_taken_action(self):
        """The other five heads move only through the shared trunk, not directly.

        Their logits receive exactly zero gradient, so what reaches them is the
        change in the hidden layers underneath -- an order of magnitude smaller
        than the head that was actually trained.  Asserting they do not move at
        all would be asserting the trunk is not shared, which it is.
        """
        head = model()
        states = np.tile(np.array([0.5, -0.2, 0.1, 0.0], dtype=np.float32), (8, 1))
        before = head.q_values(states[0]).copy()
        target = np.zeros((8, 21), dtype=np.float32)
        target[:, 4] = 1.0
        for _ in range(50):
            head.fit_batch_distribution(states, np.zeros(8, dtype=np.intp), target)
        after = head.q_values(states[0])
        taken = abs(after[0] - before[0])
        untaken = np.abs(after[1:] - before[1:]).max()
        self.assertGreater(taken, 1e-3)
        self.assertGreater(taken, 10 * untaken)

    def test_the_scalar_paths_refuse_rather_than_pretend(self):
        head = model()
        with self.assertRaises(NotImplementedError):
            head.fit_batch(np.zeros((2, 4), np.float32), np.zeros(2, np.intp), np.zeros(2, np.float32))
        with self.assertRaises(NotImplementedError):
            head.q_learning_update(np.zeros(4, np.float32), 0, 1.0, None, None, 0.01, 0.95)

    def test_a_declared_loss_other_than_cross_entropy_is_refused(self):
        with self.assertRaises(ValueError):
            model(td_loss="huber")

    def test_a_saved_head_reloads_with_its_support_and_its_predictions(self):
        head = model()
        states = np.random.default_rng(2).normal(size=(3, 4)).astype(np.float32)
        expected = head.q_values_batch(states)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "head.npz"
            head.save(path, metadata={"note": "test"})
            reloaded = CategoricalMLPQModel.load(path, learning_rate=0.01, optimizer="adam")
        self.assertEqual((reloaded.atoms, reloaded.value_min, reloaded.value_max), (21, -1.0, 9.0))
        np.testing.assert_allclose(reloaded.q_values_batch(states), expected, rtol=1e-6)

    def test_a_scalar_checkpoint_is_not_mistaken_for_a_categorical_one(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scalar.npz"
            MLPQModel(4, (16,)).save(path)
            with self.assertRaises(ValueError):
                CategoricalMLPQModel.load(path)

    def test_loading_onto_a_different_support_is_refused(self):
        """The same weights mean different values on a different grid."""
        config = validate_config(EXPERIMENTS["R02_10"])
        head = build_model(config, 62, seed=0)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "head.npz"
            head.save(path)
            with self.assertRaises(ValueError):
                load_model(replace(config, value_max=20.0), path)


class DistributionalLearnerTest(unittest.TestCase):
    def _learner(self, route: str = "R02_10", **overrides) -> ReplayQLearner:
        config = validate_config(replace(
            EXPERIMENTS[route],
            replay=ReplayConfig(capacity=256, batch_size=16, min_size=16,
                                train_every=1, target_update_every=25),
            learning_rate=0.005, **overrides))
        return ReplayQLearner(config, build_model(config, STATE_WIDTH, seed=1), seed=0)

    def test_the_learner_detects_the_head_from_the_model(self):
        self.assertTrue(self._learner().distributional)
        self.assertFalse(self._learner("R02_9").distributional)

    def test_a_route_and_a_model_that_disagree_fail_immediately(self):
        config = validate_config(replace(
            EXPERIMENTS["R02_10"],
            replay=ReplayConfig(capacity=64, batch_size=8, min_size=8)))
        scalar = MLPQModel(STATE_WIDTH, config.hidden_layers)
        with self.assertRaises(TypeError):
            ReplayQLearner(config, scalar, seed=0)

    def test_a_constant_terminal_reward_is_learned_as_its_own_value(self):
        """End to end: every transition ends with reward 2, so Q must reach 2."""
        learner = self._learner()
        state = np.full(STATE_WIDTH, 0.25, dtype=np.float32)
        for _ in range(600):
            learner.observe(Transition(state=state, action_index=0, reward=2.0, next_state=None,
                                       next_legal_mask=None, terminal=True, n_step=5))
        self.assertAlmostEqual(float(learner.model.q_values(state)[0]), 2.0, delta=0.3)

    def test_the_diagnostics_stay_comparable_with_the_scalar_arms(self):
        learner = self._learner()
        state = np.full(STATE_WIDTH, 0.25, dtype=np.float32)
        for _ in range(60):
            learner.observe(Transition(state=state, action_index=0, reward=2.0, next_state=None,
                                       next_legal_mask=None, terminal=True, n_step=5))
        diagnostics = learner.step_diagnostics()
        self.assertTrue(diagnostics["gradient_applied"])
        # mean_target is the expectation of the target distribution, so it means
        # the same thing here as in every scalar arm: with terminal rewards of 2
        # the projected target's mean is 2.
        self.assertAlmostEqual(diagnostics["mean_target"], 2.0, delta=0.05)


if __name__ == "__main__":
    unittest.main()
