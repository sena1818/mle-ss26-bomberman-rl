"""Tests for the M4 behaviour-cloning warm start.

Two claims carry the whole increment and both are checked here rather than
asserted in prose: cross entropy over the Q head really does fit the
demonstrator's action, and rescaling that head afterwards really does leave the
cloned policy alone.  If the second claim failed, a warm start would quietly
discard the cloning it just paid for.
"""

from __future__ import annotations

import unittest

import numpy as np

from agent_code.research_agent.state import state_dimension

try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "behaviour cloning is an M4 capability and requires PyTorch")
class BehaviourCloningTest(unittest.TestCase):
    def setUp(self):
        from agent_code.research_agent.models.cnn_mlp_q import CnnMlpQModel

        self.dimension = state_dimension("board_egocentric_v1")
        self.model_class = CnnMlpQModel
        generator = np.random.default_rng(4)
        # A separable toy task: the label is written into the first channel, so
        # a working supervised path has to reach well above chance quickly.
        self.labels = generator.integers(0, 6, size=24)
        self.states = generator.normal(0.0, 0.05, size=(24, self.dimension)).astype(np.float32)
        self.states[np.arange(24), self.labels] += 4.0

    def _accuracy(self, model) -> float:
        return float(np.mean(np.argmax(model.q_values_batch(self.states), axis=1) == self.labels))

    def test_cross_entropy_fitting_learns_the_demonstrated_action(self):
        model = self.model_class(self.dimension, seed=1, learning_rate=1e-3)
        before = self._accuracy(model)
        losses = [model.fit_policy_batch(self.states, self.labels) for _ in range(60)]
        self.assertLess(losses[-1], losses[0])
        self.assertGreater(self._accuracy(model), max(before, 0.8))

    def test_rescaling_the_head_preserves_every_greedy_choice(self):
        for dueling in (False, True):
            with self.subTest(dueling=dueling):
                model = self.model_class(self.dimension, dueling=dueling, seed=1, learning_rate=1e-3)
                for _ in range(20):
                    model.fit_policy_batch(self.states, self.labels)
                before = np.argmax(model.q_values_batch(self.states), axis=1)
                magnitude_before = np.abs(model.q_values_batch(self.states)).mean()
                model.rescale_head(0.1)
                after = np.argmax(model.q_values_batch(self.states), axis=1)
                np.testing.assert_array_equal(before, after)
                self.assertAlmostEqual(
                    float(np.abs(model.q_values_batch(self.states)).mean()),
                    float(magnitude_before) * 0.1,
                    places=4,
                )

    def test_a_non_positive_rescaling_is_refused(self):
        model = self.model_class(self.dimension, seed=1)
        # A negative factor would flip the ordering, i.e. clone the *worst*
        # demonstrated action.  Refusing beats silently inverting the policy.
        for factor in (0.0, -1.0):
            with self.subTest(factor=factor), self.assertRaises(ValueError):
                model.rescale_head(factor)

    def test_a_cloned_model_still_satisfies_the_shared_batch_contract(self):
        model = self.model_class(self.dimension, seed=1, learning_rate=1e-3)
        model.fit_policy_batch(self.states, self.labels)
        model.rescale_head(0.5)
        # The warm start hands this model straight to the replay learner, so the
        # TD path must keep working after a supervised phase and a rescaling.
        td_errors = model.fit_batch(self.states, self.labels, np.zeros(len(self.labels), dtype=np.float32))
        self.assertEqual(td_errors.shape, (len(self.labels),))
        self.assertTrue(np.all(np.isfinite(td_errors)))


if __name__ == "__main__":
    unittest.main()
