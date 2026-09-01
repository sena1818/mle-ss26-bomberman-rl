"""Tests for the QModel adapters behind the four main lines.

The shared contract is what matters: the runtime, the replay learner and the
target network all talk to a model through the same handful of methods, so a
new adapter is only usable if single and batch inference agree, a fitted batch
moves the selected head, a clone is independent, and a checkpoint round-trips.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from agent_code.research_agent.config import ACTIONS, EXPERIMENTS
from agent_code.research_agent.models import build_model, load_model
from agent_code.research_agent.models.linear_q import LinearQModel
from agent_code.research_agent.models.mlp_q import MLPQModel
from agent_code.research_agent.state import state_dimension

try:  # The M4 adapters are the only ones that need PyTorch.
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    TORCH_AVAILABLE = False


class SharedQModelContractTest(unittest.TestCase):
    """Run the same contract against every implemented adapter."""

    def adapters(self):
        for route in ("R01", "R02", "R02_1", "R07", "R08", "R09"):
            config = EXPERIMENTS[route]
            if config.network in {"cnn_mlp_q", "dueling_cnn_mlp_q"} and not TORCH_AVAILABLE:
                continue
            dimension = state_dimension(config.state_encoder)
            yield route, config, dimension, build_model(config, dimension, seed=3)

    def batch(self, dimension: int, size: int = 5):
        generator = np.random.default_rng(11)
        states = generator.normal(0.0, 0.3, size=(size, dimension)).astype(np.float32)
        actions = generator.integers(0, len(ACTIONS), size=size)
        targets = generator.normal(0.0, 1.0, size=size).astype(np.float32)
        return states, actions, targets

    def test_single_and_batch_inference_agree(self):
        for route, _, dimension, model in self.adapters():
            with self.subTest(route=route):
                states, _, _ = self.batch(dimension)
                single = np.stack([model.q_values(state) for state in states])
                np.testing.assert_allclose(single, model.q_values_batch(states), atol=1e-5)
                self.assertEqual(single.shape, (len(states), len(ACTIONS)))

    def test_fitting_a_batch_reduces_the_error_it_was_fitted_on(self):
        for route, _, dimension, model in self.adapters():
            with self.subTest(route=route):
                states, actions, targets = self.batch(dimension)
                first = np.abs(model.fit_batch(states, actions, targets)).mean()
                for _ in range(60):
                    last = np.abs(model.fit_batch(states, actions, targets)).mean()
                self.assertLess(last, first)

    def test_a_clone_starts_identical_and_then_moves_independently(self):
        for route, _, dimension, model in self.adapters():
            with self.subTest(route=route):
                states, actions, targets = self.batch(dimension)
                clone = model.clone()
                np.testing.assert_allclose(clone.q_values_batch(states), model.q_values_batch(states), atol=1e-6)
                for _ in range(5):
                    model.fit_batch(states, actions, targets)
                # A target network that moved with the online model would defeat
                # the entire point of having one.
                self.assertFalse(np.allclose(clone.q_values_batch(states), model.q_values_batch(states), atol=1e-6))
                clone.copy_parameters_from(model)
                np.testing.assert_allclose(clone.q_values_batch(states), model.q_values_batch(states), atol=1e-6)

    def test_a_checkpoint_round_trips_through_the_configured_loader(self):
        with tempfile.TemporaryDirectory() as temporary:
            for route, config, dimension, model in self.adapters():
                with self.subTest(route=route):
                    states, _, _ = self.batch(dimension)
                    path = Path(temporary) / f"{route}.npz"
                    model.save(path, metadata={"route": route})
                    reloaded = load_model(config, path)
                    np.testing.assert_allclose(reloaded.q_values_batch(states), model.q_values_batch(states), atol=1e-5)

    def test_model_initialization_is_reproducible_per_seed(self):
        for route, config, dimension, _ in self.adapters():
            with self.subTest(route=route):
                probe = np.zeros((1, dimension), dtype=np.float32)
                same = build_model(config, dimension, seed=7).q_values_batch(probe)
                np.testing.assert_allclose(build_model(config, dimension, seed=7).q_values_batch(probe), same)


class LinearAndMlpTest(unittest.TestCase):
    def test_a_single_row_batch_matches_the_online_update_exactly(self):
        # M2's replay arm must be comparable with its online arm, which is only
        # true if the two code paths compute the same gradient.
        state = np.random.default_rng(5).normal(size=44).astype(np.float32)
        for factory in (lambda: LinearQModel(44, seed=5, learning_rate=0.05),
                        lambda: MLPQModel(44, (16, 8), seed=5, learning_rate=0.05)):
            with self.subTest(model=factory().__class__.__name__):
                online, batched = factory(), factory()
                online.q_learning_update(state, 2, 1.7, None, None, 0.05, 0.95)
                batched.fit_batch(state[None, :], np.array([2]), np.array([1.7], dtype=np.float32))
                np.testing.assert_allclose(online.q_values(state), batched.q_values(state), atol=1e-6)

    def test_an_mlp_checkpoint_with_the_wrong_shape_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mlp.npz"
            MLPQModel(44, (8,), seed=1).save(path)
            with self.assertRaises(ValueError):
                load_model(EXPERIMENTS["R02"], path)

    def test_adam_huber_recipe_clips_an_outlier_update_and_exposes_diagnostics(self):
        model = MLPQModel(
            44, (16, 8), seed=3, learning_rate=1e-3,
            optimizer="adam", td_loss="huber", gradient_clip_norm=0.01,
        )
        states = np.ones((4, 44), dtype=np.float32)
        actions = np.array([0, 1, 2, 3])
        td_errors = model.fit_batch(states, actions, np.full(4, 100.0, dtype=np.float32))
        diagnostics = model.training_diagnostics()
        self.assertTrue(np.isfinite(td_errors).all())
        self.assertEqual(diagnostics["optimizer_steps"], 1)
        self.assertTrue(diagnostics["last_gradient_was_clipped"])
        self.assertGreater(float(diagnostics["last_gradient_l2_norm"]), 0.01)
        self.assertIsNotNone(diagnostics["last_hidden_zero_fraction"])


@unittest.skipUnless(TORCH_AVAILABLE, "the M4 adapters require PyTorch")
class CnnMlpQModelTest(unittest.TestCase):
    def setUp(self):
        self.dimension = state_dimension("board_egocentric_v1")

    def test_the_dueling_head_produces_mean_centred_advantages(self):
        from agent_code.research_agent.models.cnn_mlp_q import CnnMlpQModel

        model = CnnMlpQModel(self.dimension, dueling=True, seed=1)
        state = np.random.default_rng(2).normal(size=self.dimension).astype(np.float32)
        values = model.q_values(state)
        # V is identified only if the advantages are centred; without the shift
        # the two heads can drift by an arbitrary constant.
        network = model.network
        import torch

        from agent_code.research_agent.state import split_board_and_globals

        board, globals_ = split_board_and_globals(state[None, :])
        with torch.no_grad():
            fused = network.fused(torch.cat([
                network.board(torch.from_numpy(np.ascontiguousarray(board))),
                network.globals(torch.from_numpy(np.ascontiguousarray(globals_))),
            ], dim=1))
            state_value = float(network.value(fused))
        self.assertAlmostEqual(float(values.mean()), state_value, places=5)

    def test_a_checkpoint_refuses_to_load_into_the_other_head(self):
        from agent_code.research_agent.models.cnn_mlp_q import CnnMlpQModel

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "anchor.npz"
            CnnMlpQModel(self.dimension, dueling=False, seed=1).save(path)
            with self.assertRaises(ValueError):
                load_model(EXPERIMENTS["R08"], path)

    def test_a_state_of_the_wrong_width_is_refused_at_construction(self):
        from agent_code.research_agent.models.cnn_mlp_q import CnnMlpQModel

        with self.assertRaises(ValueError):
            CnnMlpQModel(44, seed=1)


if __name__ == "__main__":
    unittest.main()


class CnnHonoursItsDeclarationsTest(unittest.TestCase):
    """The route declares an optimizer, a loss, a clip and a step size.

    All four used to be hardcoded inside the model while the route declared
    them separately.  They agreed, which is the worst form of that bug: nothing
    was wrong, and nothing would have said so when it became wrong.  The step
    size was the one that mattered -- the runtime implements a schedule by
    assigning to ``model.learning_rate`` once a round, and a torch optimizer
    copies ``lr`` into its parameter groups at construction, so an L01 arm on
    this line would have run to completion and measured exactly nothing.
    """

    def _model(self, **overrides):
        from agent_code.research_agent.models.cnn_mlp_q import CnnMlpQModel
        return CnnMlpQModel(2029, hidden_layers=(256,), seed=0, **overrides)

    def test_setting_the_learning_rate_reaches_the_optimizer(self):
        model = self._model(learning_rate=2.5e-4)
        self.assertEqual(model.optimizer.param_groups[0]["lr"], 2.5e-4)
        model.learning_rate = 1e-5
        self.assertEqual(model.learning_rate, 1e-5)
        for group in model.optimizer.param_groups:
            self.assertEqual(group["lr"], 1e-5, "the schedule must reach the optimizer, not just the object")

    def test_the_learning_rate_schedule_actually_moves_a_cnn(self):
        """End to end against the real schedule the M3 line added."""
        from agent_code.research_agent.config import (
            EXPERIMENTS, learning_rate_for_training_round, validate_config)
        from dataclasses import replace
        config = validate_config(replace(EXPERIMENTS["R07"], learning_rate_schedule="L01"))
        model = self._model(learning_rate=config.learning_rate)
        early = learning_rate_for_training_round(config, 1, 5000)
        late = learning_rate_for_training_round(config, 5000, 5000)
        self.assertGreater(early, late, "L01 is a decay; this arm would be pointless otherwise")
        model.learning_rate = late
        self.assertEqual(model.optimizer.param_groups[0]["lr"], late)

    def test_the_declared_optimizer_and_clip_are_used(self):
        adam = self._model(optimizer="adam")
        sgd = self._model(optimizer="sgd")
        self.assertEqual(type(adam.optimizer).__name__, "Adam")
        self.assertEqual(type(sgd.optimizer).__name__, "SGD")
        self.assertEqual(self._model(gradient_clip_norm=2.0).gradient_clip_norm, 2.0)

    def test_an_undeclared_choice_fails_closed(self):
        for bad in ({"optimizer": "rmsprop"}, {"td_loss": "hinge"}, {"gradient_clip_norm": -1.0}):
            with self.assertRaises(ValueError, msg=f"{bad} should be refused"):
                self._model(**bad)

    def test_a_clone_carries_the_declarations(self):
        """The target network is a clone; it must not drift from the online net."""
        model = self._model(learning_rate=3e-4, optimizer="sgd", td_loss="mse", gradient_clip_norm=5.0)
        copy = model.clone()
        self.assertEqual(type(copy.optimizer).__name__, "SGD")
        self.assertEqual((copy.td_loss, copy.gradient_clip_norm), ("mse", 5.0))
        self.assertEqual(copy.optimizer.param_groups[0]["lr"], 3e-4)


@unittest.skipUnless(TORCH_AVAILABLE, "the hybrid route is an M4 capability and requires PyTorch")
class HybridRouteModelTest(unittest.TestCase):
    """R09 is R07's trunk with a wider scalar branch; the width follows the layout."""

    def test_the_hybrid_model_builds_reloads_and_clones(self):
        import tempfile

        from agent_code.research_agent.config import validate_config
        from agent_code.research_agent.state import state_dimension

        config = validate_config(EXPERIMENTS["R09"])
        dimension = state_dimension("hybrid_v1")
        model = build_model(config, dimension, seed=3)
        state = np.random.default_rng(0).random(dimension).astype(np.float32)
        self.assertEqual(model.q_values(state).shape, (len(ACTIONS),))
        linear_layers = [layer for layer in model.network.globals if layer.__class__.__name__ == "Linear"]
        self.assertEqual([layer.in_features for layer in linear_layers], [68, 128])
        narrow = build_model(validate_config(EXPERIMENTS["R07"]), state_dimension("board_egocentric_v2"), seed=3)
        narrow_layers = [layer for layer in narrow.network.globals if layer.__class__.__name__ == "Linear"]
        self.assertEqual([layer.in_features for layer in narrow_layers], [6])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hybrid.npz"
            model.save(path)
            reloaded = load_model(config, path)
        np.testing.assert_allclose(reloaded.q_values(state), model.q_values(state), rtol=0, atol=1e-6)
        np.testing.assert_allclose(model.clone().q_values(state), model.q_values(state), rtol=0, atol=1e-6)

    def test_the_hybrid_model_can_be_behaviour_cloned(self):
        from agent_code.research_agent.config import validate_config
        from agent_code.research_agent.state import state_dimension

        config = validate_config(EXPERIMENTS["R09"])
        dimension = state_dimension("hybrid_v1")
        model = build_model(config, dimension, seed=1)
        generator = np.random.default_rng(4)
        labels = generator.integers(0, 6, size=32)
        states = generator.normal(0.0, 0.05, size=(32, dimension)).astype(np.float32)
        # Write the label into the handcrafted tail, where the wide branch reads it.
        states[np.arange(32), 7 * 17 * 17 + 6 + labels] += 4.0
        losses = [model.fit_policy_batch(states, labels) for _ in range(80)]
        self.assertLess(losses[-1], losses[0])
        accuracy = float(np.mean(np.argmax(model.q_values_batch(states), axis=1) == labels))
        self.assertGreater(accuracy, 0.8)
