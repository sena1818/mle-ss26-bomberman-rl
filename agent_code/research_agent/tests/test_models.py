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
        for route in ("R01", "R02", "R07", "R08"):
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
