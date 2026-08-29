"""Tests for noisy networks (Fortunato et al. 2018).

Every way this can fail is quiet.  Noise that is drawn once and reused makes the
policy deterministic between gradient steps, so the arm explores nothing and
still trains.  A backward pass that descends through the mean weights instead of
the noisy ones sends the wrong signal to every layer below and still converges to
something.  Sigma gradients that never reach the optimizer leave the noise scale
frozen at its initialisation -- which is what happened here, because the
categorical head had its own copy of the backward loop.  None of those show up in
a score, so each one is pinned below.
"""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from agent_code.research_agent.config import (
    ACTIONS, EXPERIMENTS, EXPLORATION_SCHEDULES, ReplayConfig,
    epsilon_for_training_round, validate_config)
from agent_code.research_agent.models import build_model, load_model
from agent_code.research_agent.models.categorical_mlp_q import CategoricalMLPQModel
from agent_code.research_agent.models.mlp_q import MLPQModel

STATE_WIDTH = 8


def noisy_mlp(**kwargs) -> MLPQModel:
    settings = dict(seed=0, learning_rate=0.01, optimizer="adam", noisy=True)
    settings.update(kwargs)
    return MLPQModel(STATE_WIDTH, (16,), **settings)


class NoiseIsDrawnPerForwardPassTest(unittest.TestCase):
    def test_two_forward_passes_disagree(self):
        """If they agreed, the network would explore nothing between updates."""
        model = noisy_mlp()
        state = np.full(STATE_WIDTH, 0.3, dtype=np.float32)
        values = np.array([model.q_values(state) for _ in range(8)])
        self.assertGreater(values.std(axis=0).mean(), 1e-4)

    def test_disabling_the_noise_makes_it_deterministic_again(self):
        model = noisy_mlp()
        model.noise_enabled = False
        state = np.full(STATE_WIDTH, 0.3, dtype=np.float32)
        np.testing.assert_array_equal(model.q_values(state), model.q_values(state))

    def test_a_plain_model_is_untouched_by_any_of_this(self):
        """Every arm run before noisy nets existed has to take the same path."""
        plain, other = MLPQModel(STATE_WIDTH, (16,), seed=3), MLPQModel(STATE_WIDTH, (16,), seed=3)
        state = np.full(STATE_WIDTH, 0.3, dtype=np.float32)
        np.testing.assert_array_equal(plain.q_values(state), other.q_values(state))
        self.assertEqual(plain.weight_sigmas, [])
        batch = np.tile(state, (4, 1))
        actions = np.zeros(4, dtype=np.intp)
        targets = np.ones(4, dtype=np.float32)
        plain.fit_batch(batch, actions, targets)
        other.fit_batch(batch, actions, targets)
        np.testing.assert_array_equal(plain.q_values(state), other.q_values(state))


class TheNoiseScaleIsTrainedTest(unittest.TestCase):
    def _train(self, model, steps=200):
        batch = np.tile(np.full(STATE_WIDTH, 0.3, dtype=np.float32), (8, 1))
        actions = np.zeros(8, dtype=np.intp)
        targets = np.full(8, 2.0, dtype=np.float32)
        for _ in range(steps):
            model.fit_batch(batch, actions, targets)

    def test_sigma_moves(self):
        """Frozen sigma means the gradients never reached the optimizer."""
        model = noisy_mlp()
        before = [one.copy() for one in model.weight_sigmas]
        self._train(model)
        moved = max(float(np.abs(after - start).max())
                    for after, start in zip(model.weight_sigmas, before))
        self.assertGreater(moved, 1e-5)

    def test_the_scalar_and_categorical_heads_both_train_it(self):
        """The categorical head used to have its own backward loop and did not."""
        head = CategoricalMLPQModel(STATE_WIDTH, (16,), atoms=7, value_min=-1.0, value_max=5.0,
                                    dueling=True, noisy=True, seed=0, learning_rate=0.01,
                                    optimizer="adam")
        batch = np.tile(np.full(STATE_WIDTH, 0.3, dtype=np.float32), (8, 1))
        target = np.zeros((8, 7), dtype=np.float32)
        target[:, 2] = 1.0
        before = float(np.abs(head.weight_sigmas[0]).mean())
        for _ in range(200):
            head.fit_batch_distribution(batch, np.zeros(8, dtype=np.intp), target)
        self.assertNotAlmostEqual(float(np.abs(head.weight_sigmas[0]).mean()), before, places=5)

    def test_a_noisy_model_still_learns_its_target(self):
        model = noisy_mlp()
        self._train(model, steps=600)
        model.noise_enabled = False
        state = np.full(STATE_WIDTH, 0.3, dtype=np.float32)
        self.assertAlmostEqual(float(model.q_values(state)[0]), 2.0, delta=0.2)

    def test_the_noise_scale_is_reported_so_a_collapse_is_visible(self):
        """Sigma going to zero is noisy nets switching itself off mid-run."""
        model = noisy_mlp()
        self.assertIsNotNone(model.training_diagnostics()["mean_noise_scale"])
        self.assertIsNone(MLPQModel(STATE_WIDTH, (16,)).training_diagnostics()["mean_noise_scale"])


class NoisyPersistenceTest(unittest.TestCase):
    def test_a_saved_noisy_model_reloads_with_its_scales(self):
        model = noisy_mlp()
        for one in model.weight_sigmas:
            one += 0.01
        model.noise_enabled = False
        state = np.full(STATE_WIDTH, 0.3, dtype=np.float32)
        expected = model.q_values(state)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "noisy.npz"
            model.save(path)
            reloaded = MLPQModel.load(path, learning_rate=0.01, optimizer="adam")
        self.assertTrue(reloaded.noisy)
        reloaded.noise_enabled = False
        np.testing.assert_allclose(reloaded.q_values(state), expected, rtol=1e-6)
        for saved, restored in zip(model.weight_sigmas, reloaded.weight_sigmas):
            np.testing.assert_allclose(restored, saved, rtol=1e-6)

    def test_a_target_network_clone_carries_the_noise(self):
        model = noisy_mlp()
        clone = model.clone()
        self.assertTrue(clone.noisy)
        for saved, copied in zip(model.weight_sigmas, clone.weight_sigmas):
            np.testing.assert_array_equal(copied, saved)
        with self.assertRaises(ValueError):
            MLPQModel(STATE_WIDTH, (16,)).copy_parameters_from(model)


class NoisyIsDeclaredNotImpliedTest(unittest.TestCase):
    def test_e12_is_epsilon_zero_because_the_weights_do_the_exploring(self):
        self.assertEqual(EXPLORATION_SCHEDULES["E12"]["epsilon"], 0.0)
        config = validate_config(EXPERIMENTS["R02_11"])
        self.assertEqual(config.exploration_version, "E12")
        for round_number in (1, 2500, 5000):
            self.assertEqual(epsilon_for_training_round(config, round_number, 5000), 0.0)

    def test_noisy_without_e12_is_refused_and_e12_without_noisy_too(self):
        """Two exploration mechanisms at once is not a factor anybody could read."""
        with self.assertRaises(ValueError):
            validate_config(replace(EXPERIMENTS["R02_11"], exploration_version="E02"))
        with self.assertRaises(ValueError):
            validate_config(replace(EXPERIMENTS["R02_9"], exploration_version="E12"))

    def test_dueling_is_refused_on_a_scalar_head(self):
        with self.assertRaises(ValueError):
            validate_config(replace(EXPERIMENTS["R02_9"], dueling=True))

    def test_the_rainbow_route_builds_the_head_it_declares(self):
        config = validate_config(EXPERIMENTS["R02_11"])
        model = build_model(config, 62, seed=1)
        self.assertIsInstance(model, CategoricalMLPQModel)
        self.assertTrue(model.dueling and model.noisy)
        self.assertEqual(model.layer_sizes[-1], (len(ACTIONS) + 1) * config.atoms)

    def test_a_checkpoint_that_disagrees_about_dueling_or_noise_is_refused(self):
        config = validate_config(EXPERIMENTS["R02_11"])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rainbow.npz"
            build_model(config, 62, seed=1).save(path)
            plain = validate_config(replace(
                EXPERIMENTS["R02_10"], replay=ReplayConfig()))
            with self.assertRaises(ValueError):
                load_model(plain, path)


if __name__ == "__main__":
    unittest.main()
