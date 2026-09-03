"""Tests for the evaluation-only ensemble.

The claims that carry it: the average is over action *values* and not over
argmaxes, a member cannot change without the manifest saying so, exploration
that is switched off reaches every member, and nothing about it can be trained
by accident.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from agent_code.research_agent.config import ACTIONS, EXPERIMENTS, validate_config
from agent_code.research_agent.models import build_model, load_model
from agent_code.research_agent.models.ensemble import (
    MANIFEST_SUFFIX, EnsembleQModel, write_manifest)
from agent_code.research_agent.state import state_dimension


ROUTE = "R02_9"


class _Constant:
    """A member with fixed Q-values, so the arithmetic is checkable by hand."""

    model_type = "stub"

    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)
        self.input_dim = len(values)
        self.noise_enabled = True
        self.noisy = True

    def q_values(self, state):
        return self.values.copy()

    def q_values_batch(self, states):
        return np.tile(self.values, (len(np.atleast_2d(states)), 1))


class TheAverageIsOverValuesNotVotesTest(unittest.TestCase):
    def test_a_confident_member_outweighs_two_indifferent_ones(self):
        """A majority vote would pick UP; the values say BOMB, and so does the mean.

        Two members mildly prefer UP, one strongly prefers BOMB. Averaging the
        argmaxes discards exactly the information that separates 'nearly
        indifferent' from 'certain', which on a Q function is the calibrated
        part.
        """
        mild_a = _Constant([0.11, 0, 0, 0, 0, 0.10])
        mild_b = _Constant([0.11, 0, 0, 0, 0, 0.10])
        certain = _Constant([0.00, 0, 0, 0, 0, 3.00])
        votes = [int(np.argmax(m.q_values(None))) for m in (mild_a, mild_b, certain)]
        self.assertEqual(votes.count(ACTIONS.index("UP")), 2)  # the vote would say UP
        ensemble = EnsembleQModel([mild_a, mild_b, certain], manifest_path=Path("x"), route=ROUTE)
        self.assertEqual(int(np.argmax(ensemble.q_values(None))), ACTIONS.index("BOMB"))

    def test_the_mean_is_exact_for_values_and_batches(self):
        members = [_Constant([1, 2, 3, 4, 5, 6]), _Constant([3, 2, 1, 0, -1, -2])]
        ensemble = EnsembleQModel(members, manifest_path=Path("x"), route=ROUTE)
        np.testing.assert_allclose(ensemble.q_values(None), [2, 2, 2, 2, 2, 2], atol=1e-6)
        batch = ensemble.q_values_batch(np.zeros((3, 6), dtype=np.float32))
        self.assertEqual(batch.shape, (3, len(ACTIONS)))
        np.testing.assert_allclose(batch[0], [2, 2, 2, 2, 2, 2], atol=1e-6)

    def test_switching_noise_off_reaches_every_member(self):
        members = [_Constant([1] * 6), _Constant([1] * 6)]
        ensemble = EnsembleQModel(members, manifest_path=Path("x"), route=ROUTE)
        self.assertTrue(ensemble.noisy and ensemble.noise_enabled)
        ensemble.noise_enabled = False
        self.assertFalse(any(member.noise_enabled for member in members))
        self.assertFalse(ensemble.noise_enabled)

    def test_it_refuses_every_training_path(self):
        ensemble = EnsembleQModel([_Constant([1] * 6)], manifest_path=Path("x"), route=ROUTE)
        for call in (lambda: ensemble.fit_batch(None, None, None),
                     ensemble.clone,
                     lambda: ensemble.copy_parameters_from(None),
                     lambda: ensemble.save(Path("x"))):
            with self.assertRaises(NotImplementedError):
                call()

    def test_members_of_different_kinds_are_refused(self):
        one, other = _Constant([1] * 6), _Constant([1] * 6)
        other.model_type = "something_else"
        with self.assertRaises(ValueError):
            EnsembleQModel([one, other], manifest_path=Path("x"), route=ROUTE)


class AManifestIsProvenanceTest(unittest.TestCase):
    def _members(self, directory: Path, count: int = 3) -> tuple[list[Path], object]:
        config = validate_config(EXPERIMENTS[ROUTE])
        dimension = state_dimension(config.state_encoder)
        paths = []
        for index in range(count):
            model = build_model(config, dimension, seed=index)
            path = directory / f"member{index}.npz"
            model.save(path)
            paths.append(path)
        return paths, config

    def test_a_manifest_round_trips_through_the_ordinary_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, config = self._members(root)
            manifest = root / f"members{MANIFEST_SUFFIX}"
            write_manifest(manifest, route=ROUTE, members=paths,
                           input_dim=state_dimension(config.state_encoder))
            # The point of the suffix dispatch: callers hand over a path.
            model = load_model(config, manifest)
        self.assertIsInstance(model, EnsembleQModel)
        self.assertEqual(len(model.members), 3)
        self.assertIn("ensemble[3x", model.model_type)

    def test_the_ensemble_is_the_mean_of_its_members_on_a_real_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, config = self._members(root)
            manifest = root / f"members{MANIFEST_SUFFIX}"
            write_manifest(manifest, route=ROUTE, members=paths,
                           input_dim=state_dimension(config.state_encoder))
            ensemble = load_model(config, manifest)
            singles = [load_model(config, path) for path in paths]
        state = np.random.default_rng(0).random(state_dimension(config.state_encoder)).astype(np.float32)
        expected = np.mean([single.q_values(state) for single in singles], axis=0)
        np.testing.assert_allclose(ensemble.q_values(state), expected, rtol=0, atol=1e-6)

    def test_a_swapped_member_fails_the_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, config = self._members(root)
            manifest = root / f"members{MANIFEST_SUFFIX}"
            write_manifest(manifest, route=ROUTE, members=paths,
                           input_dim=state_dimension(config.state_encoder))
            build_model(config, state_dimension(config.state_encoder), seed=99).save(paths[1])
            with self.assertRaises(ValueError) as caught:
                load_model(config, manifest)
        self.assertIn("digest", str(caught.exception))

    def test_a_manifest_of_another_route_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, config = self._members(root)
            manifest = root / f"members{MANIFEST_SUFFIX}"
            write_manifest(manifest, route="R02_11", members=paths,
                           input_dim=state_dimension(config.state_encoder))
            with self.assertRaises(ValueError) as caught:
                load_model(config, manifest)
        self.assertIn("R02_11", str(caught.exception))

    def test_a_member_outside_the_manifest_directory_is_refused(self):
        """A manifest and its members have to travel together, or a submitted agent breaks."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, config = self._members(root, count=1)
            nested = root / "deeper"
            nested.mkdir()
            with self.assertRaises(ValueError):
                write_manifest(nested / f"m{MANIFEST_SUFFIX}", route=ROUTE, members=paths,
                               input_dim=state_dimension(config.state_encoder))

    def test_the_manifest_records_every_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, config = self._members(root)
            manifest = root / f"members{MANIFEST_SUFFIX}"
            written = write_manifest(manifest, route=ROUTE, members=paths,
                                     input_dim=state_dimension(config.state_encoder))
            reloaded = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(written, reloaded)
        self.assertEqual(len(reloaded["members"]), 3)
        for entry in reloaded["members"]:
            self.assertEqual(len(entry["sha256"]), 64)
            self.assertNotIn("/", entry["path"])
