"""Tests for the exported tournament agent.

The submission is one directory copied into somebody else's checkout with none
of this project's environment variables set. What these hold: the export pins
the route rather than falling back to the linear baseline, addresses its weights
relative to itself, drops the per-action log, carries nothing from the research
tree that a match does not need, and refuses to overwrite an existing folder.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
ROOT_DIR = SCRIPTS.parent
for path in (str(SCRIPTS), str(ROOT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import package_submission  # noqa: E402
from agent_code.research_agent.models.ensemble import MANIFEST_SUFFIX, write_manifest  # noqa: E402

CHECKPOINT = ROOT_DIR / "frozen_opponents" / "R02_9_eps0_seed1001_round10000.npz"
ROUTE = "R02_9"


class TheExportIsSelfContainedTest(unittest.TestCase):
    def _export(self, root: Path, **overrides) -> Path:
        arguments = dict(route=ROUTE, model=CHECKPOINT, name="tournament_agent",
                         destination_root=root)
        arguments.update(overrides)
        return package_submission.export(**arguments)

    def test_the_route_is_pinned_rather_than_left_to_the_r01_fallback(self):
        """Without this the framework's import would silently play the linear baseline."""
        with tempfile.TemporaryDirectory() as directory:
            package = self._export(Path(directory))
            declaration = json.loads((package / "submission.json").read_text())
        self.assertEqual(declaration["route"], ROUTE)
        self.assertEqual(declaration["model"], "model.npz")
        self.assertEqual(declaration["network"], "mlp_q")
        self.assertEqual(declaration["state_representation"], "handcrafted_v3")
        self.assertEqual(declaration["state_dimension"], 62)
        self.assertFalse(declaration["action_log"])

    def test_the_pinned_route_reaches_the_agent_with_no_variables_set(self):
        """config.active_config reads the file only when BOMBERMAN_EXPERIMENT is absent."""
        import os
        from unittest.mock import patch

        from agent_code.research_agent import config as config_module

        with tempfile.TemporaryDirectory() as directory:
            package = self._export(Path(directory))
            declaration_path = package / "submission.json"
            with patch.object(config_module, "__file__", str(package / "config.py")):
                environment = {k: v for k, v in os.environ.items() if not k.startswith("BOMBERMAN_")}
                with patch.dict(os.environ, environment, clear=True):
                    self.assertEqual(config_module.submission_declaration()["route"], ROUTE)
                    self.assertEqual(config_module.active_config().name, ROUTE)
                # An experiment job still wins, so no finished run changes meaning.
                with patch.dict(os.environ, {**environment, "BOMBERMAN_EXPERIMENT": "R01"}, clear=True):
                    self.assertEqual(config_module.active_config().name, "R01")
            self.assertTrue(declaration_path.is_file())

    def test_the_weights_are_addressed_relative_to_the_agent_directory(self):
        """An absolute path is the most common way a submitted agent fails their setup."""
        with tempfile.TemporaryDirectory() as directory:
            package = self._export(Path(directory))
            declaration = json.loads((package / "submission.json").read_text())
            self.assertFalse(Path(declaration["model"]).is_absolute())
            self.assertNotIn("..", Path(declaration["model"]).parts)
            self.assertTrue((package / declaration["model"]).is_file())

    def test_nothing_from_the_research_tree_travels_that_a_match_does_not_need(self):
        with tempfile.TemporaryDirectory() as directory:
            package = self._export(Path(directory))
            shipped = {str(path.relative_to(package)) for path in package.rglob("*") if path.is_file()}
        self.assertIn("callbacks.py", shipped)
        self.assertIn("runtime/experiment.py", shipped)
        self.assertIn("models/mlp_q.py", shipped)
        for unwanted in ("tests", "artifacts", "logs", "__pycache__"):
            self.assertFalse([name for name in shipped if name.startswith(unwanted + "/")],
                             f"{unwanted} was shipped")
        self.assertFalse([name for name in shipped if name.endswith(".pyc")])

    def test_an_existing_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._export(root)
            with self.assertRaises(SystemExit):
                self._export(root)

    def test_an_ensemble_travels_with_its_members_and_stays_relative(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            members_dir = root / "src" / "members"
            members_dir.mkdir(parents=True)
            for index in range(2):
                (members_dir / f"m{index}.npz").write_bytes(CHECKPOINT.read_bytes())
            manifest = root / "src" / f"members{MANIFEST_SUFFIX}"
            write_manifest(manifest, route=ROUTE,
                           members=sorted(members_dir.glob("*.npz")), input_dim=62)
            package = self._export(root / "out", model=manifest)
            declaration = json.loads((package / "submission.json").read_text())
            shipped = json.loads((package / declaration["model"]).read_text())
            members = sorted((package / "members").glob("*.npz"))
            self.assertTrue(declaration["model"].endswith(MANIFEST_SUFFIX))
            self.assertEqual(len(members), 2)
            for entry in shipped["members"]:
                self.assertTrue(entry["path"].startswith("members/"))
                self.assertTrue((package / entry["path"]).is_file())


class TheExportIsVerifiedByPlayingTest(unittest.TestCase):
    """The one test that runs the package the way the tournament will."""

    def test_it_plays_from_a_tree_holding_nothing_else_of_this_project(self):
        with tempfile.TemporaryDirectory() as directory:
            package = package_submission.export(
                route=ROUTE, model=CHECKPOINT, name="tournament_agent",
                destination_root=Path(directory))
            report = package_submission.verify(
                package, rounds=2, opponents=("rule_based_agent",), scenario="classic")
        self.assertEqual(report["rounds"], 2)
        self.assertGreater(report["steps"], 0)
        self.assertEqual(report["timeouts"], 0)
        # The budget is per step, so the worst decision is the one that matters.
        self.assertLess(report["max_decision_seconds"], 0.25)
        self.assertGreater(report["decisions_timed"], 100)
        # A packaged agent writes no per-action record: nothing in the
        # tournament reads it and it would only grow inside their directory.
        self.assertFalse(report["packaged_action_log_written"])


if __name__ == "__main__":
    unittest.main()
