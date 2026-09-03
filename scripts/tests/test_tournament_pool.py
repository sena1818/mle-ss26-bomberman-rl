"""Tests for the tournament proxy pool.

The pool exists because every number this project selected on came from one
opponent table, and the tournament is played against none of the agents on it.
What the tests hold: every pool reaches every board and every train seed as its
own job, a frozen seat carries the digest of the checkpoint it will play, the
report pairs on the board seed, and the seed it selects is the one with the
highest mean over pools -- not over the table it was trained against.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import tournament_pool  # noqa: E402
from experiment_lib import write_json  # noqa: E402

TRAIN_SEEDS = (1001, 1002)
BOARDS = (4001, 4002, 4003)


def _source_run(root: Path) -> Path:
    run_dir = root / "source"
    write_json(run_dir / "experiment_config.snapshot.json", {
        "agent": {"name": "research_agent"}, "route": "R07",
        "resolved_runtime_config": {"feature_dimension": 2029}})
    for train_seed in TRAIN_SEEDS:
        agent_dir = run_dir / "jobs" / f"train_seed{train_seed}" / "agent"
        (agent_dir / "checkpoints").mkdir(parents=True)
        (agent_dir / "checkpoints" / f"R07_A06_loot-crate_seed{train_seed}_round10000_updates00000100.npz").write_bytes(b"w")
        (agent_dir / "latest_model.npz").write_bytes(b"w")
    return run_dir


def _finished(root: Path, name: str, score_by_pool_seed_board) -> Path:
    """A pool run whose jobs already produced stats; scores keyed (pool, train, board)."""
    run_dir = root / name
    write_json(run_dir / "experiment_config.snapshot.json", {"agent": {"name": "research_agent"}})
    for (pool, train_seed, board), score in score_by_pool_seed_board.items():
        job_id = f"eval_tournament_pool_{pool}_round10000_train{train_seed}_seed{board}_rep01"
        write_json(run_dir / "jobs" / job_id / "official_stats.json", {
            "by_agent": {
                "research_agent": {"rounds": 30, "score": score * 30, "coins": score * 30,
                                   "kills": 0.0, "suicides": 3.0},
                "rule_based_agent": {"rounds": 30, "score": 30.0, "coins": 30.0},
            }
        })
    return run_dir


class ScaffoldReachesEveryPoolBoardAndSeedTest(unittest.TestCase):
    def _scaffold(self, root: Path, **overrides) -> Path:
        arguments = Namespace(source_run="source", run_id="pool", checkpoint_round=10000,
                              train_seeds=None, board_seeds=list(BOARDS), repeats=1,
                              pools=["rb3", "mixed_neural"], scenario="classic", rounds=30,
                              ensemble=False)
        for key, value in overrides.items():
            setattr(arguments, key, value)
        with patch.object(tournament_pool, "RUNS_ROOT", root):
            tournament_pool.scaffold(arguments)
        return root / arguments.run_id

    def test_one_job_per_pool_board_and_train_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _source_run(root)
            run_dir = self._scaffold(root)
            jobs = [json.loads(path.read_text()) for path in sorted((run_dir / "job_parameters").glob("*.json"))]
        self.assertEqual(len(jobs), len(TRAIN_SEEDS) * 2 * len(BOARDS))
        self.assertEqual({job["pool"] for job in jobs}, {"rb3", "mixed_neural"})
        self.assertEqual({job["seed"] for job in jobs}, set(BOARDS))
        self.assertEqual({job["train_seed"] for job in jobs}, set(TRAIN_SEEDS))
        for job in jobs:
            self.assertIsNotNone(tournament_pool.JOB_ID.match(job["job_id"]), job["job_id"])
            self.assertEqual(job["scenario"], "classic")
            self.assertEqual(job["checkpoint_round"], 10000)
            self.assertIn("checkpoint_search_relpath", job)

    def test_a_frozen_seat_carries_its_digest_and_a_scripted_pool_carries_none(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _source_run(root)
            run_dir = self._scaffold(root)
            jobs = [json.loads(path.read_text()) for path in (run_dir / "job_parameters").glob("*.json")]
            provenance = json.loads((run_dir / "provenance.json").read_text())
        neural = [job for job in jobs if job["pool"] == "mixed_neural"]
        scripted = [job for job in jobs if job["pool"] == "rb3"]
        self.assertTrue(neural and scripted)
        for job in neural:
            self.assertEqual(job["opponents"], ["rule_based_agent", "frozen_agent", "frozen_agent_b"])
            self.assertEqual(set(job["frozen_opponents"]), {"frozen_agent", "frozen_agent_b"})
            for entry in job["frozen_opponents"].values():
                self.assertEqual(len(entry["sha256"]), 64)
        for job in scripted:
            self.assertNotIn("frozen_opponents", job)
        self.assertEqual(set(provenance["tournament_pool"]["pools"]), {"rb3", "mixed_neural"})

    def test_the_checkpoints_are_copied_in(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _source_run(root)
            run_dir = self._scaffold(root)
            for train_seed in TRAIN_SEEDS:
                copies = list((run_dir / "jobs" / f"train_seed{train_seed}" / "agent" / "checkpoints").glob("*.npz"))
                self.assertEqual(len(copies), 1)

    def test_an_unknown_pool_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _source_run(root)
            with self.assertRaises(SystemExit):
                self._scaffold(root, pools=["rb3", "nonsense"])

    def test_an_existing_destination_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _source_run(root)
            self._scaffold(root)
            with self.assertRaises(SystemExit):
                self._scaffold(root)


class ReportPairsOnTheBoardAndSelectsOverPoolsTest(unittest.TestCase):
    def _report(self, run_dir: Path, baseline: Path | None = None) -> dict:
        out = run_dir.parent / f"{run_dir.name}.json"
        tournament_pool.report(Namespace(run_dir=run_dir, baseline=baseline, json=out))
        return json.loads(out.read_text())

    def test_the_mean_over_pools_weights_every_pool_equally(self):
        scores = {}
        for board in BOARDS:
            for train_seed in TRAIN_SEEDS:
                scores[("rb3", train_seed, board)] = 4.0
                scores[("coin3", train_seed, board)] = 2.0
        with tempfile.TemporaryDirectory() as directory:
            summary = self._report(_finished(Path(directory), "candidate", scores))
        self.assertAlmostEqual(summary["mean_over_pools"]["score"], 3.0)
        self.assertAlmostEqual(summary["pools"]["rb3"]["score"], 4.0)

    def test_the_paired_difference_cancels_the_board(self):
        """Boards of different difficulty, one candidate 0.3 better on every board."""
        base, better = {}, {}
        for index, board in enumerate(BOARDS):
            for train_seed in TRAIN_SEEDS:
                difficulty = 2.0 + index
                base[("rb3", train_seed, board)] = difficulty
                better[("rb3", train_seed, board)] = difficulty + 0.3
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = self._report(_finished(root, "better", better), _finished(root, "base", base))
        self.assertAlmostEqual(summary["pools"]["rb3"]["delta_score"], 0.3)
        self.assertAlmostEqual(summary["overall_delta_score"], 0.3)
        # Identical differences on every board: the paired t is undefined (sd 0)
        # rather than small, which is what "the board is cancelled" looks like.
        self.assertTrue(summary["overall_t"] != summary["overall_t"] or abs(summary["overall_t"]) > 2.3)

    def test_the_selected_seed_is_the_best_over_pools_not_on_one_table(self):
        scores = {}
        for board in BOARDS:
            # Seed 1001 wins big on rb3 alone; seed 1002 is better on the other two.
            scores[("rb3", 1001, board)] = 6.0
            scores[("rb3", 1002, board)] = 4.0
            for pool in ("coin3", "mixed_weak"):
                scores[(pool, 1001, board)] = 2.0
                scores[(pool, 1002, board)] = 3.5
        with tempfile.TemporaryDirectory() as directory:
            summary = self._report(_finished(Path(directory), "candidate", scores))
        self.assertEqual(summary["selected_train_seed"], 1002)
        self.assertAlmostEqual(summary["train_seeds"]["1002"]["mean_over_pools"], (4.0 + 3.5 + 3.5) / 3)


if __name__ == "__main__":
    unittest.main()


class AnEnsembleIsOneCandidateNotFiveTest(unittest.TestCase):
    """--ensemble files every training seed under one pseudo-seat.

    The submission is one model, so an ensemble has to appear in the pool as one
    row, addressed by a manifest that travels with its members.
    """

    def _scaffold(self, root: Path, **overrides) -> Path:
        arguments = Namespace(source_run="source", run_id="pool", checkpoint_round=10000,
                              train_seeds=None, board_seeds=list(BOARDS), repeats=1,
                              pools=["rb3"], scenario="classic", rounds=30, ensemble=True)
        for key, value in overrides.items():
            setattr(arguments, key, value)
        with patch.object(tournament_pool, "RUNS_ROOT", root):
            tournament_pool.scaffold(arguments)
        return root / arguments.run_id

    def test_one_job_per_board_and_a_manifest_that_names_every_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _source_run(root)
            run_dir = self._scaffold(root)
            jobs = [json.loads(path.read_text()) for path in (run_dir / "job_parameters").glob("*.json")]
            manifests = list((run_dir / "jobs").glob(f"train_seed*/agent/*{tournament_pool.MANIFEST_SUFFIX}"))
            manifest = json.loads(manifests[0].read_text())
            members = list(manifests[0].parent.glob("members/*.npz"))
        self.assertEqual(len(jobs), len(BOARDS))          # one model, not five
        self.assertEqual({job["train_seed"] for job in jobs}, {tournament_pool.ENSEMBLE_TRAIN_SEED})
        self.assertEqual(len(manifests), 1)
        self.assertEqual(manifest["kind"], "ensemble")
        self.assertEqual(manifest["route"], "R07")
        self.assertEqual(len(manifest["members"]), len(TRAIN_SEEDS))
        self.assertEqual(len(members), len(TRAIN_SEEDS))
        self.assertEqual(sorted(manifest["provenance"]["train_seeds"]), sorted(TRAIN_SEEDS))
        for job in jobs:
            self.assertTrue(job["model_relpath"].endswith(tournament_pool.MANIFEST_SUFFIX))
            # The manifest is the model; a round search would find nothing.
            self.assertNotIn("checkpoint_search_relpath", job)
            self.assertIsNotNone(tournament_pool.JOB_ID.match(job["job_id"]))

    def test_the_ordinary_scaffold_still_files_one_row_per_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _source_run(root)
            run_dir = self._scaffold(root, ensemble=False)
            jobs = [json.loads(path.read_text()) for path in (run_dir / "job_parameters").glob("*.json")]
        self.assertEqual(len(jobs), len(TRAIN_SEEDS) * len(BOARDS))
        self.assertEqual({job["train_seed"] for job in jobs}, set(TRAIN_SEEDS))
        for job in jobs:
            self.assertIn("checkpoint_search_relpath", job)
