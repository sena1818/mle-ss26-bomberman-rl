"""Tests for the shared, model-independent experiment infrastructure."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import aggregate_results  # noqa: E402
from experiment_lib import ConfigError, Experiment, resolved_runtime_config, verify_job_provenance, write_json  # noqa: E402
from run_experiment import _archive_failed_attempt, build_jobs, load_context  # noqa: E402


def config(route: str = "R01") -> dict:
    return {
        "schema_version": 1,
        "experiment_id": "test_r01",
        "route": route,
        "agent": {"name": "research_agent", "model": "linear_q", "algorithm": "q_learning", "state_representation": "handcrafted_v1"},
        "reward_version": "A00",
        "training": {"scenario": "coin-heaven", "opponents": [], "seeds": [11, 12], "budget": {"rounds": 2, "checkpoint_every": 1}},
        "evaluation": {"scenario": "classic", "opponents": [], "seeds": [21, 22], "budget": {"rounds": 2, "checkpoint_every": 1}},
        "promotion": {"primary_metric": "score"},
    }


class ExperimentInfrastructureTest(unittest.TestCase):
    def test_snapshot_is_reloadable_and_jobs_have_private_artifact_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            tmp = Path(temporary)
            config_path = tmp / "config.json"
            write_json(config_path, config())
            experiment = Experiment.load(config_path)
            snapshot_path = tmp / "snapshot.json"
            write_json(snapshot_path, experiment.snapshot())
            self.assertEqual(Experiment.load(snapshot_path), experiment)
            jobs = build_jobs(experiment, tmp / "run")
            self.assertEqual(len(jobs), 6)
            self.assertEqual(len({job["artifact_relpath"] for job in jobs}), len(jobs))
            self.assertTrue(all(not Path(job["artifact_relpath"]).is_absolute() for job in jobs))
            self.assertTrue(all(job["model_relpath"] is None or not Path(job["model_relpath"]).is_absolute() for job in jobs))
            self.assertEqual({job["mode"] for job in jobs}, {"train", "eval"})

    def test_job_files_relocate_with_the_run_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_run = root / "mac" / "run_a"
            source_run.mkdir(parents=True)
            experiment = Experiment.load(self._write_config(source_run / "input.json"))
            write_json(source_run / "experiment_config.snapshot.json", experiment.snapshot())
            job = build_jobs(experiment, source_run)[0]
            write_json(source_run / "job_parameters" / "train_seed11.json", job)
            moved_run = root / "hetzner" / "run_a"
            shutil.copytree(source_run, moved_run)
            run_dir, resolved, _ = load_context(moved_run / "job_parameters" / "train_seed11.json")
            self.assertEqual(run_dir, moved_run.resolve())
            self.assertEqual(Path(resolved["artifact_dir"]), (moved_run / "jobs" / "train_seed11").resolve())

    def test_runtime_snapshot_records_r01_hyperparameters(self):
        with tempfile.TemporaryDirectory() as temporary:
            experiment = Experiment.load(self._write_config(Path(temporary) / "config.json"))
            runtime = resolved_runtime_config(experiment)
            self.assertEqual(runtime["config"]["learning_rate"], 0.02)
            self.assertEqual(runtime["config"]["discount"], 0.95)
            self.assertEqual(runtime["config"]["epsilon"], 0.15)
            self.assertEqual(runtime["config"]["safety_filter"], "legality_only")

    def test_provenance_rejects_a_worker_on_the_wrong_commit(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            write_json(run_dir / "provenance.json", {"git_commit": "expected", "worktree_dirty": False})
            with patch("experiment_lib.git_provenance", return_value={"git_commit": "different", "worktree_dirty": False}):
                with self.assertRaises(RuntimeError):
                    verify_job_provenance(run_dir)

    def test_retry_archives_only_a_completed_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run_a"
            job_dir = run_dir / "jobs" / "train_seed11"
            job_dir.mkdir(parents=True)
            write_json(job_dir / "completion.json", {"exit_code": 1})
            (job_dir / "stderr.log").write_text("failure\n", encoding="utf-8")
            _archive_failed_attempt(run_dir, job_dir)
            archived = run_dir / "failed_attempts" / "train_seed11" / "attempt01"
            self.assertTrue((archived / "stderr.log").is_file())
            self.assertFalse(job_dir.exists())

    def test_declared_but_unimplemented_route_is_not_silently_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            payload = config("R02")
            payload["agent"]["model"] = "mlp_q"
            write_json(path, payload)
            with self.assertRaises(ConfigError):
                Experiment.load(path).require_implemented()

    def test_aggregate_and_promotion_use_fixed_rule(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run_a"
            run_dir.mkdir()
            experiment = Experiment.load(self._write_config(run_dir / "input.json"))
            write_json(run_dir / "experiment_config.snapshot.json", experiment.snapshot())
            jobs = build_jobs(experiment, run_dir)
            write_json(run_dir / "jobs.json", jobs)
            for job in jobs:
                job_dir = run_dir / job["artifact_relpath"]
                if job["mode"] != "eval":
                    checkpoint = job_dir / "agent" / "latest_model.npz"
                    checkpoint.parent.mkdir(parents=True)
                    checkpoint.write_bytes(b"model")
                    continue
                (job_dir / "agent").mkdir(parents=True)
                write_json(job_dir / "official_stats.json", {
                    "by_agent": {"research_agent": {"score": 6, "coins": 3, "kills": 1, "suicides": 1, "invalid": 0, "steps": 10}},
                    "by_round": {"round one": {"coins": 1, "kills": 1, "suicides": 0, "steps": 4}, "round two": {"coins": 2, "kills": 0, "suicides": 1, "steps": 6}},
                })
                (job_dir / "agent" / "agent.jsonl").write_text(
                    json.dumps({"kind": "action", "selected_action_was_legal": True, "inference_seconds": 0.01}) + "\n",
                    encoding="utf-8",
                )
            summary = aggregate_results.aggregate(run_dir)
            self.assertEqual(summary["metrics"]["score"]["count"], 4)
            self.assertEqual(summary["metrics"]["invalid_actions"]["mean"], 0.0)
            self.assertEqual(len(summary["checkpoint_candidates"]), 2)
            original_root = aggregate_results.PROMOTION_ROOT
            try:
                aggregate_results.PROMOTION_ROOT = Path(temporary) / "promoted"
                self.assertTrue(aggregate_results.maybe_promote(summary))
                self.assertTrue((aggregate_results.PROMOTION_ROOT / "active_model.npz").is_file())
                weaker = json.loads(json.dumps(summary))
                weaker["run_id"] = "later_but_weaker"
                weaker["metrics"]["score"]["mean"] = -1.0
                self.assertFalse(aggregate_results.maybe_promote(weaker))
            finally:
                aggregate_results.PROMOTION_ROOT = original_root

    @staticmethod
    def _write_config(path: Path) -> Path:
        write_json(path, config())
        return path


if __name__ == "__main__":
    unittest.main()
