"""Tests for the shared, model-independent experiment infrastructure."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import aggregate_results  # noqa: E402
from experiment_lib import ConfigError, Experiment, write_json  # noqa: E402
from run_experiment import build_jobs  # noqa: E402


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
            self.assertEqual(len({job["artifact_dir"] for job in jobs}), len(jobs))
            self.assertEqual({job["mode"] for job in jobs}, {"train", "eval"})

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
                if job["mode"] != "eval":
                    checkpoint = Path(job["artifact_dir"]) / "agent" / "latest_model.npz"
                    checkpoint.parent.mkdir(parents=True)
                    checkpoint.write_bytes(b"model")
                    continue
                job_dir = Path(job["artifact_dir"])
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
