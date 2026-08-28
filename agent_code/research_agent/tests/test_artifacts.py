"""Isolation contracts for per-job agent artifacts."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_code.research_agent.artifacts import append_jsonl, artifact_root, checkpoint_path, latest_model_path, model_path
from agent_code.research_agent.config import active_config


class ArtifactIsolationTest(unittest.TestCase):
    def test_all_agent_writes_follow_the_explicit_job_artifact_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            job_artifacts = Path(temporary) / "job_a" / "agent"
            environment = {
                "BOMBERMAN_ARTIFACT_DIR": str(job_artifacts),
                "BOMBERMAN_RUN_ID": "run_a_job_a",
                "BOMBERMAN_SCENARIO": "classic",
                "BOMBERMAN_SEED": "17",
            }
            with patch.dict(os.environ, environment, clear=False):
                self.assertEqual(artifact_root(), job_artifacts.resolve())
                self.assertEqual(latest_model_path(), job_artifacts.resolve() / "latest_model.npz")
                self.assertTrue(checkpoint_path(active_config(), 2, 9).is_relative_to(job_artifacts.resolve()))
                append_jsonl("test", {"ok": True})
            records = (job_artifacts / "agent.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(json.loads(records[0])["run_id"], "run_a_job_a")

    def test_evaluation_uses_a_packaged_model_when_a_job_did_not_select_one(self):
        with patch.dict(os.environ, {"BOMBERMAN_MODEL_PATH": ""}, clear=False):
            self.assertEqual(model_path().name, "model.npz")
            self.assertEqual(model_path().parent.name, "research_agent")


if __name__ == "__main__":
    unittest.main()


class ArtifactLogSyscallPatternTest(unittest.TestCase):
    """One open per log, not one per record.

    This agent writes a record per action: about 1.1 million per training job
    and 5.7 million across a five-seed arm.  Reopening the file each time costs
    nothing on a laptop, where the page cache absorbs it, and is a metadata
    operation against a cluster-wide shared server on a parallel filesystem.
    The records still flush immediately, so nothing about what a reader sees
    changes -- only the syscall pattern.
    """

    def setUp(self):
        from agent_code.research_agent import artifacts
        artifacts.close_artifact_logs()

    def test_a_thousand_records_open_the_file_once(self):
        from agent_code.research_agent import artifacts
        with tempfile.TemporaryDirectory() as temporary:
            os.environ["BOMBERMAN_ARTIFACT_DIR"] = temporary
            os.environ["BOMBERMAN_RUN_ID"] = "syscall_probe"
            real_open = Path.open
            opens = []

            def counting_open(self, *args, **kwargs):
                if self.name == "agent.jsonl":
                    opens.append(self)
                return real_open(self, *args, **kwargs)

            Path.open = counting_open
            try:
                for index in range(1000):
                    artifacts.append_jsonl("action", {"step": index})
            finally:
                Path.open = real_open
            self.assertEqual(len(opens), 1, f"agent.jsonl was opened {len(opens)} times for 1000 records")

    def test_every_record_is_readable_immediately(self):
        """The buffering must not delay what a reader sees."""
        from agent_code.research_agent import artifacts
        with tempfile.TemporaryDirectory() as temporary:
            os.environ["BOMBERMAN_ARTIFACT_DIR"] = temporary
            os.environ["BOMBERMAN_RUN_ID"] = "flush_probe"
            for index in range(5):
                artifacts.append_jsonl("action", {"step": index})
                lines = (Path(temporary) / "agent.jsonl").read_text(encoding="utf-8").splitlines()
                self.assertEqual(len(lines), index + 1, "a record must be visible as soon as it is written")

    def tearDown(self):
        from agent_code.research_agent import artifacts
        artifacts.close_artifact_logs()
        os.environ.pop("BOMBERMAN_ARTIFACT_DIR", None)
        os.environ.pop("BOMBERMAN_RUN_ID", None)
