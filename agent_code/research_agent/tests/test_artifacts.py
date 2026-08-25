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
