"""Tests for the archive identity check.

The failure this guards against is silent by construction: a directory with the
wrong contents reads and parses perfectly, and only the recorded run_id
contradicts the name.  Each case below is one way the archive can lie.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import verify_run_archives  # noqa: E402


def make_run(name: str, recorded: str | None = "__same__", config: str = "a.json",
             nested: bool = False, summary: bool = True) -> Path:
    root = Path(tempfile.mkdtemp())
    run = root / name
    run.mkdir()
    if summary:
        payload = {} if recorded is None else {"run_id": name if recorded == "__same__" else recorded}
        (run / "evaluation_summary.json").write_text(json.dumps(payload), encoding="utf-8")
    (run / "provenance.json").write_text(
        json.dumps({"config_source": f"/repo/experiments/{config}"}), encoding="utf-8")
    if nested:
        (run / name).mkdir()
    return run


class VerifyRunArchivesTest(unittest.TestCase):
    def test_a_matching_run_id_is_accepted(self):
        self.assertIsNone(verify_run_archives.check_run(make_run("arm_a")))

    def test_a_foreign_run_id_is_reported_with_its_config(self):
        finding = verify_run_archives.check_run(make_run("arm_a", recorded="arm_b", config="b.json"))
        self.assertIsNotNone(finding)
        self.assertEqual(finding["recorded_run_id"], "arm_b")
        self.assertEqual(finding["config_source"], "b.json")

    def test_a_run_nested_inside_itself_is_reported(self):
        finding = verify_run_archives.check_run(make_run("arm_a", nested=True))
        self.assertIsNotNone(finding)
        self.assertIn("nested", finding["problem"])

    def test_a_summary_without_a_run_id_is_reported(self):
        finding = verify_run_archives.check_run(make_run("arm_a", recorded=None))
        self.assertIsNotNone(finding)
        self.assertIn("no run_id", finding["problem"])

    def test_an_unfinished_run_is_not_a_finding(self):
        # Pruned or still-running directories have no summary to contradict.
        self.assertIsNone(verify_run_archives.check_run(make_run("arm_a", summary=False)))

    def test_unreadable_json_is_reported_rather_than_raised(self):
        run = make_run("arm_a")
        (run / "evaluation_summary.json").write_text("{not json", encoding="utf-8")
        finding = verify_run_archives.check_run(run)
        self.assertIsNotNone(finding)
        self.assertIn("unreadable", finding["problem"])


if __name__ == "__main__":
    unittest.main()
