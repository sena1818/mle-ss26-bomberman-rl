"""Tests for the gate that blocks four 4.4-hour arms.

The rule it enforces is pre-registered in docs/06 section 8: a candidate step
size wins only by beating the anchor's final pooled VALIDATION score by more
than the 4.7 resolution of the five-seed spread; otherwise the anchor's rate
stands.  A gate nobody tests is a gate nobody can rely on.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import decide_learning_rate as decision  # noqa: E402
from experiment_lib import write_json  # noqa: E402


def summary(points: list[tuple[int, float]]) -> dict:
    return {"evaluation_suites": {"primary": {"validation_checkpoint_curve": [
        {"checkpoint_round": one, "metrics": {"score": {"mean": value, "count": 3}}}
        for one, value in points]}}}


class DecisionRuleTest(unittest.TestCase):
    def _run(self, directory: Path, points, rate):
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / "evaluation_summary.json", summary(points))
        self._rates[directory.name] = rate
        return directory

    def setUp(self):
        self._rates = {}
        self._real = decide_learning_rate_of = decision.learning_rate_of
        decision.learning_rate_of = lambda path: self._rates[Path(path).name]

    def tearDown(self):
        decision.learning_rate_of = self._real

    def test_a_candidate_must_beat_the_anchor_by_the_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            anchor = self._run(root / "m4_anchor_x", [(1000, 2.0), (2000, 5.0), (4000, 10.0)], 2.5e-4)
            # +4.0 is a real improvement and still inside the noise the
            # five-seed spread implies, so the rule keeps the anchor.
            near = self._run(root / "m4_lr1e4_x", [(1000, 2.0), (2000, 6.0), (4000, 14.0)], 1e-4)
            a, c = decision.describe(anchor), decision.describe(near)
            margin = c["final_pooled_validation_score"] - a["final_pooled_validation_score"]
            self.assertLess(margin, decision.RESOLUTION)

    def test_the_window_is_the_mean_of_the_last_two_not_the_final_point(self):
        with tempfile.TemporaryDirectory() as tmp:
            described = decision.describe(
                self._run(Path(tmp) / "m4_anchor_x", [(1, 0.0), (2, 0.0), (3, 10.0), (4, 20.0)], 2.5e-4))
            self.assertEqual(described["final_pooled_validation_score"], 15.0)
            self.assertEqual(described["early_pooled_validation_score"], 0.0)

    def test_a_run_with_one_checkpoint_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                decision.describe(self._run(Path(tmp) / "m4_anchor_x", [(1000, 3.0)], 2.5e-4))

    def test_runs_that_are_not_this_comparison_are_refused(self):
        """A decision over the wrong runs still writes a decision file."""
        anchor = {"run": "m4_anchor_x", "learning_rate": 5e-4}      # wrong rate for an anchor
        with self.assertRaises(SystemExit):
            decision.check_identity(anchor, [{"run": "m4_lr1e4_x", "learning_rate": 1e-4},
                                             {"run": "m4_lr5e4_x", "learning_rate": 5e-4}])

    def test_a_missing_candidate_is_refused(self):
        anchor = {"run": "m4_anchor_x", "learning_rate": 2.5e-4}
        with self.assertRaises(SystemExit):
            decision.check_identity(anchor, [{"run": "m4_lr1e4_x", "learning_rate": 1e-4}])


class VerifyGateTest(unittest.TestCase):
    """The half that actually blocks the increments."""

    def test_a_missing_decision_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(decision.verify(Path(tmp)), 1)

    def test_a_corrupt_decision_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "learning_rate_decision.json").write_text("{not json", encoding="utf-8")
            self.assertEqual(decision.verify(Path(tmp)), 1)

    def test_a_decision_the_configs_do_not_carry_is_refused(self):
        """The P0: printing the mismatch and continuing is not a gate.

        The shipped downstream configs currently carry the route default, so a
        decision naming anything else must be refused -- which is exactly the
        state the line is in before someone applies a decision.
        """
        with tempfile.TemporaryDirectory() as tmp:
            write_json(Path(tmp) / "learning_rate_decision.json",
                       {"decided_learning_rate": 1e-4})
            self.assertEqual(decision.verify(Path(tmp)), 1)

    def test_a_decision_the_configs_do_carry_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            from experiment_lib import Experiment, resolved_runtime_config
            current = resolved_runtime_config(Experiment.load(
                ROOT / "experiments" / f"{decision.DOWNSTREAM_ARMS[0]}.json"))["config"]["learning_rate"]
            write_json(Path(tmp) / "learning_rate_decision.json",
                       {"decided_learning_rate": current})
            self.assertEqual(decision.verify(Path(tmp)), 0)


if __name__ == "__main__":
    unittest.main()


class RoundEndGateVerdictTest(unittest.TestCase):
    """The checker's verdict on each shape of round_end record.

    The bug this exists to prevent: the checker compared the identifier
    "TASK_COMPLETE" against the value the runtime serialises, "task_complete",
    so every benign smoke event was classified as unexplained and hard-failed
    -- the exact opposite of what the split was written to do. Source-reading
    tests did not notice.
    """

    def setUp(self):
        import check_pilot
        self.check_pilot = check_pilot

    def _verdicts(self, **round_end):
        row = {"round": 10, "updates_this_round": 5, "gradient_steps_this_round": 1,
               "epsilon": 0.05, "round_end_mispredictions": 0,
               "round_end_predicted_early": 0, "round_end_unpredicted": 0,
               "round_end_predicted_early_reasons": {}, **round_end}
        checks = self.check_pilot.check_training_job(Path("train_seed1"), [row], None)
        return {check.name.split(": ", 1)[1]: check for check in checks}

    def test_a_benign_smoke_event_passes(self):
        from agent_code.research_agent.runtime.experiment import TASK_COMPLETE
        verdicts = self._verdicts(round_end_mispredictions=11, round_end_predicted_early=11,
                                  round_end_predicted_early_reasons={TASK_COMPLETE: 11})
        self.assertTrue(verdicts["every early round-end prediction is the documented smoke approximation"].passed)
        self.assertTrue(verdicts["no round ended without being predicted"].passed)
        self.assertTrue(verdicts["the split accounts for the total"].passed)

    def test_a_round_ending_unannounced_fails(self):
        verdicts = self._verdicts(round_end_mispredictions=1, round_end_unpredicted=1)
        self.assertFalse(verdicts["no round ended without being predicted"].passed)

    def test_an_early_prediction_for_another_reason_fails(self):
        verdicts = self._verdicts(round_end_mispredictions=3, round_end_predicted_early=3,
                                  round_end_predicted_early_reasons={"truncation": 3})
        self.assertFalse(verdicts["every early round-end prediction is the documented smoke approximation"].passed)

    def test_unexplained_early_predictions_fail_closed(self):
        """A count with no reasons is unexplained, not benign."""
        from agent_code.research_agent.runtime.experiment import TASK_COMPLETE
        for reasons in ({}, {TASK_COMPLETE: 2}):
            verdicts = self._verdicts(round_end_mispredictions=5, round_end_predicted_early=5,
                                      round_end_predicted_early_reasons=reasons)
            self.assertFalse(
                verdicts["every early round-end prediction is the documented smoke approximation"].passed,
                f"reasons {reasons} account for fewer than the 5 counted")

    def test_numbers_that_do_not_reconcile_fail(self):
        from agent_code.research_agent.runtime.experiment import TASK_COMPLETE
        verdicts = self._verdicts(round_end_mispredictions=99, round_end_predicted_early=2,
                                  round_end_predicted_early_reasons={TASK_COMPLETE: 2})
        self.assertFalse(verdicts["the split accounts for the total"].passed)

    def test_a_run_from_before_the_split_keeps_the_strict_reading(self):
        row = {"round": 10, "updates_this_round": 5, "gradient_steps_this_round": 1,
               "epsilon": 0.05, "round_end_mispredictions": 11}
        checks = self.check_pilot.check_training_job(Path("train_seed1"), [row], None)
        strict = [c for c in checks if "round_end_mispredictions is zero" in c.name]
        self.assertEqual(len(strict), 1, "a pre-split run must fall back to the strict check")
        self.assertFalse(strict[0].passed)
