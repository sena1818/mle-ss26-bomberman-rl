"""Tests for the shared, model-independent experiment infrastructure."""

from __future__ import annotations

import json
import os
import shutil
import statistics
import sys
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
ROOT_DIR = SCRIPTS.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import aggregate_results  # noqa: E402
import compare_runs  # noqa: E402
import experiment_lib  # noqa: E402
import prune_runs  # noqa: E402
from agent_code.research_agent.config import (  # noqa: E402
    active_config as resolved_config,
    epsilon_for_training_round,
)
from experiment_lib import ROOT, ConfigError, Experiment, resolved_runtime_config, verify_job_provenance, write_json  # noqa: E402
from run_experiment import (  # noqa: E402
    _archive_failed_attempt, _publish_segment_checkpoints, build_jobs, execute_phase, load_context)


def config(route: str = "R01", reward_version: str = "A00", exploration_version: str = "E00") -> dict:
    return {
        "schema_version": 1,
        "experiment_id": "test_r01",
        "route": route,
        "agent": {"name": "research_agent", "model": "linear_q", "algorithm": "q_learning", "state_representation": "handcrafted_v1"},
        "reward_version": reward_version,
        "exploration_version": exploration_version,
        "training": {"scenario": "coin-heaven", "opponents": [], "seeds": [11, 12], "budget": {"rounds": 2, "checkpoint_every": 1}},
        "evaluation": {"scenario": "classic", "opponents": [], "seeds": [21, 22], "budget": {"rounds": 2, "checkpoint_every": 1}},
        "promotion": {"primary_metric": "score"},
    }


def curriculum_config() -> dict:
    payload = config()
    payload["experiment_id"] = "test_r01_c01"
    payload["training"] = {
        "scenario": "classic", "opponents": [], "seeds": [11, 12],
        "budget": {"rounds": 4, "checkpoint_every": 2},
    }
    payload["curriculum"] = {
        "source_run_id": "source_run",
        "segments": [
            {"scenario": "classic", "rounds": 2},
            {"scenario": "coin-heaven", "rounds": 2},
        ],
    }
    payload["evaluation_suites"] = {
        "coin_regression": {
            "scenario": "coin-heaven", "opponents": [], "seeds": [21, 22],
            "budget": {"rounds": 2, "checkpoint_every": 1},
        }
    }
    return payload


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

    def test_curriculum_jobs_keep_warm_start_and_regression_suite_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            write_json(path, curriculum_config())
            experiment = Experiment.load(path)
            self.assertEqual(sum(segment.rounds for segment in experiment.curriculum.segments), 4)
            jobs = build_jobs(experiment, Path(temporary) / "run")
            train_jobs = [job for job in jobs if job["mode"] == "train"]
            eval_jobs = [job for job in jobs if job["mode"] == "eval"]
            self.assertEqual(len(train_jobs), 2)
            self.assertEqual(len(eval_jobs), 8)
            self.assertTrue(all(job["input_model_relpath"].startswith("inputs/") for job in train_jobs))
            self.assertEqual({job["suite"] for job in eval_jobs}, {"primary", "coin_regression"})

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
            self.assertEqual(runtime["config"]["exploration_version"], "E00")
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
            # R03 (MLP plus SARSA) is a declared design in docs/00 with no
            # learner adapter behind it; the vocabulary accepts it, the runner
            # must not.
            payload = config("R03")
            payload["agent"]["model"] = "mlp_q"
            payload["agent"]["algorithm"] = "sarsa"
            write_json(path, payload)
            with self.assertRaises(ConfigError):
                Experiment.load(path).require_implemented()

    def test_an_implemented_route_declaration_must_match_its_registered_design(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            # R02 is implemented, but as an MLP: declaring it with R01's linear
            # model must fail rather than quietly run the wrong approximator.
            write_json(path, config("R02"))
            with self.assertRaises(ConfigError):
                Experiment.load(path).require_implemented()

    def test_a01_and_named_diagnostic_evaluation_suites_are_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            payload = config(reward_version="A01")
            payload["evaluation_suites"] = {
                "loot_crate_diagnostic": {
                    "scenario": "loot-crate",
                    "opponents": [],
                    "seeds": [31, 32],
                    "budget": {"rounds": 2, "checkpoint_every": 1},
                }
            }
            write_json(path, payload)
            experiment = Experiment.load(path)
            experiment.require_implemented()
            self.assertEqual(resolved_runtime_config(experiment)["config"]["reward_version"], "A01")
            jobs = build_jobs(experiment, Path(temporary) / "run")
            self.assertEqual(len(jobs), 10)
            self.assertEqual(
                {job.get("suite", "primary") for job in jobs if job["mode"] == "eval"},
                {"primary", "loot_crate_diagnostic"},
            )

    def test_a02_is_an_explicit_r01_reward_version(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            write_json(path, config(reward_version="A02"))
            experiment = Experiment.load(path)
            experiment.require_implemented()
            self.assertEqual(resolved_runtime_config(experiment)["config"]["reward_version"], "A02")

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
                self.assertTrue((aggregate_results.PROMOTION_ROOT / "classic" / "active_model.npz").is_file())
                weaker = json.loads(json.dumps(summary))
                weaker["run_id"] = "later_but_weaker"
                weaker["metrics"]["score"]["mean"] = -1.0
                self.assertFalse(aggregate_results.maybe_promote(weaker))
            finally:
                aggregate_results.PROMOTION_ROOT = original_root

    def test_dose_response_reward_versions_are_registered_end_to_end(self):
        """A03/A05 must pass config validation and reach the runtime config."""
        for version, death_penalty in (("A03", -1.0), ("A05", 0.0)):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "config.json"
                write_json(path, config(reward_version=version))
                experiment = Experiment.load(path)
                experiment.require_implemented()
                runtime = resolved_runtime_config(experiment)
                self.assertEqual(runtime["config"]["reward_version"], version)
                self.assertEqual(runtime["reward_specification"]["death_penalty"], death_penalty)

    def test_e01_is_a_versioned_single_variable_schedule(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            write_json(path, config(reward_version="A03", exploration_version="E01"))
            experiment = Experiment.load(path)
            experiment.require_implemented()
            runtime = resolved_runtime_config(experiment)
            self.assertEqual(runtime["config"]["reward_version"], "A03")
            self.assertEqual(runtime["config"]["exploration_version"], "E01")
            self.assertEqual(runtime["exploration_specification"]["initial_epsilon"], 0.30)
            self.assertEqual(runtime["exploration_specification"]["final_epsilon"], 0.05)

    def test_unregistered_exploration_version_is_rejected_before_any_job_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            write_json(path, config(exploration_version="E99"))
            with self.assertRaises(ConfigError):
                Experiment.load(path).require_implemented()

    def test_controlled_comparison_reports_the_exploration_dimension(self):
        shared = {
            "route": "R01",
            "state_representation": "handcrafted_v1",
            "model": "linear_q",
            "algorithm": "q_learning",
            "reward_version": "A03",
        }
        arms = [
            {"dimensions": {**shared, "exploration_version": "E00"}},
            {"dimensions": {**shared, "exploration_version": "E01"}},
        ]
        self.assertEqual(compare_runs.changed_dimensions(arms), ["exploration_version"])

    def test_a_comparison_names_the_training_recipe_dimensions_too(self):
        # An n-step or replay arm that reports "nothing changed" would make the
        # comparison table claim a single-factor result it does not have.
        shared = {
            "route": "R01", "state_representation": "handcrafted_v1", "model": "linear_q",
            "algorithm": "q_learning", "reward_version": "A06", "exploration_version": "E01",
        }
        self.assertEqual(
            compare_runs.changed_dimensions([
                {"dimensions": {**shared, "n_step": "1", "replay": ""}},
                {"dimensions": {**shared, "n_step": "5", "replay": ""}},
            ]),
            ["n_step"],
        )
        self.assertEqual(
            compare_runs.changed_dimensions([
                {"dimensions": {**shared, "n_step": "1", "replay": ""}},
                {"dimensions": {**shared, "n_step": "1", "replay": '{"capacity": 50000}'}},
            ]),
            ["replay"],
        )
        self.assertEqual(
            compare_runs.changed_dimensions([
                {
                    "dimensions": {**shared, "reward_version": "A03", "shaping": ""},
                    "reward_specification": {"event_weights": {"COIN_COLLECTED": 1.0}, "death_penalty": -1.0},
                },
                {
                    "dimensions": {**shared, "reward_version": "A06", "shaping": "potential_v1"},
                    "reward_specification": {"event_weights": {"COIN_COLLECTED": 1.0}, "death_penalty": -1.0},
                },
            ]),
            ["shaping"],
        )

    def test_unregistered_reward_version_is_rejected_before_any_job_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            write_json(path, config(reward_version="A04"))
            with self.assertRaises(ConfigError):
                Experiment.load(path).require_implemented()

    def test_checkpoint_evaluation_defaults_to_the_historical_latest_only_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            write_json(path, config())
            experiment = Experiment.load(path)
            self.assertEqual(experiment.checkpoint_evaluation.mode, "latest")
            self.assertEqual(experiment.checkpoint_evaluation.holdout_seeds, ())
            jobs = build_jobs(experiment, Path(temporary) / "run")
            eval_jobs = [job for job in jobs if job["mode"] == "eval"]
            self.assertTrue(all(job["model_relpath"].endswith("latest_model.npz") for job in eval_jobs))
            self.assertTrue(all(job["checkpoint_round"] is None for job in eval_jobs))
            self.assertTrue(all(job["seed_role"] == "validation" for job in eval_jobs))

    def test_checkpoint_mode_all_expands_one_job_per_saved_round(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            payload = config()
            # 4 rounds, checkpointed every 2 -> rounds 2 and 4; round 4 is also
            # latest_model.npz, so 'all' must not add a redundant latest job.
            payload["training"]["budget"] = {"rounds": 4, "checkpoint_every": 2}
            payload["checkpoint_evaluation"] = {"mode": "all"}
            write_json(path, payload)
            experiment = Experiment.load(path)
            jobs = [job for job in build_jobs(experiment, Path(temporary) / "run") if job["mode"] == "eval"]
            self.assertEqual({job["checkpoint_round"] for job in jobs}, {2, 4})
            self.assertEqual(len(jobs), 2 * 2 * 2)  # train seeds x checkpoints x eval seeds
            self.assertTrue(all(job["model_relpath"] is None for job in jobs))
            self.assertTrue(all(job["checkpoint_search_relpath"].endswith("checkpoints") for job in jobs))

    def test_holdout_seeds_are_separate_jobs_and_never_overlap_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            payload = config()
            payload["checkpoint_evaluation"] = {"validation_seeds": [21], "holdout_seeds": [31]}
            write_json(path, payload)
            experiment = Experiment.load(path)
            jobs = [job for job in build_jobs(experiment, Path(temporary) / "run") if job["mode"] == "eval"]
            roles = {job["seed_role"] for job in jobs}
            self.assertEqual(roles, {"validation", "holdout"})
            self.assertEqual({job["seed"] for job in jobs if job["seed_role"] == "holdout"}, {31})
            self.assertEqual(len({job["job_id"] for job in jobs}), len(jobs))

            payload["checkpoint_evaluation"] = {"validation_seeds": [21], "holdout_seeds": [21]}
            write_json(path, payload)
            with self.assertRaises(ConfigError):
                Experiment.load(path)

    def test_snapshot_round_trips_the_checkpoint_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            payload = config()
            payload["checkpoint_evaluation"] = {"mode": "all", "validation_seeds": [21], "holdout_seeds": [31]}
            payload["evaluation_suites"] = {
                "coin_regression": {
                    "scenario": "coin-heaven", "opponents": [], "seeds": [21],
                    "budget": {"rounds": 2, "checkpoint_every": 1},
                    "checkpoint_evaluation": {"mode": "latest", "validation_seeds": [21]},
                }
            }
            write_json(path, payload)
            experiment = Experiment.load(path)
            snapshot_path = Path(temporary) / "snapshot.json"
            write_json(snapshot_path, experiment.snapshot())
            self.assertEqual(Experiment.load(snapshot_path), experiment)
            # A diagnostic suite stays on 'latest' while the primary sweeps all.
            self.assertEqual(experiment.suite_checkpoints("primary").mode, "all")
            self.assertEqual(experiment.suite_checkpoints("coin_regression").mode, "latest")

    def test_snapshot_preserves_predeclared_design_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            payload = config(reward_version="A03")
            payload["_design_note"] = "Death-penalty dose-response arm."
            payload["_predeclared_design_numbers"] = {"p_star": 0.689, "model": "mean-field"}
            write_json(path, payload)
            experiment = Experiment.load(path)
            snapshot_path = Path(temporary) / "snapshot.json"
            write_json(snapshot_path, experiment.snapshot())
            reloaded = Experiment.load(snapshot_path)
            self.assertEqual(reloaded.design_note, payload["_design_note"])
            self.assertEqual(reloaded.predeclared_design_numbers, payload["_predeclared_design_numbers"])

    def test_ratio_metrics_are_averaged_per_round_not_pooled(self):
        """A pooled ratio of sums is not the mean of per-round ratios."""
        with tempfile.TemporaryDirectory() as temporary:
            job_dir = Path(temporary) / "job"
            (job_dir / "agent").mkdir(parents=True)
            write_json(job_dir / "official_stats.json", {
                "by_agent": {"a": {"score": 0, "coins": 2, "kills": 0, "suicides": 1,
                                   "invalid": 0, "steps": 6, "bombs": 3, "crates": 6, "moves": 2}},
                "by_round": {"Round 01": {"coins": 2, "kills": 0, "suicides": 0, "steps": 2},
                             "Round 02": {"coins": 0, "kills": 0, "suicides": 1, "steps": 4}},
            })
            records = [
                # Round 1: 2 actions, 1 bomb, survived  -> bomb_rate .5, safe 1.0
                {"kind": "action", "round": 1, "action": "BOMB", "position": [1, 1],
                 "selected_action_was_legal": True, "inference_seconds": 0.01},
                {"kind": "action", "round": 1, "action": "WAIT", "position": [1, 1],
                 "selected_action_was_legal": True, "inference_seconds": 0.01},
                # Round 2: 4 actions, 2 bombs, died once -> bomb_rate .5, safe .5
                {"kind": "action", "round": 2, "action": "BOMB", "position": [1, 1],
                 "selected_action_was_legal": True, "inference_seconds": 0.01},
                {"kind": "action", "round": 2, "action": "BOMB", "position": [2, 1],
                 "selected_action_was_legal": True, "inference_seconds": 0.01},
                {"kind": "action", "round": 2, "action": "WAIT", "position": [2, 1],
                 "selected_action_was_legal": True, "inference_seconds": 0.01},
                {"kind": "action", "round": 2, "action": "UP", "position": [2, 1],
                 "selected_action_was_legal": True, "inference_seconds": 0.01},
            ]
            (job_dir / "agent" / "agent.jsonl").write_text(
                "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
            sample = aggregate_results.one_evaluation(job_dir, "a")

            self.assertAlmostEqual(sample["bomb_rate"], 0.5)
            self.assertAlmostEqual(sample["wait_fraction"], (0.5 + 0.25) / 2)
            self.assertAlmostEqual(sample["approximate_safe_bomb_rate"], (1.0 + 0.5) / 2)
            self.assertAlmostEqual(sample["distinct_cells"], (1 + 2) / 2)
            self.assertEqual(sample["rounds_with_bombs"], 2)
            # Job-level, because official by_round carries no crate counter.
            self.assertAlmostEqual(sample["crates_per_round"], 3.0)
            self.assertAlmostEqual(sample["crates_per_bomb"], 2.0)
            self.assertAlmostEqual(sample["coins_per_crate"], 2 / 6)
            self.assertAlmostEqual(sample["official_wait_fraction"], (6 - 2 - 3 - 0) / 6)
            # Solo evaluation: the agent took every coin anyone took.
            self.assertAlmostEqual(sample["coins_share"], 1.0)

    def test_coins_share_is_measured_against_every_agent_on_the_board(self):
        """The G-B metric. docs/01 section 7.20: it resolves what score cannot."""
        with tempfile.TemporaryDirectory() as temporary:
            job_dir = Path(temporary) / "job"
            (job_dir / "agent").mkdir(parents=True)
            write_json(job_dir / "official_stats.json", {
                "by_agent": {
                    "research_agent": {"score": 3, "coins": 3, "kills": 0, "suicides": 0,
                                       "invalid": 0, "steps": 8, "bombs": 1, "crates": 2, "moves": 6},
                    "rule_based_agent": {"coins": 5}, "rule_based_agent_1": {"coins": 4},
                    "rule_based_agent_2": {"coins": 0},
                },
                "by_round": {"Round 01": {"coins": 3, "kills": 0, "suicides": 0, "steps": 8}},
            })
            (job_dir / "agent" / "agent.jsonl").write_text(
                json.dumps({"kind": "action", "round": 1, "action": "UP", "position": [1, 1],
                            "selected_action_was_legal": True, "inference_seconds": 0.01}) + "\n",
                encoding="utf-8")
            sample = aggregate_results.one_evaluation(job_dir, "research_agent")
            self.assertAlmostEqual(sample["coins_share"], 3 / 12)
            self.assertIn("coins_share", aggregate_results.ALL_METRICS)
            self.assertIn("coins_share", aggregate_results.METRIC_DEFINITIONS)

    def test_undefined_ratios_are_omitted_rather_than_counted_as_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            job_dir = Path(temporary) / "job"
            (job_dir / "agent").mkdir(parents=True)
            write_json(job_dir / "official_stats.json", {
                "by_agent": {"a": {"score": 0, "coins": 0, "kills": 0, "suicides": 0,
                                   "invalid": 0, "steps": 2, "bombs": 0, "crates": 0, "moves": 0}},
                "by_round": {"Round 01": {"coins": 0, "kills": 0, "suicides": 0, "steps": 2}},
            })
            (job_dir / "agent" / "agent.jsonl").write_text(
                json.dumps({"kind": "action", "round": 1, "action": "WAIT", "position": [1, 1],
                            "selected_action_was_legal": True, "inference_seconds": 0.01}) + "\n",
                encoding="utf-8")
            sample = aggregate_results.one_evaluation(job_dir, "a")
            # A policy that never bombs has no safe-bomb rate; calling it 0.0
            # would claim it bombed and died.
            self.assertIsNone(sample["approximate_safe_bomb_rate"])
            self.assertIsNone(sample["crates_per_bomb"])
            self.assertIsNone(sample["coins_per_crate"])
            # Nobody collected anything: a share of nothing is undefined, not 0.
            self.assertIsNone(sample["coins_share"])
            self.assertEqual(sample["bomb_rate"], 0.0)
            block = aggregate_results._metric_block([sample, sample])
            self.assertEqual(block["approximate_safe_bomb_rate"]["count"], 0)
            self.assertEqual(block["bomb_rate"]["count"], 2)

    def test_variance_is_reported_separately_for_training_and_evaluation_seeds(self):
        decomposition = aggregate_results._variance_decomposition({
            1: [{metric: 1.0 for metric in aggregate_results.ALL_METRICS},
                {metric: 3.0 for metric in aggregate_results.ALL_METRICS}],
            2: [{metric: 5.0 for metric in aggregate_results.ALL_METRICS},
                {metric: 7.0 for metric in aggregate_results.ALL_METRICS}],
        })
        score = decomposition["score"]
        self.assertEqual(score["train_seeds"], 2)
        # Seed means are 2 and 6; within-seed spread is identical for both.
        self.assertAlmostEqual(score["across_train_seeds_std"], statistics.stdev([2.0, 6.0]))
        self.assertAlmostEqual(score["mean_within_train_seed_std"], statistics.stdev([1.0, 3.0]))

    def test_checkpoint_holdout_reports_only_the_validation_selected_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            payload = config()
            payload["training"]["seeds"] = [11]
            payload["training"]["budget"] = {"rounds": 2, "checkpoint_every": 1}
            payload["evaluation"]["seeds"] = [21]
            payload["checkpoint_evaluation"] = {
                "mode": "all", "validation_seeds": [21], "holdout_seeds": [31],
            }
            config_path = run_dir / "input.json"
            write_json(config_path, payload)
            experiment = Experiment.load(config_path)
            write_json(run_dir / "experiment_config.snapshot.json", experiment.snapshot())
            jobs = build_jobs(experiment, run_dir)
            write_json(run_dir / "jobs.json", jobs)
            agent_dir = run_dir / "jobs" / "train_seed11" / "agent"
            checkpoints = agent_dir / "checkpoints"
            checkpoints.mkdir(parents=True)
            (checkpoints / "R01_round00001_updates00000001.npz").write_bytes(b"one")
            (checkpoints / "R01_round00002_updates00000002.npz").write_bytes(b"two")
            (agent_dir / "latest_model.npz").write_bytes(b"latest")
            for job in jobs:
                if job["mode"] != "eval":
                    continue
                checkpoint_round = int(job["checkpoint_round"])
                score = {
                    ("validation", 1): 1,
                    ("validation", 2): 2,
                    ("holdout", 1): 100,
                    ("holdout", 2): 4,
                }[(job["seed_role"], checkpoint_round)]
                job_dir = run_dir / job["artifact_relpath"]
                (job_dir / "agent").mkdir(parents=True)
                write_json(job_dir / "official_stats.json", {
                    "by_agent": {"research_agent": {
                        "score": score, "coins": score, "kills": 0, "suicides": 0,
                        "invalid": 0, "steps": 1, "bombs": 0, "crates": 0, "moves": 0,
                    }},
                    "by_round": {"Round 01": {"coins": score, "kills": 0, "suicides": 0, "steps": 1}},
                })
                (job_dir / "agent" / "agent.jsonl").write_text(
                    json.dumps({"kind": "action", "round": 1, "action": "WAIT", "position": [1, 1],
                                "selected_action_was_legal": True, "inference_seconds": 0.01}) + "\n",
                    encoding="utf-8",
                )
            summary = aggregate_results.aggregate(run_dir)
            self.assertEqual(summary["selected_checkpoint"]["checkpoint_round"], 2)
            self.assertEqual(summary["holdout_metrics"]["score"]["mean"], 4.0)
            self.assertEqual(summary["evaluation_suites"]["primary"]["all_checkpoint_holdout_metrics"]["score"]["mean"], 52.0)

    def test_runtime_copy_is_an_allowlist_not_a_deny_list(self):
        """A stray directory in the repo root must never reach a job."""
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "runtime"
            intruder = experiment_lib.ROOT / ".allowlist_probe"
            intruder.mkdir(exist_ok=True)
            (intruder / "blob.bin").write_bytes(b"0" * 4096)
            try:
                experiment_lib.copy_runtime(destination)
            finally:
                shutil.rmtree(intruder, ignore_errors=True)
            self.assertFalse((destination / ".allowlist_probe").exists())
            # What the framework actually imports must still be there.
            self.assertTrue((destination / "main.py").is_file())
            self.assertTrue((destination / "environment.py").is_file())
            self.assertTrue((destination / "settings.py").is_file())
            self.assertTrue((destination / "assets").is_dir())
            self.assertTrue((destination / "agent_code" / "research_agent" / "callbacks.py").is_file())

    def test_prune_only_touches_runtime_of_succeeded_jobs(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            for name, exit_code in (("done", 0), ("failed", 1), ("running", None)):
                job_dir = run_dir / "jobs" / name
                (job_dir / "runtime" / "assets").mkdir(parents=True)
                (job_dir / "runtime" / "assets" / "x.png").write_bytes(b"0" * 128)
                write_json(job_dir / "official_stats.json", {"by_agent": {}})
                if exit_code is not None:
                    write_json(job_dir / "completion.json", {"exit_code": exit_code})
            targets = prune_runs.runtime_directories(run_dir)
            self.assertEqual([path.parent.name for path in targets], ["done"])
            for path in targets:
                shutil.rmtree(path)
            self.assertFalse((run_dir / "jobs" / "done" / "runtime").exists())
            # A failure keeps its runtime for debugging; a live job is untouched.
            self.assertTrue((run_dir / "jobs" / "failed" / "runtime").is_dir())
            self.assertTrue((run_dir / "jobs" / "running" / "runtime").is_dir())
            self.assertTrue((run_dir / "jobs" / "done" / "official_stats.json").is_file())

    def test_slim_copy_keeps_every_model_including_curriculum_segments(self):
        """A curriculum job stores its models one level down, per segment."""
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            destination = Path(temporary) / "archive"
            write_json(run_dir / "evaluation_summary.json", {"run_id": "run"})
            write_json(run_dir / "provenance.json", {"git_commit": "abc"})
            (run_dir / "inputs" / "initial_models").mkdir(parents=True)
            (run_dir / "inputs" / "initial_models" / "train_seed11.npz").write_bytes(b"warm")
            job_dir = run_dir / "jobs" / "train_seed11"
            (job_dir / "agent").mkdir(parents=True)
            (job_dir / "agent" / "latest_model.npz").write_bytes(b"final")
            write_json(job_dir / "official_stats.json", {"by_agent": {}})
            write_json(job_dir / "curriculum_manifest.json", {"segments": []})
            for index in (1, 2):
                segment = job_dir / "segments" / f"segment{index:02d}"
                (segment / "agent" / "checkpoints").mkdir(parents=True)
                (segment / "agent" / "latest_model.npz").write_bytes(b"seg")
                (segment / "agent" / "checkpoints" / f"round{index}.npz").write_bytes(b"ckpt")
                write_json(segment / "official_stats.json", {"by_agent": {}})
            # Runtime copies must never be archived, even though they can hold a
            # packaged model.npz beside the agent.
            (job_dir / "runtime" / "agent_code" / "research_agent").mkdir(parents=True)
            (job_dir / "runtime" / "agent_code" / "research_agent" / "model.npz").write_bytes(b"junk")

            prune_runs.slim_copy(run_dir, destination, apply=True)
            archived = destination / "run"
            models = sorted(path.name for path in archived.rglob("*.npz"))
            self.assertEqual(models, ["latest_model.npz", "latest_model.npz", "latest_model.npz",
                                      "round1.npz", "round2.npz", "train_seed11.npz"])
            self.assertEqual(len(list(archived.rglob("official_stats.json"))), 3)
            self.assertTrue((archived / "jobs" / "train_seed11" / "curriculum_manifest.json").is_file())
            self.assertFalse(any("runtime" in path.parts for path in archived.rglob("*")))

    @staticmethod
    def _write_config(path: Path) -> Path:
        write_json(path, config())
        return path


class CurriculumAnnealTest(unittest.TestCase):
    """A curriculum segment is its own process with its own round counter.

    Regression cover: the runner used to pass the whole training budget as the
    schedule denominator while each segment restarted at round 1, so a decaying
    exploration schedule silently sat in its opening high-epsilon hold forever.
    """

    def _curriculum(self, anneal_mode: str | None = None) -> tuple[Experiment, object]:
        payload = curriculum_config()
        payload["exploration_version"] = "E01"
        if anneal_mode is not None:
            payload["curriculum"]["anneal_mode"] = anneal_mode
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            write_json(path, payload)
            experiment = Experiment.load(path)
        assert experiment.curriculum is not None
        return experiment, experiment.curriculum

    def test_global_offset_is_the_default_and_spans_the_whole_budget(self):
        experiment, curriculum = self._curriculum()
        self.assertEqual(curriculum.anneal_mode, "global_round_offset")
        self.assertEqual(curriculum.segment_round_offset(1), 0)
        self.assertEqual(curriculum.segment_round_offset(2), 2)
        for index in (1, 2):
            with self.subTest(segment=index):
                self.assertEqual(
                    curriculum.segment_schedule_rounds(index, experiment.training),
                    experiment.training.budget.rounds,
                )

    def test_per_segment_mode_restarts_the_schedule_inside_every_segment(self):
        experiment, curriculum = self._curriculum("per_segment")
        for index in (1, 2):
            with self.subTest(segment=index):
                self.assertEqual(curriculum.segment_round_offset(index), 0)
                self.assertEqual(curriculum.segment_schedule_rounds(index, experiment.training), 2)

    def test_an_undeclared_anneal_mode_is_rejected(self):
        with self.assertRaises(ConfigError):
            self._curriculum("whatever_feels_right")

    def test_offsets_cover_every_declared_round_exactly_once(self):
        experiment, curriculum = self._curriculum()
        covered: list[int] = []
        for index, segment in enumerate(curriculum.segments, start=1):
            offset = curriculum.segment_round_offset(index)
            covered.extend(offset + local for local in range(1, segment.rounds + 1))
        self.assertEqual(covered, list(range(1, experiment.training.budget.rounds + 1)))

    def test_a_decaying_schedule_actually_decays_across_segments(self):
        """The end-to-end property the offset exists to guarantee."""
        experiment, curriculum = self._curriculum()
        agent_config = replace(
            resolved_config(), exploration_version="E01",
        )
        epsilons = []
        for index, segment in enumerate(curriculum.segments, start=1):
            offset = curriculum.segment_round_offset(index)
            denominator = curriculum.segment_schedule_rounds(index, experiment.training)
            for local_round in range(1, segment.rounds + 1):
                epsilons.append(epsilon_for_training_round(agent_config, offset + local_round, denominator))
        self.assertEqual(epsilons[0], 0.30)
        self.assertEqual(epsilons[-1], 0.05)
        self.assertTrue(all(later <= earlier for earlier, later in zip(epsilons, epsilons[1:])))
        self.assertGreater(epsilons[0], epsilons[-1], "a segmented run must not freeze at the initial epsilon")

    def test_curriculum_snapshot_records_the_declared_mode(self):
        experiment, _ = self._curriculum("per_segment")
        self.assertEqual(experiment.snapshot()["curriculum"]["anneal_mode"], "per_segment")


class CurriculumCheckpointsAreWhereEvaluationLooksTest(unittest.TestCase):
    """A warm-started arm has to be sweepable by checkpoint, like any other.

    ``build_jobs`` addresses periodic checkpoints at ``jobs/<train job>/agent/
    checkpoints`` for every arm.  A curriculum job runs each segment in its own
    artifact directory, so its checkpoints landed under ``segments/<id>/agent/
    checkpoints`` and the two paths never met: ``checkpoint_evaluation`` modes
    ``all`` and ``rounds`` found nothing and the run failed in its evaluation
    phase.  The two-stage fine-tuning arm is the first to need both at once.
    """

    def _checkpoint(self, directory: Path, round_number: int, updates: int) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"R02_9_A06_classic_seed1001_round{round_number:05d}_updates{updates:08d}.npz"
        path.write_bytes(b"not really a checkpoint")
        return path

    def test_a_single_segment_publishes_under_the_unchanged_round_number(self):
        with tempfile.TemporaryDirectory() as temporary:
            job_dir = Path(temporary) / "jobs" / "train_seed1001"
            segment = job_dir / "segments" / "segment01_classic"
            for round_number in (250, 500):
                self._checkpoint(segment / "agent" / "checkpoints", round_number, round_number * 40)
            published = _publish_segment_checkpoints(segment, job_dir, rounds_before=0)
            self.assertEqual(published, 2)
            names = sorted(path.name for path in (job_dir / "agent" / "checkpoints").glob("*.npz"))
            self.assertEqual([name.split("_round")[1][:5] for name in names], ["00250", "00500"])

    def test_a_later_segment_is_renumbered_to_the_cumulative_round(self):
        """The round the evaluation job asks for counts the whole curriculum."""
        with tempfile.TemporaryDirectory() as temporary:
            job_dir = Path(temporary) / "jobs" / "train_seed1001"
            for index, (scenario, before) in enumerate(
                    (("classic", 0), ("loot_crate", 500)), start=1):
                segment = job_dir / "segments" / f"segment{index:02d}_{scenario}"
                for local in (250, 500):
                    self._checkpoint(segment / "agent" / "checkpoints", local, local * 40)
                _publish_segment_checkpoints(segment, job_dir, rounds_before=before)
            names = sorted(path.name for path in (job_dir / "agent" / "checkpoints").glob("*.npz"))
            self.assertEqual([name.split("_round")[1][:5] for name in names],
                             ["00250", "00500", "00750", "01000"])

    def test_two_segments_claiming_one_round_is_an_error_not_a_silent_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            job_dir = Path(temporary) / "jobs" / "train_seed1001"
            for index in (1, 2):
                segment = job_dir / "segments" / f"segment{index:02d}_classic"
                self._checkpoint(segment / "agent" / "checkpoints", 250, 10000)
                if index == 1:
                    _publish_segment_checkpoints(segment, job_dir, rounds_before=0)
                    continue
                with self.assertRaises(RuntimeError):
                    _publish_segment_checkpoints(segment, job_dir, rounds_before=0)

    def test_the_published_path_is_the_one_build_jobs_will_search(self):
        """Bind the two sides together so a rename on either fails here."""
        with tempfile.TemporaryDirectory() as temporary:
            payload = curriculum_config()
            payload["checkpoint_evaluation"] = {"mode": "all", "validation_seeds": [21, 22]}
            path = Path(temporary) / "config.json"
            write_json(path, payload)
            experiment = Experiment.load(path)
            jobs = build_jobs(experiment, Path(temporary) / "run")
            searched = {job["checkpoint_search_relpath"] for job in jobs
                        if job.get("checkpoint_search_relpath")}
            self.assertEqual(searched, {"jobs/train_seed11/agent/checkpoints",
                                        "jobs/train_seed12/agent/checkpoints"})

            job_dir = Path(temporary) / "run" / "jobs" / "train_seed11"
            segment = job_dir / "segments" / "segment01_classic"
            self._checkpoint(segment / "agent" / "checkpoints", 2, 80)
            _publish_segment_checkpoints(segment, job_dir, rounds_before=0)
            relative = (job_dir / "agent" / "checkpoints").relative_to(Path(temporary) / "run")
            self.assertIn(str(relative), searched)


class TerminalOnTruncationDeclarationTest(unittest.TestCase):
    def test_the_flag_defaults_to_false_and_reaches_the_runtime_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            write_json(path, config())
            experiment = Experiment.load(path)
            self.assertFalse(experiment.terminal_on_truncation)
            self.assertFalse(experiment.snapshot()["terminal_on_truncation"])
            self.assertFalse(resolved_runtime_config(experiment)["config"]["terminal_on_truncation"])

    def test_declaring_it_true_is_carried_into_the_resolved_runtime_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            payload = config()
            payload["terminal_on_truncation"] = True
            write_json(path, payload)
            experiment = Experiment.load(path)
            self.assertTrue(experiment.snapshot()["terminal_on_truncation"])
            self.assertTrue(resolved_runtime_config(experiment)["config"]["terminal_on_truncation"])

    def test_the_snapshot_round_trips_through_the_same_parser(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            payload = config()
            payload["terminal_on_truncation"] = True
            write_json(path, payload)
            experiment = Experiment.load(path)
            snapshot_path = Path(temporary) / "snapshot.json"
            write_json(snapshot_path, experiment.snapshot())
            self.assertEqual(Experiment.load(snapshot_path), experiment)


class ParallelPhaseTest(unittest.TestCase):
    """Jobs inside one phase are independent; the phases themselves are not."""

    @staticmethod
    def _jobs() -> list[dict]:
        return [
            {"job_id": "train_seed11", "mode": "train"},
            {"job_id": "train_seed12", "mode": "train"},
            {"job_id": "eval_a", "mode": "eval"},
            {"job_id": "eval_b", "mode": "eval"},
            {"job_id": "eval_c", "mode": "eval"},
        ]

    def test_only_the_requested_phase_runs_and_every_job_runs_once(self):
        for workers in (1, 4):
            with self.subTest(workers=workers):
                executed: list[str] = []
                lock = threading.Lock()

                def record(job_file, **kwargs):
                    with lock:
                        executed.append(Path(job_file).stem)

                with patch("run_experiment.execute_job", side_effect=record):
                    execute_phase(Path("/runs/x"), self._jobs(), "eval", workers)
                self.assertEqual(sorted(executed), ["eval_a", "eval_b", "eval_c"])

    def test_every_failure_in_a_phase_is_reported_not_just_the_first(self):
        def sometimes_fail(job_file, **kwargs):
            if Path(job_file).stem in {"eval_a", "eval_c"}:
                raise RuntimeError("boom")

        with patch("run_experiment.execute_job", side_effect=sometimes_fail):
            with self.assertRaises(RuntimeError) as raised:
                execute_phase(Path("/runs/x"), self._jobs(), "eval", 4)
        message = str(raised.exception)
        self.assertIn("eval_a", message)
        self.assertIn("eval_c", message)
        self.assertIn("2 eval job(s) failed", message)

    def test_a_phase_with_no_jobs_is_a_no_op(self):
        with patch("run_experiment.execute_job", side_effect=AssertionError("must not run")):
            execute_phase(Path("/runs/x"), [], "train", 4)


class DeclaredAxesReachTheAgentTest(unittest.TestCase):
    """Every axis a config can declare has to cross the process boundary.

    The agent is a separate process that rebuilds its config from the route name
    it is handed plus BOMBERMAN_* variables.  An axis added to the config schema
    and to resolved_runtime_config but not to that environment is silently
    ignored: the agent falls back to the route default and the arm trains as
    though nothing was declared.  That is what happened to learning_rate_schedule
    -- the L02/L03/L04 arms all ran L01 and looked identical to it, which read as
    'the schedule does not matter' (docs/01 section 7.29).

    Comparing the two sides by name is what makes the next such axis fail loudly.
    """

    ENVIRONMENT_BY_AXIS = {
        "reward_version": "BOMBERMAN_REWARD_VERSION",
        "exploration_version": "BOMBERMAN_EXPLORATION_VERSION",
        "learning_rate_schedule": "BOMBERMAN_LEARNING_RATE_SCHEDULE",
        "n_step": "BOMBERMAN_N_STEP",
        "terminal_on_truncation": "BOMBERMAN_TERMINAL_ON_TRUNCATION",
        "replay": "BOMBERMAN_REPLAY",
        # The step-size *level* is an experiment axis for the same reason the
        # schedule is (agent.learning_rate, section 7.24), and a continuation
        # stage is the first arm that relies on it while also changing route
        # defaults, so it is guarded here rather than assumed.
        "learning_rate": "BOMBERMAN_LEARNING_RATE",
    }

    def test_the_runner_exports_every_axis_in_every_environment_block(self):
        """Counting, not just presence: there are two environment blocks.

        execute_job builds one and the curriculum path builds another.  A
        variable present in only one is exported for some jobs and not others,
        which an `in` check cannot see -- so each axis is required to appear as
        often as BOMBERMAN_EXPLORATION_VERSION, an axis known to be wired into
        both.
        """
        source = (SCRIPTS / "run_experiment.py").read_text(encoding="utf-8")
        expected = source.count('"BOMBERMAN_EXPLORATION_VERSION":')
        self.assertGreaterEqual(expected, 2, "the reference axis is no longer in both blocks")
        for axis, variable in self.ENVIRONMENT_BY_AXIS.items():
            self.assertEqual(
                source.count(f'"{variable}":'), expected,
                f"{variable} ({axis}) is exported {source.count(chr(34) + variable + chr(34) + ':')} "
                f"times but the reference axis is exported {expected}; an axis missing from one "
                f"environment block is silently ignored for those jobs")

    def test_the_agent_reads_every_axis_the_runner_exports(self):
        config_source = (ROOT_DIR / "agent_code/research_agent/config.py").read_text(encoding="utf-8")
        for axis, variable in self.ENVIRONMENT_BY_AXIS.items():
            self.assertIn(variable, config_source,
                          f"the agent never reads {variable} for {axis}")

    def test_an_override_actually_changes_what_the_agent_would_run(self):
        """The end-to-end check: declare it, and see it on the other side."""
        from agent_code.research_agent.config import active_config
        for variable, value, attribute in (
            ("BOMBERMAN_LEARNING_RATE_SCHEDULE", "L01", "learning_rate_schedule"),
            ("BOMBERMAN_REWARD_VERSION", "A03", "reward_version"),
            ("BOMBERMAN_EXPLORATION_VERSION", "E02", "exploration_version"),
        ):
            with patch.dict(os.environ, {"BOMBERMAN_EXPERIMENT": "R02_8", variable: value}, clear=False):
                self.assertEqual(getattr(active_config(), attribute), value,
                                 f"{variable} did not reach the agent's config")
        # The step size is a float, so it needs its own round trip rather than
        # a string comparison.  A continuation stage declares 1e-4 against a
        # route whose own field is 5e-4; if that did not cross, the arm would
        # fine-tune at five times the intended rate and look like a scenario
        # effect.
        with patch.dict(os.environ,
                        {"BOMBERMAN_EXPERIMENT": "R02_9", "BOMBERMAN_LEARNING_RATE": "0.0001"},
                        clear=False):
            self.assertEqual(active_config().learning_rate, 1e-4)
        with patch.dict(os.environ, {"BOMBERMAN_EXPERIMENT": "R02_9"}, clear=False):
            os.environ.pop("BOMBERMAN_LEARNING_RATE", None)
            self.assertEqual(active_config().learning_rate, 5e-4,
                             "an undeclared step size must fall back to the route's own")


class AbsoluteExplorationScheduleTest(unittest.TestCase):
    """The E02 family has to be registered in both registries and fail closed.

    docs/01 section 7.21: E01 defines its hold as a fraction of the budget, so
    every budget comparison also changed the schedule.  The absolute family
    separates them, but only if a config cannot declare a schedule longer than
    the budget it is given -- that would end training mid-anneal at an epsilon
    nobody declared.
    """

    def _payload(self, exploration_version: str, rounds: int) -> dict:
        payload = config(route="R02_3", reward_version="A06", exploration_version=exploration_version)
        payload["experiment_id"] = "test_r02_3_" + exploration_version.lower()
        payload["agent"].update({
            "model": "mlp_q", "algorithm": "q_learning", "state_representation": "handcrafted_v3",
        })
        payload["training"]["budget"] = {"rounds": rounds, "checkpoint_every": max(1, rounds // 2)}
        payload["shaping"] = {"name": "potential_v1"}
        return payload

    def _load(self, exploration_version: str, rounds: int) -> Experiment:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            write_json(path, self._payload(exploration_version, rounds))
            return Experiment.load(path)

    def test_the_ablation_points_are_implemented_on_the_route_that_runs_them(self):
        for version in ("E02", "E03", "E04", "E05", "E06"):
            experiment = self._load(version, 2000)
            experiment.require_implemented()
            self.assertEqual(experiment.exploration_version, version)

    def test_a_schedule_longer_than_the_budget_is_rejected_at_config_time(self):
        experiment = self._load("E02", 1500)
        with self.assertRaises(ConfigError) as caught:
            experiment.require_implemented()
        self.assertIn("never reach its floor", str(caught.exception))

    def test_a_budget_longer_than_the_schedule_is_allowed_and_holds_the_floor(self):
        experiment = self._load("E02", 5000)
        experiment.require_implemented()
        agent_config = replace(resolved_config(), exploration_version="E02")
        self.assertEqual(epsilon_for_training_round(agent_config, 5000, 5000), 0.05)

    def test_the_snapshot_records_the_absolute_rounds_not_just_the_label(self):
        experiment = self._load("E04", 2000)
        runtime = resolved_runtime_config(experiment)
        specification = runtime["exploration_specification"]
        self.assertEqual(specification["exploration_version"], "E04")
        self.assertEqual(specification["hold_rounds"], 1000)
        self.assertEqual(specification["anneal_rounds"], 1000)


class FourMainLineDeclarationTest(unittest.TestCase):
    """The M1--M4 declarations of docs/05 must survive the config round trip.

    Every dimension a line varies -- shaping, n-step, replay -- has to reach the
    resolved runtime config and the run snapshot, because a result whose
    training recipe cannot be recovered from its run directory is not a result.
    """

    ROUTES = {
        "R01": ("linear_q", "q_learning", "handcrafted_v1"),
        "R02": ("mlp_q", "q_learning", "handcrafted_v1"),
        "R02_1": ("mlp_q", "q_learning", "handcrafted_v1"),
        "R07": ("cnn_mlp_q", "double_dqn", "board_egocentric_v2"),
        "R08": ("dueling_cnn_mlp_q", "double_dqn", "board_egocentric_v2"),
    }
    REPLAY = {"capacity": 128, "batch_size": 8, "min_size": 16, "train_every": 2, "target_update_every": 32}

    def _config(self, route: str, **overrides) -> dict:
        model, algorithm, state = self.ROUTES[route]
        payload = config(route)
        payload["agent"].update({"model": model, "algorithm": algorithm, "state_representation": state})
        payload["agent"].update(overrides.pop("agent", {}))
        payload.update(overrides)
        if algorithm == "double_dqn":
            payload["agent"].setdefault("replay", dict(self.REPLAY))
        return payload

    def _load(self, route: str, **overrides) -> Experiment:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            write_json(path, self._config(route, **overrides))
            return Experiment.load(path)

    def test_each_implemented_route_declares_the_line_it_serves(self):
        for route, lines in (("R01", ("M1", "M2")), ("R02", ("M3",)), ("R02_1", ("M3",)), ("R07", ("M4",)), ("R08", ("M4",))):
            with self.subTest(route=route):
                experiment = self._load(route)
                experiment.require_implemented()
                self.assertEqual(experiment.lines, lines)
                self.assertEqual(resolved_runtime_config(experiment)["main_lines"], list(lines))

    def test_the_state_dimension_recorded_is_the_encoder_s_own(self):
        self.assertEqual(resolved_runtime_config(self._load("R01"))["feature_dimension"], 44)
        # 8 board channels over a 17x17 egocentric window, plus 4 global scalars.
        self.assertEqual(resolved_runtime_config(self._load("R07"))["feature_dimension"], 7 * 17 * 17 + 6)

    def test_n_step_and_replay_reach_the_resolved_runtime_config(self):
        experiment = self._load("R01", agent={"n_step": 3, "replay": dict(self.REPLAY)})
        runtime = resolved_runtime_config(experiment)["config"]
        self.assertEqual(runtime["n_step"], 3)
        self.assertEqual(runtime["replay"]["capacity"], 128)
        self.assertEqual(runtime["replay"]["target_update_every"], 32)

    def test_an_absent_replay_block_means_no_replay_not_a_route_default(self):
        # A route carries a default buffer only for manual invocation through
        # environment variables.  A config file that omits the block gets fully
        # online updating, so the snapshot can never overstate what was run.
        self.assertIsNone(resolved_runtime_config(self._load("R01"))["config"]["replay"])
        self.assertIsNone(resolved_runtime_config(self._load("R02"))["config"]["replay"])

    def test_shaping_is_derived_from_the_reward_version_not_declared_twice(self):
        self.assertIsNone(resolved_runtime_config(self._load("R01"))["shaping_specification"])
        shaped = self._load("R01", reward_version="A06")
        self.assertEqual(resolved_runtime_config(shaped)["shaping_specification"]["name"], "potential_v1")
        # An echo that disagrees with the reward version is a config error, not
        # a silent override of one by the other.
        lying = self._load("R01", reward_version="A03", shaping={"name": "potential_v1"})
        with self.assertRaises(ConfigError):
            resolved_runtime_config(lying)

    def test_the_snapshot_reloads_into_an_identical_experiment(self):
        for route in self.ROUTES:
            with self.subTest(route=route):
                experiment = self._load(route, reward_version="A06", agent={"n_step": 3})
                with tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary) / "snapshot.json"
                    write_json(path, experiment.snapshot())
                    self.assertEqual(Experiment.load(path), experiment)

    def test_double_dqn_without_a_buffer_is_rejected_before_any_job_runs(self):
        payload = self._config("R07")
        payload["agent"]["replay"] = None
        payload["route"] = "R07"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            # Declared on a route that does not supply a default buffer either.
            payload["route"] = "R01"
            payload["agent"]["algorithm"] = "double_dqn"
            write_json(path, payload)
            with self.assertRaises(ConfigError):
                Experiment.load(path)

    def test_a_malformed_replay_block_is_rejected_with_the_agent_s_own_rules(self):
        for replay in ({"capacity": 4, "batch_size": 32, "min_size": 32}, {"augmentation": "rot90"}, {"nonsense": 1}):
            with self.subTest(replay=replay):
                with self.assertRaises(ConfigError):
                    experiment = self._load("R01", agent={"replay": replay})
                    resolved_runtime_config(experiment)

    def test_a_non_positive_n_step_is_rejected(self):
        with self.assertRaises(ConfigError):
            self._load("R01", agent={"n_step": 0})

    def test_every_shipped_experiment_config_is_runnable_as_declared(self):
        for path in sorted((Path(__file__).resolve().parents[2] / "experiments").glob("*.json")):
            with self.subTest(config=path.name):
                experiment = Experiment.load(path)
                experiment.require_implemented()
                resolved_runtime_config(experiment)

    def test_m3_n5_is_a_controlled_mlp_replacement_of_m2_n5(self):
        root = Path(__file__).resolve().parents[2]
        m2_path = root / "experiments" / "m2_r01_a06_e01_t02_n5.json"
        m3_path = root / "experiments" / "m3_r02_a06_e01_t02_n5.json"
        with m2_path.open(encoding="utf-8") as source:
            m2 = json.load(source)
        with m3_path.open(encoding="utf-8") as source:
            m3 = json.load(source)

        self.assertEqual(m3["route"], "R02")
        self.assertEqual(m3["agent"]["model"], "mlp_q")
        self.assertEqual(m3["agent"]["n_step"], 5)
        self.assertIsNone(m3["agent"]["replay"])
        self.assertEqual(m3["training"]["seeds"], [1001, 1002, 1003, 1004, 1005])
        self.assertEqual(len(build_jobs(Experiment.load(m3_path), root / "runs" / "test_m3_n5")), 365)

        for field in ("_design_note", "_predeclared_design_numbers", "experiment_id", "route"):
            m2.pop(field)
            m3.pop(field)
        m2["agent"]["model"] = m3["agent"]["model"]
        self.assertEqual(m3, m2)

        runtime = resolved_runtime_config(Experiment.load(m3_path))["config"]
        self.assertEqual(runtime["hidden_layers"], (64, 32))

    def test_m3_1_declares_the_stabilized_mlp_recipe(self):
        root = Path(__file__).resolve().parents[2]
        path = root / "experiments" / "m3_r02_1_a06_e01_t02_n5.json"
        experiment = Experiment.load(path)
        experiment.require_implemented()
        runtime = resolved_runtime_config(experiment)["config"]
        self.assertEqual(runtime["learning_rate"], 1e-3)
        self.assertEqual(runtime["optimizer"], "adam")
        self.assertEqual(runtime["td_loss"], "huber")
        self.assertEqual(runtime["gradient_clip_norm"], 10.0)
        self.assertEqual(experiment.n_step, 5)
        self.assertEqual(experiment.replay["min_size"], 1000)
        self.assertEqual(len(build_jobs(experiment, root / "runs" / "test_m3_1")), 365)


if __name__ == "__main__":
    unittest.main()


class AgentBlockFailsClosedTest(unittest.TestCase):
    """The agent block used to drop keys it did not recognise, in silence.

    Two step-size arms declared ``agent.learning_rate`` and had it discarded
    here; both would have trained at the route default and measured a
    difference of exactly zero after 4.4 hours per seed. The nested blocks in
    this file -- replay, initial_model -- had always rejected unknown keys; the
    block containing them had not.
    """

    def _config(self, **agent_overrides) -> dict:
        payload = config()
        payload["agent"].update(agent_overrides)
        return payload

    def test_an_unknown_agent_key_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            write_json(path, self._config(learninig_rate=1e-4))  # note the typo
            with self.assertRaises(ConfigError) as caught:
                Experiment.load(path)
            self.assertIn("learninig_rate", str(caught.exception))

    def test_a_declared_learning_rate_reaches_the_resolved_runtime_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            write_json(path, self._config(learning_rate=1e-4))
            experiment = Experiment.load(path)
            self.assertEqual(experiment.learning_rate, 1e-4)
            resolved = resolved_runtime_config(experiment)
            self.assertEqual(resolved["config"]["learning_rate"], 1e-4)

    def test_an_absent_learning_rate_keeps_the_route_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            write_json(path, self._config())
            experiment = Experiment.load(path)
            self.assertIsNone(experiment.learning_rate)
            from agent_code.research_agent.config import EXPERIMENTS
            resolved = resolved_runtime_config(experiment)
            self.assertEqual(resolved["config"]["learning_rate"],
                             EXPERIMENTS[experiment.route].learning_rate)

    def test_the_shipped_step_size_arms_actually_differ(self):
        """The regression that would have caught the empty experiment."""
        experiments = ROOT / "experiments"
        rates = {}
        for name in ("anchor", "lr1e4", "lr5e4"):
            matches = sorted(experiments.glob(f"m4_*{name}*.json"))
            self.assertEqual(len(matches), 1, f"expected exactly one {name} config")
            resolved = resolved_runtime_config(Experiment.load(matches[0]))
            rates[name] = resolved["config"]["learning_rate"]
        self.assertEqual(len(set(rates.values())), 3,
                         f"the step-size arms must resolve to three different rates, got {rates}")
        self.assertEqual(rates["lr1e4"], 1e-4)
        self.assertEqual(rates["lr5e4"], 5e-4)


class KillDiagnosticDoesNotLeakAcrossRoundsTest(unittest.TestCase):
    """A bomb dropped near the end of a round has no tick 3 or 4.

    ``open_drops`` tracks a drop for the four ticks a bomb lives, to measure how
    often A07's term is still non-zero when the follow-up transitions land. It
    was initialised once for the whole run, so a drop left open when a round
    ended kept ageing on the next round's board -- crediting follow-up ticks to
    a position the bomb never occupied, and inflating the very activation rate
    the diagnostic exists to report.
    """

    def test_the_round_loop_clears_the_pending_drops(self):
        source = (ROOT / "scripts" / "diagnose_kill_opportunities.py").read_text(encoding="utf-8")
        new_round = source.index("world.new_round()")
        step_loop = source.index("while world.running:", new_round)
        between = source[new_round:step_loop]
        self.assertIn("open_drops.clear()", between,
                      "pending drops must be cleared when a new round starts, "
                      "before that round's steps are walked")


class ShapedArmsStayOffTheNStepResonanceTest(unittest.TestCase):
    """Which arms may use which n is experiment policy, not model behaviour.

    Potential shaping contributes ``gamma^n phi(s_t+n) - phi(s_t)`` to an
    n-step target, so a potential whose transient is exactly n long telescopes
    to nothing.  A bomb lives BOMB_TIMER+1 transitions, and both potential_v1
    (A06) and potential_v2 (A07) are transient over exactly that window --
    measured in test_shaping, which is where that behaviour belongs.

    What lives here is the consequence for the configs, because it is a rule
    about which experiments may be run, and it will need per-arm exceptions the
    moment a legitimate ablation wants one.  Keeping it inside the model tests
    made every such change a two-file edit in an unrelated module.
    """

    def test_no_m4_arm_shapes_at_or_above_the_bomb_lifetime(self):
        from agent_code.research_agent.config import BOMB_TIMER
        checked = 0
        for path in sorted((ROOT / "experiments").glob("m4_*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not (payload.get("shaping") or {}).get("name"):
                continue
            n_step = payload["agent"]["n_step"]
            self.assertLess(
                n_step, BOMB_TIMER + 1,
                f"{path.name} shapes at n_step={n_step}, at or above the bomb's "
                f"{BOMB_TIMER + 1}-transition life, where the danger term telescopes away")
            checked += 1
        self.assertGreater(checked, 0, "no shaped M4 config found to check")


class SnapshotRoundTripsThroughItsOwnParserTest(unittest.TestCase):
    """A job reloads its experiment from the snapshot, so the two must agree.

    ``run_experiment.py`` calls ``Experiment.load`` on
    ``experiment_config.snapshot.json``.  Any field the snapshot writes
    somewhere ``load`` does not read is therefore recorded faithfully in the
    artifact and then dropped on the way back in -- the run's own provenance
    says one thing and the process that trained did another.  That is exactly
    what happened to ``learning_rate``: written at the top level, read from the
    agent block, and the two step-size arms trained at the route default while
    their snapshots said otherwise.

    Asserting the round trip catches the whole class, not the one instance.
    """

    def test_every_shipped_config_survives_a_snapshot_round_trip(self):
        import dataclasses
        checked = 0
        for path in sorted((ROOT / "experiments").glob("*.json")):
            experiment = Experiment.load(path)
            with tempfile.TemporaryDirectory() as temporary:
                snapshot_path = Path(temporary) / "experiment_config.snapshot.json"
                write_json(snapshot_path, experiment.snapshot())
                reloaded = Experiment.load(snapshot_path)
            for field in dataclasses.fields(experiment):
                self.assertEqual(
                    getattr(reloaded, field.name), getattr(experiment, field.name),
                    f"{path.name}: {field.name} does not survive the snapshot round trip",
                )
            checked += 1
        self.assertGreater(checked, 0)
