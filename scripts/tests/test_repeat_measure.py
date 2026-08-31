"""Tests for the repeat-measurement protocol.

The protocol is what stands between this line and its own history: nearly every
withdrawn conclusion here was one draw of an irreproducible tournament number
read as a result (docs/01 section 7.20).  Encoding it in a script only helps if
the script refuses the mistakes the habit used to make -- pooling an incomplete
repeat, comparing two different scenarios, or letting the holdout seeds pick the
checkpoint they then report.
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

import repeat_measure  # noqa: E402
from experiment_lib import write_json  # noqa: E402

TRAIN_SEEDS = (1001, 1002)
VALIDATION, HOLDOUT = (2001,), (3001,)


def _snapshot() -> dict:
    return {
        "agent": {"name": "research_agent"},
        "evaluation": {"scenario": "loot-crate", "opponents": ["rule_based_agent"], "seeds": [2001]},
        "evaluation_suites": {
            "classic_versus_opponents": {
                "scenario": "classic",
                "opponents": ["rule_based_agent", "rule_based_agent", "rule_based_agent"],
                "seeds": [2001],
            },
            "classic_solo": {"scenario": "classic", "opponents": [], "seeds": [2001]},
        },
    }


def _source_run(root: Path, rounds: tuple[int, ...] = (500, 1000)) -> Path:
    """A finished run with a checkpoint sweep on the tournament suite.

    ``rounds`` are the saved checkpoints; only those the config declared get an
    evaluation job.  The fixture declares every round except 250, which stands
    for "saved, but never evaluated".
    """
    evaluated = tuple(one for one in rounds if one != 250)
    run_dir = root / "source"
    write_json(run_dir / "experiment_config.snapshot.json", _snapshot())
    for train_seed in TRAIN_SEEDS:
        checkpoints = run_dir / "jobs" / f"train_seed{train_seed}" / "agent" / "checkpoints"
        checkpoints.mkdir(parents=True)
        for checkpoint_round in rounds:
            (checkpoints / f"R02_9_A06_classic_seed{train_seed}_round{checkpoint_round:05d}"
                           f"_updates00000100.npz").write_bytes(b"weights")
        (run_dir / "jobs" / f"train_seed{train_seed}" / "agent" / "latest_model.npz").write_bytes(b"weights")
        for role, seeds in (("validation", VALIDATION), ("holdout", HOLDOUT)):
            for seed in seeds:
                for checkpoint_round in evaluated:
                    job_id = (f"eval_classic_versus_opponents_round{checkpoint_round:05d}"
                              f"_train{train_seed}_seed{seed}")
                    write_json(run_dir / "job_parameters" / f"{job_id}.json", {
                        "job_id": job_id, "mode": "eval", "suite": "classic_versus_opponents",
                        "seed_role": role, "seed": seed, "train_seed": train_seed,
                        "checkpoint_round": checkpoint_round,
                        "checkpoint_search_relpath": f"jobs/train_seed{train_seed}/agent/checkpoints",
                        "model_relpath": None,
                        "artifact_relpath": f"jobs/{job_id}",
                    })
                job_id = f"eval_classic_versus_opponents_train{train_seed}_seed{seed}"
                write_json(run_dir / "job_parameters" / f"{job_id}.json", {
                    "job_id": job_id, "mode": "eval", "suite": "classic_versus_opponents",
                    "seed_role": role, "seed": seed, "train_seed": train_seed,
                    "checkpoint_round": None,
                    "model_relpath": f"jobs/train_seed{train_seed}/agent/latest_model.npz",
                    "artifact_relpath": f"jobs/{job_id}",
                })
    return run_dir


def _finished_repeats(root: Path, name: str, scores: dict[int, float],
                      suite: str = "classic_versus_opponents", drop: tuple[str, ...] = ()) -> Path:
    """A repeat run whose jobs have already produced official stats."""
    run_dir = root / name
    write_json(run_dir / "experiment_config.snapshot.json", _snapshot())
    infix = f"_{suite}" if suite != "primary" else ""
    for repeat, score in scores.items():
        for train_seed in TRAIN_SEEDS:
            for seed in HOLDOUT:
                job_id = f"eval{infix}_train{train_seed}_seed{seed}_rep{repeat:02d}"
                if job_id in drop:
                    continue
                write_json(run_dir / "jobs" / job_id / "official_stats.json", {
                    "by_agent": {
                        "research_agent": {"rounds": 30, "score": score * 30, "coins": score * 30,
                                           "kills": 0.0, "suicides": 9.0},
                        "rule_based_agent": {"rounds": 30, "score": 30.0, "coins": 30.0,
                                             "kills": 0.0, "suicides": 3.0},
                    }
                })
    return run_dir


class BoardSeedsAreWhatMoreSamplesMeansTest(unittest.TestCase):
    """Against a deterministic opponent a repeat carries no information.

    frozen_agent and coin_collector_agent play the same game every time for a
    given board seed, so ten repeats are bit-identical and the between-seed
    spread they appear to show is the spread of three board seeds. More board
    seeds is the only way to add samples there.
    """

    def _scaffold(self, root, **overrides):
        arguments = Namespace(source_run="source", run_id="repeat", repeats=1,
                              suite="classic_versus_opponents", seed_role="holdout",
                              checkpoint_round=None, opponents=None, eval_seeds=None,
                              frozen_route="R02_9", frozen_model=None)
        for key, value in overrides.items():
            setattr(arguments, key, value)
        with patch.object(repeat_measure, "RUNS_ROOT", root):
            repeat_measure.scaffold(arguments)
        return root / arguments.run_id

    def test_each_declared_seed_becomes_its_own_job(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _source_run(root)
            destination = self._scaffold(root, eval_seeds=[7001, 7002, 7003, 7004])
            seeds = set()
            for job_file in (destination / "job_parameters").glob("*.json"):
                seeds.add(json.loads(job_file.read_text(encoding="utf-8"))["seed"])
            self.assertEqual(seeds, {7001, 7002, 7003, 7004})

    def test_the_source_seeds_are_not_multiplied_in(self):
        """One job per (training seed, board seed), not per source job as well.

        The source's own evaluation seeds are collapsed first: replaying them
        under an explicit seed list would write one identical copy of each
        board seed per source evaluation seed.
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _source_run(root)
            destination = self._scaffold(root, eval_seeds=[7001, 7002])
            pairs = []
            for job_file in (destination / "job_parameters").glob("*.json"):
                job = json.loads(job_file.read_text(encoding="utf-8"))
                pairs.append((job["train_seed"], job["seed"], job.get("checkpoint_round")))
            self.assertEqual(len(pairs), len(set(pairs)), "a (train seed, board seed) pair was written twice")
            train_seeds = {train for train, _, _ in pairs}
            self.assertEqual(len(pairs), len(train_seeds) * 2)

    def test_job_ids_stay_distinct(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _source_run(root)
            destination = self._scaffold(root, eval_seeds=[7001, 7002])
            names = [p.name for p in (destination / "job_parameters").glob("*.json")]
            self.assertEqual(len(names), len(set(names)))

    def test_the_seeds_are_recorded_in_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _source_run(root)
            destination = self._scaffold(root, eval_seeds=[7001, 7002])
            provenance = json.loads((destination / "provenance.json").read_text(encoding="utf-8"))
            self.assertEqual(provenance["repeat_measurement"]["eval_seeds"], [7001, 7002])


class AJobWithNothingBadInItStillParsesTest(unittest.TestCase):
    """The missing key fails on the best jobs, not on random ones.

    Agent.statistics is a defaultdict(int) and agents.py only writes a key when
    the event fired, so a job with no suicides has no 'suicides' key -- and a
    job with no suicides is the one worth keeping. Reading it by index crashed
    the report for two whole transfer arms.
    """

    def test_missing_event_keys_read_as_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "official_stats.json"
            path.write_text(json.dumps({
                "by_agent": {
                    "research_agent": {"score": 12, "rounds": 3},
                    "rule_based_agent_0": {"score": 4, "rounds": 3},
                }
            }), encoding="utf-8")
            metrics = repeat_measure._job_metrics(path, "research_agent")
        self.assertEqual(metrics["suicides"], 0.0)
        self.assertEqual(metrics["coins"], 0.0)
        # score = coins + 5*kills, so 12 points with no coins is 2.4 kills.
        self.assertAlmostEqual(metrics["kills"], 12 / 5.0 / 3)

    def test_a_board_where_nobody_took_a_coin_is_not_a_crash(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "official_stats.json"
            path.write_text(json.dumps({
                "by_agent": {"research_agent": {"score": 0, "rounds": 2}}
            }), encoding="utf-8")
            metrics = repeat_measure._job_metrics(path, "research_agent")
        self.assertTrue(metrics["coins_share"] != metrics["coins_share"])  # nan


class TheOpponentIsPartOfTheMeasurementTest(unittest.TestCase):
    """Same weights, different opponents: the transfer question.

    docs/05 hard rule 8 says a score has to name its scenario and whether
    opponents were present. Three rule_based and three frozen copies of a
    trained agent are both "3 opponents", so the label has to name which, or
    two incomparable numbers sit in one table looking comparable.
    """

    def _scaffold(self, root: Path, **overrides):
        arguments = Namespace(source_run="source", run_id="repeat", repeats=1,
                              suite="classic_versus_opponents", seed_role="holdout",
                              checkpoint_round=None, opponents=None, eval_seeds=None,
                              frozen_route="R02_9", frozen_model=None)
        for key, value in overrides.items():
            setattr(arguments, key, value)
        with patch.object(repeat_measure, "RUNS_ROOT", root):
            repeat_measure.scaffold(arguments)
        return root / arguments.run_id

    def test_the_override_reaches_every_job(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _source_run(root)
            destination = self._scaffold(root, opponents=["coin_collector_agent"] * 3)
            written = sorted((destination / "job_parameters").glob("*.json"))
            self.assertTrue(written)
            for job_file in written:
                job = json.loads(job_file.read_text(encoding="utf-8"))
                self.assertEqual(job["opponents"], ["coin_collector_agent"] * 3)

    def test_the_override_is_recorded_in_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _source_run(root)
            destination = self._scaffold(root, opponents=["coin_collector_agent"] * 3)
            provenance = json.loads((destination / "provenance.json").read_text(encoding="utf-8"))
            self.assertEqual(provenance["repeat_measurement"]["opponents"],
                             ["coin_collector_agent"] * 3)

    def test_the_label_names_which_opponents_not_how_many(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _source_run(root)
            plain = self._scaffold(root, run_id="plain")
            swapped = self._scaffold(root, run_id="swapped",
                                     opponents=["coin_collector_agent"] * 3)
            suite = "classic_versus_opponents"
            self.assertNotEqual(repeat_measure._suite_label(plain, suite),
                                repeat_measure._suite_label(swapped, suite))
            self.assertIn("coin_collector_agent", repeat_measure._suite_label(swapped, suite))

    def test_a_frozen_opponent_without_a_model_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _source_run(root)
            with self.assertRaises(SystemExit) as caught:
                self._scaffold(root, opponents=["frozen_agent"] * 3)
            self.assertIn("--frozen-model", str(caught.exception))

    def test_a_model_without_a_frozen_opponent_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _source_run(root)
            with self.assertRaises(SystemExit):
                self._scaffold(root, opponents=["rule_based_agent"] * 3,
                               frozen_model="frozen_opponents/R02_9_seed1001_round05000.npz")


class ScaffoldSelectsExactlyWhatWasAskedForTest(unittest.TestCase):
    def _scaffold(self, root: Path, **overrides):
        arguments = Namespace(source_run="source", run_id="repeat", repeats=3,
                              suite="classic_versus_opponents", seed_role="holdout",
                              checkpoint_round=None, opponents=None, eval_seeds=None,
                              frozen_route="R02_9", frozen_model=None)
        for key, value in overrides.items():
            setattr(arguments, key, value)
        with patch.object(repeat_measure, "RUNS_ROOT", root):
            repeat_measure.scaffold(arguments)
        return root / arguments.run_id

    def test_the_latest_checkpoint_is_replayed_once_per_repeat(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _source_run(root)
            destination = self._scaffold(root)
            jobs = sorted(path.stem for path in (destination / "job_parameters").glob("*.json"))
            self.assertEqual(len(jobs), len(TRAIN_SEEDS) * len(HOLDOUT) * 3)
            self.assertTrue(all(job.endswith(("_rep01", "_rep02", "_rep03")) for job in jobs))
            self.assertTrue(all("_round" not in job for job in jobs))
            for train_seed in TRAIN_SEEDS:
                self.assertTrue((destination / "jobs" / f"train_seed{train_seed}"
                                 / "agent" / "latest_model.npz").is_file())

    def test_the_holdout_seeds_are_not_mixed_with_the_validation_seeds(self):
        """Selection happens on validation, reporting on holdout, never both."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _source_run(root)
            destination = self._scaffold(root, seed_role="validation")
            seeds = {json.loads(path.read_text())["seed"]
                     for path in (destination / "job_parameters").glob("*.json")}
            self.assertEqual(seeds, set(VALIDATION))

    def test_a_dose_curve_is_several_rounds_in_one_scaffold(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _source_run(root)
            destination = self._scaffold(root, checkpoint_round=[500, 1000], repeats=2)
            rounds = {json.loads(path.read_text())["checkpoint_round"]
                      for path in (destination / "job_parameters").glob("*.json")}
            self.assertEqual(rounds, {500, 1000})
            for train_seed in TRAIN_SEEDS:
                copied = sorted(path.name for path in (destination / "jobs" / f"train_seed{train_seed}"
                                                      / "agent" / "checkpoints").glob("*.npz"))
                self.assertEqual([name.split("_round")[1][:5] for name in copied], ["00500", "01000"])

    def test_a_run_with_no_latest_job_still_yields_an_unevaluated_dose(self):
        """mode "rounds" prepares no latest job, and the curve still has to work.

        Both fine-tuning arms declare that mode, so the round-250 dose had no
        latest job to template from and the whole curve came back empty.
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _source_run(root, rounds=(250, 500, 1000))
            for path in (source / "job_parameters").glob("*.json"):
                if json.loads(path.read_text())["checkpoint_round"] is None:
                    path.unlink()
            destination = self._scaffold(root, checkpoint_round=[250, 1000], repeats=2)
            jobs = [json.loads(path.read_text())
                    for path in (destination / "job_parameters").glob("*.json")]
            self.assertEqual({job["checkpoint_round"] for job in jobs}, {250, 1000})
            # One job per (training seed, evaluation seed, round, repeat) and no more:
            # every round-addressed job of the source must not become its own template.
            self.assertEqual(len(jobs), len(TRAIN_SEEDS) * len(HOLDOUT) * 2 * 2)

    def test_a_run_that_addresses_only_rounds_says_which_rounds_it_has(self):
        """mode "rounds" prepares no latest job, so the finished arm needs a round."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _source_run(root)
            for path in (source / "job_parameters").glob("*.json"):
                if json.loads(path.read_text())["checkpoint_round"] is None:
                    path.unlink()
            with self.assertRaises(SystemExit) as raised:
                self._scaffold(root)
            self.assertIn("[500, 1000]", str(raised.exception))

    def test_a_round_with_no_saved_checkpoint_is_refused(self):
        """Not evaluated is recoverable; not saved is not."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _source_run(root)
            with self.assertRaises(SystemExit):
                self._scaffold(root, checkpoint_round=[750])

    def test_a_saved_checkpoint_the_run_never_evaluated_is_built_from_its_latest_job(self):
        """A run saves every checkpoint_every rounds but evaluates only what it declared.

        A dose curve that could ask about only the declared rounds would be
        limited by a choice made before any of the doses were seen, which is
        exactly backwards: the first dose curve here was flat after round 500
        and the interesting question moved to a round the run had not evaluated.
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _source_run(root, rounds=(250, 500, 1000))
            # The run evaluated 500 and 1000; 250 exists on disk but was not asked about.
            self.assertFalse(any("round00250" in path.name
                                 for path in (source / "job_parameters").glob("*.json")))
            destination = self._scaffold(root, checkpoint_round=[250, 500], repeats=2)
            jobs = [json.loads(path.read_text())
                    for path in (destination / "job_parameters").glob("*.json")]
            self.assertEqual({job["checkpoint_round"] for job in jobs}, {250, 500})
            self.assertEqual({job["seed_role"] for job in jobs}, {"holdout"})
            for job in jobs:
                self.assertIsNone(job["model_relpath"], "a round-addressed job resolves by round")
                self.assertEqual(job["checkpoint_search_relpath"],
                                 f"jobs/train_seed{job['train_seed']}/agent/checkpoints")
            for train_seed in TRAIN_SEEDS:
                copied = sorted(path.name for path in (destination / "jobs" / f"train_seed{train_seed}"
                                                      / "agent" / "checkpoints").glob("*.npz"))
                self.assertEqual([name.split("_round")[1][:5] for name in copied], ["00250", "00500"])

    def test_an_existing_destination_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _source_run(root)
            self._scaffold(root)
            with self.assertRaises(SystemExit):
                self._scaffold(root)


class ReportPoolsAndDiscriminatesTest(unittest.TestCase):
    def _report(self, run_dir: Path, baseline: Path | None = None):
        repeat_measure.report(Namespace(run_dir=run_dir, baseline=baseline))

    def test_a_difference_below_the_noise_floor_is_reported_as_not_distinguishable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arm = _finished_repeats(root, "arm", {1: 3.60, 2: 3.70, 3: 3.65, 4: 3.55})
            control = _finished_repeats(root, "control", {1: 3.62, 2: 3.68, 3: 3.60, 4: 3.60})
            _, pooled = repeat_measure._pool(arm)
            self.assertEqual(list(pooled), [None])
            left = [pooled[None][repeat]["score"] for repeat in sorted(pooled[None])]
            _, control_pooled = repeat_measure._pool(control)
            right = [control_pooled[None][repeat]["score"] for repeat in sorted(control_pooled[None])]
            t, _ = repeat_measure._welch(left, right)
            self.assertLess(abs(t), repeat_measure.DISCRIMINATION_T)

    def test_a_real_effect_clears_the_threshold(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arm = _finished_repeats(root, "arm", {1: 4.60, 2: 4.70, 3: 4.65, 4: 4.55})
            control = _finished_repeats(root, "control", {1: 3.62, 2: 3.68, 3: 3.60, 4: 3.60})
            _, arm_pooled = repeat_measure._pool(arm)
            _, control_pooled = repeat_measure._pool(control)
            t, df = repeat_measure._welch(
                [arm_pooled[None][r]["score"] for r in sorted(arm_pooled[None])],
                [control_pooled[None][r]["score"] for r in sorted(control_pooled[None])])
            self.assertGreater(abs(t), repeat_measure.DISCRIMINATION_T)
            self.assertGreater(df, 1.0)

    def test_an_incomplete_repeat_is_dropped_rather_than_averaged_over_fewer_jobs(self):
        """Half a repeat is a different, noisier number, not a smaller sample."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arm = _finished_repeats(
                root, "arm", {1: 3.60, 2: 3.70, 3: 3.65},
                drop=("eval_classic_versus_opponents_train1002_seed3001_rep03",))
            _, pooled = repeat_measure._pool(arm)
            self.assertEqual(sorted(pooled[None]), [1, 2])

    def test_coins_share_counts_every_agents_coins(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arm = _finished_repeats(root, "arm", {1: 3.0})
            _, pooled = repeat_measure._pool(arm)
            # 3.0 coins per round for us against 1.0 for the single opponent.
            self.assertAlmostEqual(pooled[None][1]["coins_share"], 0.75)

    def test_an_unreadable_layout_is_refused_rather_than_pooled_as_nothing(self):
        """The failure mode this replaced: a silent empty pool with a default name.

        Repeat runs scaffolded by hand put the repeat index in whatever position
        the author liked, and a reader that skipped what it could not parse
        reported an empty directory as a suite it had never seen.
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = _finished_repeats(root, "arm", {1: 3.6, 2: 3.7})
            for job_dir in sorted((run_dir / "jobs").glob("eval*")):
                job_dir.rename(job_dir.with_name(job_dir.name.replace("_rep", "_r7000_rep")))
            with self.assertRaises(SystemExit):
                repeat_measure._pool(run_dir)

    def test_a_counter_that_never_fired_reads_as_zero_not_as_a_crash(self):
        """``Agent.statistics`` is a defaultdict; a zero counter is ABSENT.

        This crashed on exactly the best-behaved jobs -- the ones where the
        agent never once killed itself -- so treating it as a broken job would
        have deleted the strongest samples from the pool.  ``rep10k_opponents``
        had two of them and ``rep10k_anchor`` had none.
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = _finished_repeats(root, "arm", {1: 3.6})
            stats_path = next((run_dir / "jobs").glob("eval*"))/ "official_stats.json"
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            for agent in stats["by_agent"].values():
                agent.pop("suicides", None)
            write_json(stats_path, stats)
            metrics = repeat_measure._job_metrics(stats_path, "research_agent")
            self.assertEqual(metrics["suicides"], 0.0)

    def test_two_scenarios_are_refused_rather_than_tabulated_side_by_side(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tournament = _finished_repeats(root, "arm", {1: 3.6, 2: 3.7})
            loot_crate = _finished_repeats(root, "control", {1: 3.6, 2: 3.7}, suite="primary")
            with self.assertRaises(SystemExit):
                self._report(tournament, baseline=loot_crate)


if __name__ == "__main__":
    unittest.main()
