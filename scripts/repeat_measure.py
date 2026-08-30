#!/usr/bin/env python3
"""Repeat one finished run's evaluation suite, and compare two such repeats.

Why this exists as a script rather than as a habit.  ``rule_based_agent`` calls
``np.random.seed()`` with no argument (agent_code/rule_based_agent/callbacks.py),
so an evaluation against opponents is not reproducible: replaying the identical
checkpoints, scenario seeds and round count still lands somewhere else.  Section
7.20 measured how far -- the fifteen-job pooled tournament number has sd 0.089,
which makes the smallest difference two *single* measurements can support 0.246,
about 7% of the current score.  Nearly every conclusion this line has had to
withdraw was a single draw read as a result.

The protocol that replaced it is: repeat the same jobs n times, pool each repeat
into one number, and compare two arms with Welch on those n numbers.  This
script is that protocol, so it stops being retyped per experiment.

    scaffold  build a run directory that replays one suite n times
    report    pool each repeat, and optionally Welch against a baseline

A repeat measures one *checkpoint*, so a dose-response curve is several
``--checkpoint-round`` values in one scaffold.  Selection stays on the
validation seeds and the quoted number comes from the holdout seeds
(``--seed-role``); the two never mix, because a dose chosen on the same seeds
that report it is not a measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import statistics
import sys
from pathlib import Path

from experiment_lib import FROZEN_OPPONENT_AGENT, ROOT, RUNS_ROOT, git_provenance, write_json

# Job ids are built by run_experiment.build_jobs; this is the same shape read
# back.  The suite is absent for the primary suite, and the round is absent for
# a ``latest`` checkpoint, exactly as they are absent from the id.
JOB_ID = re.compile(
    r"^eval(?:_(?P<suite>.+?))?(?:_round(?P<round>\d{5}))?"
    r"_train(?P<train>\d+)_seed(?P<seed>\d+)(?:_rep(?P<repeat>\d+))?$"
)
METRICS = ("score", "coins", "kills", "suicides", "coins_share")
# The threshold docs/01 section 7.20.6 fixed for this line.  Two arms of ten
# repeats each is roughly 18 degrees of freedom, where 2.3 is close to the 95%
# two-sided critical value; it is quoted as a constant so that no arm silently
# gets an easier one.
DISCRIMINATION_T = 2.3


def _job_metrics(stats_path: Path, agent_name: str) -> dict[str, float]:
    """Per-round metrics for one evaluation job, from the official stats only.

    ``coins_share`` needs every agent's coins, not just ours: a lower coin count
    against three opponents can mean the agent got worse or that the opponents
    got there first, and section 7.19 exists because those were once conflated.
    """
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    mine = stats["by_agent"][agent_name]
    rounds = int(mine["rounds"])
    if rounds < 1:
        raise ValueError(f"{stats_path} reports {rounds} rounds")
    # ``Agent.statistics`` is a defaultdict(int) and agents.py writes a key only
    # when the event actually fired (EVENT_STAT_MAP), so coins, kills and
    # suicides are all absent from a job where that never happened.  Indexing
    # them directly fails *preferentially on the best jobs*: a run with no
    # suicides at all is exactly the one worth keeping, and it was the one that
    # raised KeyError.  aggregate_results.py:237 has always used .get here.
    total_coins = sum(agent.get("coins", 0) for agent in stats["by_agent"].values())
    coins = mine.get("coins", 0)
    # Older stats files predate the explicit kills field; the official score is
    # coins + 5 * kills by definition, so it can always be recovered.
    kills = mine["kills"] if "kills" in mine else (mine["score"] - coins) / 5.0
    return {
        "score": mine["score"] / rounds,
        "coins": coins / rounds,
        "kills": kills / rounds,
        "suicides": mine.get("suicides", 0) / rounds,
        "coins_share": (coins / total_coins) if total_coins else float("nan"),
        "rounds": float(rounds),
    }


def _suite_label(run_dir: Path, suite: str) -> str:
    """Scenario and opponents, so no number is ever quoted without them.

    docs/05 hard rule 8: a score that does not say which scenario it came from
    and whether opponents were present is not comparable to any other score.
    """
    snapshot = json.loads((run_dir / "experiment_config.snapshot.json").read_text(encoding="utf-8"))
    phase = snapshot["evaluation"] if suite == "primary" else snapshot["evaluation_suites"][suite]
    opponents = phase.get("opponents") or []
    # An opponent override replaces the snapshot's list for every job, so the
    # label has to come from the override when there is one -- otherwise a
    # transfer measurement would be labelled with the opponents it did not play.
    provenance_path = run_dir / "provenance.json"
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        override = (provenance.get("repeat_measurement") or {}).get("opponents")
        if override:
            opponents = list(override)
    if not opponents:
        return f"{phase['scenario']} solo"
    # Which opponents, not how many.  Three rule_based and three frozen copies
    # of a trained agent are both "3 opponents" and the scores are not
    # comparable -- that is the whole point of a transfer measurement.
    kinds = ", ".join(f"{opponents.count(name)}x{name}" for name in sorted(set(opponents)))
    return f"{phase['scenario']} + {kinds}"


def scaffold(args: argparse.Namespace) -> None:
    source = RUNS_ROOT / args.source_run
    destination = RUNS_ROOT / args.run_id
    if not source.is_dir():
        raise SystemExit(f"source run is unavailable: {source}")
    if destination.exists():
        raise SystemExit(f"refusing to overwrite {destination}")

    frozen = None
    if args.opponents and FROZEN_OPPONENT_AGENT in args.opponents:
        if not args.frozen_model:
            raise SystemExit(
                f"--opponents includes {FROZEN_OPPONENT_AGENT}; pass --frozen-model with the "
                "repository-relative checkpoint it should play. There is no default: a "
                "frozen opponent that silently fell back to something else would read as a "
                "weaker opponent rather than as a mistake.")
        model = ROOT / args.frozen_model
        if not model.is_file():
            raise SystemExit(f"frozen opponent checkpoint is unavailable: {model}")
        frozen = {"route": args.frozen_route, "model_path": args.frozen_model,
                  "sha256": hashlib.sha256(model.read_bytes()).hexdigest()}
    elif args.frozen_model:
        raise SystemExit(f"--frozen-model given but {FROZEN_OPPONENT_AGENT} is not among --opponents")

    wanted_rounds = set(args.checkpoint_round) if args.checkpoint_round else None
    candidates: list[dict] = []
    for job_file in sorted((source / "job_parameters").glob("*.json")):
        job = json.loads(job_file.read_text(encoding="utf-8"))
        if job.get("mode") != "eval" or job.get("suite") != args.suite:
            continue
        if job.get("seed_role") == args.seed_role:
            candidates.append(job)
    if not candidates:
        raise SystemExit(f"no {args.seed_role} jobs of suite {args.suite!r} in {source}")

    if wanted_rounds is None:
        selected = [job for job in candidates if job.get("checkpoint_round") is None]
        if not selected:
            # checkpoint_evaluation mode "rounds" addresses the final model by its
            # round number and prepares no ``latest`` job at all, so "give me the
            # finished arm" has to be spelled with the round.  Say which rounds
            # exist rather than only that this one does not.
            available = sorted({job["checkpoint_round"] for job in candidates})
            raise SystemExit(
                f"no {args.seed_role} job addresses the latest checkpoint in {source}; "
                f"this run addresses its checkpoints by round. Pass --checkpoint-round; "
                f"rounds evaluated: {available}")
    else:
        selected = [job for job in candidates if job.get("checkpoint_round") in wanted_rounds]
        # A run saves a checkpoint every ``checkpoint_every`` rounds but only
        # prepares evaluation jobs for the rounds its config named, and a dose
        # curve should be able to ask about a saved checkpoint the original run
        # did not happen to evaluate.  The template is the run's own latest-job
        # for the same suite, seed role, training seed and evaluation seed, so
        # nothing about the evaluation changes except which weights it loads.
        missing = sorted(wanted_rounds - {job.get("checkpoint_round") for job in selected})
        if missing:
            # One template per (training seed, evaluation seed).  A round-addressed
            # job serves as well as a latest one -- the round is overwritten either
            # way, and a run that declares checkpoint_evaluation mode "rounds" has
            # no latest job at all, which is how both fine-tuning arms are written.
            templates: dict[tuple[int, int], dict] = {}
            for job in candidates:
                key = (job["train_seed"], job["seed"])
                if key not in templates or job.get("checkpoint_round") is None:
                    templates[key] = job
            if not templates:
                raise SystemExit(
                    f"rounds {missing} were never evaluated in {source} and there is no job "
                    "to build them from")
            for template in templates.values():
                for checkpoint_round in missing:
                    job_id = (f"eval_{args.suite}_round{checkpoint_round:05d}"
                              f"_train{template['train_seed']}_seed{template['seed']}")
                    selected.append(dict(
                        template, job_id=job_id, model_relpath=None,
                        checkpoint_round=checkpoint_round,
                        checkpoint_search_relpath=str(
                            Path("jobs") / f"train_seed{template['train_seed']}" / "agent" / "checkpoints"),
                        artifact_relpath=str(Path("jobs") / job_id)))
            print(f"rounds {missing} were not evaluated by {args.source_run}; "
                  "built from its latest-checkpoint jobs")
    if not selected:
        raise SystemExit(
            f"no {args.seed_role} jobs of suite {args.suite!r} "
            f"at rounds {sorted(wanted_rounds)} in {source}")

    (destination / "job_parameters").mkdir(parents=True)
    shutil.copy2(source / "experiment_config.snapshot.json", destination / "experiment_config.snapshot.json")
    provenance = git_provenance()
    provenance["repeat_measurement_of"] = args.source_run
    provenance["repeat_measurement"] = {
        "suite": args.suite, "seed_role": args.seed_role, "repeats": args.repeats,
        "checkpoint_rounds": sorted(wanted_rounds) if wanted_rounds else "latest",
    }
    if args.eval_seeds:
        provenance["repeat_measurement"]["eval_seeds"] = list(args.eval_seeds)
    if args.opponents:
        provenance["repeat_measurement"]["opponents"] = list(args.opponents)
        if frozen is not None:
            provenance["repeat_measurement"]["frozen_opponent"] = dict(frozen)
    write_json(destination / "provenance.json", provenance)

    # Copy in exactly the weights the selected jobs address, so a repeat run is
    # self-contained and cannot silently follow the source run's later edits.
    copied: set[Path] = set()
    for job in selected:
        train_seed = job["train_seed"]
        agent_dir = destination / "jobs" / f"train_seed{train_seed}" / "agent"
        if job.get("checkpoint_round") is None:
            origin = source / "jobs" / f"train_seed{train_seed}" / "agent" / "latest_model.npz"
            target = agent_dir / "latest_model.npz"
        else:
            search = source / job["checkpoint_search_relpath"]
            matches = sorted(search.glob(f"*_round{int(job['checkpoint_round']):05d}_*.npz"))
            if len(matches) != 1:
                raise SystemExit(
                    f"train seed {train_seed}: expected one round-{job['checkpoint_round']} "
                    f"checkpoint under {search}, found {len(matches)}"
                )
            origin, target = matches[0], agent_dir / "checkpoints" / matches[0].name
        if target in copied:
            continue
        if not origin.is_file():
            raise SystemExit(f"checkpoint is unavailable: {origin}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, target)
        copied.add(target)

    # One job per (source job, evaluation seed, repeat).  Overriding the seed
    # keeps the checkpoint and the opponents fixed and varies only the board,
    # which is the sole remaining source of variation once the opponents are
    # deterministic.
    seed_variants = list(args.eval_seeds) if args.eval_seeds else [None]
    if args.eval_seeds:
        # Collapse the source's own evaluation seeds first: with an explicit
        # seed list, replaying fifteen source jobs would otherwise multiply
        # every board seed by three identical copies of it.
        by_train: dict[int, dict] = {}
        for job in selected:
            by_train.setdefault((job["train_seed"], job.get("checkpoint_round")), job)
        selected = list(by_train.values())
    written = 0
    for job in selected:
        for eval_seed in seed_variants:
            for repeat in range(1, args.repeats + 1):
                base = job["job_id"]
                if eval_seed is not None:
                    base = re.sub(r"_seed\d+", f"_seed{eval_seed}", base)
                job_id = f"{base}_rep{repeat:02d}"
                payload = dict(job, job_id=job_id, artifact_relpath=str(Path("jobs") / job_id))
                if eval_seed is not None:
                    payload["seed"] = int(eval_seed)
                if args.opponents:
                    payload["opponents"] = list(args.opponents)
                    if frozen is not None:
                        payload["frozen_opponent"] = dict(frozen)
                write_json(destination / "job_parameters" / f"{job_id}.json", payload)
                written += 1
    print(f"{destination}: {written} jobs "
          f"({len(selected)} source jobs x {len(seed_variants)} eval seeds "
          f"x {args.repeats} repeats), {len(copied)} checkpoints")
    # The destination's label, not the source's: with an opponent override they
    # differ, and printing the source's here would announce the opponents this
    # scaffold is specifically not going to play.
    print(f"suite {args.suite} ({_suite_label(destination, args.suite)}), {args.seed_role} seeds")


def _pool(run_dir: Path) -> tuple[str, dict[str | None, dict[int, dict[str, float]]]]:
    """Return the suite and, per checkpoint round, per repeat, one pooled number."""
    snapshot = json.loads((run_dir / "experiment_config.snapshot.json").read_text(encoding="utf-8"))
    agent_name = snapshot["agent"]["name"]
    grouped: dict[str | None, dict[int, list[dict[str, float]]]] = {}
    suites: set[str] = set()
    for job_dir in sorted((run_dir / "jobs").glob("eval*")):
        stats_path = job_dir / "official_stats.json"
        match = JOB_ID.match(job_dir.name)
        if match is None or match.group("repeat") is None or not stats_path.is_file():
            continue
        suites.add(match.group("suite") or "primary")
        round_tag = match.group("round")
        grouped.setdefault(round_tag, {}).setdefault(
            int(match.group("repeat")), []).append(_job_metrics(stats_path, agent_name))
    if not grouped:
        raise SystemExit(
            f"{run_dir} has no readable repeat jobs. Job directories must be named the way "
            "run_experiment builds them, with the repeat suffix appended last: "
            "eval[_<suite>][_round<NNNNN>]_train<seed>_seed<seed>_rep<NN>. "
            "Reporting an empty pool as a number is how a run with the wrong layout becomes a result.")
    if len(suites) > 1:
        raise SystemExit(f"{run_dir} mixes suites {sorted(suites)}; a pooled number must name one")
    pooled: dict[str | None, dict[int, dict[str, float]]] = {}
    for round_tag, repeats in grouped.items():
        expected = max(len(jobs) for jobs in repeats.values())
        complete = {repeat: jobs for repeat, jobs in repeats.items() if len(jobs) == expected}
        dropped = sorted(set(repeats) - set(complete))
        if dropped:
            print(f"  ! round {round_tag or 'latest'}: repeats {dropped} are incomplete and are dropped",
                  file=sys.stderr)
        pooled[round_tag] = {
            repeat: {metric: statistics.fmean(job[metric] for job in jobs) for metric in METRICS}
            for repeat, jobs in complete.items()
        }
    return suites.pop(), pooled


def _welch(left: list[float], right: list[float]) -> tuple[float, float]:
    """Welch's t and its Satterthwaite degrees of freedom, without scipy."""
    n_left, n_right = len(left), len(right)
    if n_left < 2 or n_right < 2:
        return float("nan"), float("nan")
    var_left = statistics.variance(left) / n_left
    var_right = statistics.variance(right) / n_right
    if var_left + var_right == 0:
        return float("nan"), float("nan")
    t = (statistics.fmean(left) - statistics.fmean(right)) / math.sqrt(var_left + var_right)
    df = (var_left + var_right) ** 2 / (
        var_left ** 2 / (n_left - 1) + var_right ** 2 / (n_right - 1))
    return t, df


def report(args: argparse.Namespace) -> None:
    suite, pooled = _pool(args.run_dir)
    label = _suite_label(args.run_dir, suite)
    print(f"{args.run_dir.name}: suite {suite} -- {label}")
    baseline = None
    if args.baseline is not None:
        baseline_suite, baseline_pooled = _pool(args.baseline)
        baseline_label = _suite_label(args.baseline, baseline_suite)
        if (baseline_suite, baseline_label) != (suite, label):
            raise SystemExit(
                f"refusing to compare {suite} ({label}) with {baseline_suite} ({baseline_label}): "
                "scores from different scenarios or opponent counts are not comparable")
        if len(baseline_pooled) != 1:
            raise SystemExit("the baseline must be a single checkpoint, not a dose curve")
        baseline = next(iter(baseline_pooled.values()))
        print(f"baseline {args.baseline.name}: {len(baseline)} repeats")
    for round_tag in sorted(pooled, key=lambda tag: (tag is not None, tag)):
        repeats = pooled[round_tag]
        print(f"\n-- checkpoint {'latest' if round_tag is None else 'round ' + str(int(round_tag))}"
              f"  ({len(repeats)} repeats)")
        for metric in METRICS:
            values = [repeats[repeat][metric] for repeat in sorted(repeats)]
            mean, sd = statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else float("nan"))
            line = f"   {metric:12s} {mean:8.4f}  sd {sd:6.4f}"
            if baseline is not None:
                against = [baseline[repeat][metric] for repeat in sorted(baseline)]
                t, df = _welch(values, against)
                verdict = "distinguishable" if abs(t) > DISCRIMINATION_T else "not distinguishable"
                line += (f"   delta {mean - statistics.fmean(against):+8.4f}"
                         f"   t {t:+6.2f} (df {df:4.1f})  {verdict}")
            print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("scaffold", help="Prepare a run directory that replays one suite n times.")
    build.add_argument("--source-run", required=True, help="Run id whose checkpoints and jobs are replayed.")
    build.add_argument("--run-id", required=True, help="Run id to create under runs/.")
    build.add_argument("--repeats", type=int, required=True)
    build.add_argument("--suite", default="classic_versus_opponents")
    build.add_argument("--seed-role", default="holdout", choices=("validation", "holdout"))
    build.add_argument("--eval-seeds", nargs="+", type=int, metavar="SEED",
                       help="Replace every job's evaluation seed, one job per seed. Against a "
                            "deterministic opponent a repeat is bit-identical and carries no "
                            "information, so board seeds are what 'more samples' has to mean "
                            "there. Recorded in provenance.")
    build.add_argument("--opponents", nargs="+", metavar="AGENT",
                       help="Replace every job's opponent list. Same weights, different "
                            "opponents -- the transfer question. Recorded in provenance and "
                            "in the suite label, so an overridden run can never be compared "
                            "against a differently-opposed one by accident.")
    build.add_argument("--frozen-route", default="R02_9",
                       help="Route whose encoder and network frozen_agent opponents build.")
    build.add_argument("--frozen-model",
                       help="Repository-relative checkpoint the frozen_agent opponents play.")
    build.add_argument("--checkpoint-round", type=int, action="append",
                       help="Repeatable. Omit to replay the latest checkpoint.")
    build.set_defaults(handler=scaffold)
    summarize = subparsers.add_parser("report", help="Pool each repeat and optionally Welch against a baseline.")
    summarize.add_argument("--run-dir", type=Path, required=True)
    summarize.add_argument("--baseline", type=Path, help="A second repeat run, measured on the same suite.")
    summarize.set_defaults(handler=report)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
