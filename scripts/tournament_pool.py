#!/usr/bin/env python3
"""Measure a checkpoint against a pool of opponent tables, and pick a seed on it.

Why a pool.  Every number this project selected on came from one opponent
table -- three ``rule_based_agent`` -- and docs/01 section 7.43 measured what
that buys: the rainbow arm's 4.9x kill advantage on that table fell to 1.1-1.9x
on any opponent it had not trained against.  The tournament is played against
other groups' agents, none of which is ``rule_based_agent``, and the pool is the
closest proxy this repository can build: the best dodger in the framework, a
noisy copy of it standing in for a middling entrant, a coin-only collector, a
random walker standing in for a broken one, a table mixing them, and a table of
trained neural agents.  A candidate is scored on the *mean over pools*, each
pool weighted equally, so that no single opponent's quirks decide the
submission.

    scaffold  build a run directory that plays one checkpoint on every pool
    report    per-pool and pooled means; paired against a baseline on the board seed

Pairing on the board seed is what makes two candidates comparable at this
resolution (docs/01 section 7.45.4): board difficulty is the largest variance
term and both candidates play the same boards.  Against a deterministic
opponent a repeat is bit-identical, so the sample is the number of board seeds;
``rule_based_agent`` and its noisy copy draw fresh entropy every round, so for
them repeats do add information and ``--repeats`` is honoured.
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

from experiment_lib import FROZEN_OPPONENT_AGENTS, ROOT, RUNS_ROOT, git_provenance, write_json
from repeat_measure import DISCRIMINATION_T, METRICS, _job_metrics

SUITE = "tournament_pool"
DEFAULT_BOARD_SEEDS = tuple(range(4001, 4007))
# The pools.  ``frozen_opponents`` entries carry no digest here; scaffold fills
# it in from the file, so a swapped checkpoint changes the recorded digest and
# a missing one fails the scaffold rather than the job.
POOLS: dict[str, dict] = {
    "rb3": {
        "opponents": ["rule_based_agent"] * 3,
        "why": "the framework's best dodger; the table every earlier number was measured on",
    },
    "noisyrb3": {
        "opponents": ["rule_based_noisy_agent"] * 3,
        "why": "rule_based with 15% random actions: a middling entrant that dodges but errs",
    },
    "coin3": {
        "opponents": ["coin_collector_agent"] * 3,
        "why": "collects and avoids, never attacks: a pure coin race",
    },
    "mixed_weak": {
        "opponents": ["rule_based_agent", "coin_collector_agent", "peaceful_agent"],
        "why": "one of each level, including a random walker standing in for a broken entrant",
    },
    "mixed_neural": {
        "opponents": ["rule_based_agent", "frozen_agent", "frozen_agent_b"],
        "frozen_opponents": {
            "frozen_agent": {"route": "R02_11",
                             "model_path": "frozen_opponents/R02_11_rainbow_seed1005_round10000.npz"},
            "frozen_agent_b": {"route": "R07",
                               "model_path": "frozen_opponents/R07_oppbc_seed1004_round10000.npz"},
        },
        "why": "trained neural agents on the board: the M3 rainbow and the M4 opponents+BC checkpoints",
    },
}
DEFAULT_POOLS = ("rb3", "noisyrb3", "coin3", "mixed_weak", "mixed_neural")
JOB_ID = re.compile(
    rf"^eval_{SUITE}_(?P<pool>[A-Za-z0-9_-]+?)"
    r"(?:_round(?P<round>\d{5}))?_train(?P<train>\d+)_seed(?P<seed>\d+)_rep(?P<repeat>\d+)$"
)


def _resolve_source(value: str) -> Path:
    path = Path(value)
    if path.is_dir():
        return path.resolve()
    candidate = RUNS_ROOT / value
    if candidate.is_dir():
        return candidate
    raise SystemExit(f"source run is unavailable: {value}")


def _train_seeds(source: Path) -> list[int]:
    seeds = sorted(int(path.name.removeprefix("train_seed"))
                   for path in (source / "jobs").glob("train_seed*") if path.is_dir())
    if not seeds:
        raise SystemExit(f"{source} has no jobs/train_seed* directories")
    return seeds


def _checkpoint(source: Path, train_seed: int, checkpoint_round: int | None) -> Path:
    agent_dir = source / "jobs" / f"train_seed{train_seed}" / "agent"
    if checkpoint_round is None:
        origin = agent_dir / "latest_model.npz"
        if not origin.is_file():
            raise SystemExit(f"checkpoint is unavailable: {origin}")
        return origin
    matches = sorted((agent_dir / "checkpoints").glob(f"*_round{checkpoint_round:05d}_*.npz"))
    if len(matches) != 1:
        raise SystemExit(
            f"train seed {train_seed}: expected one round-{checkpoint_round} checkpoint under "
            f"{agent_dir / 'checkpoints'}, found {len(matches)}")
    return matches[0]


def _pool_table(name: str) -> dict:
    try:
        pool = POOLS[name]
    except KeyError as exc:
        raise SystemExit(f"unknown pool {name!r}; pools: {', '.join(POOLS)}") from exc
    frozen = {}
    for seat, entry in pool.get("frozen_opponents", {}).items():
        if seat not in FROZEN_OPPONENT_AGENTS:
            raise SystemExit(f"pool {name} seats {seat!r}, which is not a frozen seat")
        model = ROOT / entry["model_path"]
        if not model.is_file():
            raise SystemExit(f"pool {name}: frozen checkpoint is unavailable: {model}")
        frozen[seat] = {**entry, "sha256": hashlib.sha256(model.read_bytes()).hexdigest()}
    return {"opponents": list(pool["opponents"]), "frozen_opponents": frozen, "why": pool.get("why", "")}


def scaffold(args: argparse.Namespace) -> None:
    source = _resolve_source(args.source_run)
    destination = RUNS_ROOT / args.run_id
    if destination.exists():
        raise SystemExit(f"refusing to overwrite {destination}")
    train_seeds = list(args.train_seeds) if args.train_seeds else _train_seeds(source)
    pools = {name: _pool_table(name) for name in (args.pools or DEFAULT_POOLS)}
    checkpoint_round = args.checkpoint_round

    (destination / "job_parameters").mkdir(parents=True)
    shutil.copy2(source / "experiment_config.snapshot.json",
                 destination / "experiment_config.snapshot.json")
    provenance = git_provenance()
    provenance["tournament_pool_of"] = str(source)
    provenance["tournament_pool"] = {
        "checkpoint_round": checkpoint_round if checkpoint_round is not None else "latest",
        "train_seeds": train_seeds, "board_seeds": list(args.board_seeds),
        "repeats": args.repeats, "scenario": args.scenario, "rounds": args.rounds,
        "pools": pools,
    }
    write_json(destination / "provenance.json", provenance)

    round_tag = "" if checkpoint_round is None else f"_round{checkpoint_round:05d}"
    written = 0
    for train_seed in train_seeds:
        origin = _checkpoint(source, train_seed, checkpoint_round)
        agent_dir = destination / "jobs" / f"train_seed{train_seed}" / "agent"
        if checkpoint_round is None:
            target = agent_dir / "latest_model.npz"
            model_relpath: str | None = str(target.relative_to(destination))
        else:
            target = agent_dir / "checkpoints" / origin.name
            model_relpath = None
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, target)
        for pool_name, pool in pools.items():
            for board_seed in args.board_seeds:
                for repeat in range(1, args.repeats + 1):
                    job_id = (f"eval_{SUITE}_{pool_name}{round_tag}"
                              f"_train{train_seed}_seed{board_seed}_rep{repeat:02d}")
                    payload = {
                        "job_id": job_id, "mode": "eval", "seed": int(board_seed),
                        "train_seed": int(train_seed), "phase": "evaluation", "suite": SUITE,
                        "pool": pool_name, "seed_role": "holdout",
                        "scenario": args.scenario, "opponents": pool["opponents"],
                        "budget": {"rounds": int(args.rounds), "checkpoint_every": int(args.rounds)},
                        "artifact_relpath": str(Path("jobs") / job_id),
                        "model_relpath": model_relpath,
                        "checkpoint_round": checkpoint_round,
                    }
                    if pool["frozen_opponents"]:
                        payload["frozen_opponents"] = pool["frozen_opponents"]
                    if checkpoint_round is not None:
                        payload["checkpoint_search_relpath"] = str(
                            (agent_dir / "checkpoints").relative_to(destination))
                    write_json(destination / "job_parameters" / f"{job_id}.json", payload)
                    written += 1
    print(f"{destination}: {written} jobs "
          f"({len(train_seeds)} train seeds x {len(pools)} pools x {len(args.board_seeds)} boards "
          f"x {args.repeats} repeats)")
    for name, pool in pools.items():
        print(f"  {name:13s} {', '.join(pool['opponents'])}")


def _collect(run_dir: Path) -> dict[str, dict[tuple[int, int, int], dict[str, float]]]:
    """Per pool, per (train seed, board seed, repeat), one job's metrics."""
    snapshot = json.loads((run_dir / "experiment_config.snapshot.json").read_text(encoding="utf-8"))
    agent_name = snapshot["agent"]["name"]
    pools: dict[str, dict[tuple[int, int, int], dict[str, float]]] = {}
    for job_dir in sorted((run_dir / "jobs").glob(f"eval_{SUITE}_*")):
        match = JOB_ID.match(job_dir.name)
        stats = job_dir / "official_stats.json"
        if match is None or not stats.is_file():
            continue
        key = (int(match.group("train")), int(match.group("seed")), int(match.group("repeat")))
        pools.setdefault(match.group("pool"), {})[key] = _job_metrics(stats, agent_name)
    if not pools:
        raise SystemExit(f"{run_dir} has no finished {SUITE} jobs")
    return pools


def _by_board(jobs: dict[tuple[int, int, int], dict[str, float]], metric: str,
              train_seed: int | None = None) -> dict[int, float]:
    """One number per board seed: the mean over train seeds and repeats."""
    grouped: dict[int, list[float]] = {}
    for (train, board, _repeat), metrics in jobs.items():
        if train_seed is not None and train != train_seed:
            continue
        grouped.setdefault(board, []).append(metrics[metric])
    return {board: statistics.fmean(values) for board, values in grouped.items()}


def _paired(left: dict[int, float], right: dict[int, float]) -> tuple[float, float, int]:
    """right minus left over the boards both have: mean, paired t, n."""
    common = sorted(set(left) & set(right))
    diffs = [right[board] - left[board] for board in common]
    if len(diffs) < 2:
        return (statistics.fmean(diffs) if diffs else float("nan")), float("nan"), len(diffs)
    mean = statistics.fmean(diffs)
    sd = statistics.stdev(diffs)
    t = mean / (sd / math.sqrt(len(diffs))) if sd else float("nan")
    return mean, t, len(diffs)


def report(args: argparse.Namespace) -> None:
    pools = _collect(args.run_dir)
    baseline = _collect(args.baseline) if args.baseline is not None else None
    train_seeds = sorted({key[0] for jobs in pools.values() for key in jobs})
    print(f"{args.run_dir.name}: {len(pools)} pools, train seeds {train_seeds}")
    summary: dict = {"pools": {}, "train_seeds": {}}

    print(f"\n{'pool':13s} {'n':>3s} " + " ".join(f"{metric:>12s}" for metric in METRICS)
          + ("      paired vs baseline (score)" if baseline else ""))
    overall: dict[str, list[float]] = {metric: [] for metric in METRICS}
    overall_diffs: list[float] = []
    for pool_name in sorted(pools):
        jobs = pools[pool_name]
        row = {}
        for metric in METRICS:
            boards = _by_board(jobs, metric)
            row[metric] = statistics.fmean(boards.values())
            overall[metric].append(row[metric])
        line = f"{pool_name:13s} {len(jobs):3d} " + " ".join(f"{row[metric]:12.4f}" for metric in METRICS)
        if baseline is not None and pool_name in baseline:
            mean, t, n = _paired(_by_board(baseline[pool_name], "score"), _by_board(jobs, "score"))
            flag = "*" if abs(t) > DISCRIMINATION_T else " "
            line += f"   delta {mean:+8.4f}  t {t:+6.2f}{flag} (n {n})"
            row["delta_score"], row["t"], row["n_paired"] = mean, t, n
            for board, value in _by_board(jobs, "score").items():
                base = _by_board(baseline[pool_name], "score").get(board)
                if base is not None:
                    overall_diffs.append(value - base)
        print(line)
        summary["pools"][pool_name] = row
    pooled = {metric: statistics.fmean(values) for metric, values in overall.items()}
    line = f"{'MEAN OVER POOLS':13s} {'':3s} " + " ".join(f"{pooled[metric]:12.4f}" for metric in METRICS)
    if overall_diffs and len(overall_diffs) > 1:
        mean = statistics.fmean(overall_diffs)
        sd = statistics.stdev(overall_diffs)
        t = mean / (sd / math.sqrt(len(overall_diffs))) if sd else float("nan")
        flag = "*" if abs(t) > DISCRIMINATION_T else " "
        line += f"   delta {mean:+8.4f}  t {t:+6.2f}{flag} (n {len(overall_diffs)})"
        summary["overall_delta_score"], summary["overall_t"] = mean, t
    print(line)
    summary["mean_over_pools"] = pooled
    print(f"\n* = |t| > {DISCRIMINATION_T}, paired on the board seed within each pool.")

    # The seed a submission is chosen from: mean over pools of the per-seed mean.
    print(f"\n{'train seed':11s} " + " ".join(f"{name:>12s}" for name in sorted(pools)) + f" {'MEAN':>12s}")
    ranking = []
    for train_seed in train_seeds:
        per_pool = []
        for pool_name in sorted(pools):
            boards = _by_board(pools[pool_name], "score", train_seed=train_seed)
            per_pool.append(statistics.fmean(boards.values()) if boards else float("nan"))
        mean = statistics.fmean(value for value in per_pool if not math.isnan(value))
        ranking.append((mean, train_seed))
        summary["train_seeds"][str(train_seed)] = {
            "per_pool": dict(zip(sorted(pools), per_pool)), "mean_over_pools": mean}
        print(f"{train_seed:<11d} " + " ".join(f"{value:12.4f}" for value in per_pool) + f" {mean:12.4f}")
    best = max(ranking)
    summary["selected_train_seed"] = best[1]
    print(f"\nselected train seed {best[1]} at {best[0]:.4f} (highest mean over pools; "
          "quote it on a different pool run than the one it was picked on)")
    if args.json is not None:
        write_json(args.json, summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("scaffold", help="Prepare a run directory that plays one checkpoint on every pool.")
    build.add_argument("--source-run", required=True,
                       help="Run id under runs/, or a path to a run directory, holding the checkpoints.")
    build.add_argument("--run-id", required=True, help="Run id to create under runs/.")
    build.add_argument("--checkpoint-round", type=int, help="Omit to play each seed's latest checkpoint.")
    build.add_argument("--train-seeds", nargs="+", type=int, metavar="SEED",
                       help="Default: every train_seed* directory of the source run.")
    build.add_argument("--board-seeds", nargs="+", type=int, default=list(DEFAULT_BOARD_SEEDS), metavar="SEED")
    build.add_argument("--repeats", type=int, default=1)
    build.add_argument("--pools", nargs="+", metavar="POOL", help=f"Default: {' '.join(DEFAULT_POOLS)}")
    build.add_argument("--scenario", default="classic")
    build.add_argument("--rounds", type=int, default=30, help="Rounds per job.")
    build.set_defaults(handler=scaffold)
    summarize = subparsers.add_parser("report", help="Per-pool and pooled means, optionally paired against a baseline.")
    summarize.add_argument("--run-dir", type=Path, required=True)
    summarize.add_argument("--baseline", type=Path, help="Another pool run on the same boards.")
    summarize.add_argument("--json", type=Path, help="Also write the summary here.")
    summarize.set_defaults(handler=report)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
