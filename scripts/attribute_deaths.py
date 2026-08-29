#!/usr/bin/env python3
"""Split a run's deaths into own-bomb and killed-by-whom, per evaluation job.

``official_stats.json`` counts ``suicides``, which is ``KILLED_SELF`` only.
There is no counter for being blown up by somebody else: ``GOT_KILLED`` is
absent from ``agents.EVENT_STAT_MAP``, and an agent killed by its own bomb
raises both events anyway.  So in every arm that has opponents, the share of
deaths caused by an opponent is simply not in the statistics, and a suicide rate
of 0.5 says nothing about how often the agent is killed rather than kills
itself.

This reads the framework's own game log instead, where ``evaluate_explosions``
writes one line per death naming the bomb's owner.  Nothing is re-simulated,
which matters here: ``rule_based_agent.setup`` calls ``np.random.seed()`` with
no argument and shuffles with the unseeded ``random`` module, so a game with
those opponents cannot be replayed and the replay-based diagnostic in
``diagnose_bomb_escape.py`` does not apply to them.  Parsing the log of the run
that actually happened is exact regardless.

The acceptance test is that own-bomb deaths counted from the log equal the
``suicides`` field the framework wrote independently into the statistics.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import re
from pathlib import Path

from experiment_lib import write_json
from framework_log import usable_jobs

OWN_BOMB = re.compile(r"Agent <(?P<victim>[^>]+)> blown up by own bomb")
OTHER_BOMB = re.compile(r"Agent <(?P<victim>[^>]+)> blown up by agent <(?P<killer>[^>]+)>'s bomb")
ROUND_START = re.compile(r"STARTING ROUND #(?P<round>\d+)")


def read_log(job_dir: Path):
    """Yield the lines of a job's game log, compressed or not."""
    path = job_dir / "framework_game.log.gz"
    opener = gzip.open
    if not path.exists():
        path = job_dir / "framework_game.log"
        opener = open
    if not path.exists():
        return
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        yield from handle


def attribute_job(job_dir: Path, agent: str) -> dict:
    """Classify how each of the agent's rounds ended, and who ended it.

    ``evaluate_explosions`` loops over every live explosion, so an agent standing
    where its own blast and an opponent's blast overlap is logged twice, once as
    a suicide and once as a kill for the opponent.  It still dies only once.
    Counting log lines therefore over-counts deaths; rounds are grouped instead,
    and a round holding both kinds of line is reported as its own category rather
    than being assigned arbitrarily to one side.
    """
    rounds = 0
    per_round_own: collections.Counter = collections.Counter()
    per_round_killers: dict[int, set[str]] = collections.defaultdict(set)
    for line in read_log(job_dir):
        start = ROUND_START.search(line)
        if start:
            rounds += 1
            continue
        match = OWN_BOMB.search(line)
        if match and match.group("victim") == agent:
            per_round_own[rounds] += 1
            continue
        match = OTHER_BOMB.search(line)
        if match and match.group("victim") == agent:
            per_round_killers[rounds].add(match.group("killer"))

    own_only = both = other_only = 0
    by_killer: collections.Counter = collections.Counter()
    for index in range(1, rounds + 1):
        hit_self = per_round_own.get(index, 0) > 0
        killers = per_round_killers.get(index, set())
        if hit_self and killers:
            both += 1
        elif hit_self:
            own_only += 1
        elif killers:
            other_only += 1
        # One death per round, so a shared blast is credited to nobody twice.
        if killers and not hit_self:
            for killer in killers:
                by_killer[killer] += 1 / len(killers)
    deaths = own_only + other_only + both
    return {
        "job_id": job_dir.name,
        "rounds": rounds,
        "own_bomb_deaths": own_only + both,
        "died_to_own_bomb_only": own_only,
        "killed_by_others_only": other_only,
        "caught_in_overlapping_blasts": both,
        "by_killer": {k: round(v, 3) for k, v in by_killer.items()},
        "survived": rounds - deaths,
    }


def official_suicides(job_dir: Path, agent: str) -> int | None:
    """The framework's own suicide count, written independently of the log."""
    path = job_dir / "official_stats.json"
    if not path.exists():
        return None
    stats = json.loads(path.read_text(encoding="utf-8"))
    return int(stats.get("by_agent", {}).get(agent, {}).get("suicides", 0))


def train_seed_of(job_id: str) -> str:
    match = re.search(r"train(\d+)", job_id)
    return match.group(1) if match else "?"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--job-prefix", required=True,
                        help="Which evaluation jobs to attribute, e.g. eval_classic_versus_opponents.")
    parser.add_argument("--agent", default="research_agent")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--include-degraded", action="store_true",
                        help="Include jobs whose agents were overridden or skipped "
                             "for slow think time; they describe a stalled node.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs = sorted(p for p in (args.run_dir / "jobs").iterdir() if p.name.startswith(args.job_prefix))
    if not jobs:
        raise SystemExit(f"No jobs matching {args.job_prefix!r} in {args.run_dir}")
    jobs = usable_jobs(jobs, args.include_degraded)

    results, mismatched = [], []
    for job in jobs:
        entry = attribute_job(job, args.agent)
        expected = official_suicides(job, args.agent)
        entry["official_suicides"] = expected
        if expected is not None and expected != entry["own_bomb_deaths"]:
            mismatched.append(entry)
        results.append(entry)

    if mismatched:
        print("LOG AND OFFICIAL STATISTICS DISAGREE -- findings below are void:")
        for entry in mismatched:
            print(f"  {entry['job_id']}: log {entry['own_bomb_deaths']} vs stats {entry['official_suicides']}")

    rounds = sum(r["rounds"] for r in results)
    own = sum(r["died_to_own_bomb_only"] for r in results)
    others = sum(r["killed_by_others_only"] for r in results)
    both = sum(r["caught_in_overlapping_blasts"] for r in results)
    survived = sum(r["survived"] for r in results)
    killers: collections.Counter = collections.Counter()
    for r in results:
        killers.update(r["by_killer"])

    print("=" * 62)
    print(f"run          {args.run_dir.name}")
    print(f"suite        {args.job_prefix}")
    print(f"jobs         {len(results)}   (log agrees with stats on {len(results) - len(mismatched)})")
    print(f"rounds       {rounds}")
    if rounds:
        print(f"  died to its own bomb alone      {own:5d}  ({own / rounds:6.1%})")
        print(f"  killed by an opponent alone     {others:5d}  ({others / rounds:6.1%})")
        print(f"  caught in overlapping blasts    {both:5d}  ({both / rounds:6.1%})")
        print(f"  survived the round              {survived:5d}  ({survived / rounds:6.1%})")
        deaths = own + others + both
        if deaths:
            print(f"  deaths with an opponent's bomb involved   "
                  f"{(others + both) / deaths:6.1%}")
    if killers:
        print(f"  killed by: {dict(killers.most_common())}")

    per_seed = collections.defaultdict(lambda: collections.Counter())
    for r in results:
        seed = train_seed_of(r["job_id"])
        per_seed[seed].update({"rounds": r["rounds"], "own": r["died_to_own_bomb_only"],
                               "others": r["killed_by_others_only"],
                               "both": r["caught_in_overlapping_blasts"],
                               "survived": r["survived"]})
    print("-" * 62)
    print("per training seed (rate per round)")
    print(f"  {'seed':<8s}{'own only':>11s}{'opponent':>11s}{'both':>9s}{'survived':>11s}")
    for seed in sorted(per_seed):
        counts = per_seed[seed]
        n = counts["rounds"] or 1
        print(f"  {seed:<8s}{counts['own'] / n:>11.3f}{counts['others'] / n:>11.3f}"
              f"{counts['both'] / n:>9.3f}{counts['survived'] / n:>11.3f}")

    if args.out:
        write_json(args.out, {
            "run_dir": str(args.run_dir), "job_prefix": args.job_prefix, "agent": args.agent,
            "rounds": rounds, "died_to_own_bomb_only": own, "killed_by_others_only": others,
            "caught_in_overlapping_blasts": both, "survived": survived, "killers": dict(killers),
            "mismatched_jobs": [m["job_id"] for m in mismatched], "jobs": results,
        })
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
