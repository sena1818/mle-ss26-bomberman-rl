#!/usr/bin/env python3
"""Reading the framework's game log without simulating anything.

Kept separate from ``diagnose_bomb_escape`` on purpose: that module imports
``environment``, ``settings`` and the agent's encoder, and ``attribute_deaths``
exists precisely to answer its question without any of that.
"""

from __future__ import annotations

import gzip
import re
from pathlib import Path

TIMED_OUT = re.compile(r"Agent <(?P<name>[^>]+)> exceeded think time by ")
SKIPPED = re.compile(r"Skipping agent <(?P<name>[^>]+)> because of last slow think time")


def open_game_log(job_dir: Path):
    """The job's game log, gzipped or not, or None if it was never archived."""
    path = job_dir / "framework_game.log.gz"
    if path.exists():
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    path = job_dir / "framework_game.log"
    if path.exists():
        return open(path, "rt", encoding="utf-8", errors="replace")
    return None


def think_time_violations(job_dir: Path) -> tuple[int, int]:
    """How often the framework overrode or skipped an agent -- not the agent's doing.

    ``poll_and_run_agents`` logs ``chose action X`` BEFORE it checks the clock,
    and then, if the agent was late, logs a warning and applies WAIT instead
    (environment.py, the ``exceeded think time`` branch).  Once an agent's
    carried-over budget goes negative it is not polled at all on the next step
    -- ``Skipping agent`` -- and no action is performed, not even WAIT.

    Two consequences, and the second is the important one:

    * ``read_actions`` sees only the first line, so a replay applies the action
      the agent asked for while the real game applied something else.  That is
      what a MISMATCH means in these scripts.
    * Such a job is not worth replaying even if it could be.  The agent was not
      in control for those steps, so every ``it had the chance and did not take
      it`` count describes a stalled node rather than a policy.

    Written for m4_opponents train_seed 1005, whose six evaluation jobs hit a
    node stall of about nine seconds on 2026-08-29; anchor and dueling were
    clean, as were that seed's later repeat-measurement jobs.
    """
    handle = open_game_log(job_dir)
    if handle is None:
        return (0, 0)
    overridden = skipped = 0
    with handle:
        for line in handle:
            overridden += bool(TIMED_OUT.search(line))
            skipped += bool(SKIPPED.search(line))
    return overridden, skipped


def usable_jobs(jobs: list[Path], include_degraded: bool) -> list[Path]:
    """Drop the jobs the framework had to override, and say which and why."""
    degraded = [(job, *think_time_violations(job)) for job in jobs]
    degraded = [entry for entry in degraded if entry[1] or entry[2]]
    if not degraded:
        return jobs
    print(f"  {len(degraded)} of {len(jobs)} jobs lost steps to think-time overrides:")
    for job, overridden, skipped in degraded:
        print(f"    {job.name}: {overridden} forced to WAIT, {skipped} steps not polled")
    if include_degraded:
        print("  --include-degraded given: replaying them anyway; expect MISMATCH.")
        return jobs
    print("  Excluded.  Pass --include-degraded to measure them anyway.")
    keep = [job for job in jobs if job not in {entry[0] for entry in degraded}]
    if not keep:
        raise SystemExit("Every job is degraded; there is nothing to measure.")
    return keep
