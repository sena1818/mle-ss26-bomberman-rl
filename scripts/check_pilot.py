#!/usr/bin/env python3
"""Check that a training run is operationally healthy before it is believed.

The M4 pilot exists because two schedule bugs were found by reading round_end
records, not by reasoning: a hold counted in rounds that collected a tenth of
the transitions ``replay.min_size`` needed, and a step-counted anneal so long
it never reached its floor.  Both would have run to completion and produced
plausible-looking numbers.  This script turns the checks that caught them into
something a cluster job can run unattended.

It reports on the *mechanics* of a run -- did the buffer fill, did gradients
happen, are the values finite, did exploration finish.  It says nothing about
whether the agent is any good; ``aggregate_results.py`` answers that.

Exit code 0 if every applicable check passed, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment_lib import RUNS_ROOT  # noqa: E402


class Check:
    """One named condition, its measured value, and whether it holds."""

    def __init__(self, name: str, passed: bool | None, detail: str):
        self.name, self.passed, self.detail = name, passed, detail

    def render(self) -> str:
        mark = {True: "PASS", False: "FAIL", None: "SKIP"}[self.passed]
        return f"[{mark}] {self.name}\n       {self.detail}"


def round_end_records(job_dir: Path) -> list[dict]:
    files = sorted((job_dir / "agent").glob("*.jsonl"))
    if not files:
        return []
    records = []
    for line in files[0].read_text(encoding="utf-8").splitlines():
        if '"round_end"' not in line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("kind") == "round_end":
            records.append(record)
    return records


def training_job_dirs(run_dir: Path) -> list[Path]:
    return sorted(path for path in (run_dir / "jobs").glob("train_*") if path.is_dir())




def declared_epsilon_floor(job_dir: Path) -> float | None:
    """The floor the schedule DECLARES, read from the job's agent_setup record.

    Deliberately not ``min(observed epsilon)``: a run whose anneal never
    completes has its stalled value as the minimum, so an observed floor makes
    the check pass exactly when it should fail.
    """
    files = sorted((job_dir / "agent").glob("*.jsonl"))
    if not files:
        return None
    for line in files[0].read_text(encoding="utf-8").splitlines():
        if '"agent_setup"' not in line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("kind") != "agent_setup":
            continue
        specification = record.get("exploration_specification") or {}
        return specification.get("final_epsilon")
    return None

def replay_min_size(job_dir: Path) -> int | None:
    """Read the buffer threshold this job actually ran with.

    It comes from the job's own ``agent_setup`` record rather than from the
    config on disk, so a run stays checkable after the config is edited -- the
    same reason every other artifact in this project is self-describing.
    """
    files = sorted((job_dir / "agent").glob("*.jsonl"))
    if not files:
        return None
    for line in files[0].read_text(encoding="utf-8").splitlines():
        if '"agent_setup"' not in line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("kind") == "agent_setup":
            return (record.get("replay") or {}).get("min_size")
    return None

def check_training_job(job_dir: Path, rows: list[dict], min_size: int | None) -> list[Check]:
    name = job_dir.name
    checks: list[Check] = []
    steps = sum(row.get("updates_this_round", 0) for row in rows)
    gradient_rows = [row for row in rows if row.get("gradient_steps_this_round", 0) > 0]

    # 1. The buffer filled, and we can say when.
    if min_size is None:
        checks.append(Check(f"{name}: replay reached min_size", None,
                            "this route does no replay, or the job recorded no replay settings"))
        filled = None
    else:
        filled = next((row for row in rows
                       if (row.get("learner_step") or {}).get("replay_size", 0) >= min_size), None)
        checks.append(Check(
            f"{name}: replay reached min_size ({min_size})",
            filled is not None,
            f"at round {filled['round']}" if filled else
            f"never; the buffer held "
            f"{(rows[-1].get('learner_step') or {}).get('replay_size') if rows else 0}"
            f" of {min_size} after {len(rows)} rounds."
            " Lower min_size or lengthen the exploration hold.",
        ))

    # 2. A gradient step happened, and exploration had not already decayed.
    first = gradient_rows[0] if gradient_rows else None
    if first is None:
        checks.append(Check(f"{name}: a gradient step happened", False,
                            "no round applied a gradient; nothing was learned"))
    else:
        epsilon = first.get("epsilon")
        initial = max((row.get("epsilon") or 0.0) for row in rows) if rows else 0.0
        # Learning must start while most of the schedule is still ahead of it.
        healthy = epsilon is None or initial <= 0 or epsilon >= 0.9 * initial
        checks.append(Check(
            f"{name}: first gradient step happens at full exploration",
            healthy,
            f"round {first['round']}, epsilon {epsilon}, replay "
            f"{(first.get('learner_step') or {}).get('replay_size')} (schedule starts at {initial})"
            + ("" if healthy else
               " -- exploration had already decayed before the first weight moved."
               " The hold and min_size are in different units or badly matched."),
        ))

    # 3. Exploration finished inside the budget.
    epsilons = [row.get("epsilon") for row in rows if row.get("epsilon") is not None]
    declared_floor = declared_epsilon_floor(job_dir)
    if not epsilons or declared_floor is None:
        checks.append(Check(f"{name}: exploration reached its floor", None,
                            "constant-epsilon schedule, or the job recorded no schedule"))
    else:
        # The floor comes from the DECLARED schedule, never from the smallest
        # epsilon this run happened to reach.  Taking the observed minimum is
        # what makes this check vacuous: a run that stalls at 0.96 has 0.96 as
        # its minimum and "reaches its floor" on the first round.  That is the
        # precise failure this script was written to catch.
        reached = next((row for row in rows if row.get("epsilon", 1.0) <= declared_floor + 1e-9), None)
        checks.append(Check(
            f"{name}: exploration reached its declared floor ({declared_floor}) inside the budget",
            reached is not None,
            f"first reached at round {reached['round']} ({reached['round'] / len(rows):.0%}"
            f" of the budget)" if reached else
            f"NEVER; epsilon was still {epsilons[-1]:.3f} at round {len(rows)}, against a declared"
            f" floor of {declared_floor}. A step-counted anneal advances only as the agent"
            " survives, so one that is too long never completes.",
        ))

    # 4. The numbers a diverging run would show.
    diagnostics = [row["learner_gradient_step"] for row in rows if row.get("learner_gradient_step")]
    if not diagnostics:
        checks.append(Check(f"{name}: gradient diagnostics are finite", None,
                            "no model reported per-step diagnostics"))
    else:
        last = diagnostics[-1]
        values = {key: last.get(key) for key in ("loss", "q_mean", "q_max", "last_gradient_l2_norm")}
        finite = all(value is None or (value == value and abs(value) < 1e6) for value in values.values())
        norms = [one["last_gradient_l2_norm"] for one in diagnostics
                 if one.get("last_gradient_l2_norm") is not None]
        pinned = sum(1 for norm in norms if norm >= 9.99)
        checks.append(Check(
            f"{name}: gradient diagnostics are finite",
            finite,
            ", ".join(f"{key} {value:.4g}" for key, value in values.items() if value is not None),
        ))
        if norms:
            checks.append(Check(
                f"{name}: gradient norm is not pinned at the clip",
                pinned < 0.5 * len(norms),
                f"min {min(norms):.3f} max {max(norms):.3f}; {pinned}/{len(norms)}"
                f" rounds at the clip"
                + ("" if pinned < 0.5 * len(norms) else
                   " -- a run whose gradients sit on the clip is diverging, not training"),
            ))

    # 5. The invariant every finished run in this project is checked against.
    mispredictions = rows[-1].get("round_end_mispredictions") if rows else None
    checks.append(Check(
        f"{name}: round_end_mispredictions is zero",
        mispredictions == 0,
        f"final value {mispredictions} (this is a cumulative counter -- read the final"
        " value, never the sum over rounds)",
    ))

    segments = []
    for low, high in ((1, 1000), (1001, 2000), (2001, 3000), (3001, 5000), (5001, 10000)):
        window = [row.get("updates_this_round", 0) for row in rows if low <= row["round"] <= high]
        if window:
            segments.append(f"{low}-{high}: {statistics.fmean(window):.1f}")
    checks.append(Check(
        f"{name}: survival is improving", None,
        f"{steps} environment steps over {len(rows)} rounds."
        f" Mean steps/round by segment -- {'; '.join(segments)}."
        " A round is not a unit of experience; this is the ratio that makes"
        " round-counted schedules misleading.",
    ))
    return checks


def check_learning_curve(run_dir: Path) -> list[Check]:
    """The one check that needs the evaluation leg to have run."""
    summary_path = run_dir / "evaluation_summary.json"
    if not summary_path.is_file():
        # A hard failure, not a SKIP.  SKIP does not affect the exit code, so a
        # run whose evaluation leg never happened would report success -- and
        # the learning-curve checks are the only ones that can say whether the
        # agent learned anything at all.  Pass --training-only to ask for the
        # mechanics alone; do not get there by leaving the summary missing.
        return [Check("validation score rises", False,
                      "no evaluation_summary.json. Run aggregate_results.py first, or pass"
                      " --training-only if the mechanics are all you meant to check.")]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    curve = summary.get("evaluation_suites", {}).get("primary", {}).get("validation_checkpoint_curve", [])
    points = [(row["checkpoint_round"], row["metrics"]["score"]["mean"])
              for row in curve if row["metrics"]["score"]["count"]]
    if len(points) < 2:
        return [Check("validation score rises", None,
                      f"only {len(points)} evaluated checkpoint(s); need at least two")]
    rendered = ", ".join(f"{one}: {value:.2f}" for one, value in points)
    # G-A as docs/06 pre-registers it: the mean of the last two evaluated
    # checkpoints minus the mean of the first two, against the 4.7 resolution
    # from docs/01 section 7.21.  "last > first" is a weaker and different
    # claim -- it passes on a single lucky final checkpoint, and it is not what
    # the arm was registered against.
    window = min(2, len(points) // 2)
    early = statistics.fmean(value for _, value in points[:window])
    late = statistics.fmean(value for _, value in points[-window:])
    resolution = 4.7
    checks = [Check(
        f"G-A: mean(last {window}) - mean(first {window}) > {resolution}",
        late - early > resolution,
        f"{late:.2f} - {early:.2f} = {late - early:+.2f} ({rendered})"
        + ("" if late - early > resolution else
           " -- below the resolution of the five-seed spread, so this is not a learning signal"),
    )]
    last = curve[-1]["metrics"]
    floors = {"distinct_cells": 20.0, "bomb_rate": 0.01}
    for metric, floor in floors.items():
        value = last.get(metric, {}).get("mean")
        checks.append(Check(
            f"final policy is not degenerate: {metric} > {floor}",
            value is not None and value > floor,
            f"{metric} = {value}" if value is not None else f"{metric} undefined",
        ))
    wait = last.get("wait_fraction", {}).get("mean")
    checks.append(Check("final policy is not degenerate: wait_fraction < 0.6",
                        wait is not None and wait < 0.6,
                        f"wait_fraction = {wait}" if wait is not None else "undefined"))
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, help="runs/<run-id>, absolute or relative")
    parser.add_argument("--training-only", action="store_true",
                        help="skip the learning-curve checks that need evaluation jobs")
    arguments = parser.parse_args()

    run_dir = Path(arguments.run_dir)
    if not run_dir.is_absolute() and not run_dir.exists():
        run_dir = RUNS_ROOT / run_dir.name
    if not run_dir.is_dir():
        raise SystemExit(f"No such run directory: {run_dir}")

    jobs = training_job_dirs(run_dir)
    if not jobs:
        raise SystemExit(f"No training jobs under {run_dir / 'jobs'}")

    checks: list[Check] = []
    for job_dir in jobs:
        rows = round_end_records(job_dir)
        if not rows:
            checks.append(Check(f"{job_dir.name}: has round_end records", False, "none found"))
            continue
        checks.extend(check_training_job(job_dir, rows, replay_min_size(job_dir)))

    if not arguments.training_only:
        checks.extend(check_learning_curve(run_dir))

    print(f"Pilot checks for {run_dir.name}\n")
    for check in checks:
        print(check.render())
    failed = [check for check in checks if check.passed is False]
    skipped = [check for check in checks if check.passed is None]
    print(f"\n{len(checks) - len(failed) - len(skipped)} passed, {len(failed)} failed, "
          f"{len(skipped)} not applicable")
    if failed:
        print("\nDo not start the five-seed arms until these are addressed.")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
