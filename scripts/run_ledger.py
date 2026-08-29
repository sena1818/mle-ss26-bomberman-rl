#!/usr/bin/env python3
"""One row per run directory: what it declared, what it produced, which commit.

The documents cite runs by directory name, and the name is the only thing tying
a published table to the data behind it.  Section 7.17 records what that costs
when it goes wrong -- a local directory held a different arm's results while the
docs quoted it as a control.  This is the index that makes the set auditable:
generated from each run's own snapshot and provenance, never written by hand, so
it cannot drift from what is on disk.

    scripts/run_ledger.py                    # every run under runs/
    scripts/run_ledger.py --runs-root run_archives
    scripts/run_ledger.py --markdown         # the form the docs carry
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiment_lib import RUNS_ROOT

HEADER = ("run", "route", "A/E/L", "training", "jobs", "kind", "commit")


def describe(run: Path) -> tuple[str, ...]:
    snapshot = run / "experiment_config.snapshot.json"
    if not snapshot.is_file():
        return (run.name, "", "", "", "", "no snapshot", "")
    declared = json.loads(snapshot.read_text(encoding="utf-8"))
    provenance = {}
    if (run / "provenance.json").is_file():
        provenance = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
    config = (declared.get("resolved_runtime_config") or {}).get("config") or {}
    training = declared.get("training") or {}
    budget = training.get("budget") or {}
    replay = config.get("replay") or {}
    derived = provenance.get("repeat_measurement_of") or provenance.get("opponent_substitution_of")
    completions = list(run.glob("jobs/*/completion.json"))
    failed = sum(1 for one in completions
                 if json.loads(one.read_text(encoding="utf-8")).get("exit_code"))
    recipe = "%s/%s/%s%s%s" % (
        declared.get("reward_version", ""), declared.get("exploration_version", ""),
        config.get("learning_rate_schedule", ""),
        "/PER" if replay.get("sampling") == "prioritized" else "",
        "/dueling" if config.get("dueling") else "")
    return (
        run.name,
        declared.get("route", ""),
        recipe,
        "%s%s x%s" % (training.get("scenario", ""),
                      "+%dopp" % len(training.get("opponents") or []) if training.get("opponents") else "",
                      budget.get("rounds", "")),
        "%d%s" % (len(completions), " (%d FAILED)" % failed if failed else ""),
        "summary" if (run / "evaluation_summary.json").is_file()
        else ("repeat of %s" % derived if derived else "unfinished"),
        provenance.get("git_commit", "")[:7],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--markdown", action="store_true", help="Emit a Markdown table.")
    arguments = parser.parse_args()
    rows = [describe(run) for run in sorted(arguments.runs_root.iterdir())
            if run.is_dir() and run.name != "promoted"]
    if not rows:
        raise SystemExit(f"no run directories under {arguments.runs_root}")
    if arguments.markdown:
        print("| " + " | ".join(HEADER) + " |")
        print("|" + "|".join("---" for _ in HEADER) + "|")
        for row in rows:
            print("| " + " | ".join(f"`{c}`" if i == 0 and c else str(c)
                                    for i, c in enumerate(row)) + " |")
    else:
        widths = [max(len(str(row[i])) for row in rows + [HEADER]) for i in range(len(HEADER))]
        line = "  ".join(h.ljust(w) for h, w in zip(HEADER, widths))
        print(line)
        print("-" * len(line))
        for row in rows:
            print("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))
    print()
    print(f"{len(rows)} run directories under {arguments.runs_root}")


if __name__ == "__main__":
    main()
