#!/usr/bin/env python3
"""Check that every archived run directory holds the run it is named after.

A run directory is identified only by its name, and every table in the docs
cites runs by that name.  Copying archives between machines is the one step that
can put the wrong contents under a name without anything failing: ``rsync`` of a
directory into a destination that already contains it nests instead of merging,
and the obvious cleanup -- compare the two listings, delete the duplicate -- only
compares file *names*, which match no matter which run's numbers are inside.
That is exactly how ``runs/m3_3lx_oppeval_20260826`` came to hold M3.5's results
while the docs quoted it as the solo-trained control (docs/01 section 7.17).

Every ``evaluation_summary.json`` records the ``run_id`` it was produced under,
and ``provenance.json`` records the config that produced it, so the check is a
comparison the archive can answer about itself.  Run it after copying archives
and before quoting a run in a document.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def check_run(run_dir: Path) -> dict | None:
    """Return a finding for one run directory, or None when it is consistent."""
    summary = run_dir / "evaluation_summary.json"
    if not summary.exists():
        return None  # an unfinished or pruned run has nothing to contradict
    try:
        recorded = json.loads(summary.read_text(encoding="utf-8")).get("run_id")
    except (OSError, json.JSONDecodeError) as exc:
        return {"run_dir": run_dir.name, "problem": f"unreadable evaluation_summary.json: {exc}"}
    if recorded is None:
        return {"run_dir": run_dir.name, "problem": "evaluation_summary.json has no run_id"}
    if recorded != run_dir.name:
        config = "unknown"
        provenance = run_dir / "provenance.json"
        if provenance.exists():
            try:
                config = Path(json.loads(provenance.read_text(encoding="utf-8"))
                              .get("config_source", "unknown")).name
            except (OSError, json.JSONDecodeError):
                pass
        return {"run_dir": run_dir.name, "problem": "contents belong to a different run",
                "recorded_run_id": recorded, "config_source": config}
    nested = run_dir / run_dir.name
    if nested.is_dir():
        return {"run_dir": run_dir.name,
                "problem": "a copy of itself is nested inside; an rsync landed in the wrong place"}
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--quiet", action="store_true", help="Print only the findings.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.runs_root.is_dir():
        raise SystemExit(f"No such directory: {args.runs_root}")
    directories = sorted(p for p in args.runs_root.iterdir() if p.is_dir())
    findings = [finding for finding in (check_run(p) for p in directories) if finding]

    checked = sum(1 for p in directories if (p / "evaluation_summary.json").exists())
    if not args.quiet:
        print(f"checked {checked} finished runs under {args.runs_root}")
    for finding in findings:
        print(f"  MISMATCH {finding['run_dir']}: {finding['problem']}")
        if "recorded_run_id" in finding:
            print(f"           its evaluation_summary.json says run_id={finding['recorded_run_id']}")
            print(f"           produced by {finding['config_source']}")
    if findings:
        raise SystemExit(f"{len(findings)} run directory(ies) do not hold the run they are named after")
    if not args.quiet:
        print("every finished run holds the run it is named after")


if __name__ == "__main__":
    main()
