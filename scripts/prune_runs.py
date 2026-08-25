#!/usr/bin/env python3
"""Reclaim disk from finished run directories without losing any result.

Three tiers, from safest to most aggressive:

``--drop-runtime`` (default)
    Delete ``jobs/*/runtime`` and ``jobs/*/segments/*/runtime`` for jobs that
    completed successfully.  A job's private framework copy is dead weight once
    it exits: both framework logs were already copied into the job directory,
    aggregation reads only ``official_stats.json`` and the agent JSONL, and
    ``provenance.json`` plus ``command.json`` record exactly which commit and
    command produced the result.  A failed job keeps its runtime for debugging.

``--compress-logs``
    Gzip ``framework_game.log``, ``framework_agent.log`` and ``agent.jsonl``.
    These are highly repetitive text.  The aggregator reads ``.jsonl.gz``
    transparently, so a compressed run can still be re-summarized.

``--slim-copy DEST``
    Write a separate, tiny archive holding only what a report or a re-run needs:
    the config snapshot, provenance, job list, evaluation summary, every trained
    model, and each job's official statistics.  The source run is not modified.

Nothing here ever touches ``evaluation_summary.json``, ``official_stats.json``
or any ``.npz`` model in the source run.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path


# Kept in a slim copy because a report, a re-aggregation or a warm start needs
# them.  Everything else in a job directory can be regenerated or is a log.
SLIM_RUN_FILES = (
    "experiment_config.snapshot.json",
    "provenance.json",
    "jobs.json",
    "evaluation_summary.json",
)
SLIM_JOB_FILES = ("official_stats.json", "completion.json", "job.snapshot.json", "command.json")
# A curriculum job keeps its real artifacts one level down, one directory per
# scenario segment.  Missing these silently drops most of that run's models.
SLIM_JOB_EXTRA = ("curriculum_manifest.json",)
COMPRESSIBLE = ("framework_game.log", "framework_agent.log", "agent/agent.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, action="append", required=True,
                        help="Repeat per run. Only completed jobs inside it are touched.")
    parser.add_argument("--drop-runtime", action="store_true",
                        help="Delete the private framework copy of every succeeded job (default when no tier is given).")
    parser.add_argument("--compress-logs", action="store_true", help="Gzip framework logs and the agent JSONL.")
    parser.add_argument("--slim-copy", type=Path, help="Write a minimal archive of each run into this directory.")
    parser.add_argument("--apply", action="store_true",
                        help="Actually delete or rewrite. Without it the script only reports what it would reclaim.")
    return parser.parse_args()


def human(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


def tree_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def job_succeeded(job_dir: Path) -> bool:
    """Only a job with a recorded zero exit code is safe to prune."""
    completion = job_dir / "completion.json"
    if not completion.is_file():
        return False
    try:
        return int(json.loads(completion.read_text(encoding="utf-8")).get("exit_code", 1)) == 0
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False


def runtime_directories(run_dir: Path) -> list[Path]:
    """Every prunable runtime, including the per-segment ones a curriculum makes."""
    found: list[Path] = []
    jobs_root = run_dir / "jobs"
    if not jobs_root.is_dir():
        return found
    for job_dir in sorted(jobs_root.iterdir()):
        if not job_dir.is_dir() or not job_succeeded(job_dir):
            continue
        if (job_dir / "runtime").is_dir():
            found.append(job_dir / "runtime")
        for segment in sorted((job_dir / "segments").glob("*/runtime")):
            if segment.is_dir():
                found.append(segment)
    return found


def compressible_files(run_dir: Path) -> list[Path]:
    found: list[Path] = []
    jobs_root = run_dir / "jobs"
    if not jobs_root.is_dir():
        return found
    for job_dir in sorted(jobs_root.iterdir()):
        if not job_dir.is_dir() or not job_succeeded(job_dir):
            continue
        for relative in COMPRESSIBLE:
            candidate = job_dir / relative
            if candidate.is_file():
                found.append(candidate)
    return found


def compress(path: Path, apply: bool) -> int:
    """Gzip one file in place, returning the bytes saved."""
    before = path.stat().st_size
    target = path.with_suffix(path.suffix + ".gz")
    if not apply:
        # Text logs here compress by roughly an order of magnitude; report a
        # deliberately conservative estimate rather than reading every byte.
        return int(before * 0.9)
    with path.open("rb") as source, gzip.open(target, "wb", compresslevel=6) as sink:
        shutil.copyfileobj(source, sink)
    saved = before - target.stat().st_size
    path.unlink()
    return saved


def slim_copy(run_dir: Path, destination: Path, apply: bool) -> int:
    """Copy only the result-bearing files of one run into destination."""
    target = destination / run_dir.name
    written = 0
    plan: list[tuple[Path, Path]] = []
    for name in SLIM_RUN_FILES:
        source = run_dir / name
        if source.is_file():
            plan.append((source, target / name))
    # A warm-started curriculum reads its initial models from the run root.
    for model in sorted(run_dir.glob("inputs/**/*.npz")):
        plan.append((model, target / model.relative_to(run_dir)))
    jobs_root = run_dir / "jobs"
    if jobs_root.is_dir():
        for job_dir in sorted(jobs_root.iterdir()):
            if not job_dir.is_dir():
                continue
            # A plain job holds its artifacts directly; a curriculum job holds
            # them once per segment as well.
            scopes = [job_dir] + [path for path in sorted(job_dir.glob("segments/*")) if path.is_dir()]
            for scope in scopes:
                for name in SLIM_JOB_FILES + SLIM_JOB_EXTRA:
                    source = scope / name
                    if source.is_file():
                        plan.append((source, target / "jobs" / source.relative_to(jobs_root)))
                # Every trained model, including periodic checkpoints: this is
                # the actual product of a run and it is only a few kilobytes.
                for model in sorted(scope.glob("agent/**/*.npz")):
                    plan.append((model, target / "jobs" / model.relative_to(jobs_root)))
    for source, sink in plan:
        written += source.stat().st_size
        if apply:
            sink.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, sink)
    return written


def main() -> None:
    args = parse_args()
    drop_runtime = args.drop_runtime or not (args.compress_logs or args.slim_copy)
    total_reclaimed = 0
    for run_dir in args.run_dir:
        run_dir = run_dir.resolve()
        if not (run_dir / "jobs").is_dir():
            print(f"skip {run_dir.name}: no jobs directory")
            continue
        before = tree_size(run_dir)
        reclaimed = 0
        if drop_runtime:
            targets = runtime_directories(run_dir)
            size = sum(tree_size(path) for path in targets)
            reclaimed += size
            print(f"{run_dir.name}: {len(targets)} runtime copies, {human(size)}")
            if args.apply:
                for path in targets:
                    shutil.rmtree(path, ignore_errors=True)
        if args.compress_logs:
            targets = compressible_files(run_dir)
            size = sum(compress(path, args.apply) for path in targets)
            reclaimed += size
            print(f"{run_dir.name}: {len(targets)} logs compressed, {human(size)}")
        if args.slim_copy:
            written = slim_copy(run_dir, args.slim_copy.resolve(), args.apply)
            print(f"{run_dir.name}: slim copy {human(written)} (source {human(before)}, "
                  f"{before / max(1, written):.0f}x smaller)")
        total_reclaimed += reclaimed
    verb = "reclaimed" if args.apply else "would reclaim"
    print(f"\ntotal {verb}: {human(total_reclaimed)}")
    if not args.apply:
        print("dry run: nothing was changed. Re-run with --apply to act.")


if __name__ == "__main__":
    main()
