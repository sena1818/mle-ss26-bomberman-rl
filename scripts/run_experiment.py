#!/usr/bin/env python3
"""Prepare and run isolated, declarative Bomberman R01-R07 experiment jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import shutil
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from experiment_lib import (
    ConfigError,
    Experiment,
    ROOT,
    RUNS_ROOT,
    copy_runtime,
    git_provenance,
    resolved_runtime_config,
    safe_identifier,
    verify_job_provenance,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="Create config/provenance snapshots and job parameter files.")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--run-id", help="Unique run id; default includes UTC time and random suffix.")
    prepare.add_argument("--allow-dirty", action="store_true", help="Allow a dirty checkout (intended only for local smoke runs).")
    run = subparsers.add_parser("run", help="Prepare and execute all training then greedy evaluation jobs locally.")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--run-id")
    run.add_argument("--allow-dirty", action="store_true")
    run.add_argument(
        "--promote",
        action="store_true",
        help="Explicitly update the scenario promotion pointer after local aggregation.",
    )
    run.add_argument(
        "--jobs", type=int, default=1, metavar="N",
        help="Run up to N jobs of the same phase concurrently (default 1). "
             "Jobs within a phase are independent; training always finishes before evaluation starts.",
    )
    job = subparsers.add_parser("job", help="Run exactly one already-prepared job parameter file.")
    job.add_argument("--job-file", type=Path, required=True)
    job.add_argument("--retry", action="store_true", help="Archive one completed failed attempt, then retry this job.")
    job.add_argument("--keep-runtime", action="store_true",
                     help="Keep the private framework copy after success. Only for debugging: it is the dominant disk cost.")
    return parser.parse_args()


def make_run_id(experiment: Experiment) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{experiment.experiment_id}_{stamp}_{secrets.token_hex(4)}"


def build_jobs(experiment: Experiment, run_dir: Path) -> list[dict]:
    jobs: list[dict] = []
    for seed in experiment.training.seeds:
        job_id = f"train_seed{seed}"
        job = {
            "job_id": job_id, "mode": "train", "seed": seed,
            "phase": "training", "scenario": experiment.training.scenario,
            "opponents": list(experiment.training.opponents), "budget": experiment.training.budget.__dict__,
            "artifact_relpath": str(Path("jobs") / job_id), "model_relpath": None,
        }
        if experiment.curriculum is not None:
            job["curriculum"] = {
                "source_run_id": experiment.curriculum.source_run_id,
                "segments": [segment.__dict__ for segment in experiment.curriculum.segments],
            }
            job["input_model_relpath"] = str(Path("inputs") / "initial_models" / f"train_seed{seed}.npz")
        jobs.append(job)

    def add_evaluations(phase, suite: str) -> None:
        policy = experiment.suite_checkpoints(suite)
        checkpoint_rounds = policy.checkpoint_rounds(experiment.training)
        seed_roles = [("validation", policy.validation_seeds)]
        if policy.holdout_seeds:
            seed_roles.append(("holdout", policy.holdout_seeds))
        for train_seed in experiment.training.seeds:
            agent_dir = Path("jobs") / f"train_seed{train_seed}" / "agent"
            for checkpoint_round in checkpoint_rounds:
                # ``latest`` keeps the historical job id and model_relpath byte
                # for byte, so existing runs stay reproducible and comparable.
                if checkpoint_round is None:
                    model_relpath: str | None = str(agent_dir / "latest_model.npz")
                    round_tag = ""
                else:
                    # The saved file name also encodes an update count that is
                    # unknown until training has run, so the job addresses the
                    # checkpoint by round and the worker resolves the file.
                    model_relpath = None
                    round_tag = f"_round{checkpoint_round:05d}"
                for role, seeds in seed_roles:
                    for seed in seeds:
                        suffix = "" if suite == "primary" else f"_{suite}"
                        job_id = f"eval{suffix}{round_tag}_train{train_seed}_seed{seed}"
                        payload = {
                            "job_id": job_id, "mode": "eval", "seed": seed, "train_seed": train_seed,
                            "phase": "evaluation", "suite": suite, "scenario": phase.scenario,
                            "opponents": list(phase.opponents), "budget": phase.budget.__dict__,
                            "artifact_relpath": str(Path("jobs") / job_id),
                            "model_relpath": model_relpath,
                            "checkpoint_round": checkpoint_round,
                            "seed_role": role,
                        }
                        if checkpoint_round is not None:
                            payload["checkpoint_search_relpath"] = str(agent_dir / "checkpoints")
                        jobs.append(payload)

    add_evaluations(experiment.evaluation, "primary")
    for suite in experiment.evaluation_suites:
        add_evaluations(suite.phase, suite.name)
    counts = Counter(job["job_id"] for job in jobs)
    duplicates = sorted(job_id for job_id, count in counts.items() if count > 1)
    if duplicates:
        raise ConfigError(f"Job expansion produced duplicate job ids: {', '.join(duplicates)}")
    return jobs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def materialize_curriculum_inputs(experiment: Experiment, run_dir: Path) -> list[dict]:
    """Copy warm-start checkpoints into the new run, making it portable."""
    if experiment.curriculum is None:
        return []
    source_root = RUNS_ROOT / experiment.curriculum.source_run_id
    if not source_root.is_dir():
        raise FileNotFoundError(f"Curriculum source run is unavailable: {source_root}")
    try:
        source_provenance = json.loads((source_root / "provenance.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Curriculum source provenance is unavailable: {source_root / 'provenance.json'}") from exc
    records = []
    for seed in experiment.training.seeds:
        source = source_root / "jobs" / f"train_seed{seed}" / "agent" / "latest_model.npz"
        if not source.is_file():
            raise FileNotFoundError(f"Curriculum source checkpoint is unavailable: {source}")
        destination = run_dir / "inputs" / "initial_models" / f"train_seed{seed}.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        records.append({
            "train_seed": seed,
            "source_run_id": experiment.curriculum.source_run_id,
            "source_git_commit": source_provenance.get("git_commit"),
            "source_checkpoint_relpath": str(source.relative_to(source_root)),
            "input_checkpoint_relpath": str(destination.relative_to(run_dir)),
            "sha256": _sha256(destination),
        })
    return records


def prepare(config_path: Path, requested_run_id: str | None, allow_dirty: bool) -> Path:
    experiment = Experiment.load(config_path)
    experiment.require_implemented()
    provenance = git_provenance()
    if provenance["worktree_dirty"] and not allow_dirty:
        raise RuntimeError("Refusing a long experiment from a dirty checkout. Commit first, or use --allow-dirty only for a local smoke run.")
    run_id = safe_identifier(requested_run_id, "run_id") if requested_run_id else make_run_id(experiment)
    run_dir = RUNS_ROOT / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    snapshot = experiment.snapshot()
    snapshot["resolved_runtime_config"] = resolved_runtime_config(experiment)
    snapshot["curriculum_inputs"] = materialize_curriculum_inputs(experiment, run_dir)
    write_json(run_dir / "experiment_config.snapshot.json", snapshot)
    write_json(run_dir / "provenance.json", {**provenance, "config_source": str(config_path.resolve())})
    jobs = build_jobs(experiment, run_dir)
    write_json(run_dir / "jobs.json", jobs)
    for job_payload in jobs:
        write_json(run_dir / "job_parameters" / f"{job_payload['job_id']}.json", job_payload)
    print(run_dir)
    return run_dir


def _resolve_run_relative(run_dir: Path, value: str | None, label: str) -> Path | None:
    if value is None:
        return None
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"{label} must be a path relative to the run directory: {value!r}")
    resolved = (run_dir / relative).resolve()
    if not resolved.is_relative_to(run_dir):
        raise RuntimeError(f"{label} escapes the run directory: {value!r}")
    return resolved


def _resolve_checkpoint_by_round(run_dir: Path, job: dict) -> Path:
    """Find the one saved checkpoint for a round number.

    Checkpoint file names embed an update count that only exists after the
    training job has run, so a prepared evaluation job can only name the round.
    Requiring exactly one match keeps a retried or duplicated training job from
    silently changing which weights an evaluation used.
    """
    search_dir = _resolve_run_relative(run_dir, job.get("checkpoint_search_relpath"), "checkpoint_search_relpath")
    if search_dir is None:
        raise RuntimeError(f"Job {job.get('job_id')} requests a checkpoint round without a search directory")
    round_number = int(job["checkpoint_round"])
    matches = sorted(search_dir.glob(f"*_round{round_number:05d}_*.npz"))
    if not matches:
        raise FileNotFoundError(
            f"No checkpoint for round {round_number} under {search_dir}. "
            "Run the training job first and synchronize its artifacts."
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous checkpoint for round {round_number} under {search_dir}: "
            + ", ".join(match.name for match in matches)
        )
    return matches[0]


def load_context(job_file: Path) -> tuple[Path, dict, Experiment]:
    job = json.loads(job_file.read_text(encoding="utf-8"))
    job_file = job_file.resolve()
    if job_file.parent.name != "job_parameters":
        raise RuntimeError("Job file must belong to its run's job_parameters directory")
    run_dir = job_file.parent.parent
    artifact_dir = _resolve_run_relative(run_dir, job.get("artifact_relpath"), "artifact_relpath")
    if artifact_dir is None:
        raise RuntimeError("Job file is missing artifact_relpath")
    model_path = _resolve_run_relative(run_dir, job.get("model_relpath"), "model_relpath")
    if model_path is None and job.get("checkpoint_round") is not None:
        model_path = _resolve_checkpoint_by_round(run_dir, job)
    input_model_path = _resolve_run_relative(run_dir, job.get("input_model_relpath"), "input_model_relpath")
    experiment = Experiment.load(run_dir / "experiment_config.snapshot.json")
    experiment.require_implemented()
    job = {
        **job,
        "artifact_dir": str(artifact_dir),
        "model_path": str(model_path) if model_path else None,
        "input_model_path": str(input_model_path) if input_model_path else None,
    }
    return run_dir, job, experiment


def _archive_failed_attempt(run_dir: Path, job_dir: Path) -> None:
    completion_path = job_dir / "completion.json"
    if not completion_path.is_file():
        raise RuntimeError("Cannot retry a job without completion.json; it may still be running or was interrupted.")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if int(completion.get("exit_code", 1)) == 0:
        raise RuntimeError("Cannot retry a completed successful job.")
    archive_root = run_dir / "failed_attempts" / job_dir.name
    attempt = 1
    while (archive_root / f"attempt{attempt:02d}").exists():
        attempt += 1
    destination = archive_root / f"attempt{attempt:02d}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(job_dir), str(destination))


def _run_curriculum_segment(
    *,
    run_dir: Path,
    job: dict,
    experiment: Experiment,
    segment_dir: Path,
    segment_index: int,
    segment: dict,
    input_model: Path,
    keep_runtime: bool = False,
) -> tuple[int, Path, Path]:
    """Run one scenario segment and return its exit code, model, and stats paths."""
    segment_id = f"segment{segment_index:02d}_{segment['scenario'].replace('-', '_')}"
    curriculum = experiment.curriculum
    assert curriculum is not None
    # Each segment is its own process with its own round counter.  Passing the
    # declared offset and denominator is what keeps a decaying exploration
    # schedule from restarting inside every segment.
    round_offset = curriculum.segment_round_offset(segment_index)
    schedule_rounds = curriculum.segment_schedule_rounds(segment_index, experiment.training)
    runtime = segment_dir / "runtime"
    copy_runtime(runtime)
    (runtime / "logs").mkdir()
    agent_dir = segment_dir / "agent"
    stats_path = segment_dir / "official_stats.json"
    segment_seed = int(job["seed"]) * 1000 + segment_index
    framework_log = runtime / "agent_code" / experiment.agent_name / "logs" / f"{experiment.agent_name}.log"
    game_log = runtime / "logs" / "game.log"
    environment = os.environ.copy()
    environment.update({
        "BOMBERMAN_RUN_ID": f"{run_dir.name}_{job['job_id']}_{segment_id}",
        "BOMBERMAN_EXPERIMENT": experiment.route,
        "BOMBERMAN_REWARD_VERSION": experiment.reward_version,
        "BOMBERMAN_EXPLORATION_VERSION": experiment.exploration_version,
        "BOMBERMAN_TERMINAL_ON_TRUNCATION": "1" if experiment.terminal_on_truncation else "0",
        "BOMBERMAN_TRAINING_ROUNDS": str(schedule_rounds),
        "BOMBERMAN_ROUND_OFFSET": str(round_offset),
        "BOMBERMAN_ARTIFACT_DIR": str(agent_dir.resolve()),
        "BOMBERMAN_SCENARIO": segment["scenario"],
        "BOMBERMAN_SEED": str(segment_seed),
        "BOMBERMAN_AGENT_SEED": str(segment_seed),
        "BOMBERMAN_CHECKPOINT_EVERY": str(job["budget"]["checkpoint_every"]),
        "BOMBERMAN_CONTINUE": "1",
        "BOMBERMAN_MODEL_PATH": str(input_model.resolve()),
    })
    command = [
        sys.executable, "main.py", "play", "--agents", experiment.agent_name, *job["opponents"],
        "--scenario", segment["scenario"], "--seed", str(segment_seed), "--n-rounds", str(segment["rounds"]),
        "--no-gui", "--save-stats", str(stats_path),
        "--match-name", f"{run_dir.name}_{job['job_id']}_{segment_id}", "--train", "1",
    ]
    write_json(segment_dir / "command.json", {
        "command": command,
        "cwd": str(runtime),
        "environment": {key: environment[key] for key in environment if key.startswith("BOMBERMAN_")},
    })
    with (segment_dir / "stdout.log").open("w", encoding="utf-8") as stdout, (segment_dir / "stderr.log").open("w", encoding="utf-8") as stderr:
        result = subprocess.run(command, cwd=runtime, env=environment, stdout=stdout, stderr=stderr, text=True)
    if framework_log.exists():
        shutil.copy2(framework_log, segment_dir / "framework_agent.log")
    if game_log.exists():
        shutil.copy2(game_log, segment_dir / "framework_game.log")
    output_model = agent_dir / "latest_model.npz"
    write_json(segment_dir / "completion.json", {
        "exit_code": result.returncode,
        "job_id": job["job_id"],
        "segment_index": segment_index,
        "segment_seed": segment_seed,
    })
    if not result.returncode:
        # A curriculum job copies the runtime once per segment, so keeping them
        # multiplied the waste by the number of segments.
        _discard_runtime(runtime, keep_runtime)
    return result.returncode, output_model, stats_path


def _combine_training_stats(segment_stats: list[tuple[str, Path]], agent_name: str) -> dict:
    """Preserve a concise root training stats file without flattening segment artifacts."""
    by_agent: dict[str, float] = {}
    by_round: dict[str, dict] = {}
    for segment_id, stats_path in segment_stats:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        for key, value in stats["by_agent"].get(agent_name, {}).items():
            if isinstance(value, (int, float)):
                by_agent[key] = by_agent.get(key, 0.0) + value
        for round_id, values in stats["by_round"].items():
            by_round[f"{segment_id}:{round_id}"] = values
    return {"by_agent": {agent_name: by_agent}, "by_round": by_round}


def execute_curriculum_job(run_dir: Path, job: dict, experiment: Experiment, job_dir: Path, *, keep_runtime: bool = False) -> None:
    """Continue one model through a frozen sequence of private scenario segments."""
    input_model = Path(job["input_model_path"])
    if not input_model.is_file():
        raise FileNotFoundError(f"Curriculum warm-start checkpoint is unavailable: {input_model}")
    manifest: list[dict] = []
    stats: list[tuple[str, Path]] = []
    previous_model = input_model
    try:
        for index, segment in enumerate(job["curriculum"]["segments"], start=1):
            segment_id = f"segment{index:02d}_{segment['scenario'].replace('-', '_')}"
            segment_dir = job_dir / "segments" / segment_id
            result, output_model, stats_path = _run_curriculum_segment(
                run_dir=run_dir, job=job, experiment=experiment, segment_dir=segment_dir,
                segment_index=index, segment=segment, input_model=previous_model,
                keep_runtime=keep_runtime,
            )
            record = {
                "segment_index": index,
                "scenario": segment["scenario"],
                "rounds": segment["rounds"],
                "input_checkpoint": str(previous_model.relative_to(run_dir)),
                "output_checkpoint": str(output_model.relative_to(run_dir)),
                "exit_code": result,
            }
            if result:
                manifest.append(record)
                write_json(job_dir / "curriculum_manifest.json", {"segments": manifest})
                write_json(job_dir / "completion.json", {"exit_code": result, "job_id": job["job_id"]})
                raise subprocess.CalledProcessError(result, ["curriculum", job["job_id"], segment_id])
            if not output_model.is_file():
                raise FileNotFoundError(f"Curriculum segment did not produce latest_model.npz: {output_model}")
            record["output_sha256"] = _sha256(output_model)
            manifest.append(record)
            stats.append((segment_id, stats_path))
            previous_model = output_model
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise
    final_model = job_dir / "agent" / "latest_model.npz"
    final_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(previous_model, final_model)
    write_json(job_dir / "curriculum_manifest.json", {"segments": manifest, "final_model": str(final_model.relative_to(run_dir))})
    write_json(job_dir / "official_stats.json", _combine_training_stats(stats, experiment.agent_name))
    write_json(job_dir / "completion.json", {"exit_code": 0, "job_id": job["job_id"]})


def _discard_runtime(runtime: Path, keep_runtime: bool) -> None:
    """Delete a succeeded job's private framework copy.

    Nothing reads it afterwards: the two framework logs were already copied into
    the job directory, aggregation reads only ``official_stats.json`` and the
    agent JSONL, and ``provenance.json`` plus ``command.json`` record exactly
    which commit and command produced the result.  Keeping it multiplied one
    repository checkout by the number of jobs.
    """
    if keep_runtime or not runtime.exists():
        return
    shutil.rmtree(runtime, ignore_errors=True)


def execute_job(job_file: Path, *, retry: bool = False, keep_runtime: bool = False) -> None:
    run_dir, job, experiment = load_context(job_file)
    verify_job_provenance(run_dir)
    job_dir = Path(job["artifact_dir"])
    if job_dir.exists():
        if not retry:
            raise FileExistsError(f"Job artifact directory already exists (refusing overwrite): {job_dir}")
        _archive_failed_attempt(run_dir, job_dir)
    if job["mode"] == "eval" and not Path(job["model_path"]).is_file():
        raise FileNotFoundError(f"Evaluation checkpoint is unavailable: {job['model_path']}")
    job_dir.mkdir(parents=True)
    write_json(job_dir / "job.snapshot.json", job)
    if job["mode"] == "train" and "curriculum" in job:
        execute_curriculum_job(run_dir, job, experiment, job_dir, keep_runtime=keep_runtime)
        print(f"completed {job['job_id']}: {job_dir}")
        return
    runtime = job_dir / "runtime"
    copy_runtime(runtime)
    # ``environment.py`` expects this framework-owned location to exist.  It
    # remains inside the private runtime and is copied into the job artifacts.
    (runtime / "logs").mkdir()
    agent_dir = job_dir / "agent"
    stats_path = job_dir / "official_stats.json"
    framework_log = runtime / "agent_code" / experiment.agent_name / "logs" / f"{experiment.agent_name}.log"
    game_log = runtime / "logs" / "game.log"
    environment = os.environ.copy()
    environment.update({
        "BOMBERMAN_RUN_ID": f"{run_dir.name}_{job['job_id']}",
        "BOMBERMAN_EXPERIMENT": experiment.route,
        "BOMBERMAN_REWARD_VERSION": experiment.reward_version,
        "BOMBERMAN_EXPLORATION_VERSION": experiment.exploration_version,
        "BOMBERMAN_TERMINAL_ON_TRUNCATION": "1" if experiment.terminal_on_truncation else "0",
        "BOMBERMAN_TRAINING_ROUNDS": str(experiment.training.budget.rounds),
        "BOMBERMAN_ROUND_OFFSET": "0",
        "BOMBERMAN_ARTIFACT_DIR": str(agent_dir.resolve()),
        "BOMBERMAN_SCENARIO": job["scenario"],
        "BOMBERMAN_SEED": str(job["seed"]),
        "BOMBERMAN_AGENT_SEED": str(job["seed"]),
        "BOMBERMAN_CHECKPOINT_EVERY": str(job["budget"]["checkpoint_every"]),
        "BOMBERMAN_CONTINUE": "0",
    })
    command = [sys.executable, "main.py", "play", "--agents", experiment.agent_name, *job["opponents"],
               "--scenario", job["scenario"], "--seed", str(job["seed"]), "--n-rounds", str(job["budget"]["rounds"]),
               "--no-gui", "--save-stats", str(stats_path), "--match-name", f"{run_dir.name}_{job['job_id']}"]
    if job["mode"] == "train":
        command.extend(("--train", "1"))
    else:
        environment["BOMBERMAN_MODEL_PATH"] = job["model_path"]
    write_json(job_dir / "command.json", {"command": command, "cwd": str(runtime), "environment": {key: environment[key] for key in environment if key.startswith("BOMBERMAN_")}})
    with (job_dir / "stdout.log").open("w", encoding="utf-8") as stdout, (job_dir / "stderr.log").open("w", encoding="utf-8") as stderr:
        result = subprocess.run(command, cwd=runtime, env=environment, stdout=stdout, stderr=stderr, text=True)
    if framework_log.exists():
        shutil.copy2(framework_log, job_dir / "framework_agent.log")
    if game_log.exists():
        shutil.copy2(game_log, job_dir / "framework_game.log")
    write_json(job_dir / "completion.json", {"exit_code": result.returncode, "job_id": job["job_id"]})
    if result.returncode:
        # A failed job keeps its runtime so the exact tree can be inspected.
        raise subprocess.CalledProcessError(result.returncode, command)
    _discard_runtime(runtime, keep_runtime)
    print(f"completed {job['job_id']}: {job_dir}")


def execute_phase(run_dir: Path, jobs: list[dict], mode: str, workers: int) -> None:
    """Run one phase's jobs, optionally concurrently.

    Jobs inside a phase are independent by construction: each owns a private
    artifact directory and a private framework copy, and communicates only
    through files.  The phases themselves are ordered, because evaluation reads
    checkpoints that training produces.

    Concurrency is thread-based on purpose.  Every job's real work happens in a
    ``subprocess.run`` that releases the GIL; the in-process part is a few
    milliseconds of file copying.
    """
    job_files = [
        run_dir / "job_parameters" / f"{payload['job_id']}.json"
        for payload in jobs
        if payload["mode"] == mode
    ]
    if not job_files:
        return
    if workers <= 1:
        for job_file in job_files:
            execute_job(job_file)
        return
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        submitted = {pool.submit(execute_job, job_file): job_file for job_file in job_files}
        for future in as_completed(submitted):
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001 - reported, then re-raised below
                failures.append(f"{submitted[future].stem}: {exc}")
    if failures:
        # Every failure is reported rather than only the first, because a shared
        # cause (a missing dependency, a full disk) shows up as a pattern.
        raise RuntimeError(
            f"{len(failures)} {mode} job(s) failed:\n  " + "\n  ".join(sorted(failures))
        )


def execute_all(
    config_path: Path,
    requested_run_id: str | None,
    allow_dirty: bool,
    *,
    promote: bool = False,
    workers: int = 1,
) -> None:
    run_dir = prepare(config_path, requested_run_id, allow_dirty)
    jobs = json.loads((run_dir / "jobs.json").read_text(encoding="utf-8"))
    for mode in ("train", "eval"):
        execute_phase(run_dir, jobs, mode, workers)
    aggregate_command = [sys.executable, "scripts/aggregate_results.py", "--run-dir", str(run_dir)]
    if promote:
        aggregate_command.append("--promote")
    subprocess.run(aggregate_command, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    try:
        if args.command == "prepare":
            prepare(args.config, args.run_id, args.allow_dirty)
        elif args.command == "job":
            execute_job(args.job_file, retry=args.retry, keep_runtime=args.keep_runtime)
        else:
            execute_all(args.config, args.run_id, args.allow_dirty, promote=args.promote, workers=args.jobs)
    except (ConfigError, FileExistsError, FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
