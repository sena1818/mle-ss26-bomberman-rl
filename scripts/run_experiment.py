#!/usr/bin/env python3
"""Prepare and run isolated, declarative Bomberman R01-R07 experiment jobs."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from experiment_lib import ConfigError, Experiment, ROOT, RUNS_ROOT, copy_runtime, git_provenance, safe_identifier, write_json


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
    job = subparsers.add_parser("job", help="Run exactly one already-prepared job parameter file.")
    job.add_argument("--job-file", type=Path, required=True)
    return parser.parse_args()


def make_run_id(experiment: Experiment) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{experiment.experiment_id}_{stamp}_{secrets.token_hex(4)}"


def build_jobs(experiment: Experiment, run_dir: Path) -> list[dict]:
    jobs: list[dict] = []
    for seed in experiment.training.seeds:
        job_id = f"train_seed{seed}"
        jobs.append({
            "job_id": job_id, "mode": "train", "seed": seed,
            "phase": "training", "scenario": experiment.training.scenario,
            "opponents": list(experiment.training.opponents), "budget": experiment.training.budget.__dict__,
            "artifact_dir": str((run_dir / "jobs" / job_id).resolve()), "model_path": None,
        })
    for train_seed in experiment.training.seeds:
        model_path = run_dir / "jobs" / f"train_seed{train_seed}" / "agent" / "latest_model.npz"
        for seed in experiment.evaluation.seeds:
            job_id = f"eval_train{train_seed}_seed{seed}"
            jobs.append({
                "job_id": job_id, "mode": "eval", "seed": seed, "train_seed": train_seed,
                "phase": "evaluation", "scenario": experiment.evaluation.scenario,
                "opponents": list(experiment.evaluation.opponents), "budget": experiment.evaluation.budget.__dict__,
                "artifact_dir": str((run_dir / "jobs" / job_id).resolve()), "model_path": str(model_path.resolve()),
            })
    return jobs


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
    write_json(run_dir / "experiment_config.snapshot.json", experiment.snapshot())
    write_json(run_dir / "provenance.json", {**provenance, "config_source": str(config_path.resolve())})
    jobs = build_jobs(experiment, run_dir)
    write_json(run_dir / "jobs.json", jobs)
    for job_payload in jobs:
        write_json(run_dir / "job_parameters" / f"{job_payload['job_id']}.json", job_payload)
    print(run_dir)
    return run_dir


def load_context(job_file: Path) -> tuple[Path, dict, Experiment]:
    job = json.loads(job_file.read_text(encoding="utf-8"))
    artifact_dir = Path(job["artifact_dir"]).resolve()
    run_dir = artifact_dir.parents[1]
    experiment = Experiment.load(run_dir / "experiment_config.snapshot.json")
    experiment.require_implemented()
    if job_file.resolve().parent != (run_dir / "job_parameters").resolve():
        raise RuntimeError("Job file must belong to its run's job_parameters directory")
    return run_dir, job, experiment


def execute_job(job_file: Path) -> None:
    run_dir, job, experiment = load_context(job_file)
    job_dir = Path(job["artifact_dir"])
    if job_dir.exists():
        raise FileExistsError(f"Job artifact directory already exists (refusing overwrite): {job_dir}")
    if job["mode"] == "eval" and not Path(job["model_path"]).is_file():
        raise FileNotFoundError(f"Evaluation checkpoint is unavailable: {job['model_path']}")
    job_dir.mkdir(parents=True)
    write_json(job_dir / "job.snapshot.json", job)
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
        raise subprocess.CalledProcessError(result.returncode, command)
    print(f"completed {job['job_id']}: {job_dir}")


def execute_all(config_path: Path, requested_run_id: str | None, allow_dirty: bool) -> None:
    run_dir = prepare(config_path, requested_run_id, allow_dirty)
    jobs = json.loads((run_dir / "jobs.json").read_text(encoding="utf-8"))
    for mode in ("train", "eval"):
        for payload in jobs:
            if payload["mode"] == mode:
                execute_job(run_dir / "job_parameters" / f"{payload['job_id']}.json")
    subprocess.run([sys.executable, "scripts/aggregate_results.py", "--run-dir", str(run_dir), "--promote"], cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    try:
        if args.command == "prepare":
            prepare(args.config, args.run_id, args.allow_dirty)
        elif args.command == "job":
            execute_job(args.job_file)
        else:
            execute_all(args.config, args.run_id, args.allow_dirty)
    except (ConfigError, FileExistsError, FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
