#!/usr/bin/env python3
"""Aggregate isolated greedy-evaluation jobs and deterministically promote best."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
from pathlib import Path
from typing import Iterable

from experiment_lib import Experiment, ROOT, RUNS_ROOT, write_json


METRICS = ("score", "coins", "kills", "suicides", "invalid_actions", "steps", "inference_seconds")
PROMOTION_ROOT = RUNS_ROOT / "promoted"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--promote", action="store_true", help="Allow this summarizer to update promoted best/active files.")
    return parser.parse_args()


def mean_std(values: list[float]) -> dict[str, float | int]:
    return {
        "mean": statistics.fmean(values) if values else 0.0,
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "count": len(values),
    }


def action_metrics(path: Path) -> tuple[int, list[float]]:
    invalid_actions = 0
    timings: list[float] = []
    if not path.exists():
        return invalid_actions, timings
    for line in path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("kind") != "action":
            continue
        invalid_actions += int(not event.get("selected_action_was_legal", False))
        timings.append(float(event["inference_seconds"]))
    return invalid_actions, timings


def one_evaluation(job_dir: Path, agent_name: str) -> dict[str, float]:
    stats = json.loads((job_dir / "official_stats.json").read_text(encoding="utf-8"))
    by_round = stats["by_round"]
    agent = stats["by_agent"][agent_name]
    selected_invalid, timings = action_metrics(job_dir / "agent" / "agent.jsonl")
    rounds = max(1, len(by_round))
    return {
        # Official per-agent stats are authoritative.  Normalize each job by
        # its fixed number of rounds before comparing different seed jobs.
        "score": float(agent.get("score", 0.0)) / rounds,
        "coins": float(agent.get("coins", 0.0)) / rounds,
        "kills": float(agent.get("kills", 0.0)) / rounds,
        "suicides": float(agent.get("suicides", 0.0)) / rounds,
        "invalid_actions": float(agent.get("invalid", 0.0)) / rounds,
        "steps": float(agent.get("steps", 0.0)) / rounds,
        "inference_seconds": statistics.fmean(timings) if timings else 0.0,
        "selected_invalid_actions": float(selected_invalid) / rounds,
    }


def aggregate(run_dir: Path) -> dict:
    run_dir = run_dir.resolve()
    experiment = Experiment.load(run_dir / "experiment_config.snapshot.json")
    jobs = json.loads((run_dir / "jobs.json").read_text(encoding="utf-8"))
    samples: list[dict[str, float]] = []
    candidate_samples: dict[int, list[dict[str, float]]] = {}
    missing: list[str] = []
    for job in jobs:
        if job["mode"] != "eval":
            continue
        job_dir = Path(job["artifact_dir"])
        if not (job_dir / "official_stats.json").is_file():
            missing.append(job["job_id"])
            continue
        sample = one_evaluation(job_dir, experiment.agent_name)
        samples.append(sample)
        candidate_samples.setdefault(int(job["train_seed"]), []).append(sample)
    if missing:
        raise RuntimeError(f"Cannot summarize incomplete evaluation jobs: {', '.join(missing)}")
    summary = {
        "experiment_id": experiment.experiment_id,
        "run_id": run_dir.name,
        "primary_metric": "score",
        "evaluation_jobs": len([job for job in jobs if job["mode"] == "eval"]),
        "metrics": {metric: mean_std([sample[metric] for sample in samples]) for metric in METRICS},
        "invalid_actions_definition": "official per-agent INVALID_ACTION events per round; agent JSONL separately records selection legality",
        "checkpoint_candidates": [
            {
                "train_seed": seed,
                "checkpoint": str((run_dir / "jobs" / f"train_seed{seed}" / "agent" / "latest_model.npz").resolve()),
                "metrics": {metric: mean_std([sample[metric] for sample in candidate_samples[seed]]) for metric in METRICS},
            }
            for seed in sorted(candidate_samples)
        ],
    }
    write_json(run_dir / "evaluation_summary.json", summary)
    return summary


def promotion_key(summary: dict) -> tuple[float, float, float, float, str]:
    """Fixed, explainable tie-break order: score, stability, survival, then ids."""
    metrics = summary["metrics"]
    return (
        float(metrics["score"]["mean"]),
        -float(metrics["score"]["std"]),
        -float(metrics["suicides"]["mean"]),
        float(metrics["coins"]["mean"]),
        # A lexical id final tie-break keeps promotion independent of job order.
        summary["run_id"],
    )


def best_source_checkpoint(summary: dict) -> Path:
    """Choose the checkpoint with the best fixed greedy-evaluation summary."""
    candidates = summary["checkpoint_candidates"]
    if not candidates:
        raise FileNotFoundError("No training checkpoint is available for promotion")
    def candidate_key(candidate: dict) -> tuple[float, float, float, float, int]:
        metrics = candidate["metrics"]
        return (
            float(metrics["score"]["mean"]),
            -float(metrics["score"]["std"]),
            -float(metrics["suicides"]["mean"]),
            float(metrics["coins"]["mean"]),
            -int(candidate["train_seed"]),
        )
    chosen = max(candidates, key=candidate_key)
    checkpoint = Path(chosen["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"No training checkpoint is available for promotion: {checkpoint}")
    return checkpoint


def maybe_promote(summary: dict) -> bool:
    PROMOTION_ROOT.mkdir(parents=True, exist_ok=True)
    best_path = PROMOTION_ROOT / "best_summary.json"
    current = json.loads(best_path.read_text(encoding="utf-8")) if best_path.exists() else None
    if current is not None and promotion_key(summary) <= promotion_key(current):
        return False
    source = best_source_checkpoint(summary)
    staging = PROMOTION_ROOT / ".active_model.staging.npz"
    shutil.copy2(source, staging)
    os.replace(staging, PROMOTION_ROOT / "active_model.npz")
    best_staging = PROMOTION_ROOT / ".best_model.staging.npz"
    shutil.copy2(source, best_staging)
    os.replace(best_staging, PROMOTION_ROOT / "best_model.npz")
    write_json(best_path, summary)
    write_json(PROMOTION_ROOT / "promotion_rule.json", {
        "rule": "maximize mean official score; then minimize score std; then minimize mean suicides; then maximize mean coins; then lexical run_id",
        "writer": "scripts/aggregate_results.py only",
    })
    return True


def main() -> None:
    args = parse_args()
    try:
        summary = aggregate(args.run_dir)
        promoted = maybe_promote(summary) if args.promote else False
    except (FileNotFoundError, RuntimeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(json.dumps({"summary": str(args.run_dir.resolve() / "evaluation_summary.json"), "promoted": promoted}, sort_keys=True))


if __name__ == "__main__":
    main()
