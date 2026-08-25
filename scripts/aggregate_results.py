#!/usr/bin/env python3
"""Aggregate isolated greedy-evaluation jobs and deterministically promote best.

Three groups of metrics are reported, and they are *not* interchangeable:

``ROUND_METRICS``
    Averaged per round inside one job, then averaged across jobs.  Only
    quantities the framework actually records per round, plus our own agent
    JSONL (which carries a ``round`` field on every action), can be computed
    this way.

``JOB_METRICS``
    ``official_stats.json`` stores ``bombs``/``crates``/``moves``/``invalid``
    only in ``by_agent``, which is a whole-job accumulator: ``end_round()``
    writes just ``steps``/``coins``/``kills``/``suicides`` into ``by_round``.
    Any ratio built from crates is therefore a job-level ratio, aggregated as
    mean +/- std across jobs.  Labelling it as per-round would be wrong.

``METRICS``
    The historical set, computed exactly as before so older runs stay
    comparable.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
from collections import Counter
from pathlib import Path

from experiment_lib import Experiment, ROOT, RUNS_ROOT, write_json


METRICS = ("score", "coins", "kills", "suicides", "invalid_actions", "steps", "inference_seconds")
ROUND_METRICS = ("bomb_rate", "wait_fraction", "approximate_safe_bomb_rate", "distinct_cells")
JOB_METRICS = ("crates_per_round", "crates_per_bomb", "coins_per_crate", "official_wait_fraction")
DIAGNOSTIC_METRICS = ("rounds_with_bombs", "selected_invalid_actions")
ALL_METRICS = METRICS + ROUND_METRICS + JOB_METRICS + DIAGNOSTIC_METRICS
PROMOTION_ROOT = RUNS_ROOT / "promoted"
_ROUND_KEY = re.compile(r"Round\s+(\d+)")

METRIC_DEFINITIONS = {
    "score": {"granularity": "job", "formula": "official by_agent.score / rounds"},
    "coins": {"granularity": "job", "formula": "official by_agent.coins / rounds"},
    "kills": {"granularity": "job", "formula": "official by_agent.kills / rounds"},
    "suicides": {"granularity": "job", "formula": "official by_agent.suicides / rounds"},
    "invalid_actions": {"granularity": "job", "formula": "official by_agent.invalid / rounds"},
    "steps": {"granularity": "job", "formula": "official by_agent.steps / rounds"},
    "inference_seconds": {"granularity": "action", "formula": "mean over every logged action"},
    "bomb_rate": {
        "granularity": "round",
        "formula": "mean over rounds of (BOMB actions in round / actions in round)",
        "source": "agent JSONL",
    },
    "wait_fraction": {
        "granularity": "round",
        "formula": "mean over rounds of (WAIT actions in round / actions in round)",
        "source": "agent JSONL",
        "note": "separates 'survived by playing well' from 'survived by freezing'",
    },
    "approximate_safe_bomb_rate": {
        "granularity": "round",
        "formula": "mean over rounds that dropped at least one bomb of (1 - suicides / bombs)",
        "source": "agent JSONL bombs + official by_round.suicides",
        "note": (
            "APPROXIMATE, not a safe-bomb rate. Bombs still unexploded when the round ends are "
            "counted in the denominator without ever having created risk, and with opponents "
            "GOT_KILLED is not counted as a suicide. An exact figure requires tracking each own "
            "BOMB_EXPLODED through the full two-step lethal window."
        ),
    },
    "distinct_cells": {
        "granularity": "round",
        "formula": "mean over rounds of the number of distinct visited cells",
        "source": "agent JSONL position field",
        "note": "absent for runs recorded before the position field existed; detects two-cycle policies",
    },
    "crates_per_round": {"granularity": "job", "formula": "official by_agent.crates / rounds"},
    "crates_per_bomb": {"granularity": "job", "formula": "official by_agent.crates / by_agent.bombs"},
    "coins_per_crate": {
        "granularity": "job",
        "formula": "official by_agent.coins / by_agent.crates",
        "note": "reward-hacking probe: high crates with low coins means crates are farmed, not cashed in",
    },
    "official_wait_fraction": {
        "granularity": "job",
        "formula": "(steps - moves - bombs - invalid) / steps",
        "note": "job-level cross-check of the per-round wait_fraction; WAIT is the residual action",
    },
    "rounds_with_bombs": {
        "granularity": "job",
        "formula": "number of rounds in the job that dropped at least one bomb",
        "note": "sample size behind approximate_safe_bomb_rate; a near-zero value makes that metric meaningless",
    },
    "selected_invalid_actions": {
        "granularity": "job",
        "formula": "agent-side count of selected actions the mask considered illegal, per round",
        "note": "cross-checks the legality mask against the official INVALID_ACTION counter",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--promote", action="store_true", help="Allow this summarizer to update promoted best/active files.")
    return parser.parse_args()


def mean_std(values: list[float | None]) -> dict[str, float | int]:
    """Aggregate, ignoring undefined samples but reporting how many were used."""
    present = [float(value) for value in values if value is not None]
    return {
        "mean": statistics.fmean(present) if present else 0.0,
        "std": statistics.stdev(present) if len(present) > 1 else 0.0,
        "count": len(present),
    }


def _ratio(numerator: float, denominator: float) -> float | None:
    """Return a ratio, or ``None`` when it is undefined rather than zero.

    A round with no bombs has no safe-bomb rate; folding it in as ``0.0`` would
    silently claim the agent bombed and died.
    """
    return numerator / denominator if denominator else None


def rounds_by_number(by_round: dict) -> dict[int, dict]:
    """Map official per-round statistics onto 1-based round numbers."""
    mapped: dict[int, dict] = {}
    for index, (key, value) in enumerate(by_round.items(), start=1):
        match = _ROUND_KEY.search(key)
        mapped[int(match.group(1)) if match else index] = value
    return mapped


def agent_round_actions(path: Path) -> tuple[dict[int, Counter], dict[int, set], list[float], int]:
    """Return per-round action counts, visited cells, timings, illegal count."""
    actions: dict[int, Counter] = {}
    visited: dict[int, set] = {}
    timings: list[float] = []
    illegal = 0
    if not path.exists():
        return actions, visited, timings, illegal
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            if event.get("kind") != "action":
                continue
            illegal += int(not event.get("selected_action_was_legal", False))
            timings.append(float(event["inference_seconds"]))
            # ``action`` and ``position`` were added to the record at different
            # times.  Older runs simply contribute no round-level metrics rather
            # than making the whole aggregation fail.
            action = event.get("action")
            if action is None:
                continue
            round_number = int(event.get("round", 0))
            actions.setdefault(round_number, Counter())[action] += 1
            position = event.get("position")
            if position is not None:
                visited.setdefault(round_number, set()).add((int(position[0]), int(position[1])))
    return actions, visited, timings, illegal


def round_level_metrics(actions: dict[int, Counter], visited: dict[int, set], official_rounds: dict[int, dict]) -> dict[str, float | None]:
    """Compute each ratio inside a round first, then average over rounds."""
    if not actions:
        return {metric: None for metric in ROUND_METRICS}
    bomb_rates: list[float] = []
    wait_fractions: list[float] = []
    safe_rates: list[float] = []
    distinct: list[float] = []
    for round_number, counter in sorted(actions.items()):
        total = sum(counter.values())
        if not total:
            continue
        bombs = counter.get("BOMB", 0)
        bomb_rates.append(bombs / total)
        wait_fractions.append(counter.get("WAIT", 0) / total)
        if bombs:
            suicides = float(official_rounds.get(round_number, {}).get("suicides", 0.0))
            safe_rates.append(max(0.0, 1.0 - suicides / bombs))
        if round_number in visited:
            distinct.append(float(len(visited[round_number])))
    return {
        "bomb_rate": statistics.fmean(bomb_rates) if bomb_rates else None,
        "wait_fraction": statistics.fmean(wait_fractions) if wait_fractions else None,
        # Undefined, not zero, when the greedy policy never dropped a bomb.
        "approximate_safe_bomb_rate": statistics.fmean(safe_rates) if safe_rates else None,
        "distinct_cells": statistics.fmean(distinct) if distinct else None,
    }


def one_evaluation(job_dir: Path, agent_name: str) -> dict[str, float | None]:
    stats = json.loads((job_dir / "official_stats.json").read_text(encoding="utf-8"))
    by_round = stats["by_round"]
    agent = stats["by_agent"][agent_name]
    actions, visited, timings, selected_invalid = agent_round_actions(job_dir / "agent" / "agent.jsonl")
    rounds = max(1, len(by_round))
    steps = float(agent.get("steps", 0.0))
    crates = float(agent.get("crates", 0.0))
    bombs = float(agent.get("bombs", 0.0))
    moves = float(agent.get("moves", 0.0))
    invalid = float(agent.get("invalid", 0.0))
    sample: dict[str, float | None] = {
        # Official per-agent stats are authoritative.  Normalize each job by
        # its fixed number of rounds before comparing different seed jobs.
        "score": float(agent.get("score", 0.0)) / rounds,
        "coins": float(agent.get("coins", 0.0)) / rounds,
        "kills": float(agent.get("kills", 0.0)) / rounds,
        "suicides": float(agent.get("suicides", 0.0)) / rounds,
        "invalid_actions": invalid / rounds,
        "steps": steps / rounds,
        "inference_seconds": statistics.fmean(timings) if timings else 0.0,
        "selected_invalid_actions": float(selected_invalid) / rounds,
        "crates_per_round": crates / rounds,
        "crates_per_bomb": _ratio(crates, bombs),
        "coins_per_crate": _ratio(float(agent.get("coins", 0.0)), crates),
        "official_wait_fraction": _ratio(steps - moves - bombs - invalid, steps),
        "rounds_with_bombs": float(sum(1 for counter in actions.values() if counter.get("BOMB", 0))),
    }
    sample.update(round_level_metrics(actions, visited, rounds_by_number(by_round)))
    return sample


def job_artifact_dir(run_dir: Path, job: dict) -> Path:
    """Resolve the portable job artifact path without trusting host paths."""
    relative = Path(job["artifact_relpath"])
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"Invalid artifact_relpath for {job.get('job_id')}: {relative}")
    path = (run_dir / relative).resolve()
    if not path.is_relative_to(run_dir):
        raise RuntimeError(f"artifact_relpath escapes run directory for {job.get('job_id')}")
    return path


def _metric_block(samples: list[dict]) -> dict:
    return {metric: mean_std([sample.get(metric) for sample in samples]) for metric in ALL_METRICS}


def _variance_decomposition(by_train_seed: dict[int, list[dict]]) -> dict:
    """Separate training randomness from evaluation-map randomness.

    Reporting one pooled standard deviation over all evaluation jobs hides
    which of the two dominates.  In the A00 coin pilot the spread across
    training seeds was far larger than the spread across evaluation seeds.
    """
    decomposition: dict[str, dict[str, float]] = {}
    for metric in ALL_METRICS:
        seed_means: list[float] = []
        within_stds: list[float] = []
        for samples in by_train_seed.values():
            present = [float(sample[metric]) for sample in samples if sample.get(metric) is not None]
            if not present:
                continue
            seed_means.append(statistics.fmean(present))
            if len(present) > 1:
                within_stds.append(statistics.stdev(present))
        decomposition[metric] = {
            "across_train_seeds_std": statistics.stdev(seed_means) if len(seed_means) > 1 else 0.0,
            "mean_within_train_seed_std": statistics.fmean(within_stds) if within_stds else 0.0,
            "train_seeds": len(seed_means),
        }
    return decomposition


def _candidate_checkpoint_path(run_dir: Path, train_seed: int, checkpoint_round: int | None) -> str:
    agent_dir = run_dir / "jobs" / f"train_seed{train_seed}" / "agent"
    if checkpoint_round is None:
        return str((agent_dir / "latest_model.npz").resolve())
    matches = sorted((agent_dir / "checkpoints").glob(f"*_round{checkpoint_round:05d}_*.npz"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one checkpoint for round {checkpoint_round} of train seed {train_seed}, "
            f"found {len(matches)}"
        )
    return str(matches[0].resolve())


def aggregate(run_dir: Path) -> dict:
    run_dir = run_dir.resolve()
    experiment = Experiment.load(run_dir / "experiment_config.snapshot.json")
    jobs = json.loads((run_dir / "jobs.json").read_text(encoding="utf-8"))
    # suite -> seed_role -> samples, and suite -> seed_role -> candidate key -> samples
    samples: dict[str, dict[str, list[dict]]] = {}
    candidates: dict[str, dict[str, dict[tuple[int, int | None], list[dict]]]] = {}
    by_train_seed: dict[str, dict[str, dict[int, list[dict]]]] = {}
    scenarios: dict[str, set[str]] = {}
    missing: list[str] = []
    for job in jobs:
        if job["mode"] != "eval":
            continue
        job_dir = job_artifact_dir(run_dir, job)
        if not (job_dir / "official_stats.json").is_file():
            missing.append(job["job_id"])
            continue
        sample = one_evaluation(job_dir, experiment.agent_name)
        suite = job.get("suite", "primary")
        role = job.get("seed_role", "validation")
        train_seed = int(job["train_seed"])
        checkpoint_round = job.get("checkpoint_round")
        samples.setdefault(suite, {}).setdefault(role, []).append(sample)
        candidates.setdefault(suite, {}).setdefault(role, {}).setdefault((train_seed, checkpoint_round), []).append(sample)
        by_train_seed.setdefault(suite, {}).setdefault(role, {}).setdefault(train_seed, []).append(sample)
        scenarios.setdefault(suite, set()).add(job["scenario"])
    if missing:
        raise RuntimeError(f"Cannot summarize incomplete evaluation jobs: {', '.join(missing)}")

    def candidate_records(suite: str, role: str) -> list[dict]:
        grouped = candidates.get(suite, {}).get(role, {})
        records = []
        for (train_seed, checkpoint_round) in sorted(grouped, key=lambda key: (key[0], -1 if key[1] is None else key[1])):
            records.append({
                "train_seed": train_seed,
                "checkpoint_round": checkpoint_round,
                "checkpoint": _candidate_checkpoint_path(run_dir, train_seed, checkpoint_round),
                "metrics": _metric_block(grouped[(train_seed, checkpoint_round)]),
            })
        return records

    def checkpoint_curve_records(suite: str, role: str) -> list[dict]:
        """Aggregate every seed at one training round into a learning curve.

        This is deliberately separate from ``selected_checkpoint``.  A dose
        response study may care about whether bomb activity changes at round
        500 even when official score is still zero, so score-based model
        selection must not hide the curve.
        """
        by_candidate = candidates.get(suite, {}).get(role, {})
        by_round: dict[int | None, list[dict]] = {}
        by_round_seed: dict[int | None, dict[int, list[dict]]] = {}
        for (train_seed, checkpoint_round), candidate_samples in by_candidate.items():
            by_round.setdefault(checkpoint_round, []).extend(candidate_samples)
            by_round_seed.setdefault(checkpoint_round, {})[train_seed] = candidate_samples
        return [
            {
                "checkpoint_round": checkpoint_round,
                "metrics": _metric_block(items),
                "variance_decomposition": _variance_decomposition(by_round_seed[checkpoint_round]),
            }
            for checkpoint_round, items in sorted(
                by_round.items(), key=lambda item: 10 ** 9 if item[0] is None else int(item[0])
            )
        ]

    def suite_summary(name: str) -> dict:
        by_role = samples.get(name, {})
        validation = by_role.get("validation", [])
        if not validation:
            raise RuntimeError(f"Evaluation suite {name!r} has no completed validation jobs")
        suite_scenarios = scenarios[name]
        if len(suite_scenarios) != 1:
            raise RuntimeError(f"Evaluation suite {name!r} mixes scenarios: {sorted(suite_scenarios)}")
        policy = experiment.suite_checkpoints(name)
        validation_candidates = candidate_records(name, "validation")
        validation_curve = checkpoint_curve_records(name, "validation")
        selected = max(validation_candidates, key=candidate_key) if validation_candidates else None
        summary = {
            "scenario": next(iter(suite_scenarios)),
            "evaluation_jobs": sum(len(items) for items in by_role.values()),
            "checkpoint_mode": policy.mode,
            "selection_seeds": list(policy.validation_seeds),
            "holdout_seeds": list(policy.holdout_seeds),
            "checkpoint_candidates": validation_candidates,
            "validation_checkpoint_curve": validation_curve,
            "selected_checkpoint": selected,
            "variance_decomposition": _variance_decomposition(by_train_seed[name]["validation"]),
        }
        pooled_validation = _metric_block(validation)
        if policy.mode == "all":
            # A pool spanning different training rounds is a learning curve,
            # not the performance of any deployable model.  Report the
            # validation result of the selected checkpoint under ``metrics``;
            # preserve the pool separately for diagnostics only.
            summary["metrics"] = selected["metrics"] if selected is not None else pooled_validation
            summary["all_checkpoint_validation_metrics"] = pooled_validation
            summary["all_checkpoint_variance_decomposition"] = summary.pop("variance_decomposition")
        else:
            # Historical latest-only runs retain their existing aggregate
            # meaning and remain comparable to prior summaries.
            summary["metrics"] = pooled_validation
        holdout = by_role.get("holdout", [])
        if holdout:
            holdout_candidates = candidate_records(name, "holdout")
            holdout_curve = checkpoint_curve_records(name, "holdout")
            summary["holdout_checkpoint_candidates"] = holdout_candidates
            summary["holdout_checkpoint_curve"] = holdout_curve
            summary["all_checkpoint_holdout_metrics"] = _metric_block(holdout)
            summary["all_checkpoint_holdout_variance_decomposition"] = _variance_decomposition(
                by_train_seed[name]["holdout"]
            )
            if selected is not None:
                selected_holdout = next(
                    (
                        candidate for candidate in holdout_candidates
                        if candidate["train_seed"] == selected["train_seed"]
                        and candidate["checkpoint_round"] == selected["checkpoint_round"]
                    ),
                    None,
                )
                if selected_holdout is not None:
                    # This is the only holdout result suitable for a final
                    # claim: the matching weights were selected exclusively
                    # on validation seeds above.
                    summary["selected_holdout_checkpoint"] = selected_holdout
                    summary["holdout_metrics"] = selected_holdout["metrics"]
        return summary

    primary = suite_summary("primary")
    suites = {name: suite_summary(name) for name in sorted(samples)}
    selected = primary["selected_checkpoint"]
    summary = {
        "experiment_id": experiment.experiment_id,
        "run_id": run_dir.name,
        "primary_metric": "score",
        "evaluation_jobs": sum(len(items) for roles in samples.values() for items in roles.values()),
        "primary_evaluation_jobs": primary["evaluation_jobs"],
        "primary_scenario": primary["scenario"],
        "reward_version": experiment.reward_version,
        "metrics": primary["metrics"],
        "invalid_actions_definition": "official per-agent INVALID_ACTION events per round; agent JSONL separately records selection legality",
        "metric_definitions": METRIC_DEFINITIONS,
        "checkpoint_candidates": primary["checkpoint_candidates"],
        "selected_checkpoint": selected,
        "evaluation_suites": suites,
    }
    if "holdout_metrics" in primary:
        summary["holdout_metrics"] = primary["holdout_metrics"]
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


def candidate_key(candidate: dict) -> tuple[float, float, float, float, int, int]:
    """Rank one checkpoint. Selection always uses validation metrics only."""
    metrics = candidate["metrics"]
    checkpoint_round = candidate.get("checkpoint_round")
    return (
        float(metrics["score"]["mean"]),
        -float(metrics["score"]["std"]),
        -float(metrics["suicides"]["mean"]),
        float(metrics["coins"]["mean"]),
        -int(candidate["train_seed"]),
        # Prefer the earlier checkpoint on an exact tie: less training for the
        # same measured result. ``latest`` sorts last, as the largest round.
        -(10 ** 9 if checkpoint_round is None else int(checkpoint_round)),
    )


def best_source_checkpoint(summary: dict) -> Path:
    """Choose the checkpoint with the best fixed greedy-evaluation summary."""
    candidates = summary["checkpoint_candidates"]
    if not candidates:
        raise FileNotFoundError("No training checkpoint is available for promotion")
    chosen = max(candidates, key=candidate_key)
    checkpoint = Path(chosen["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"No training checkpoint is available for promotion: {checkpoint}")
    return checkpoint


def maybe_promote(summary: dict) -> bool:
    scenario = summary.get("primary_scenario", "legacy")
    root = PROMOTION_ROOT / scenario
    root.mkdir(parents=True, exist_ok=True)
    best_path = root / "best_summary.json"
    current = json.loads(best_path.read_text(encoding="utf-8")) if best_path.exists() else None
    if current is not None and promotion_key(summary) <= promotion_key(current):
        return False
    source = best_source_checkpoint(summary)
    staging = root / ".active_model.staging.npz"
    shutil.copy2(source, staging)
    os.replace(staging, root / "active_model.npz")
    best_staging = root / ".best_model.staging.npz"
    shutil.copy2(source, best_staging)
    os.replace(best_staging, root / "best_model.npz")
    write_json(best_path, summary)
    write_json(root / "promotion_rule.json", {
        "rule": "maximize mean official score; then minimize score std; then minimize mean suicides; then maximize mean coins; then lexical run_id",
        "checkpoint_rule": "same order within a run, then smaller training seed, then earlier checkpoint round",
        "selection_split": "validation evaluation seeds only; holdout seeds are reported, never used to select",
        "primary_scenario": scenario,
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
