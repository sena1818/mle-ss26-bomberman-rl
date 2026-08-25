#!/usr/bin/env python3
"""Put several finished runs side by side as one controlled comparison.

Built for arm studies: several runs that differ in exactly one declared
dimension.  The script refuses to line up runs that differ in more than one,
because a table that silently mixes two changed factors is worse than no table.

It reads only ``evaluation_summary.json`` files and never writes into a run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiment_lib import write_json


# Ordered for reading: the activity metrics come first because official score
# is expected to stay near zero while the policy is still degenerate.
REPORTED = (
    "bomb_rate",
    "crates_per_round",
    "approximate_safe_bomb_rate",
    "rounds_with_bombs",
    "wait_fraction",
    "score",
    "coins",
    "suicides",
    "steps",
    "coins_per_crate",
    "distinct_cells",
)
COMPARABLE_FIELDS = ("route", "state_representation", "model", "algorithm", "reward_version", "exploration_version")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True,
                        help="Repeat once per arm. Order is preserved in the table.")
    parser.add_argument("--suite", default="primary", help="Evaluation suite to compare (default: primary).")
    parser.add_argument("--split", default="validation", choices=("validation", "holdout"),
                        help="Which seed split to report. Holdout is the honest one for a final claim.")
    parser.add_argument("--checkpoint-round", type=int,
                        help="For a checkpoint sweep, compare this fixed training round across all train seeds.")
    parser.add_argument("--out", type=Path, help="Optional path for a machine-readable copy of the table.")
    parser.add_argument("--allow-multiple-differences", action="store_true",
                        help="Compare runs that differ in more than one dimension. Say why in your notes.")
    return parser.parse_args()


def load_arm(run_dir: Path, suite: str, split: str, checkpoint_round: int | None) -> dict:
    summary = json.loads((run_dir / "evaluation_summary.json").read_text(encoding="utf-8"))
    snapshot = json.loads((run_dir / "experiment_config.snapshot.json").read_text(encoding="utf-8"))
    suites = summary.get("evaluation_suites", {})
    if suite not in suites:
        raise SystemExit(f"error: {run_dir.name} has no evaluation suite {suite!r}; available: {sorted(suites)}")
    block = suites[suite]
    key = "metrics" if split == "validation" else "holdout_metrics"
    variance_key = "variance_decomposition" if split == "validation" else "holdout_variance_decomposition"
    if checkpoint_round is not None:
        curve_key = "validation_checkpoint_curve" if split == "validation" else "holdout_checkpoint_curve"
        curve = block.get(curve_key, [])
        matched = next((entry for entry in curve if entry["checkpoint_round"] == checkpoint_round), None)
        if matched is None:
            available = [entry["checkpoint_round"] for entry in curve]
            raise SystemExit(
                f"error: {run_dir.name} has no {split} checkpoint round {checkpoint_round}; "
                f"available: {available}"
            )
        metrics = matched["metrics"]
        variance = matched["variance_decomposition"]
    else:
        if key not in block:
            raise SystemExit(
                f"error: {run_dir.name} suite {suite!r} has no {split} split. "
                "Declare holdout_seeds in checkpoint_evaluation to produce one."
            )
        metrics = block[key]
        variance = block.get(variance_key, {})
    agent = snapshot.get("agent", {})
    return {
        "run_id": summary.get("run_id", run_dir.name),
        "experiment_id": summary.get("experiment_id", ""),
        "scenario": block.get("scenario", ""),
        "evaluation_jobs": block.get("evaluation_jobs", 0),
        "checkpoint_mode": block.get("checkpoint_mode", "latest"),
        "dimensions": {
            "route": snapshot.get("route", ""),
            "state_representation": agent.get("state_representation", ""),
            "model": agent.get("model", ""),
            "algorithm": agent.get("algorithm", ""),
            "reward_version": snapshot.get("reward_version", ""),
            "exploration_version": snapshot.get("exploration_version", "E00"),
        },
        "reward_specification": snapshot.get("resolved_runtime_config", {}).get("reward_specification", {}),
        "predeclared": snapshot.get("_predeclared_design_numbers", {}),
        "metrics": metrics,
        "variance": variance,
    }


def changed_dimensions(arms: list[dict]) -> list[str]:
    return [
        field for field in COMPARABLE_FIELDS
        if len({arm["dimensions"].get(field, "") for arm in arms}) > 1
    ]


def render(arms: list[dict], changed: list[str], split: str, checkpoint_round: int | None) -> str:
    labels = [
        "_".join(
            part for part in (arm["dimensions"]["reward_version"], arm["dimensions"]["exploration_version"])
            if part
        ) or arm["run_id"]
        for arm in arms
    ]
    width = max(14, max(len(label) for label in labels) + 2)
    lines = [
        f"suite scenario : {arms[0]['scenario']}",
        f"seed split     : {split}",
        f"checkpoint     : {checkpoint_round if checkpoint_round is not None else 'validation-selected model'}",
        f"changed        : {', '.join(changed) if changed else 'nothing (identical declarations)'}",
        "",
        "metric".ljust(30) + "".join(label.rjust(width) for label in labels),
        "-" * (30 + width * len(labels)),
    ]
    for metric in REPORTED:
        cells = []
        for arm in arms:
            entry = arm["metrics"].get(metric)
            if entry is None or entry.get("count", 0) == 0:
                cells.append("n/a".rjust(width))
                continue
            cells.append(f"{entry['mean']:.4g}+-{entry['std']:.3g}".rjust(width))
        lines.append(metric.ljust(30) + "".join(cells))
    lines.append("")
    lines.append("predeclared p* (mean-field design model, not a bound)".ljust(30))
    predeclared = "".join(
        (str(arm["predeclared"].get("p_star", "-")) if arm["predeclared"] else "-").rjust(width) for arm in arms
    )
    lines.append("p_star".ljust(30) + predeclared)
    death = "".join(
        str(arm["reward_specification"].get("death_penalty", "-")).rjust(width) for arm in arms
    )
    lines.append("death_penalty".ljust(30) + death)
    lines.append("")
    lines.append("n/a means the metric was undefined for every job (for example no bomb was ever dropped).")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if len(args.run_dir) < 2:
        raise SystemExit("error: pass --run-dir at least twice")
    arms = [load_arm(run_dir, args.suite, args.split, args.checkpoint_round) for run_dir in args.run_dir]
    scenarios = {arm["scenario"] for arm in arms}
    if len(scenarios) > 1:
        raise SystemExit(f"error: refusing to compare different scenarios: {sorted(scenarios)}")
    changed = changed_dimensions(arms)
    if len(changed) > 1 and not args.allow_multiple_differences:
        raise SystemExit(
            "error: these runs differ in more than one dimension "
            f"({', '.join(changed)}), so the comparison is not controlled. "
            "Pass --allow-multiple-differences to override deliberately."
        )
    table = render(arms, changed, args.split, args.checkpoint_round)
    print(table)
    if args.out:
        write_json(args.out, {
            "split": args.split,
            "suite": args.suite,
            "checkpoint_round": args.checkpoint_round,
            "changed_dimensions": changed,
            "arms": arms,
        })


if __name__ == "__main__":
    main()
