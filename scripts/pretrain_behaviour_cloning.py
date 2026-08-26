"""Fit an M4 network to demonstrations, producing a warm-start checkpoint.

This is the ``+ BC warm start`` rung of the M4 ladder in docs/05 section 5.4.
It is a *single* declared increment: the anchor must already learn from scratch
before a run is allowed to start from cloned weights, otherwise a working
warm start would hide a broken base.

What is and is not fitted here matters.  Cross entropy over the Q head fits the
demonstrator's ``argmax`` -- the policy -- and nothing about returns; a cloned
head is calibrated only up to scale.  So the head is rescaled afterwards to the
magnitude TD targets actually occupy.  Scaling by a positive factor leaves every
``argmax`` untouched, so the cloned policy survives exactly while the first DQN
updates no longer have to spend themselves shrinking an arbitrary scale.

The resulting checkpoint is referenced from an experiment config through its
``initial_model`` block, which records the file's SHA-256 so a run can always be
traced back to the demonstrations it started from.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_code.research_agent.config import ACTIONS, EXPERIMENTS  # noqa: E402
from agent_code.research_agent.models import build_model  # noqa: E402
from experiment_lib import git_provenance, write_json  # noqa: E402


DEFAULT_OUTPUT = ROOT / "pretraining" / "models"


def _accuracy(model, states: np.ndarray, action_indices: np.ndarray, batch_size: int) -> float:
    correct = 0
    for start in range(0, len(states), batch_size):
        chunk = states[start:start + batch_size].astype(np.float32)
        predictions = np.argmax(model.q_values_batch(chunk), axis=1)
        correct += int(np.sum(predictions == action_indices[start:start + batch_size]))
    return correct / len(states)


def pretrain(
    *, route: str, demonstrations: Path, output: Path, epochs: int, batch_size: int,
    validation_fraction: float, seed: int, target_q_scale: float,
) -> dict:
    config = EXPERIMENTS[route]
    with np.load(demonstrations) as data:
        states = data["states"]
        action_indices = data["action_indices"]
    metadata_path = demonstrations.with_suffix(".json")
    demonstration_metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    declared_encoder = demonstration_metadata.get("state_encoder")
    if declared_encoder is not None and declared_encoder != config.state_encoder:
        raise ValueError(
            f"{demonstrations.name} was recorded with encoder {declared_encoder!r}, "
            f"but route {route} consumes {config.state_encoder!r}."
        )

    generator = np.random.default_rng(seed)
    order = generator.permutation(len(states))
    split = int(len(order) * (1.0 - validation_fraction))
    if split < batch_size or len(order) - split < 1:
        raise ValueError("The dataset is too small for the requested batch size and validation split.")
    train_index, validation_index = order[:split], order[split:]

    model = build_model(config, states.shape[1], seed=seed)
    if not hasattr(model, "fit_policy_batch"):
        raise TypeError(f"Route {route} uses {config.network!r}, which cannot be behaviour-cloned.")

    history = []
    for epoch in range(1, epochs + 1):
        shuffled = generator.permutation(train_index)
        losses = []
        for start in range(0, len(shuffled) - batch_size + 1, batch_size):
            rows = shuffled[start:start + batch_size]
            losses.append(model.fit_policy_batch(states[rows].astype(np.float32), action_indices[rows]))
        record = {
            "epoch": epoch,
            "mean_cross_entropy": float(np.mean(losses)),
            "train_accuracy": _accuracy(model, states[train_index], action_indices[train_index], batch_size),
            "validation_accuracy": _accuracy(model, states[validation_index], action_indices[validation_index], batch_size),
        }
        history.append(record)
        print(f"epoch {epoch:2d}  loss {record['mean_cross_entropy']:.4f}  "
              f"train {record['train_accuracy']:.3f}  val {record['validation_accuracy']:.3f}")

    sample = states[validation_index[:batch_size]].astype(np.float32)
    magnitude_before = float(np.abs(model.q_values_batch(sample)).mean())
    factor = target_q_scale / magnitude_before if magnitude_before > 0 else 1.0
    policy_before = np.argmax(model.q_values_batch(sample), axis=1)
    model.rescale_head(factor)
    policy_after = np.argmax(model.q_values_batch(sample), axis=1)
    # The rescaling claim is checked rather than asserted in a comment: a head
    # that reordered its actions would silently discard the cloning result.
    if not np.array_equal(policy_before, policy_after):
        raise RuntimeError("Rescaling the head changed the greedy policy; the warm start would be invalid.")

    summary = {
        "route": route,
        "model": config.network,
        "demonstrations": str(demonstrations),
        "demonstration_sha256": demonstration_metadata.get("sha256"),
        "demonstrator": demonstration_metadata.get("demonstrator"),
        "states": int(len(states)),
        "train_states": int(len(train_index)),
        "validation_states": int(len(validation_index)),
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "history": history,
        "final_validation_accuracy": history[-1]["validation_accuracy"],
        "majority_class_accuracy": float(np.bincount(action_indices, minlength=len(ACTIONS)).max() / len(action_indices)),
        "head_rescaling": {
            "target_q_scale": target_q_scale,
            "mean_abs_q_before": magnitude_before,
            "factor": factor,
            "mean_abs_q_after": float(np.abs(model.q_values_batch(sample)).mean()),
        },
        "pretrained_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        **git_provenance(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output, metadata={"pretraining": "behaviour_cloning", "route": route,
                                 "demonstration_sha256": summary["demonstration_sha256"]})
    summary["sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    write_json(output.with_suffix(".json"), summary)
    print(f"\nwrote {output}\n  sha256 {summary['sha256']}\n"
          f"  validation accuracy {summary['final_validation_accuracy']:.3f} "
          f"(majority class {summary['majority_class_accuracy']:.3f})")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--route", default="R07", choices=sorted(EXPERIMENTS))
    parser.add_argument("--demonstrations", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-q-scale", type=float, default=1.0,
                        help="mean |Q| the head is rescaled to; positive scaling preserves the policy")
    args = parser.parse_args(argv)

    output = args.output or DEFAULT_OUTPUT / f"{args.route.lower()}_bc_{args.demonstrations.stem}.npz"
    pretrain(
        route=args.route, demonstrations=args.demonstrations, output=output, epochs=args.epochs,
        batch_size=args.batch_size, validation_fraction=args.validation_fraction, seed=args.seed,
        target_q_scale=args.target_q_scale,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
