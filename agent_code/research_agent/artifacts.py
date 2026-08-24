"""Run-isolated artifacts and structured logs for research experiments.

The official framework imports this module from the agent directory.  The
experiment runner therefore passes an absolute ``BOMBERMAN_ARTIFACT_DIR`` for
every job.  Nothing in this module writes to a shared active model.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .config import ExperimentConfig


_LEGACY_ARTIFACT_ROOT = Path(__file__).resolve().parent / "artifacts"


def run_id() -> str:
    """Return a filesystem-safe identifier supplied by the reproducible runner."""
    value = os.environ.get("BOMBERMAN_RUN_ID", "manual")
    if not value.replace("-", "").replace("_", "").isalnum():
        raise ValueError("BOMBERMAN_RUN_ID may contain only letters, digits, '-' and '_'.")
    return value


def artifact_root() -> Path:
    """Return the dedicated directory for this job.

    Direct manual runs are still safe: their data goes below ``artifacts/manual``
    rather than the historic shared artifact root.  The reproducible runner
    always sets the environment variable and never relies on that fallback.
    """
    selected = os.environ.get("BOMBERMAN_ARTIFACT_DIR")
    root = Path(selected).expanduser().resolve() if selected else _LEGACY_ARTIFACT_ROOT / "manual" / run_id()
    root.mkdir(parents=True, exist_ok=True)
    return root


def model_path() -> Path:
    """Return the explicit model requested by an evaluation job."""
    selected = os.environ.get("BOMBERMAN_MODEL_PATH")
    if not selected:
        raise RuntimeError(
            "BOMBERMAN_MODEL_PATH is required for evaluation; evaluation never reads a shared active model."
        )
    return Path(selected).expanduser().resolve()


def checkpoint_path(config: ExperimentConfig, round_number: int, updates: int) -> Path:
    scenario = os.environ.get("BOMBERMAN_SCENARIO", "unknown")
    seed = os.environ.get("BOMBERMAN_SEED", "unknown")
    file_name = (
        f"{config.name}_{config.reward_version}_{scenario}_seed{seed}_"
        f"round{round_number:05d}_updates{updates:08d}.npz"
    )
    return artifact_root() / "checkpoints" / file_name


def latest_model_path() -> Path:
    """Mutable convenience snapshot, private to one training job only."""
    return artifact_root() / "latest_model.npz"


def checkpoint_interval() -> int:
    value = int(os.environ.get("BOMBERMAN_CHECKPOINT_EVERY", "25"))
    if value < 1:
        raise ValueError("BOMBERMAN_CHECKPOINT_EVERY must be positive.")
    return value


def append_jsonl(kind: str, payload: dict) -> None:
    """Append one self-describing event record without a non-stdlib dependency."""
    log_path = artifact_root() / "agent.jsonl"
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        "run_id": run_id(),
        **payload,
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
