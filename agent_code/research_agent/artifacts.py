"""Run-isolated artifacts and structured logs for research experiments.

The official framework imports this module from the agent directory.  The
experiment runner therefore passes an absolute ``BOMBERMAN_ARTIFACT_DIR`` for
every job.  Nothing in this module writes to a shared active model.
"""

from __future__ import annotations

import atexit
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .config import ExperimentConfig, submission_declaration


_LEGACY_ARTIFACT_ROOT = Path(__file__).resolve().parent / "artifacts"
_PACKAGED_MODEL_NAME = "model.npz"
_MANUAL_RUN_ID = f"manual_{os.getpid()}"


def run_id() -> str:
    """Return a filesystem-safe identifier supplied by the reproducible runner."""
    # A direct framework invocation has no runner to allocate a job directory.
    # Keep that convenience path process-private instead of letting every manual
    # invocation overwrite ``artifacts/manual/manual``.
    value = os.environ.get("BOMBERMAN_RUN_ID", _MANUAL_RUN_ID)
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
    """Return an explicit job model, or the model packaged with this agent.

    Experiment jobs must set ``BOMBERMAN_MODEL_PATH`` so evaluation is tied to
    one exact checkpoint.  The fallback makes an exported final agent work with
    the unmodified official framework, which provides no experiment variables.
    """
    selected = os.environ.get("BOMBERMAN_MODEL_PATH")
    if selected:
        return Path(selected).expanduser().resolve()
    here = Path(__file__).resolve().parent
    declaration = submission_declaration()
    if declaration is not None:
        # Relative to the agent directory, never absolute: the tournament
        # unpacks this folder somewhere nobody here can predict, and a leading
        # slash is the most common way a submitted agent fails their setup.
        return here / declaration["model"]
    return here / _PACKAGED_MODEL_NAME


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


# One open handle per artifact log, per process.  Reopening the file for every
# record costs nothing on a laptop -- the page cache absorbs it -- but this
# agent writes one record per action, which is roughly 1.1 million open/close
# cycles per training job and 5.7 million per five-seed arm.  On a parallel
# filesystem every open is a metadata operation against a server shared by the
# whole cluster, so that pattern is both slow for the job and antisocial.
_LOG_HANDLES: dict[Path, "object"] = {}

def _log_handle(log_path: Path):
    handle = _LOG_HANDLES.get(log_path)
    if handle is None or handle.closed:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("a", encoding="utf-8")
        _LOG_HANDLES[log_path] = handle
    return handle


def close_artifact_logs() -> None:
    """Flush and close every open artifact log; registered to run at exit."""
    for handle in list(_LOG_HANDLES.values()):
        if not handle.closed:
            handle.flush()
            handle.close()
    _LOG_HANDLES.clear()


atexit.register(close_artifact_logs)


def action_logging_enabled() -> bool:
    """Whether this process writes the per-action record.

    Every diagnostic in docs/01 is computed from that record, so it stays on for
    experiments.  A packaged agent turns it off: it is one flushed line per
    step, the tournament plays many rounds back to back, and nothing there will
    ever read the file -- it would only be unbounded growth inside a directory
    the organisers unpack.
    """
    declaration = submission_declaration()
    return True if declaration is None else bool(declaration.get("action_log", False))


def append_jsonl(kind: str, payload: dict) -> None:
    """Append one self-describing event record without a non-stdlib dependency."""
    if not action_logging_enabled():
        return
    log_path = artifact_root() / "agent.jsonl"
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        "run_id": run_id(),
        **payload,
    }
    handle = _log_handle(log_path)
    handle.write(json.dumps(record, sort_keys=True) + "\n")
    # Still flushed on every record, so a reader sees the same thing it always
    # did and a crash loses nothing.  The cost removed here is the open and the
    # close, not the write: on a parallel filesystem an open is a metadata
    # operation against a server the whole cluster shares, while appending to
    # an already-open file is the streaming write such a filesystem is built
    # for.  Behaviour is unchanged; only the syscall pattern is.
    handle.flush()
