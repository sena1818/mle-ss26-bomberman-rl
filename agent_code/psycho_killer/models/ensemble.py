"""Several checkpoints of one route, averaged into a single greedy policy.

Why this exists.  The submission is one model, and on this project a single
training seed is a genuine lottery: the rainbow arm's seeds span 3.79 to 7.13
against three rule_based (docs/01 section 7.39.5), and even on the opponent
proxy pool, where that spread collapses, the seeds still differ by more than the
gap between the two leading arms.  Averaging the value estimates of every seed
is the standard, and here the cheapest, way to spend that variance: the members
already exist, and a five-member CNN forward pass costs 1.3 ms against the
framework's 0.5 s budget.

What is averaged is the *action value*, not the greedy action.  A vote over
argmaxes throws away how strongly each member preferred its choice, and on a
Q function the magnitudes are the calibrated part -- a member that is nearly
indifferent should not outvote one that is certain.  For a distributional head
``q_values`` already returns the expectation of each action's distribution, so
the same average is the expectation of the mixture and needs no special case.

This is an evaluation-only model.  It has no gradient path on purpose: an
ensemble that could be trained would be a different agent design, not a way of
reading the checkpoints an experiment already produced, and every arm this is
built from was trained as a single network.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..config import ACTIONS


MANIFEST_SUFFIX = ".ensemble.json"


class EnsembleQModel:
    """Mean of several members' Q-values; greedy selection is unchanged.

    The members are loaded through the same ``load_model`` every other route
    uses, so an ensemble of a categorical head is a categorical head and an
    ensemble of a CNN is a CNN -- this class never learns what it is holding.
    """

    def __init__(self, members: list, *, manifest_path: Path, route: str):
        if not members:
            raise ValueError("An ensemble needs at least one member.")
        self.members = members
        self.manifest_path = Path(manifest_path)
        self.route = route
        self.metadata = {"ensemble": {"route": route, "members": len(members)}}
        # A member's own model_type, so a run snapshot still says what was run.
        kinds = {getattr(member, "model_type", type(member).__name__) for member in members}
        if len(kinds) != 1:
            raise ValueError(f"An ensemble must hold one kind of model; found {sorted(kinds)}.")
        self.member_type = kinds.pop()

    @property
    def model_type(self) -> str:
        return f"ensemble[{len(self.members)}x{self.member_type}]"

    @property
    def noisy(self) -> bool:
        return any(getattr(member, "noisy", False) for member in self.members)

    @property
    def noise_enabled(self) -> bool:
        return any(getattr(member, "noise_enabled", False) for member in self.members)

    @noise_enabled.setter
    def noise_enabled(self, value: bool) -> None:
        # The runtime switches noise off for evaluation; it has to reach every
        # member, or an ensemble would explore where a single model would not.
        for member in self.members:
            if hasattr(member, "noise_enabled"):
                member.noise_enabled = bool(value)

    def q_values(self, state: np.ndarray) -> np.ndarray:
        total = self.members[0].q_values(state).astype(np.float64)
        for member in self.members[1:]:
            total += member.q_values(state)
        return (total / len(self.members)).astype(np.float32)

    def q_values_batch(self, states: np.ndarray) -> np.ndarray:
        total = self.members[0].q_values_batch(states).astype(np.float64)
        for member in self.members[1:]:
            total += member.q_values_batch(states)
        return (total / len(self.members)).astype(np.float32)

    def _refuse(self, what: str):
        raise NotImplementedError(
            f"An ensemble is evaluation-only and cannot {what}. Every member was trained as a "
            "single network; training the average would be a different agent design.")

    def fit_batch(self, *args, **kwargs):
        self._refuse("be fitted")

    def clone(self):
        self._refuse("serve as a target network")

    def copy_parameters_from(self, other):
        self._refuse("take parameters")

    def save(self, path: Path, *, metadata: dict | None = None) -> None:
        self._refuse("be saved as one checkpoint")

    @classmethod
    def load(cls, config, path: Path) -> "EnsembleQModel":
        """Load every member a manifest names, verifying each digest.

        Member paths are relative to the manifest, so a run directory that
        carries both stays movable -- which is what the submitted agent folder
        has to be.
        """
        from . import load_model

        path = Path(path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("kind") != "ensemble":
            raise ValueError(f"{path} is not an ensemble manifest.")
        declared = manifest.get("route")
        if declared != config.name:
            raise ValueError(
                f"{path} is an ensemble of route {declared!r} but the job declares {config.name!r}. "
                "Averaging two different agent designs is not this class's job.")
        members = []
        for entry in manifest["members"]:
            member_path = (path.parent / entry["path"]).resolve()
            if not member_path.is_file():
                raise FileNotFoundError(f"Ensemble member is unavailable: {member_path}")
            digest = hashlib.sha256(member_path.read_bytes()).hexdigest()
            if digest != entry["sha256"]:
                raise ValueError(
                    f"Ensemble member {entry['path']} has digest {digest}, but the manifest "
                    f"declares {entry['sha256']}. A member may not change silently.")
            members.append(load_model(config, member_path))
        model = cls(members, manifest_path=path, route=declared)
        if model.q_values(np.zeros(members[0].input_dim if hasattr(members[0], "input_dim")
                                   else manifest["input_dim"], dtype=np.float32)).shape != (len(ACTIONS),):
            raise ValueError(f"{path} does not produce the frozen six-action interface.")
        return model


def write_manifest(path: Path, *, route: str, members: list[Path], input_dim: int,
                   provenance: dict | None = None) -> dict:
    """Write a manifest beside its members, recording each member's digest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    for member in members:
        member = Path(member)
        if not member.is_file():
            raise FileNotFoundError(f"Ensemble member is unavailable: {member}")
        try:
            relative = member.resolve().relative_to(path.parent.resolve())
        except ValueError as exc:
            raise ValueError(
                f"Ensemble member {member} is not under the manifest's directory {path.parent}; "
                "a manifest and its members have to travel together.") from exc
        entries.append({"path": str(relative),
                        "sha256": hashlib.sha256(member.read_bytes()).hexdigest()})
    manifest = {"kind": "ensemble", "route": route, "input_dim": int(input_dim),
                "members": entries}
    if provenance:
        manifest["provenance"] = provenance
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
