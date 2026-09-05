"""The small QModel interface shared by all route-specific model adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class QModel(Protocol):
    """A six-action Q-value model used by a learner.

    The action order is the frozen order in ``config.ACTIONS``.  Model adapters
    keep their own training internals -- SGD by hand, or an optimizer object --
    but must expose this inference, batch-fitting and persistence interface to
    the shared runtime.

    ``q_values``/``q_learning_update`` are the online single-transition path used
    by M1 and M3 without replay.  ``q_values_batch``/``fit_batch``/``clone`` are
    the batch path a replay learner needs; a model that implements one must
    implement all three, because a target network is a clone that is fitted
    never and queried in batches.
    """

    def q_values(self, state: np.ndarray) -> np.ndarray: ...

    def q_values_batch(self, states: np.ndarray) -> np.ndarray: ...

    def fit_batch(self, states: np.ndarray, action_indices: np.ndarray, targets: np.ndarray,
                  weights: np.ndarray | None = None) -> np.ndarray:
        """Fit the selected heads towards ``targets``; return the TD errors."""

    def clone(self) -> "QModel":
        """Return an independent copy, used as a frozen target network."""

    def copy_parameters_from(self, other: "QModel") -> None:
        """Overwrite this model's parameters with another's, in place."""

    def save(self, path: Path, *, metadata: dict | None = None) -> None: ...


# The optimizer / loss / clip triple is declared per route and applied by every
# gradient-based adapter.  Validating it in each adapter let the two drift: the
# CNN accepted only what it happened to hardcode, and neither could tell you
# which values were supported without reading both.  One place, one answer.
SUPPORTED_OPTIMIZERS = frozenset({"sgd", "adam"})
# "cross_entropy" belongs to the distributional head only: C51 regresses a
# probability vector onto a projected target distribution, so a squared or Huber
# error over a scalar has nothing to apply to.  Declared rather than implied, so
# a run snapshot says which loss actually ran.
SUPPORTED_TD_LOSSES = frozenset({"mse", "huber", "cross_entropy"})


def validate_training_declarations(optimizer: str, td_loss: str,
                                   gradient_clip_norm: float | None) -> None:
    """Refuse a declaration no adapter implements, naming what is supported."""
    if optimizer not in SUPPORTED_OPTIMIZERS:
        raise ValueError(
            f"Unsupported optimizer {optimizer!r}; supported: {sorted(SUPPORTED_OPTIMIZERS)}.")
    if td_loss not in SUPPORTED_TD_LOSSES:
        raise ValueError(
            f"Unsupported TD loss {td_loss!r}; supported: {sorted(SUPPORTED_TD_LOSSES)}.")
    if gradient_clip_norm is not None and gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive when declared.")
