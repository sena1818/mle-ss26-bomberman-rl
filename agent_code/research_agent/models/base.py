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

    def fit_batch(self, states: np.ndarray, action_indices: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Fit the selected heads towards ``targets``; return the TD errors."""

    def clone(self) -> "QModel":
        """Return an independent copy, used as a frozen target network."""

    def copy_parameters_from(self, other: "QModel") -> None:
        """Overwrite this model's parameters with another's, in place."""

    def save(self, path: Path, *, metadata: dict | None = None) -> None: ...
