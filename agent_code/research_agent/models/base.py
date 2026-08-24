"""The small QModel interface shared by all route-specific model adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class QModel(Protocol):
    """A six-action Q-value model used by a learner.

    The action order is the frozen order in ``config.ACTIONS``.  Future model
    adapters may have their own private training internals, but must expose
    this inference and persistence interface to the shared runtime.
    """

    def q_values(self, state: np.ndarray) -> np.ndarray: ...

    def save(self, path: Path, *, metadata: dict | None = None) -> None: ...
