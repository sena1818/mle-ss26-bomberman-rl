"""QModel adapters selected by the shared experiment runtime.

R01 is the only implemented adapter.  Names for later routes belong in this
package so that callbacks and training callbacks never need route-specific
imports.
"""

from __future__ import annotations

from pathlib import Path

from ..config import ExperimentConfig
from .base import QModel
from .linear_q import LinearQModel


def build_model(config: ExperimentConfig, input_dim: int) -> QModel:
    if config.network == "linear_q":
        return LinearQModel(input_dim)
    raise NotImplementedError(f"QModel adapter {config.network!r} has not been implemented yet.")


def load_model(config: ExperimentConfig, path: Path) -> QModel:
    if config.network == "linear_q":
        return LinearQModel.load(path)
    raise NotImplementedError(f"QModel adapter {config.network!r} has not been implemented yet.")


__all__ = ("QModel", "LinearQModel", "build_model", "load_model")
