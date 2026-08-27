"""QModel adapters selected by the shared experiment runtime.

One adapter per function approximator, named after the ``agent.model`` value
that selects it.  Callbacks and training hooks never import a route-specific
model: they go through ``build_model``/``load_model``.

``cnn_mlp_q`` and ``dueling_cnn_mlp_q`` are imported inside their branch on
purpose.  They are the only adapters that need PyTorch, and the M1--M3 lines
must keep running in a NumPy-only environment.
"""

from __future__ import annotations

from pathlib import Path

from ..config import ExperimentConfig
from .base import QModel
from .linear_q import LinearQModel
from .mlp_q import MLPQModel


_CNN_NETWORKS = {"cnn_mlp_q": False, "dueling_cnn_mlp_q": True}


def build_model(config: ExperimentConfig, input_dim: int, *, seed: int) -> QModel:
    if config.network == "linear_q":
        return LinearQModel(input_dim, seed=seed, learning_rate=config.learning_rate)
    if config.network == "mlp_q":
        return MLPQModel(
            input_dim, config.hidden_layers, seed=seed, learning_rate=config.learning_rate,
            optimizer=config.optimizer, td_loss=config.td_loss, gradient_clip_norm=config.gradient_clip_norm,
        )
    if config.network in _CNN_NETWORKS:
        from .cnn_mlp_q import CnnMlpQModel

        return CnnMlpQModel(
            input_dim,
            hidden_layers=config.hidden_layers,
            dueling=_CNN_NETWORKS[config.network],
            seed=seed,
            learning_rate=config.learning_rate,
            optimizer=config.optimizer,
            td_loss=config.td_loss,
            gradient_clip_norm=config.gradient_clip_norm,
        )
    raise NotImplementedError(f"QModel adapter {config.network!r} has not been implemented yet.")


def load_model(config: ExperimentConfig, path: Path) -> QModel:
    if config.network == "linear_q":
        model = LinearQModel.load(path)
        model.learning_rate = config.learning_rate
        return model
    if config.network == "mlp_q":
        model = MLPQModel.load(
            path,
            learning_rate=config.learning_rate,
            optimizer=config.optimizer,
            td_loss=config.td_loss,
            gradient_clip_norm=config.gradient_clip_norm,
        )
        if model.layer_sizes[1:-1] != tuple(config.hidden_layers):
            raise ValueError(
                f"Checkpoint {path} has hidden layers {model.layer_sizes[1:-1]}, "
                f"but {config.name} declares {tuple(config.hidden_layers)}."
            )
        return model
    if config.network in _CNN_NETWORKS:
        from .cnn_mlp_q import CnnMlpQModel

        model = CnnMlpQModel.load(
            path, learning_rate=config.learning_rate, optimizer=config.optimizer,
            td_loss=config.td_loss, gradient_clip_norm=config.gradient_clip_norm)
        if model.dueling != _CNN_NETWORKS[config.network]:
            raise ValueError(f"Checkpoint {path} is a {model.model_type}, but {config.name} declares {config.network}.")
        return model
    raise NotImplementedError(f"QModel adapter {config.network!r} has not been implemented yet.")


__all__ = ("QModel", "LinearQModel", "MLPQModel", "build_model", "load_model")
