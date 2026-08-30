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


def _as_declared(config: ExperimentConfig, model: QModel) -> QModel:
    """Refuse a model that does not match what the route declared.

    R02_12 and R02_13 were trained for 10000 rounds each with noisy layers
    declared, validated and snapshotted, and neither model had a single sigma
    parameter: this factory's ``mlp_q`` branch simply did not forward
    ``config.noisy``, and ``MLPQModel`` defaults it to False.  Both arms ran to
    completion, reported healthy diagnostics for everything they did have, and
    measured a factor that was never switched on.

    ``validate_config`` could not have caught it -- it validates the config,
    which was correct.  The gap is between the config and the object, so the
    check belongs here.
    """
    declared = bool(config.noisy)
    actual = bool(getattr(model, "noisy", False))
    if declared != actual:
        raise ValueError(
            f"{config.name} declares noisy={declared} but build_model produced "
            f"noisy={actual} for network {config.network!r}. A declared factor that "
            "does not reach the model trains an arm that measures nothing.")
    if declared and not getattr(model, "weight_sigmas", None):
        raise ValueError(
            f"{config.name} declares noisy but the model holds no sigma parameters.")
    return model


def build_model(config: ExperimentConfig, input_dim: int, *, seed: int) -> QModel:
    if config.network == "linear_q":
        return LinearQModel(input_dim, seed=seed, learning_rate=config.learning_rate)
    if config.network == "mlp_q":
        model = MLPQModel(
            input_dim, config.hidden_layers, seed=seed, learning_rate=config.learning_rate,
            optimizer=config.optimizer, td_loss=config.td_loss, gradient_clip_norm=config.gradient_clip_norm,
            noisy=config.noisy,
        )
        return _as_declared(config, model)
    if config.network == "categorical_mlp_q":
        from .categorical_mlp_q import CategoricalMLPQModel

        model = CategoricalMLPQModel(
            input_dim, config.hidden_layers, seed=seed, learning_rate=config.learning_rate,
            atoms=config.atoms, value_min=config.value_min, value_max=config.value_max,
            dueling=config.dueling, noisy=config.noisy,
            optimizer=config.optimizer, td_loss=config.td_loss,
            gradient_clip_norm=config.gradient_clip_norm,
        )
        return _as_declared(config, model)
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
    if config.network == "categorical_mlp_q":
        from .categorical_mlp_q import CategoricalMLPQModel

        model = CategoricalMLPQModel.load(
            path, learning_rate=config.learning_rate, optimizer=config.optimizer,
            td_loss=config.td_loss, gradient_clip_norm=config.gradient_clip_norm)
        if model.layer_sizes[1:-1] != tuple(config.hidden_layers):
            raise ValueError(
                f"Checkpoint {path} has hidden layers {model.layer_sizes[1:-1]}, "
                f"but {config.name} declares {tuple(config.hidden_layers)}.")
        if (model.dueling, model.noisy) != (config.dueling, config.noisy):
            raise ValueError(
                f"Checkpoint {path} is dueling={model.dueling} noisy={model.noisy}, but "
                f"{config.name} declares dueling={config.dueling} noisy={config.noisy}.")
        if (model.atoms, model.value_min, model.value_max) != (
                config.atoms, config.value_min, config.value_max):
            raise ValueError(
                f"Checkpoint {path} has support ({model.atoms} atoms, "
                f"[{model.value_min}, {model.value_max}]) but {config.name} declares "
                f"({config.atoms} atoms, [{config.value_min}, {config.value_max}]); "
                "the same weights mean different values on a different support.")
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
