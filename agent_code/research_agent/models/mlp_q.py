"""The small dependency-free MLP six-head Q model used by the M3 line."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import ACTIONS


class MLPQModel:
    """A ReLU MLP trained by one-step online Q-learning with SGD.

    The implementation intentionally relies only on NumPy, which is already a
    dependency of the official project.  Hidden layers use He initialization;
    the final Q head starts close to zero like R01's linear head.
    """

    def __init__(self, input_dim: int, hidden_layers: tuple[int, ...], seed: int = 0, learning_rate: float = 0.02):
        if input_dim < 1 or not hidden_layers or any(width < 1 for width in hidden_layers):
            raise ValueError("MLPQModel requires a positive input dimension and non-empty positive hidden layers.")
        self.layer_sizes = (int(input_dim), *(int(width) for width in hidden_layers), len(ACTIONS))
        generator = np.random.default_rng(seed)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for layer, (fan_in, fan_out) in enumerate(zip(self.layer_sizes, self.layer_sizes[1:])):
            scale = np.sqrt(2.0 / fan_in) if layer < len(self.layer_sizes) - 2 else 0.01
            self.weights.append(generator.normal(0.0, scale, size=(fan_out, fan_in)).astype(np.float32))
            self.biases.append(np.zeros(fan_out, dtype=np.float32))
        # Only the batch path uses this; the online path is handed a rate by
        # its learner, exactly as the linear model is.
        self.learning_rate = float(learning_rate)
        self.metadata: dict = {}

    def q_values(self, state: np.ndarray) -> np.ndarray:
        """Return six Q-values in the frozen ``config.ACTIONS`` order."""
        return self._forward(state)

    def q_learning_update(
        self,
        state: np.ndarray,
        action_index: int,
        reward: float,
        next_state: np.ndarray | None,
        next_legal_mask: np.ndarray | None,
        learning_rate: float,
        discount: float,
    ) -> float:
        """Apply SGD to the selected Q head using the masked Q-learning target."""
        prediction, activations, pre_activations = self._forward(state, retain_cache=True)
        predicted_q = float(prediction[action_index])
        if next_state is None:
            target = float(reward)
        else:
            next_q = self.q_values(next_state)
            if next_legal_mask is None or not np.any(next_legal_mask):
                raise ValueError("Non-terminal Q-learning updates require at least one legal next action.")
            target = float(reward) + discount * float(np.max(next_q[next_legal_mask]))
        td_error = target - predicted_q

        # Gradient ascent on td_error * Q(s, a), equivalent to SGD on
        # 1/2(target - Q(s, a))^2 with the target held constant.
        delta = np.zeros(len(ACTIONS), dtype=np.float32)
        delta[action_index] = td_error
        weight_gradients: list[np.ndarray] = [np.empty_like(weight) for weight in self.weights]
        bias_gradients: list[np.ndarray] = [np.empty_like(bias) for bias in self.biases]
        for layer in range(len(self.weights) - 1, -1, -1):
            weight_gradients[layer] = np.outer(delta, activations[layer]).astype(np.float32)
            bias_gradients[layer] = delta
            if layer:
                delta = (self.weights[layer].T @ delta) * (pre_activations[layer - 1] > 0.0)
        for layer, (weight_gradient, bias_gradient) in enumerate(zip(weight_gradients, bias_gradients)):
            self.weights[layer] += learning_rate * weight_gradient
            self.biases[layer] += learning_rate * bias_gradient
        return td_error

    def q_values_batch(self, states: np.ndarray) -> np.ndarray:
        """Return one row of six Q-values per state in the batch."""
        return self._forward_batch(np.asarray(states, dtype=np.float32))[0]

    def fit_batch(self, states: np.ndarray, action_indices: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """One SGD step on the mean squared TD error of the selected heads."""
        states = np.asarray(states, dtype=np.float32)
        action_indices = np.asarray(action_indices, dtype=np.intp)
        predictions, activations, pre_activations = self._forward_batch(states)
        rows = np.arange(len(action_indices))
        td_errors = np.asarray(targets, dtype=np.float32) - predictions[rows, action_indices]

        delta = np.zeros_like(predictions)
        delta[rows, action_indices] = td_errors / len(action_indices)
        for layer in range(len(self.weights) - 1, -1, -1):
            weight_gradient = delta.T @ activations[layer]
            bias_gradient = delta.sum(axis=0)
            if layer:
                delta = (delta @ self.weights[layer]) * (pre_activations[layer - 1] > 0.0)
            self.weights[layer] += (self.learning_rate * weight_gradient).astype(np.float32)
            self.biases[layer] += (self.learning_rate * bias_gradient).astype(np.float32)
        return td_errors

    def clone(self) -> "MLPQModel":
        copy = MLPQModel(self.layer_sizes[0], self.layer_sizes[1:-1], learning_rate=self.learning_rate)
        copy.copy_parameters_from(self)
        return copy

    def copy_parameters_from(self, other: "MLPQModel") -> None:
        if other.layer_sizes != self.layer_sizes:
            raise ValueError(f"Cannot copy parameters between MLP shapes {other.layer_sizes} and {self.layer_sizes}.")
        self.weights = [weight.copy() for weight in other.weights]
        self.biases = [bias.copy() for bias in other.biases]

    def _forward_batch(self, states: np.ndarray):
        """Batched forward pass returning the per-layer cache used by backprop."""
        if states.ndim != 2 or states.shape[1] != self.layer_sizes[0]:
            raise ValueError(f"Expected a batch of states with width {self.layer_sizes[0]}, received {states.shape}.")
        activation = states
        activations = [activation]
        pre_activations: list[np.ndarray] = []
        for layer, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            linear = activation @ weight.T + bias
            pre_activations.append(linear)
            activation = linear if layer == len(self.weights) - 1 else np.maximum(linear, 0.0)
            activations.append(activation)
        return activation, activations, pre_activations

    def save(self, path: Path, *, metadata: dict | None = None) -> None:
        """Persist all parameters, architecture, and JSON-safe experiment metadata."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata = dict(metadata or {})
        payload: dict[str, np.ndarray] = {
            "model_type": np.asarray("mlp_q"),
            "layer_sizes": np.asarray(self.layer_sizes, dtype=np.int64),
            "metadata": np.asarray(json.dumps(self.metadata, sort_keys=True)),
        }
        for layer, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            payload[f"weights_{layer}"] = weight
            payload[f"biases_{layer}"] = bias
        np.savez(path, **payload)

    @classmethod
    def load(cls, path: Path) -> "MLPQModel":
        """Load and validate a self-describing R02 checkpoint."""
        with np.load(path, allow_pickle=False) as data:
            required = {"model_type", "layer_sizes", "metadata"}
            if not required.issubset(data.files) or str(data["model_type"].item()) != "mlp_q":
                raise ValueError(f"Checkpoint {path} is not an MLP Q model.")
            layer_sizes = tuple(int(size) for size in data["layer_sizes"])
            if len(layer_sizes) < 3 or layer_sizes[0] < 1 or any(size < 1 for size in layer_sizes[1:]) or layer_sizes[-1] != len(ACTIONS):
                raise ValueError(f"Checkpoint {path} has an invalid MLP shape {layer_sizes}.")
            model = cls(layer_sizes[0], layer_sizes[1:-1])
            weights: list[np.ndarray] = []
            biases: list[np.ndarray] = []
            for layer, (fan_in, fan_out) in enumerate(zip(layer_sizes, layer_sizes[1:])):
                weight_key, bias_key = f"weights_{layer}", f"biases_{layer}"
                if weight_key not in data or bias_key not in data:
                    raise ValueError(f"Checkpoint {path} is missing MLP layer {layer}.")
                weight = data[weight_key].astype(np.float32)
                bias = data[bias_key].astype(np.float32)
                if weight.shape != (fan_out, fan_in) or bias.shape != (fan_out,):
                    raise ValueError(f"Checkpoint {path} has invalid parameters for MLP layer {layer}.")
                weights.append(weight)
                biases.append(bias)
            try:
                metadata = json.loads(str(data["metadata"].item()))
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Checkpoint {path} has invalid MLP metadata.") from exc
        if not isinstance(metadata, dict):
            raise ValueError(f"Checkpoint {path} metadata must be a JSON object.")
        model.weights = weights
        model.biases = biases
        model.metadata = metadata
        return model

    def _forward(self, state: np.ndarray, *, retain_cache: bool = False):
        activation = np.asarray(state, dtype=np.float32)
        if activation.shape != (self.layer_sizes[0],):
            raise ValueError(f"Expected state shape {(self.layer_sizes[0],)}, received {activation.shape}.")
        activations = [activation]
        pre_activations: list[np.ndarray] = []
        for layer, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            linear = weight @ activation + bias
            pre_activations.append(linear)
            activation = linear if layer == len(self.weights) - 1 else np.maximum(linear, 0.0)
            activations.append(activation)
        if retain_cache:
            return activation, activations, pre_activations
        return activation
