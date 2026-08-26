"""The dependency-free linear QModel adapter used by the M1 and M2 lines."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import ACTIONS


class LinearQModel:
    """A multi-head linear approximation Q(s, ·) = W @ features + b."""

    def __init__(self, input_dim: int, seed: int = 0, learning_rate: float = 0.02):
        generator = np.random.default_rng(seed)
        self.weights = generator.normal(0.0, 0.01, size=(len(ACTIONS), input_dim)).astype(np.float32)
        self.bias = np.zeros(len(ACTIONS), dtype=np.float32)
        # Only the batch path uses this; the online path is still handed a rate
        # by its learner, so every published R01 result stays bit-reproducible.
        self.learning_rate = float(learning_rate)

    def q_values(self, state: np.ndarray) -> np.ndarray:
        return self.weights @ state + self.bias

    def q_values_batch(self, states: np.ndarray) -> np.ndarray:
        return np.asarray(states, dtype=np.float32) @ self.weights.T + self.bias

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
        prediction = float(self.q_values(state)[action_index])
        if next_state is None:
            target = reward
        else:
            next_q = self.q_values(next_state)
            assert next_legal_mask is not None and np.any(next_legal_mask)
            target = reward + discount * float(np.max(next_q[next_legal_mask]))
        td_error = target - prediction
        self.weights[action_index] += learning_rate * td_error * state
        self.bias[action_index] += learning_rate * td_error
        return td_error

    def fit_batch(self, states: np.ndarray, action_indices: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """One SGD step on the mean squared TD error of the selected heads."""
        states = np.asarray(states, dtype=np.float32)
        action_indices = np.asarray(action_indices, dtype=np.intp)
        predictions = self.q_values_batch(states)[np.arange(len(action_indices)), action_indices]
        td_errors = np.asarray(targets, dtype=np.float32) - predictions
        scale = self.learning_rate / len(action_indices)
        # Rows of the same action accumulate, hence add.at rather than fancy +=.
        np.add.at(self.weights, action_indices, scale * td_errors[:, None] * states)
        np.add.at(self.bias, action_indices, scale * td_errors)
        return td_errors

    def clone(self) -> "LinearQModel":
        copy = LinearQModel(self.weights.shape[1], learning_rate=self.learning_rate)
        copy.copy_parameters_from(self)
        return copy

    def copy_parameters_from(self, other: "LinearQModel") -> None:
        self.weights = other.weights.copy()
        self.bias = other.bias.copy()

    def save(self, path: Path, *, metadata: dict | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, weights=self.weights, bias=self.bias, metadata=np.asarray(json.dumps(metadata or {}, sort_keys=True)))

    @classmethod
    def load(cls, path: Path) -> "LinearQModel":
        data = np.load(path)
        if data["weights"].shape[0] != len(ACTIONS) or data["bias"].shape != (len(ACTIONS),):
            raise ValueError(f"Checkpoint {path} is not a six-action linear Q model.")
        model = cls(input_dim=data["weights"].shape[1])
        model.weights = data["weights"].astype(np.float32)
        model.bias = data["bias"].astype(np.float32)
        return model
