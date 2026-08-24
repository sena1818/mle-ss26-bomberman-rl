"""R01's dependency-free linear QModel adapter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import ACTIONS


class LinearQModel:
    """A multi-head linear approximation Q(s, ·) = W @ features + b."""

    def __init__(self, input_dim: int, seed: int = 0):
        generator = np.random.default_rng(seed)
        self.weights = generator.normal(0.0, 0.01, size=(len(ACTIONS), input_dim)).astype(np.float32)
        self.bias = np.zeros(len(ACTIONS), dtype=np.float32)

    def q_values(self, state: np.ndarray) -> np.ndarray:
        return self.weights @ state + self.bias

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
