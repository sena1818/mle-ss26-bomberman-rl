"""C51: an MLP that predicts the return *distribution* instead of its mean.

Bellemare et al. 2017.  A scalar Q head answers "what is the average return of
this action"; a categorical head answers "with what probability does it return
each of these 51 values".  The training signal is correspondingly richer -- a
cross-entropy against a full projected target rather than one squared error --
and the greedy policy is unchanged, because the action is still chosen by the
expectation of that distribution.

Everything below the head is the shared ``MLPQModel``: the same He
initialization, backprop loop, Adam and gradient clipping.  Only three things
differ, and they are the definition of C51:

* the head is ``len(ACTIONS) * atoms`` wide, softmaxed per action;
* the target is a distribution projected back onto the fixed support, because
  ``r + gamma^n * z`` generally falls between two atoms;
* the loss is cross-entropy, whose gradient with respect to the logits is
  exactly ``p - target`` and therefore needs no special backward pass.

The support is fixed and declared.  It has to be: an atom grid that does not
cover the returns the environment produces silently clips every target that
falls outside it.  ``value_min``/``value_max`` are route fields for that reason,
and docs/01 section 7.33 records the measurement they were chosen from.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import ACTIONS
from .mlp_q import MLPQModel


class CategoricalMLPQModel(MLPQModel):
    """A distributional Q model over a fixed, declared value support."""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: tuple[int, ...],
        seed: int = 0,
        learning_rate: float = 0.02,
        *,
        atoms: int = 51,
        value_min: float = -2.0,
        value_max: float = 12.0,
        optimizer: str = "adam",
        td_loss: str = "cross_entropy",
        gradient_clip_norm: float | None = None,
    ):
        if atoms < 2:
            raise ValueError("A categorical head needs at least two atoms.")
        if not value_max > value_min:
            raise ValueError("value_max must be greater than value_min.")
        if td_loss != "cross_entropy":
            raise ValueError(
                f"A categorical head is trained by cross-entropy, not {td_loss!r}; "
                "declaring anything else would make the run snapshot say the wrong thing.")
        super().__init__(
            input_dim, hidden_layers, seed=seed, learning_rate=learning_rate,
            optimizer=optimizer, td_loss=td_loss, gradient_clip_norm=gradient_clip_norm,
            output_dim=len(ACTIONS) * int(atoms),
        )
        self.atoms = int(atoms)
        self.value_min = float(value_min)
        self.value_max = float(value_max)
        self.support = np.linspace(self.value_min, self.value_max, self.atoms, dtype=np.float32)
        self.delta_z = float((self.value_max - self.value_min) / (self.atoms - 1))

    # ---- inference -------------------------------------------------------

    def distribution_batch(self, states: np.ndarray) -> np.ndarray:
        """Return ``(batch, actions, atoms)`` probabilities."""
        logits, _, _ = self._forward_batch(np.asarray(states, dtype=np.float32))
        return self._softmax(logits.reshape(-1, len(ACTIONS), self.atoms))

    def q_values_batch(self, states: np.ndarray) -> np.ndarray:
        """The expectation of each action's distribution: one row of six."""
        return self.distribution_batch(states) @ self.support

    def q_values(self, state: np.ndarray) -> np.ndarray:
        return self.q_values_batch(np.asarray(state, dtype=np.float32)[None, :])[0]

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exponentials = np.exp(shifted)
        return exponentials / exponentials.sum(axis=-1, keepdims=True)

    # ---- training --------------------------------------------------------

    def fit_batch(self, states, action_indices, targets, weights=None):
        raise NotImplementedError(
            "A categorical head has no scalar regression path; the learner must call "
            "fit_batch_distribution with a projected target distribution.")

    def q_learning_update(self, *args, **kwargs):
        raise NotImplementedError(
            "A categorical head is a batch learner only; online single-transition "
            "updates would need their own projection and are not implemented.")

    def fit_batch_distribution(self, states: np.ndarray, action_indices: np.ndarray,
                               target_probabilities: np.ndarray,
                               weights: np.ndarray | None = None) -> np.ndarray:
        """One step of cross-entropy against the projected target distribution.

        Returns the per-sample cross-entropy, which is what a prioritized buffer
        would rank by and what the learner logs.
        """
        states = np.asarray(states, dtype=np.float32)
        action_indices = np.asarray(action_indices, dtype=np.intp)
        target_probabilities = np.asarray(target_probabilities, dtype=np.float32)
        if target_probabilities.shape != (len(action_indices), self.atoms):
            raise ValueError(
                f"Expected target probabilities of shape {(len(action_indices), self.atoms)}, "
                f"received {target_probabilities.shape}.")
        logits, activations, pre_activations = self._forward_batch(states)
        probabilities = self._softmax(logits.reshape(-1, len(ACTIONS), self.atoms))
        rows = np.arange(len(action_indices))
        selected = probabilities[rows, action_indices]
        losses = -np.sum(target_probabilities * np.log(np.clip(selected, 1e-8, None)), axis=1)

        # Softmax composed with cross-entropy: dL/dlogits is exactly
        # (p - target) for the selected action and zero for every other.
        #
        # The sign is (target - p), not (p - target), because ``_apply_gradients``
        # *ascends*: the scalar path hands it (target - Q) and it adds. Passing
        # the descent direction here would train the head away from its target,
        # which is exactly what the first version did -- the probability mass
        # walked to the far end of the support and the cross-entropy pinned at
        # -log(1e-8). ``test_a_delta_target_is_learned`` is that failure frozen.
        head = np.zeros_like(probabilities)
        head[rows, action_indices] = target_probabilities - selected
        if weights is not None:
            head *= np.asarray(weights, dtype=np.float32)[:, None, None]
        delta = head.reshape(len(action_indices), -1) / len(action_indices)

        self.last_hidden_zero_fraction = self._hidden_zero_fraction(pre_activations)
        weight_gradients: list[np.ndarray] = [np.empty_like(weight) for weight in self.weights]
        bias_gradients: list[np.ndarray] = [np.empty_like(bias) for bias in self.biases]
        for layer in range(len(self.weights) - 1, -1, -1):
            weight_gradients[layer] = delta.T @ activations[layer]
            bias_gradients[layer] = delta.sum(axis=0)
            if layer:
                delta = (delta @ self.weights[layer]) * (pre_activations[layer - 1] > 0.0)
        self._apply_gradients(weight_gradients, bias_gradients, self.learning_rate)
        return losses.astype(np.float32)

    def project_targets(self, rewards: np.ndarray, discounts: np.ndarray, terminals: np.ndarray,
                        next_probabilities: np.ndarray) -> np.ndarray:
        """Project ``r + gamma^n * z`` back onto the fixed support.

        ``r + gamma^n * z`` almost never lands on an atom, so its mass is split
        between the two neighbours in proportion to how close it is to each.
        This is the categorical Bellman operator of Bellemare et al. 2017,
        algorithm 1.

        A terminal transition collapses every atom of ``Tz`` onto ``r``, so all
        of ``next_probabilities`` -- whatever it holds for the unused next state
        -- lands on the same one or two atoms and sums to one there.  That makes
        the terminal case fall out of the same arithmetic instead of needing a
        branch that could disagree with it.
        """
        rewards = np.asarray(rewards, dtype=np.float64)[:, None]
        discounts = np.asarray(discounts, dtype=np.float64)[:, None]
        terminals = np.asarray(terminals, dtype=bool)[:, None]
        next_probabilities = np.asarray(next_probabilities, dtype=np.float64)
        batch = next_probabilities.shape[0]

        projected_values = np.where(terminals, rewards, rewards + discounts * self.support[None, :])
        projected_values = np.clip(projected_values, self.value_min, self.value_max)
        position = (projected_values - self.value_min) / self.delta_z
        lower = np.floor(position).astype(np.intp)
        upper = np.ceil(position).astype(np.intp)
        rows = np.repeat(np.arange(batch, dtype=np.intp)[:, None], self.atoms, axis=1)

        target = np.zeros((batch, self.atoms), dtype=np.float64)
        np.add.at(target, (rows, lower), next_probabilities * (upper - position))
        np.add.at(target, (rows, upper), next_probabilities * (position - lower))
        # A value that lands exactly on an atom gives both terms zero weight, so
        # its mass would vanish; it belongs entirely to that atom.
        exact = lower == upper
        if exact.any():
            np.add.at(target, (rows[exact], lower[exact]), next_probabilities[exact])
        return target.astype(np.float32)

    # ---- persistence -----------------------------------------------------

    def clone(self) -> "CategoricalMLPQModel":
        copy = CategoricalMLPQModel(
            self.layer_sizes[0], self.layer_sizes[1:-1], learning_rate=self.learning_rate,
            atoms=self.atoms, value_min=self.value_min, value_max=self.value_max,
            optimizer=self.optimizer, td_loss=self.td_loss,
            gradient_clip_norm=self.gradient_clip_norm)
        copy.copy_parameters_from(self)
        return copy

    def save(self, path: Path, *, metadata: dict | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata = dict(metadata or {})
        payload: dict[str, np.ndarray] = {
            "model_type": np.asarray("categorical_mlp_q"),
            "layer_sizes": np.asarray(self.layer_sizes, dtype=np.int64),
            "atoms": np.asarray(self.atoms, dtype=np.int64),
            "value_min": np.asarray(self.value_min, dtype=np.float64),
            "value_max": np.asarray(self.value_max, dtype=np.float64),
            "metadata": np.asarray(json.dumps(self.metadata, sort_keys=True)),
        }
        for layer, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            payload[f"weights_{layer}"] = weight
            payload[f"biases_{layer}"] = bias
        np.savez(path, **payload)

    @classmethod
    def load(cls, path: Path, *, learning_rate: float = 0.02, optimizer: str = "adam",
             td_loss: str = "cross_entropy",
             gradient_clip_norm: float | None = None) -> "CategoricalMLPQModel":
        with np.load(path, allow_pickle=False) as data:
            if str(data["model_type"].item()) != "categorical_mlp_q":
                raise ValueError(f"Checkpoint {path} is not a categorical MLP Q model.")
            layer_sizes = tuple(int(size) for size in data["layer_sizes"])
            atoms = int(data["atoms"].item())
            value_min = float(data["value_min"].item())
            value_max = float(data["value_max"].item())
            if layer_sizes[-1] != len(ACTIONS) * atoms:
                raise ValueError(
                    f"Checkpoint {path} has a head of {layer_sizes[-1]} for {atoms} atoms; "
                    f"expected {len(ACTIONS) * atoms}.")
            model = cls(layer_sizes[0], layer_sizes[1:-1], learning_rate=learning_rate,
                        atoms=atoms, value_min=value_min, value_max=value_max,
                        optimizer=optimizer, td_loss=td_loss,
                        gradient_clip_norm=gradient_clip_norm)
            weights, biases = [], []
            for layer, (fan_in, fan_out) in enumerate(zip(layer_sizes, layer_sizes[1:])):
                weight = data[f"weights_{layer}"].astype(np.float32)
                bias = data[f"biases_{layer}"].astype(np.float32)
                if weight.shape != (fan_out, fan_in) or bias.shape != (fan_out,):
                    raise ValueError(f"Checkpoint {path} has invalid parameters for layer {layer}.")
                weights.append(weight)
                biases.append(bias)
            metadata = json.loads(str(data["metadata"].item()))
        model.weights = weights
        model.biases = biases
        model.metadata = metadata if isinstance(metadata, dict) else {}
        return model
