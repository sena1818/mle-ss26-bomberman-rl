"""The small dependency-free MLP six-head Q model used by the M3 line."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import ACTIONS
from .base import validate_training_declarations


class MLPQModel:
    """A ReLU MLP Q model with explicit, route-declared optimization.

    The implementation intentionally relies only on NumPy, which is already a
    dependency of the official project.  Hidden layers use He initialization;
    the final Q head starts close to zero like R01's linear head.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: tuple[int, ...],
        seed: int = 0,
        learning_rate: float = 0.02,
        *,
        optimizer: str = "sgd",
        td_loss: str = "mse",
        gradient_clip_norm: float | None = None,
        output_dim: int | None = None,
        noisy: bool = False,
        noisy_sigma: float = 0.5,
        weight_decay: float = 0.0,
    ):
        if input_dim < 1 or not hidden_layers or any(width < 1 for width in hidden_layers):
            raise ValueError("MLPQModel requires a positive input dimension and non-empty positive hidden layers.")
        # ``output_dim`` exists for the distributional head, which predicts
        # ``actions * atoms`` numbers rather than one per action.  Everything
        # below the head -- initialization, backprop, Adam, clipping -- is the
        # same arithmetic, so the subclass widens the head instead of copying it.
        width = len(ACTIONS) if output_dim is None else int(output_dim)
        self.layer_sizes = (int(input_dim), *(int(width_) for width_ in hidden_layers), width)
        generator = np.random.default_rng(seed)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for layer, (fan_in, fan_out) in enumerate(zip(self.layer_sizes, self.layer_sizes[1:])):
            scale = np.sqrt(2.0 / fan_in) if layer < len(self.layer_sizes) - 2 else 0.01
            self.weights.append(generator.normal(0.0, scale, size=(fan_out, fan_in)).astype(np.float32))
            self.biases.append(np.zeros(fan_out, dtype=np.float32))
        validate_training_declarations(optimizer, td_loss, gradient_clip_norm)
        self.learning_rate = float(learning_rate)
        self.optimizer = optimizer
        self.td_loss = td_loss
        self.gradient_clip_norm = gradient_clip_norm
        # Noisy networks (Fortunato et al. 2018): every weight gains a learned
        # noise scale, and the network decides for itself where and how much to
        # explore.  It replaces epsilon-greedy rather than adding to it.
        #
        # One deliberate deviation from the paper: the noise is a *training*
        # mechanism here and evaluation uses the mean weights alone.  Every
        # other exploration mechanism on this line is switched off at evaluation
        # (epsilon is set to 0), the reported numbers are greedy by definition,
        # and a stochastic evaluation would also cost the bit-exact determinism
        # the solo suite relies on (docs/01 section 7.14.1).
        self.noisy = bool(noisy)
        self.noise_enabled = self.noisy
        self._noise_generator = np.random.default_rng(seed + 977 if self.noisy else 0)
        if self.noisy:
            # sigma_0 / sqrt(fan_in), the factorised initialisation of the paper.
            self.weight_sigmas = [
                np.full_like(weight, noisy_sigma / np.sqrt(weight.shape[1]))
                for weight in self.weights]
            self.bias_sigmas = [
                np.full_like(bias, noisy_sigma / np.sqrt(fan_in))
                for bias, fan_in in zip(self.biases, self.layer_sizes[:-1])]
        else:
            self.weight_sigmas, self.bias_sigmas = [], []
        self._weight_noise: list[np.ndarray] = []
        self._bias_noise: list[np.ndarray] = []
        self.noisy_sigma = float(noisy_sigma)
        # Decoupled, in the sense of Loshchilov & Hutter 2019 (AdamW): the decay
        # is applied to the parameter, not added to the gradient.  With Adam the
        # two are not the same thing -- an L2 term added to the gradient is
        # divided by the same running RMS as everything else, so the effective
        # decay ends up inversely proportional to the gradient magnitude of each
        # weight, which is not the penalty anybody intended.  This line of work
        # already learned that lesson once, in section 7.38.1, where a loss
        # rescaling was read as a step-size change and Adam's scale invariance
        # made it nothing at all.
        #
        # Biases and the noise scales are excluded.  Decaying a bias shifts the
        # function without constraining its complexity, and decaying sigma is a
        # second, undeclared annealing of the exploration schedule.
        self.weight_decay = float(weight_decay)
        # Applied in accumulated bursts, not every step, and that is a
        # correctness requirement rather than an optimisation.  The per-step
        # factor is ``1 - learning_rate * weight_decay`` = 1 - 5e-8 for this
        # recipe; the weights are float32 with a magnitude around 0.1, so one
        # step moves a weight by 5e-9 against an ulp of 7.5e-9 and rounds
        # straight back to where it started.  Multiplying every step would have
        # been exactly a no-op -- a declared factor that never reaches the
        # model, which is what section 7.42 is about.  The factor is therefore
        # compounded here and applied once it is far enough from 1 to survive
        # the rounding, which is first-order identical and, unlike the naive
        # version, actually happens.
        self._pending_decay = 1.0
        self._adam_step = 0
        self._weight_momentum = [np.zeros_like(weight) for weight in self.weights]
        self._weight_variance = [np.zeros_like(weight) for weight in self.weights]
        self._bias_momentum = [np.zeros_like(bias) for bias in self.biases]
        self._bias_variance = [np.zeros_like(bias) for bias in self.biases]
        self._weight_sigma_momentum = [np.zeros_like(one) for one in self.weight_sigmas]
        self._weight_sigma_variance = [np.zeros_like(one) for one in self.weight_sigmas]
        self._bias_sigma_momentum = [np.zeros_like(one) for one in self.bias_sigmas]
        self._bias_sigma_variance = [np.zeros_like(one) for one in self.bias_sigmas]
        self.last_gradient_l2_norm: float | None = None
        self.last_gradient_was_clipped = False
        self.last_hidden_zero_fraction: float | None = None
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
        """Apply one declared optimizer step using the masked Q-learning target."""
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
        delta[action_index] = self._loss_gradient(np.asarray([td_error], dtype=np.float32))[0]
        weight_gradients: list[np.ndarray] = [np.empty_like(weight) for weight in self.weights]
        bias_gradients: list[np.ndarray] = [np.empty_like(bias) for bias in self.biases]
        for layer in range(len(self.weights) - 1, -1, -1):
            weight_gradients[layer] = np.outer(delta, activations[layer]).astype(np.float32)
            bias_gradients[layer] = delta
            if layer:
                delta = (self.weights[layer].T @ delta) * (pre_activations[layer - 1] > 0.0)
        self._apply_gradients(weight_gradients, bias_gradients, learning_rate)
        return td_error

    def q_values_batch(self, states: np.ndarray) -> np.ndarray:
        """Return one row of six Q-values per state in the batch."""
        return self._forward_batch(np.asarray(states, dtype=np.float32))[0]

    def fit_batch(self, states: np.ndarray, action_indices: np.ndarray, targets: np.ndarray,
                  weights: np.ndarray | None = None) -> np.ndarray:
        """One SGD step on the mean TD loss of the selected heads.

        ``weights`` are the per-sample importance-sampling weights prioritized
        replay needs.  ``None`` means uniform and takes the identical arithmetic
        path as before, so every arm run without prioritization is unchanged.
        The returned TD errors are the *unweighted* ones: they are what the
        buffer prioritizes by, and weighting them there would compound the
        correction with the thing it corrects.
        """
        states = np.asarray(states, dtype=np.float32)
        action_indices = np.asarray(action_indices, dtype=np.intp)
        predictions, activations, pre_activations = self._forward_batch(states)
        rows = np.arange(len(action_indices))
        td_errors = np.asarray(targets, dtype=np.float32) - predictions[rows, action_indices]

        gradient = self._loss_gradient(td_errors)
        if weights is not None:
            gradient = gradient * np.asarray(weights, dtype=np.float32)
        delta = np.zeros_like(predictions)
        delta[rows, action_indices] = gradient / len(action_indices)
        self.last_hidden_zero_fraction = self._hidden_zero_fraction(pre_activations)
        self._backpropagate(delta, activations, pre_activations)
        return td_errors

    def _backpropagate(self, delta: np.ndarray, activations, pre_activations) -> None:
        """One backward pass and one optimizer step, noisy or not.

        With noise on, the layer's Jacobian is the *effective* weight
        ``mu + sigma * epsilon``, not ``mu`` -- using ``mu`` would send the wrong
        signal to every layer below.  The gradient with respect to a noise scale
        is the gradient with respect to that effective weight times the noise
        that was drawn, which is why the sample has to survive from the forward
        pass to here rather than being redrawn.
        """
        weights, _, noisy = self._effective_parameters()
        weight_gradients: list[np.ndarray] = [np.empty_like(weight) for weight in self.weights]
        bias_gradients: list[np.ndarray] = [np.empty_like(bias) for bias in self.biases]
        for layer in range(len(self.weights) - 1, -1, -1):
            weight_gradients[layer] = delta.T @ activations[layer]
            bias_gradients[layer] = delta.sum(axis=0)
            if layer:
                delta = (delta @ weights[layer]) * (pre_activations[layer - 1] > 0.0)
        if noisy:
            sigma_weight_gradients = [gradient * noise for gradient, noise
                                      in zip(weight_gradients, self._weight_noise)]
            sigma_bias_gradients = [gradient * noise for gradient, noise
                                    in zip(bias_gradients, self._bias_noise)]
        else:
            sigma_weight_gradients, sigma_bias_gradients = [], []
        self._apply_gradients(weight_gradients, bias_gradients, self.learning_rate,
                              sigma_weight_gradients, sigma_bias_gradients)

    def clone(self) -> "MLPQModel":
        copy = MLPQModel(
            self.layer_sizes[0], self.layer_sizes[1:-1], learning_rate=self.learning_rate,
            optimizer=self.optimizer, td_loss=self.td_loss, gradient_clip_norm=self.gradient_clip_norm,
            noisy=self.noisy, noisy_sigma=self.noisy_sigma, weight_decay=self.weight_decay,
        )
        copy.copy_parameters_from(self)
        return copy

    def copy_parameters_from(self, other: "MLPQModel") -> None:
        if other.layer_sizes != self.layer_sizes:
            raise ValueError(f"Cannot copy parameters between MLP shapes {other.layer_sizes} and {self.layer_sizes}.")
        if other.noisy != self.noisy:
            raise ValueError("Cannot copy parameters between a noisy and a plain MLP.")
        self.weights = [weight.copy() for weight in other.weights]
        self.biases = [bias.copy() for bias in other.biases]
        self.weight_sigmas = [one.copy() for one in other.weight_sigmas]
        self.bias_sigmas = [one.copy() for one in other.bias_sigmas]

    def _sample_noise(self) -> None:
        """Draw one factorised noise sample per layer (Fortunato et al., section 3.2).

        Factorised rather than independent: a layer needs ``fan_in * fan_out``
        noise values, and drawing them as the outer product of two vectors costs
        ``fan_in + fan_out`` draws instead.  ``f(x) = sign(x) * sqrt(|x|)`` is the
        transform the paper uses to keep the resulting products well scaled.
        """
        def transform(size: int) -> np.ndarray:
            raw = self._noise_generator.standard_normal(size).astype(np.float32)
            return np.sign(raw) * np.sqrt(np.abs(raw))

        self._weight_noise, self._bias_noise = [], []
        for weight in self.weights:
            outputs, inputs = weight.shape
            epsilon_in, epsilon_out = transform(inputs), transform(outputs)
            self._weight_noise.append(np.outer(epsilon_out, epsilon_in))
            self._bias_noise.append(epsilon_out)

    def _effective_parameters(self):
        """The weights of the most recent forward pass, and whether they are noisy.

        This never draws: the backward pass has to use exactly the noise the
        forward pass used, so sampling belongs at the entry to a forward and
        nowhere else.  Drawing here would give backprop a different network from
        the one whose output it is correcting.
        """
        if not (self.noisy and self.noise_enabled):
            return self.weights, self.biases, False
        if not self._weight_noise:
            self._sample_noise()
        weights = [mu + sigma * noise for mu, sigma, noise
                   in zip(self.weights, self.weight_sigmas, self._weight_noise)]
        biases = [mu + sigma * noise for mu, sigma, noise
                  in zip(self.biases, self.bias_sigmas, self._bias_noise)]
        return weights, biases, True

    def _forward_batch(self, states: np.ndarray):
        """Batched forward pass returning the per-layer cache used by backprop."""
        if states.ndim != 2 or states.shape[1] != self.layer_sizes[0]:
            raise ValueError(f"Expected a batch of states with width {self.layer_sizes[0]}, received {states.shape}.")
        activation = states
        activations = [activation]
        pre_activations: list[np.ndarray] = []
        # A fresh draw per forward pass: the noise IS the exploration, so reusing
        # one sample would make the policy deterministic between gradient steps
        # and noisy nets would explore nothing at all.
        if self.noisy and self.noise_enabled:
            self._sample_noise()
        weights, biases, _ = self._effective_parameters()
        for layer, (weight, bias) in enumerate(zip(weights, biases)):
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
        if self.noisy:
            payload["noisy"] = np.asarray(True)
            for layer, (weight, bias) in enumerate(zip(self.weight_sigmas, self.bias_sigmas)):
                payload[f"weight_sigmas_{layer}"] = weight
                payload[f"bias_sigmas_{layer}"] = bias
        np.savez(path, **payload)

    def training_diagnostics(self) -> dict[str, float | bool | int | None]:
        """Return small, JSON-safe signals for diagnosing fitted-Q stability."""
        parameter_sq = sum(float(np.sum(parameter.astype(np.float64) ** 2)) for parameter in self.weights + self.biases)
        noise_scale = None
        if self.noisy:
            # The one number that says whether the network still explores: sigma
            # collapsing to zero is noisy nets turning itself off, and the arm
            # would then be greedy training with no exploration at all.
            noise_scale = float(np.mean([np.mean(np.abs(one)) for one in self.weight_sigmas]))
        return {
            "mean_noise_scale": noise_scale,
            "optimizer_steps": self._adam_step if self.optimizer == "adam" else 0,
            "parameter_l2_norm": float(np.sqrt(parameter_sq)),
            "last_gradient_l2_norm": self.last_gradient_l2_norm,
            "last_gradient_was_clipped": self.last_gradient_was_clipped,
            "last_hidden_zero_fraction": self.last_hidden_zero_fraction,
        }

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        learning_rate: float = 0.02,
        optimizer: str = "sgd",
        td_loss: str = "mse",
        gradient_clip_norm: float | None = None,
        weight_decay: float = 0.0,
    ) -> "MLPQModel":
        """Load and validate a self-describing R02 checkpoint."""
        with np.load(path, allow_pickle=False) as data:
            required = {"model_type", "layer_sizes", "metadata"}
            if not required.issubset(data.files) or str(data["model_type"].item()) != "mlp_q":
                raise ValueError(f"Checkpoint {path} is not an MLP Q model.")
            layer_sizes = tuple(int(size) for size in data["layer_sizes"])
            if len(layer_sizes) < 3 or layer_sizes[0] < 1 or any(size < 1 for size in layer_sizes[1:]) or layer_sizes[-1] != len(ACTIONS):
                raise ValueError(f"Checkpoint {path} has an invalid MLP shape {layer_sizes}.")
            model = cls(
                layer_sizes[0], layer_sizes[1:-1], learning_rate=learning_rate,
                optimizer=optimizer, td_loss=td_loss, gradient_clip_norm=gradient_clip_norm,
                weight_decay=weight_decay,
                noisy=bool(data["noisy"].item()) if "noisy" in data.files else False,
            )
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
            if model.noisy:
                model.weight_sigmas = [data[f"weight_sigmas_{layer}"].astype(np.float32)
                                       for layer in range(len(weights))]
                model.bias_sigmas = [data[f"bias_sigmas_{layer}"].astype(np.float32)
                                     for layer in range(len(biases))]
                model._weight_sigma_momentum = [np.zeros_like(one) for one in model.weight_sigmas]
                model._weight_sigma_variance = [np.zeros_like(one) for one in model.weight_sigmas]
                model._bias_sigma_momentum = [np.zeros_like(one) for one in model.bias_sigmas]
                model._bias_sigma_variance = [np.zeros_like(one) for one in model.bias_sigmas]
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
        if self.noisy and self.noise_enabled:
            self._sample_noise()
        weights, biases, _ = self._effective_parameters()
        for layer, (weight, bias) in enumerate(zip(weights, biases)):
            linear = weight @ activation + bias
            pre_activations.append(linear)
            activation = linear if layer == len(self.weights) - 1 else np.maximum(linear, 0.0)
            activations.append(activation)
        if retain_cache:
            self.last_hidden_zero_fraction = self._hidden_zero_fraction(pre_activations)
            return activation, activations, pre_activations
        return activation

    def _loss_gradient(self, td_errors: np.ndarray) -> np.ndarray:
        """Return the update direction for MSE or unit-delta Huber loss."""
        if self.td_loss == "mse":
            return td_errors
        return np.clip(td_errors, -1.0, 1.0)

    @staticmethod
    def _hidden_zero_fraction(pre_activations: list[np.ndarray]) -> float | None:
        hidden = pre_activations[:-1]
        if not hidden:
            return None
        return float(np.mean(np.concatenate([layer.reshape(-1) for layer in hidden]) <= 0.0))

    def _apply_gradients(
        self,
        weight_gradients: list[np.ndarray],
        bias_gradients: list[np.ndarray],
        learning_rate: float,
        sigma_weight_gradients: list[np.ndarray] | None = None,
        sigma_bias_gradients: list[np.ndarray] | None = None,
    ) -> None:
        sigma_weight_gradients = sigma_weight_gradients or []
        sigma_bias_gradients = sigma_bias_gradients or []
        # The noise scales are parameters like any other, so they join the clip
        # norm and the optimizer state rather than getting their own rules.
        gradients = (weight_gradients + bias_gradients
                     + sigma_weight_gradients + sigma_bias_gradients)
        squared_norm = sum(float(np.sum(gradient.astype(np.float64) ** 2)) for gradient in gradients)
        gradient_norm = float(np.sqrt(squared_norm))
        self.last_gradient_l2_norm = gradient_norm
        self.last_gradient_was_clipped = False
        if self.gradient_clip_norm is not None and gradient_norm > self.gradient_clip_norm:
            scale = self.gradient_clip_norm / gradient_norm
            weight_gradients = [gradient * scale for gradient in weight_gradients]
            bias_gradients = [gradient * scale for gradient in bias_gradients]
            sigma_weight_gradients = [gradient * scale for gradient in sigma_weight_gradients]
            sigma_bias_gradients = [gradient * scale for gradient in sigma_bias_gradients]
            self.last_gradient_was_clipped = True

        if self.optimizer == "sgd":
            for layer, (weight_gradient, bias_gradient) in enumerate(zip(weight_gradients, bias_gradients)):
                self.weights[layer] += (learning_rate * weight_gradient).astype(np.float32)
                self.biases[layer] += (learning_rate * bias_gradient).astype(np.float32)
            for layer, (weight_gradient, bias_gradient) in enumerate(
                    zip(sigma_weight_gradients, sigma_bias_gradients)):
                self.weight_sigmas[layer] += (learning_rate * weight_gradient).astype(np.float32)
                self.bias_sigmas[layer] += (learning_rate * bias_gradient).astype(np.float32)
            return

        if self.weight_decay:
            self._pending_decay *= 1.0 - learning_rate * self.weight_decay
            if self._pending_decay < 1.0 - 1e-5:
                for layer, weight in enumerate(self.weights):
                    self.weights[layer] = (weight * self._pending_decay).astype(np.float32)
                self._pending_decay = 1.0
        self._adam_step += 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        correction1 = 1.0 - beta1 ** self._adam_step
        correction2 = 1.0 - beta2 ** self._adam_step
        for parameters, gradients_for_parameters, momentum, variance in (
            (self.weights, weight_gradients, self._weight_momentum, self._weight_variance),
            (self.biases, bias_gradients, self._bias_momentum, self._bias_variance),
            (self.weight_sigmas, sigma_weight_gradients,
             self._weight_sigma_momentum, self._weight_sigma_variance),
            (self.bias_sigmas, sigma_bias_gradients,
             self._bias_sigma_momentum, self._bias_sigma_variance),
        ):
            for parameter, gradient, first, second in zip(parameters, gradients_for_parameters, momentum, variance):
                first *= beta1
                first += (1.0 - beta1) * gradient
                second *= beta2
                second += (1.0 - beta2) * (gradient * gradient)
                parameter += (learning_rate * (first / correction1) / (np.sqrt(second / correction2) + epsilon)).astype(np.float32)
