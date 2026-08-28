"""Replay-based Q-learning with a target network: DQN and Double DQN.

Used by the M2 replay arm, the M3 replay arm and the whole M4 line.  Which of
the two targets is computed is decided by ``config.algorithm``:

``q_learning``  target = r + gamma^n * max_a' Q_target(s', a')
``double_dqn``  target = r + gamma^n * Q_target(s', argmax_a' Q_online(s', a'))

Both mask illegal next actions.  Masking is not cosmetic: an unmasked max can
bootstrap from an action the agent could never take, which inflates the target
exactly where the state is most constrained.
"""

from __future__ import annotations

import numpy as np

from ..config import ACTIONS, ExperimentConfig
from ..models.base import QModel
from ..replay import ReplayBuffer
from ..state import quantised_board_spec, state_dimension
from ..symmetry import D4_ORDER, transform_action_indices, transform_board_states, transform_legal_masks
from .base import Transition


class ReplayQLearner:
    """Uniform replay plus a periodically synchronised target network."""

    def __init__(self, config: ExperimentConfig, model: QModel, *, seed: int = 0):
        if config.replay is None:
            raise ValueError("ReplayQLearner requires a declared replay configuration.")
        for capability in ("q_values_batch", "fit_batch", "clone", "copy_parameters_from"):
            if not hasattr(model, capability):
                raise TypeError(f"ReplayQLearner requires a QModel implementing {capability}.")
        self.config = config
        self.settings = config.replay
        self.model = model
        self.target_model = model.clone()
        board_size, quantisation = quantised_board_spec(config.state_encoder)
        self.buffer = ReplayBuffer(
            self.settings.capacity,
            state_dimension(config.state_encoder),
            len(ACTIONS),
            seed=seed,
            quantised_board=board_size,
            quantisation=quantisation,
            sampling=self.settings.sampling,
            priority_exponent=self.settings.priority_exponent,
        )
        self.rng = np.random.default_rng(seed)
        self.observed_transitions = 0
        self.gradient_steps = 0
        self.last_step: dict = {"gradient_applied": False, "gradient_steps": 0, "replay_size": 0}

    def select_action(self, state: np.ndarray, legal_mask: np.ndarray, epsilon: float, generator: np.random.Generator) -> int:
        legal_indices = np.flatnonzero(legal_mask)
        if len(legal_indices) == 0:
            raise ValueError("No legal action was available.")
        if generator.random() < epsilon:
            return int(generator.choice(legal_indices))
        return int(np.argmax(np.where(legal_mask, self.model.q_values(state), -np.inf)))

    def step_diagnostics(self) -> dict:
        """Report what the most recent ``observe`` did, not what it returned.

        Most observes here do not update anything: the buffer is below
        ``min_size``, or the step is not a multiple of ``train_every``.  Those
        return a TD error of ``0.0``, which reads exactly like a converged one.
        ``gradient_applied`` is the field that separates them, and
        ``replay_size`` is what says which of the two reasons applied.
        """
        return dict(self.last_step)

    def observe(self, transition: Transition) -> float:
        self.buffer.append(
            transition.state,
            transition.action_index,
            transition.reward,
            transition.next_state,
            transition.next_legal_mask,
            transition.terminal,
            self.config.discount ** transition.n_step,
        )
        self.observed_transitions += 1
        if len(self.buffer) < self.settings.min_size or self.observed_transitions % self.settings.train_every:
            self.last_step = {
                "gradient_applied": False,
                "gradient_steps": self.gradient_steps,
                "replay_size": len(self.buffer),
            }
            return 0.0
        beta = self._importance_sampling_exponent()
        batch = self.buffer.sample(self.settings.batch_size, beta=beta)
        if self.settings.augmentation == "d4":
            batch = self._augment(batch)
        targets = self._targets(batch)
        td_errors = self.model.fit_batch(
            batch["states"], batch["action_indices"], targets, weights=batch["weights"])
        self.buffer.update_priorities(batch["indices"], td_errors)
        self.gradient_steps += 1
        synchronised = self.gradient_steps % self.settings.target_update_every == 0
        if synchronised:
            self.target_model.copy_parameters_from(self.model)
        self.last_step = {
            "gradient_applied": True,
            "gradient_steps": self.gradient_steps,
            "replay_size": len(self.buffer),
            "target_synchronised": synchronised,
            "mean_abs_td_error": float(np.abs(td_errors).mean()),
            "mean_target": float(np.mean(targets)),
        }
        if self.settings.sampling == "prioritized":
            # Recorded so a finished run can show the prioritization was live
            # and how far beta had annealed, rather than being taken on trust.
            self.last_step["importance_sampling_exponent"] = beta
            self.last_step["mean_importance_weight"] = float(np.mean(batch["weights"]))
            self.last_step["min_importance_weight"] = float(np.min(batch["weights"]))
        # Q statistics and the gradient norm come from the batch the model has
        # just put through the network.  Recomputing them here would cost
        # another forward pass -- about a quarter of the step -- for numbers the
        # model already holds.
        if hasattr(self.model, "training_diagnostics"):
            self.last_step.update(self.model.training_diagnostics())
        return float(np.abs(td_errors).mean())

    def _importance_sampling_exponent(self) -> float:
        """Anneal beta from its declared start to 1.0 over the declared steps.

        Beta has to reach 1 for the correction to be unbiased, and it starts
        below 1 because the bias matters least early, when the value function is
        wrong anyway and the benefit of replaying surprising transitions is
        largest.  This is the schedule in Schaul et al. 2016 section 3.4.
        """
        if self.settings.sampling != "prioritized":
            return 1.0
        start = self.settings.importance_sampling_start
        progress = min(1.0, self.gradient_steps / self.settings.importance_sampling_steps)
        return start + progress * (1.0 - start)

    def end_round(self) -> None:
        """The buffer and the target network deliberately survive a round."""

    def _targets(self, batch: dict[str, np.ndarray]) -> np.ndarray:
        """Return the bootstrapped regression targets for one sampled batch."""
        masks = batch["next_legal_masks"]
        target_q = np.where(masks, self.target_model.q_values_batch(batch["next_states"]), -np.inf)
        if self.config.algorithm == "double_dqn":
            online_q = np.where(masks, self.model.q_values_batch(batch["next_states"]), -np.inf)
            chosen = np.argmax(online_q, axis=1)
            next_value = target_q[np.arange(len(chosen)), chosen]
        elif self.config.algorithm == "q_learning":
            next_value = np.max(target_q, axis=1)
        else:
            raise NotImplementedError(f"ReplayQLearner does not implement algorithm {self.config.algorithm!r}.")
        # A terminal row stores no next state; its bootstrap term is dropped
        # here rather than relying on whatever the unused row happens to hold.
        continues = ~batch["terminals"]
        next_value = np.where(continues, next_value, 0.0)
        return batch["rewards"] + batch["discounts"] * next_value

    def _augment(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Relabel each sampled transition under a random board symmetry.

        Bomberman's dynamics commute with the eight symmetries of the square, so
        a rotated board with correspondingly rotated actions is a real, legal
        transition rather than synthetic noise.  Only an agent-centred
        representation qualifies; ``config.validate_config`` enforces that.
        """
        augmented = {key: value.copy() for key, value in batch.items()}
        transforms = self.rng.integers(0, D4_ORDER, size=len(augmented["action_indices"]))
        for transform in range(1, D4_ORDER):
            rows = np.flatnonzero(transforms == transform)
            if rows.size == 0:
                continue
            augmented["states"][rows] = transform_board_states(augmented["states"][rows], transform)
            augmented["next_states"][rows] = transform_board_states(augmented["next_states"][rows], transform)
            augmented["action_indices"][rows] = transform_action_indices(augmented["action_indices"][rows], transform)
            augmented["next_legal_masks"][rows] = transform_legal_masks(augmented["next_legal_masks"][rows], transform)
        return augmented
