"""Shared ExperimentRuntime: the only route-agnostic callback implementation."""

from __future__ import annotations

import os
from dataclasses import asdict
from time import perf_counter
from typing import Any

import numpy as np

from ..artifacts import append_jsonl, checkpoint_interval, checkpoint_path, latest_model_path, model_path, run_id
from ..config import (
    ACTIONS,
    MAX_STEPS,
    ExperimentConfig,
    epsilon_for_training_round,
    epsilon_for_training_step,
    learning_rate_for_training_round,
    learning_rate_specification,
    exploration_specification,
    shaping_specification,
)
from ..learners import Learner, Transition, build_learner
from ..models import QModel, build_model, load_model
from ..shaping import PotentialShaping, build_shaping
from ..state import encode_state, legal_action_mask, state_dimension
from .transitions import EncodedTransition, NStepAssembler


REWARD_TABLES = {
    # Frozen baseline: only official positive scoring events reach the learner.
    "A00": {
        "COIN_COLLECTED": 1.0,
        "KILLED_OPPONENT": 5.0,
    },
    # Safety ablation: the only change from A00 is a terminal death cost.
    "A01": {
        "COIN_COLLECTED": 1.0,
        "KILLED_OPPONENT": 5.0,
    },
    # Sparse-credit ablation: keep A01 safety, then lightly reward the two
    # official events that connect a safe bomb to a hidden collectible.
    "A02": {
        "COIN_COLLECTED": 1.0,
        "KILLED_OPPONENT": 5.0,
        "CRATE_DESTROYED": 0.1,
        "COIN_FOUND": 0.2,
    },
    # A03 and A05 are the other two levels of the death-penalty dose-response
    # study.  Their event tables are byte-identical to A02 on purpose: the only
    # thing that varies across A02/A03/A05 is DEATH_PENALTIES.
    "A03": {
        "COIN_COLLECTED": 1.0,
        "KILLED_OPPONENT": 5.0,
        "CRATE_DESTROYED": 0.1,
        "COIN_FOUND": 0.2,
    },
    "A05": {
        "COIN_COLLECTED": 1.0,
        "KILLED_OPPONENT": 5.0,
        "CRATE_DESTROYED": 0.1,
        "COIN_FOUND": 0.2,
    },
    # A06 is A03 plus potential-based shaping.  Its event table is identical to
    # A03 on purpose: the shaping term must be the only variable in the
    # comparison, and shaping is not an event weight.
    "A06": {
        "COIN_COLLECTED": 1.0,
        "KILLED_OPPONENT": 5.0,
        "CRATE_DESTROYED": 0.1,
        "COIN_FOUND": 0.2,
    },
    # A07 is A06 plus one shaping term.  Its event table is identical to A06 on
    # purpose, for the same reason A06's is identical to A03's: the shaping term
    # has to be the only variable in the comparison.
    "A07": {
        "COIN_COLLECTED": 1.0,
        "KILLED_OPPONENT": 5.0,
        "CRATE_DESTROYED": 0.1,
        "COIN_FOUND": 0.2,
    },
}
DEATH_PENALTIES = {
    "A00": 0.0,
    # The official framework emits both KILLED_SELF and GOT_KILLED for one
    # self-inflicted death.  This is deliberately one penalty per death, not
    # one penalty per event label.
    "A01": -5.0,
    "A02": -5.0,
    # Dose-response levels.  D = 1.0 comes from inverting the mean-field design
    # model at p_target = 0.55; D = 0.0 is the no-risk-term control arm that
    # separates "signal starvation" from "risk term too large".
    "A03": -1.0,
    "A05": 0.0,
    "A06": -1.0,
    "A07": -1.0,
}
# Frozen, human-readable provenance for every registered reward version.  The
# runner copies this into the run snapshot so a result can always be traced back
# to the exact weights that produced it, without re-reading this source file.
REWARD_DESIGN_NOTES = {
    "A00": "official events only; frozen baseline",
    "A01": "A00 + one death penalty of 5.0 per death",
    "A02": "A01 + CRATE_DESTROYED 0.1 + COIN_FOUND 0.2",
    "A03": "A02 with the death penalty lowered from 5.0 to 1.0; nothing else changes",
    "A05": "A02 with the death penalty removed; control arm of the D dose-response study",
    "A06": "A03 plus potential-based shaping; the event weights are identical to A03",
    "A07": "A06 plus one shaping term for opponents standing in a bomb's blast; the event weights are identical to A06",
}


def reward_specification(reward_version: str) -> dict:
    """Return the full, serializable definition of one reward version.

    This is the single source of truth consumed by the runtime, the run
    snapshot, and the tests, so a registered version can never be described in
    one place and implemented differently in another.
    """
    try:
        return {
            "reward_version": reward_version,
            "event_weights": dict(REWARD_TABLES[reward_version]),
            # Reported as the signed value actually added to the reward.
            "death_penalty": DEATH_PENALTIES[reward_version],
            "death_penalty_events": ["KILLED_SELF", "GOT_KILLED"],
            "death_penalty_applications_per_death": 1,
            # Shaping belongs in the reward specification because it changes the
            # learner's reward.  It is derived from the version, never declared
            # twice, so a config cannot say A03 and train with a shaped reward.
            "shaping": shaping_specification(reward_version),
            "notes": REWARD_DESIGN_NOTES.get(reward_version, ""),
        }
    except KeyError as exc:
        raise ValueError(f"No reward table is registered for {reward_version!r}") from exc


def reward_for_events(reward_version: str, events: list[str]) -> float:
    """Return one versioned training reward without changing official scoring."""
    try:
        rewards = REWARD_TABLES[reward_version]
        death_penalty = DEATH_PENALTIES[reward_version]
    except KeyError as exc:
        raise ValueError(f"No reward table is registered for {reward_version!r}") from exc
    reward = sum(rewards.get(event, 0.0) for event in events)
    if death_penalty and {"KILLED_SELF", "GOT_KILLED"}.intersection(events):
        reward += death_penalty
    return reward


def _is_unusable_action(action: str | None) -> bool:
    """Report an action the framework substituted rather than the agent chose.

    ``Agent.last_action`` is ``None`` before the first act, and the world
    records ``"ERROR"`` for a silenced agent exception.  Neither has a
    six-action index, so neither may become a learning target.
    """
    return action not in ACTIONS


ROUND_CONTINUES = None
TASK_COMPLETE = "task_complete"
TRUNCATION = "truncation"


def round_end_reason(new_game_state: dict | None) -> str | None:
    """Return why the round ends after the step that produced this state.

    ``None`` means it continues.  This mirrors ``environment.time_to_stop`` in
    the order that function checks its conditions, so that terminality is known
    at delivery time and no step has to be held back.

    The two reasons are not interchangeable and must not share one switch:

    * ``TASK_COMPLETE`` -- nothing collectable or destructible is left and no
      opponent is alive.  The remaining true return is zero, so this is a real
      terminal state and its target is ``r`` regardless of configuration.
    * ``TRUNCATION`` -- the step limit was reached while the world was still
      going.  The MDP has not ended, so the target bootstraps unless
      ``terminal_on_truncation`` says otherwise (Pardo et al. 2018).

    One approximation is unavoidable and is therefore measured rather than
    assumed.  ``time_to_stop`` requires ``len(self.explosions) == 0``, but an
    explosion stays in that list for two further steps as harmless smoke, and
    ``explosion_map`` only exposes its dangerous stage.  The condition below is
    therefore *necessary* but can fire up to two steps early.  ``ExperimentRuntime``
    counts every disagreement between this prediction and what the framework
    actually did and writes the count into ``round_end``; see
    ``round_end_mispredictions``.
    """
    if new_game_state is None:
        return ROUND_CONTINUES
    if (not new_game_state["others"]
            and not (new_game_state["field"] == 1).any()
            and not new_game_state["coins"]
            and not new_game_state["bombs"]
            and not new_game_state["explosion_map"].any()):
        return TASK_COMPLETE
    if int(new_game_state["step"]) >= MAX_STEPS:
        return TRUNCATION
    return ROUND_CONTINUES


class ExperimentRuntime:
    """Deep module hiding route-specific model and learner adapters.

    Its interface is intentionally limited to the official lifecycle:
    ``select_action``, ``observe``, and ``end_round``.  Algorithm-specific
    state never reaches callbacks: the replay learner owns its buffer and
    target network, and the n-step window lives in ``runtime.transitions``.

    All four main lines of docs/05 run through this one class.  What differs
    between them -- the state encoder, the Q-model, the update rule, potential
    shaping, the bootstrap length and whether updates come from a buffer -- is
    resolved from the declared config here and nowhere else.
    """

    def __init__(self, config: ExperimentConfig, *, train: bool, agent_seed: int, logger: Any):
        self.config = config
        self.train = train
        self.logger = logger
        self.rng = np.random.default_rng(agent_seed)
        self.agent_seed = agent_seed
        self.model: QModel | None = None
        self.learner: Learner | None = None
        self.training_updates = 0
        # Observing a transition and applying a gradient are different events.
        # A replay learner below ``min_size``, or on a step that is not a
        # multiple of ``train_every``, observes without updating anything.
        # Counting the two together is what made a stalled run indistinguishable
        # from a converged one.
        self.gradient_steps = 0
        self.round_number: int | None = None
        # Only the identity of the last delivered step is kept, never the step
        # itself: keeping the transition is what used to delay learning by one
        # action.  See docs/05 section 1.10.
        self._delivered_key: tuple[int, int] | None = None
        self._predicted_round_end = False
        self.round_end_mispredictions = 0
        # Shaping is derived from the reward version and shares the learner's
        # gamma, which is what makes the policy-invariance argument hold.
        self.shaping: PotentialShaping | None = build_shaping(config)
        self._assembler = NStepAssembler(config.n_step, config.discount)
        self._reset_round_metrics()
        append_jsonl("agent_setup", {
            "experiment": config.name,
            "main_lines": list(config.lines),
            "runtime_config": asdict(config),
            "terminal_on_truncation": config.terminal_on_truncation,
            "n_step": config.n_step,
            "replay": asdict(config.replay) if config.replay is not None else None,
            "shaping_specification": shaping_specification(config.reward_version),
            "round_offset": self._round_offset(),
            "feature_version": config.feature_version,
            "reward_version": config.reward_version,
            "reward_specification": reward_specification(config.reward_version),
            "exploration_version": config.exploration_version,
            "exploration_specification": exploration_specification(config.exploration_version),
            "learning_rate_schedule": config.learning_rate_schedule,
            "learning_rate_specification": learning_rate_specification(config.learning_rate_schedule),
            "training_rounds": self._training_rounds() if train else None,
            "agent_seed": agent_seed,
            "training": train,
        })

    def warm_up(self) -> None:
        """Build or load the model before the first *timed* callback.

        The official framework gives ``act`` 0.5 s but does not time ``setup``.
        For the NumPy routes construction costs microseconds, but an M4 route
        has to import PyTorch, build the network and run its first forward pass
        -- about three seconds on this machine, which would exceed the timeout
        on step 1 of every round.  Doing it here moves that cost to where it is
        free, and makes a missing evaluation checkpoint fail at setup instead of
        mid-game.  Steady-state inference is unaffected (0.7 ms for the CNN).
        """
        input_dim = state_dimension(self.config.state_encoder)
        self._ensure_initialized(input_dim)
        assert self.model is not None
        self.model.q_values(np.zeros(input_dim, dtype=np.float32))

    def select_action(self, game_state: dict) -> str:
        started = perf_counter()
        state = encode_state(game_state, self.config.state_encoder)
        assert state is not None
        self._ensure_initialized(state.shape[0])
        assert self.learner is not None
        mask = legal_action_mask(game_state)
        epsilon = self._epsilon_for_game_state(game_state)
        action_index = self.learner.select_action(state, mask, epsilon, self.rng)
        action = ACTIONS[action_index]
        position = game_state["self"][3]
        append_jsonl("action", {
            "round": int(game_state["round"]),
            "step": int(game_state["step"]),
            "action": action,
            # The pre-action cell.  Recorded so an offline summary can count
            # distinct visited cells and detect a two-cycle policy; evaluation
            # jobs receive no events, so this is the only positional record.
            "position": [int(position[0]), int(position[1])],
            "selected_action_was_legal": bool(mask[action_index]),
            "epsilon": epsilon,
            "inference_seconds": perf_counter() - started,
        })
        return action

    def observe(self, old_game_state: dict | None, action: str, new_game_state: dict | None, events: list[str]) -> None:
        """Official ``game_events_occurred`` adapter; learns from the step at once.

        One-step Q-learning does not need the next action to form its target, so
        a transition can be learned the moment it is delivered -- and must be, if
        the next action is to be chosen from parameters that already include it.
        ``round_end_reason`` supplies the terminality the framework will not
        announce until afterwards, so nothing has to be held back.

        ``end_round`` then only has to tell the redelivered survivor step (this
        one, already learned) from the fatal step that never arrived here.
        """
        if old_game_state is None or self.model is None or self.learner is None:
            return
        round_number = int(old_game_state["round"])
        if self.round_number != round_number:
            self.round_number = round_number
            self._start_round()
        elif self._predicted_round_end:
            # The previous step was predicted to end the round and did not.  Only
            # the smoke-stage approximation in ``round_end_reason`` can cause this.
            self.round_end_mispredictions += 1
            self.logger.warning("Predicted a round end at step %s that did not happen", self._delivered_key)
        if _is_unusable_action(action):
            # A timeout or a silenced agent error has no six-action index.  The
            # step is skipped rather than mapped onto an action never chosen.
            self.logger.warning("Skipping a transition with unusable action %r", action)
            self._delivered_key = None
            self._predicted_round_end = False
            return
        reason = round_end_reason(new_game_state)
        self._commit(
            EncodedTransition(
                key=(round_number, int(old_game_state["step"])),
                state=encode_state(old_game_state, self.config.state_encoder),
                action_index=ACTIONS.index(action),
                next_state=encode_state(new_game_state, self.config.state_encoder),
                next_legal_mask=legal_action_mask(new_game_state) if new_game_state is not None else None,
                events=list(events),
                potential=self._potential(old_game_state),
                next_potential=self._potential(new_game_state),
            ),
            terminal=self._is_terminal(reason),
        )
        self._delivered_key = (round_number, int(old_game_state["step"]))
        self._predicted_round_end = reason is not ROUND_CONTINUES

    def _is_terminal(self, reason: str | None) -> bool:
        """Map a round-end reason onto a TD target.

        A completed task is a real terminal state: the remaining return is zero,
        so ``target = r`` regardless of configuration.  A step-limit truncation
        is not, and bootstraps unless ``terminal_on_truncation`` overrides it.
        """
        if reason == TASK_COMPLETE:
            return True
        if reason == TRUNCATION:
            return self.config.terminal_on_truncation
        return False

    def end_round(self, last_game_state: dict | None, last_action: str, events: list[str]) -> None:
        """Official ``end_of_round`` adapter; commits the fatal step, if any.

        Two framework paths reach this hook and they are not symmetric.  A
        surviving agent has *already* had this exact step delivered through
        ``game_events_occurred``, because ``send_game_events`` runs before
        ``time_to_stop``; ``observe`` has therefore already learned from it once,
        with the target its end reason implies.  A dead agent has not:
        ``send_game_events`` skips a dead agent, so its final -- and only fatal --
        transition arrives here for the first time.  Committing unconditionally
        double-counts the first case; ignoring this payload drops the death
        penalty in the second.  The ``(round, step)`` key tells them apart.
        """
        if last_game_state is None or self.model is None or self.learner is None:
            return
        key = (int(last_game_state["round"]), int(last_game_state["step"]))
        if key == self._delivered_key:
            # Survived: already learned exactly once in ``observe``.  The events
            # here are a superset (they add SURVIVED_ROUND), which no registered
            # reward version weights, so re-reading them would change nothing.
            if not self._predicted_round_end:
                self.round_end_mispredictions += 1
                self.logger.warning("Round ended at step %s without being predicted", key)
        elif _is_unusable_action(last_action):
            # No six-action index, so the fatal step cannot become a target.
            self.logger.warning("Skipping a final transition with unusable action %r", last_action)
        else:
            # Died: this transition never reached ``game_events_occurred`` and is
            # the only one carrying KILLED_SELF / GOT_KILLED.
            self._commit(
                EncodedTransition(
                    key=key,
                    state=encode_state(last_game_state, self.config.state_encoder),
                    action_index=ACTIONS.index(last_action),
                    next_state=None,
                    next_legal_mask=None,
                    events=list(events),
                    potential=self._potential(last_game_state),
                    next_potential=self._terminal_potential(),
                ),
                terminal=True,
            )
        self._delivered_key = None
        self._predicted_round_end = False
        # A truncated round leaves shorter windows behind; they are bootstrapped
        # from the last observed state rather than silently discarded.
        self._apply_all(self._assembler.flush())
        self.learner.end_round()
        round_number = int(last_game_state["round"])
        metadata = {
            "experiment": self.config.name,
            "feature_version": self.config.feature_version,
            "reward_version": self.config.reward_version,
            "exploration_version": self.config.exploration_version,
            "runtime_config": asdict(self.config),
            "agent_seed": self.agent_seed,
            "round": round_number,
            # Observed transitions, which is what the checkpoint filename has
            # always encoded.  Gradient steps are the separate, smaller number.
            "updates": self.training_updates,
            "gradient_steps": self.gradient_steps,
        }
        latest_path = latest_model_path()
        self.model.save(latest_path, metadata=metadata)
        saved_checkpoint = None
        if round_number % checkpoint_interval() == 0:
            saved_checkpoint = checkpoint_path(self.config, round_number, self.training_updates)
            self.model.save(saved_checkpoint, metadata=metadata)
        append_jsonl("round_end", {
            **metadata,
            "official_reward": self.round_reward,
            "shaping_reward": self.round_shaping_reward,
            # Kept over observes for continuity with every published run.
            # ``mean_abs_td_error_per_gradient_step`` is the one to read: the
            # other divides by steps that never updated anything.
            "mean_abs_td_error": self.round_abs_td_error / max(1, self.round_updates),
            "mean_abs_td_error_per_gradient_step": (
                self.round_gradient_abs_td_error / self.round_gradient_steps
                if self.round_gradient_steps else None
            ),
            "updates_this_round": self.round_updates,
            # The exploration value actually in force at the end of this round.
            # A step-counted schedule cannot be validated against a round budget
            # at config time, so it is recorded per round and checked from the
            # trace instead of trusted.
            "epsilon": self.round_end_epsilon,
            "gradient_steps_this_round": self.round_gradient_steps,
            "gradient_steps": self.gradient_steps,
            # ``learner_step`` is the buffer state as the round ended;
            # ``learner_gradient_step`` is the last step that actually updated.
            "learner_step": self.round_last_step or None,
            "learner_gradient_step": self.round_last_gradient_step or None,
            "model_diagnostics": self._round_model_diagnostics(),
            "round_end_mispredictions": self.round_end_mispredictions,
            "events": self.round_event_counts,
            "checkpoint": str(saved_checkpoint) if saved_checkpoint else None,
            "latest_model": str(latest_path),
        })
        self.logger.info("%s round=%d updates=%d official_reward=%+.1f checkpoint=%s", self.config.name, round_number, self.training_updates, self.round_reward, saved_checkpoint)

    def _commit(self, pending: EncodedTransition, *, terminal: bool, events: list[str] | None = None) -> None:
        """Resolve one transition and hand it to the n-step window.

        Resolving is not the same as learning from it.  With ``n_step = 1`` the
        window emits it immediately and the two coincide, which is why the
        historical behaviour is unchanged; with a longer window it is held until
        its full return is known.  Either way it is emitted exactly once.
        """
        resolved_events = pending.events if events is None else events
        reward = reward_for_events(self.config.reward_version, resolved_events)
        shaping_reward = 0.0
        if self.shaping is not None:
            # phi(terminal) = 0 by construction, so a terminal transition uses
            # the declared terminal potential rather than the observed successor.
            next_potential = self._terminal_potential() if terminal else pending.next_potential
            shaping_reward = self.shaping.shaping_reward(pending.potential, next_potential)
        self._apply_all(self._assembler.push(Transition(
            state=pending.state,
            action_index=pending.action_index,
            reward=reward + shaping_reward,
            next_state=None if terminal else pending.next_state,
            next_legal_mask=None if terminal else pending.next_legal_mask,
            terminal=terminal,
        )))
        # Reward and event counts are accounted per game step, not per learner
        # update, so that a shaped or n-step run stays comparable with A03/n=1.
        self.round_reward += reward
        self.round_shaping_reward += shaping_reward
        for event in resolved_events:
            self.round_event_counts[event] = self.round_event_counts.get(event, 0) + 1
        self.logger.debug("reward=%+.2f shaping=%+.4f terminal=%s", reward, shaping_reward, terminal)

    def _apply_all(self, transitions: list[Transition]) -> None:
        """Send every completed n-step transition to the learner exactly once."""
        assert self.learner is not None
        for transition in transitions:
            td_error = self.learner.observe(transition)
            self.training_updates += 1
            self.round_updates += 1
            self.round_abs_td_error += abs(td_error)
            step = self.learner.step_diagnostics() if hasattr(self.learner, "step_diagnostics") else {}
            if step.get("gradient_applied", True):
                self.gradient_steps += 1
                self.round_gradient_steps += 1
                self.round_gradient_abs_td_error += abs(td_error)
                self._record_model_diagnostics()
            if step:
                self.round_last_step = step
                if step.get("gradient_applied"):
                    self.round_last_gradient_step = step

    def _record_model_diagnostics(self) -> None:
        """Accumulate optional model-side stability diagnostics for this round."""
        if self.model is None or not hasattr(self.model, "training_diagnostics"):
            return
        diagnostics = self.model.training_diagnostics()
        gradient = diagnostics.get("last_gradient_l2_norm")
        if gradient is not None:
            self.round_gradient_norm_sum += float(gradient)
            self.round_gradient_norm_count += 1
        activation = diagnostics.get("last_hidden_zero_fraction")
        if activation is not None:
            self.round_hidden_zero_sum += float(activation)
            self.round_hidden_zero_count += 1
        self.round_gradient_clipped_updates += int(bool(diagnostics.get("last_gradient_was_clipped")))
        self.latest_model_diagnostics = diagnostics

    def _round_model_diagnostics(self) -> dict[str, Any] | None:
        if self.latest_model_diagnostics is None:
            return None
        return {
            "parameter_l2_norm": self.latest_model_diagnostics.get("parameter_l2_norm"),
            "optimizer_steps": self.latest_model_diagnostics.get("optimizer_steps"),
            "mean_gradient_l2_norm": self.round_gradient_norm_sum / max(1, self.round_gradient_norm_count),
            "gradient_clipped_updates": self.round_gradient_clipped_updates,
            "mean_hidden_zero_fraction": self.round_hidden_zero_sum / max(1, self.round_hidden_zero_count),
        }

    def _potential(self, game_state: dict | None) -> float:
        """Return phi(s), or zero when no shaping is configured."""
        return 0.0 if self.shaping is None else self.shaping.potential(game_state)

    def _terminal_potential(self) -> float:
        return 0.0 if self.shaping is None else self.shaping.terminal_potential

    def _ensure_initialized(self, input_dim: int) -> None:
        if self.model is not None:
            return
        continue_training = self.train and os.environ.get("BOMBERMAN_CONTINUE", "0") == "1"
        selected_path = model_path() if not self.train or continue_training else None
        if selected_path is not None and selected_path.exists():
            self.logger.info("Loading trained model from %s", selected_path)
            self.model = load_model(self.config, selected_path)
            if self.model.q_values(np.zeros(input_dim, dtype=np.float32)).shape != (len(ACTIONS),):
                raise ValueError(f"Checkpoint {selected_path} is incompatible with the frozen six-action interface.")
        elif not self.train:
            raise FileNotFoundError(
                f"No evaluation model found at {selected_path}. Set BOMBERMAN_MODEL_PATH for an experiment job, "
                "or package model.npz beside this agent's callbacks.py."
            )
        else:
            self.logger.info("Creating a fresh %s QModel adapter", self.config.network)
            self.model = build_model(self.config, input_dim, seed=self.agent_seed)
        self.learner = build_learner(self.config, self.model, seed=self.agent_seed, training=self.train)

    def _start_round(self) -> None:
        """Reset per-round metrics and guarantee an empty n-step window.

        ``end_round`` always flushes, so a leftover window means a round ended
        without the official hook firing.  Carrying it into the next round would
        mix two rounds' rewards into one return, so it is dropped loudly.
        """
        leftover = self._assembler.pending_count()
        if leftover:
            self.logger.warning("Discarding %d unflushed transitions from the previous round", leftover)
            self._assembler.reset()
        # A round that ended without the official hook leaves these set; the new
        # round must not inherit the previous round's delivery identity.
        self._delivered_key = None
        self._predicted_round_end = False
        self._apply_learning_rate_schedule()
        self._reset_round_metrics()

    def _apply_learning_rate_schedule(self) -> None:
        """Set this round's step size on the model.

        A constant schedule (L00) is left strictly alone rather than reassigned
        to the same value, so every arm run before the schedule existed keeps
        the identical code path and stays bit-comparable.
        """
        if not self.train or self.model is None:
            return
        if self.config.learning_rate_schedule == "L00":
            return
        self.model.learning_rate = learning_rate_for_training_round(
            self.config,
            round_number=self.round_number,
            training_rounds=self._training_rounds(),
        )

    def _reset_round_metrics(self) -> None:
        self.round_updates = 0
        self.round_gradient_steps = 0
        self.round_gradient_abs_td_error = 0.0
        self.round_last_step: dict[str, Any] = {}
        # Separate, because most observes are not gradient steps: with
        # train_every 4 only a quarter are, so the last observe of a round
        # almost never carries loss or Q statistics.  Recording only that one
        # made every round report them as null.
        self.round_last_gradient_step: dict[str, Any] = {}
        self.round_end_epsilon: float | None = None
        self.round_reward = 0.0
        self.round_shaping_reward = 0.0
        self.round_abs_td_error = 0.0
        self.round_event_counts: dict[str, int] = {}
        self.round_gradient_norm_sum = 0.0
        self.round_gradient_norm_count = 0
        self.round_gradient_clipped_updates = 0
        self.round_hidden_zero_sum = 0.0
        self.round_hidden_zero_count = 0
        self.latest_model_diagnostics: dict[str, Any] | None = None

    def _training_rounds(self) -> int:
        """Read the immutable budget supplied by the experiment runner.

        E00 does not mathematically need the value, but recording it for all
        jobs makes action-level exploration logs self-describing.  A manual
        historical E00 invocation remains supported through its fixed epsilon.
        """
        raw = os.environ.get("BOMBERMAN_TRAINING_ROUNDS")
        if raw is None:
            if self.config.exploration_version == "E00":
                return 1
            raise ValueError("BOMBERMAN_TRAINING_ROUNDS is required for a non-constant exploration schedule.")
        try:
            rounds = int(raw)
        except ValueError as exc:
            raise ValueError("BOMBERMAN_TRAINING_ROUNDS must be a positive integer.") from exc
        if rounds < 1:
            raise ValueError("BOMBERMAN_TRAINING_ROUNDS must be a positive integer.")
        return rounds

    def _round_offset(self) -> int:
        """Return how many training rounds preceded this process.

        Every curriculum segment is a separate game process whose round counter
        restarts at 1.  Without an offset a segmented run would silently sit in
        the first, highest-epsilon part of the schedule forever.  A standalone
        job has no preceding rounds and therefore an offset of zero.
        """
        raw = os.environ.get("BOMBERMAN_ROUND_OFFSET", "0")
        try:
            offset = int(raw)
        except ValueError as exc:
            raise ValueError("BOMBERMAN_ROUND_OFFSET must be a non-negative integer.") from exc
        if offset < 0:
            raise ValueError("BOMBERMAN_ROUND_OFFSET must be a non-negative integer.")
        return offset

    def _epsilon_for_game_state(self, game_state: dict) -> float:
        if not self.train:
            return 0.0
        # ``training_updates`` is the cumulative transition count, which the P0
        # invariant ``updates_this_round == act_steps`` makes equal to the
        # environment steps this job has taken.
        by_step = epsilon_for_training_step(self.config, self.training_updates)
        if by_step is not None:
            self.round_end_epsilon = by_step
            return by_step
        self.round_end_epsilon = epsilon_for_training_round(
            self.config,
            round_number=self._round_offset() + int(game_state["round"]),
            training_rounds=self._training_rounds(),
        )
        return epsilon_for_training_round(
            self.config,
            round_number=self._round_offset() + int(game_state["round"]),
            training_rounds=self._training_rounds(),
        )
