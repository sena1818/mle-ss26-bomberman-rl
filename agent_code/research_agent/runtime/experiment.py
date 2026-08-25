"""Shared ExperimentRuntime: the only route-agnostic callback implementation."""

from __future__ import annotations

import os
from dataclasses import asdict
from time import perf_counter
from typing import Any

import numpy as np

from ..artifacts import append_jsonl, checkpoint_interval, checkpoint_path, latest_model_path, model_path, run_id
from ..config import ACTIONS, ExperimentConfig
from ..learners import Learner, Transition, build_learner
from ..models import QModel, build_model, load_model
from ..state import encode_state, legal_action_mask


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
}
DEATH_PENALTIES = {
    "A00": 0.0,
    # The official framework emits both KILLED_SELF and GOT_KILLED for one
    # self-inflicted death.  This is deliberately one penalty per death, not
    # one penalty per event label.
    "A01": -5.0,
    "A02": -5.0,
}


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


class ExperimentRuntime:
    """Deep module hiding route-specific model and learner adapters.

    Its interface is intentionally limited to the official lifecycle:
    ``select_action``, ``observe``, and ``end_round``.  Algorithm-specific
    state never reaches callbacks: a future SarsaLearner owns its action cache,
    while a future DqnLearner owns replay and target-network state.
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
        self.round_number: int | None = None
        self._reset_round_metrics()
        append_jsonl("agent_setup", {
            "experiment": config.name,
            "runtime_config": asdict(config),
            "feature_version": config.feature_version,
            "reward_version": config.reward_version,
            "agent_seed": agent_seed,
            "training": train,
        })

    def select_action(self, game_state: dict) -> str:
        started = perf_counter()
        state = encode_state(game_state, self.config.state_encoder)
        assert state is not None
        self._ensure_initialized(state.shape[0])
        assert self.learner is not None
        mask = legal_action_mask(game_state)
        action_index = self.learner.select_action(state, mask, self.config.epsilon if self.train else 0.0, self.rng)
        action = ACTIONS[action_index]
        append_jsonl("action", {
            "round": int(game_state["round"]),
            "step": int(game_state["step"]),
            "action": action,
            "selected_action_was_legal": bool(mask[action_index]),
            "inference_seconds": perf_counter() - started,
        })
        return action

    def observe(self, old_game_state: dict | None, action: str, new_game_state: dict | None, events: list[str], *, terminal: bool) -> None:
        if old_game_state is None or self.model is None or self.learner is None:
            return
        round_number = int(old_game_state["round"])
        if self.round_number != round_number:
            self.round_number = round_number
            self._reset_round_metrics()
        state = encode_state(old_game_state, self.config.state_encoder)
        next_state = None if terminal else encode_state(new_game_state, self.config.state_encoder)
        next_mask = None if terminal else legal_action_mask(new_game_state)
        reward = reward_for_events(self.config.reward_version, events)
        td_error = self.learner.observe(Transition(
            state=state,
            action_index=ACTIONS.index(action),
            reward=reward,
            next_state=next_state,
            next_legal_mask=next_mask,
            terminal=terminal,
        ))
        self.training_updates += 1
        self.round_updates += 1
        self.round_reward += reward
        self.round_abs_td_error += abs(td_error)
        for event in events:
            self.round_event_counts[event] = self.round_event_counts.get(event, 0) + 1
        self.logger.debug("reward=%+.2f td_error=%+.4f", reward, td_error)

    def end_round(self, last_game_state: dict | None, last_action: str, events: list[str]) -> None:
        if last_game_state is None or self.model is None or self.learner is None:
            return
        self.observe(last_game_state, last_action, None, events, terminal=True)
        self.learner.end_round()
        round_number = int(last_game_state["round"])
        metadata = {
            "experiment": self.config.name,
            "feature_version": self.config.feature_version,
            "reward_version": self.config.reward_version,
            "runtime_config": asdict(self.config),
            "agent_seed": self.agent_seed,
            "round": round_number,
            "updates": self.training_updates,
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
            "mean_abs_td_error": self.round_abs_td_error / max(1, self.round_updates),
            "updates_this_round": self.round_updates,
            "events": self.round_event_counts,
            "checkpoint": str(saved_checkpoint) if saved_checkpoint else None,
            "latest_model": str(latest_path),
        })
        self.logger.info("%s round=%d updates=%d official_reward=%+.1f checkpoint=%s", self.config.name, round_number, self.training_updates, self.round_reward, saved_checkpoint)

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
        self.learner = build_learner(self.config, self.model)

    def _reset_round_metrics(self) -> None:
        self.round_updates = 0
        self.round_reward = 0.0
        self.round_abs_td_error = 0.0
        self.round_event_counts: dict[str, int] = {}
