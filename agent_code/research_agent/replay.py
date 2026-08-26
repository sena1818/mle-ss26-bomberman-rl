"""A fixed-capacity uniform experience replay buffer.

Preallocated NumPy arrays rather than a deque of Python objects: the M4 line
stores 2316-float board states, and a ring of flat arrays keeps sampling to one
fancy-index instead of a per-item copy.  Capacity is declared per route in
``config.ReplayConfig`` and recorded in the run snapshot.

A terminal transition stores no next state.  Rather than keeping a ``None``
column, the buffer keeps the next-state row unused and marks it through
``terminal``; the learner is required to ignore it, which the tests check.
"""

from __future__ import annotations

import numpy as np


class ReplayBuffer:
    """Uniformly sampled transitions with an explicit, frozen column layout."""

    def __init__(self, capacity: int, state_dimension: int, action_count: int, *, seed: int = 0):
        if capacity < 1 or state_dimension < 1 or action_count < 1:
            raise ValueError("ReplayBuffer needs a positive capacity, state dimension and action count.")
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self.states = np.zeros((self.capacity, state_dimension), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, state_dimension), dtype=np.float32)
        self.action_indices = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_legal_masks = np.zeros((self.capacity, action_count), dtype=bool)
        self.terminals = np.zeros(self.capacity, dtype=bool)
        # gamma**n for the stored transition; n-step transitions of different
        # lengths coexist in one buffer, so the discount travels with the row.
        self.discounts = np.ones(self.capacity, dtype=np.float32)
        self._next_index = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def append(
        self,
        state: np.ndarray,
        action_index: int,
        reward: float,
        next_state: np.ndarray | None,
        next_legal_mask: np.ndarray | None,
        terminal: bool,
        discount: float,
    ) -> None:
        index = self._next_index
        self.states[index] = state
        self.action_indices[index] = action_index
        self.rewards[index] = reward
        self.terminals[index] = terminal
        self.discounts[index] = discount
        if terminal or next_state is None:
            self.next_states[index] = 0.0
            self.next_legal_masks[index] = False
            self.terminals[index] = True
        else:
            self.next_states[index] = next_state
            self.next_legal_masks[index] = next_legal_mask
        self._next_index = (index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Return one uniformly drawn batch as a dictionary of column arrays."""
        if batch_size > self._size:
            raise ValueError(f"Cannot sample {batch_size} transitions from a buffer holding {self._size}.")
        indices = self.rng.integers(0, self._size, size=batch_size)
        return {
            "states": self.states[indices],
            "action_indices": self.action_indices[indices],
            "rewards": self.rewards[indices],
            "next_states": self.next_states[indices],
            "next_legal_masks": self.next_legal_masks[indices],
            "terminals": self.terminals[indices],
            "discounts": self.discounts[indices],
        }
