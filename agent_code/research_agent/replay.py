"""A fixed-capacity uniform experience replay buffer.

Preallocated NumPy arrays rather than a deque of Python objects: the M4 line
stores 2316-float board states, and a ring of flat arrays keeps sampling to one
fancy-index instead of a per-item copy.  Capacity is declared per route in
``config.ReplayConfig`` and recorded in the run snapshot.

A terminal transition stores no next state.  Rather than keeping a ``None``
column, the buffer keeps the next-state row unused and marks it through
``terminal``; the learner is required to ignore it, which the tests check.

States may be stored either as plain ``float32`` (the default, and what the
handcrafted routes need, since their entries are arbitrary reals) or, for the
spatial routes, as ``uint8`` codes on the 1/20 grid every board channel lives
on.  Two float32 copies of a 2029-long state cost 16.2 KB per transition, so a
100k buffer would be 1.6 GB per job and eight concurrent jobs would not fit on
the training host; the codes cost 4.1 KB and 406 MB.  The encoding is exactly
invertible -- ``_encode_board`` refuses anything off the grid rather than
rounding it -- so this is a memory layout, not an approximation.
"""

from __future__ import annotations

import numpy as np


class ReplayBuffer:
    """Uniformly sampled transitions with an explicit, frozen column layout."""

    def __init__(
        self,
        capacity: int,
        state_dimension: int,
        action_count: int,
        *,
        seed: int = 0,
        quantised_board: int = 0,
        quantisation: int = 0,
    ):
        if capacity < 1 or state_dimension < 1 or action_count < 1:
            raise ValueError("ReplayBuffer needs a positive capacity, state dimension and action count.")
        if quantised_board:
            if not 0 < quantised_board <= state_dimension:
                raise ValueError("quantised_board must be a prefix length of the state vector.")
            if not 1 <= quantisation <= 255:
                raise ValueError("quantisation must be a positive integer step count no larger than 255.")
        self.capacity = int(capacity)
        self.state_dimension = int(state_dimension)
        self.quantised_board = int(quantised_board)
        self.quantisation = int(quantisation)
        self.rng = np.random.default_rng(seed)
        if self.quantised_board:
            tail = self.state_dimension - self.quantised_board
            self.state_codes = np.zeros((self.capacity, self.quantised_board), dtype=np.uint8)
            self.next_state_codes = np.zeros((self.capacity, self.quantised_board), dtype=np.uint8)
            self.state_tails = np.zeros((self.capacity, tail), dtype=np.float32)
            self.next_state_tails = np.zeros((self.capacity, tail), dtype=np.float32)
            # Decoding through a table rather than a division makes every stored
            # code map to one fixed float32, whatever the batch shape.
            self._levels = (np.arange(256, dtype=np.float32) / np.float32(self.quantisation))
            self.states = None
            self.next_states = None
        else:
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
        self._store_state(index, state, into_next=False)
        self.action_indices[index] = action_index
        self.rewards[index] = reward
        self.terminals[index] = terminal
        self.discounts[index] = discount
        if terminal or next_state is None:
            self._store_state(index, None, into_next=True)
            self.next_legal_masks[index] = False
            self.terminals[index] = True
        else:
            self._store_state(index, next_state, into_next=True)
            self.next_legal_masks[index] = next_legal_mask
        self._next_index = (index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)


    def _store_state(self, index: int, vector: np.ndarray | None, *, into_next: bool) -> None:
        if not self.quantised_board:
            target = self.next_states if into_next else self.states
            target[index] = 0.0 if vector is None else vector
            return
        codes = self.next_state_codes if into_next else self.state_codes
        tails = self.next_state_tails if into_next else self.state_tails
        if vector is None:
            codes[index] = 0
            tails[index] = 0.0
            return
        board = np.asarray(vector[:self.quantised_board], dtype=np.float32)
        codes[index] = self._encode_board(board)
        tails[index] = vector[self.quantised_board:]

    def _encode_board(self, board: np.ndarray) -> np.ndarray:
        """Return the uint8 codes of a board, refusing anything off the grid.

        Silently rounding would turn a representation change into an invisible
        loss of precision in every stored transition, so an off-grid value is an
        error naming the offender instead.
        """
        scaled = board * np.float32(self.quantisation)
        codes = np.rint(scaled)
        if not np.all(np.abs(scaled - codes) <= 1e-3):
            worst = float(board[np.argmax(np.abs(scaled - codes))])
            raise ValueError(
                f"Board value {worst!r} is not a multiple of 1/{self.quantisation}; "
                "this buffer cannot store it without loss."
            )
        if codes.min() < 0 or codes.max() > 255:
            raise ValueError("Board values must lie in [0, 255/quantisation] to be stored as uint8 codes.")
        return codes.astype(np.uint8)

    def _read_states(self, indices: np.ndarray, *, from_next: bool) -> np.ndarray:
        if not self.quantised_board:
            return (self.next_states if from_next else self.states)[indices]
        codes = (self.next_state_codes if from_next else self.state_codes)[indices]
        tails = (self.next_state_tails if from_next else self.state_tails)[indices]
        return np.concatenate([self._levels[codes], tails], axis=1)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Return one uniformly drawn batch as a dictionary of column arrays."""
        if batch_size > self._size:
            raise ValueError(f"Cannot sample {batch_size} transitions from a buffer holding {self._size}.")
        indices = self.rng.integers(0, self._size, size=batch_size)
        return {
            "states": self._read_states(indices, from_next=False),
            "action_indices": self.action_indices[indices],
            "rewards": self.rewards[indices],
            "next_states": self._read_states(indices, from_next=True),
            "next_legal_masks": self.next_legal_masks[indices],
            "terminals": self.terminals[indices],
            "discounts": self.discounts[indices],
        }
